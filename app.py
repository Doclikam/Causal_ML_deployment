# Full updated app.py with diagnostics + safe Xpatient builder

import os
from io import BytesIO
from typing import Optional

import joblib
import matplotlib.pyplot as plt  # kept if you want later
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from utils.infer import infer_new_patient_fixed

# ----------------- STYLE & CONFIG -----------------
COLOR_RT = "#1f77b4"        # calm blue
COLOR_CHEMO = "#2ca02c"     # medical green
COLOR_BENEFIT = "#1f77b4"   # blue for benefit
COLOR_HARM = "#d62728"      # red for harm

st.set_page_config(
    page_title="H&N Chemo-RT Decision Aid",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Head & Neck Cancer – Chemo-RT Decision Aid")

st.markdown(
    """
This tool uses data-driven models to compare **radiotherapy alone (RT)** with  
**concurrent chemoradiotherapy (Chemo-RT)** for an individual patient.

It estimates:
- The chance of being **alive and well (no death from any cause)** over time
- The **extra time alive and well** that Chemo-RT may offer, in **months**
- How similar patients in the dataset have done

> ⚠️ These are **model-based estimates from retrospective data**.  
> They are intended to support – not replace – clinical judgment and guidelines.
"""
)

# Defaults for loading artifacts
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/Doclikam/Causal_ML_deployment/main/outputs/"
DEFAULT_OUTDIR = "outputs"
INTERVAL_DAYS = 30

# ----------------- SIDEBAR: SETTINGS -----------------
st.sidebar.header("Time horizon")

max_period_months = st.sidebar.number_input(
    "Max follow-up (months) for curves",
    value=60,
    min_value=6,
    max_value=156,
    step=6
)

rmst_horizon_months = st.sidebar.number_input(
    "Key time point for summary (months)",
    value=36,
    min_value=6,
    max_value=int(max_period_months),
    step=6
)

# Advanced data/model settings (hidden by default for clinicians)
BASE_URL = DEFAULT_BASE_URL
OUTDIR = DEFAULT_OUTDIR

with st.sidebar.expander("Advanced: data source & model files", expanded=False):
    st.markdown(
        "These options are mainly for developers. "
        "Defaults should work for routine use."
    )
    BASE_URL = st.text_input(
        "BASE_URL (raw GitHub path to outputs/)",
        value=DEFAULT_BASE_URL
    )
    OUTDIR = st.text_input(
        "Local outputs folder (if model files are stored locally)",
        value=DEFAULT_OUTDIR
    )
    st.markdown(
        "**Note**: The app first looks in the local folder, then falls back to GitHub."
    )

# ----------------- HELPERS -----------------
def load_csv_with_fallback(filename: str) -> Optional[pd.DataFrame]:
    """Try local OUTDIR/filename; if not found, try BASE_URL/filename."""
    local_path = os.path.join(OUTDIR, filename)
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass

    if BASE_URL:
        url = BASE_URL.rstrip("/") + "/" + filename
        try:
            return pd.read_csv(url)
        except Exception:
            return None
    return None


def load_joblib_with_fallback(filename: str):
    """Try local OUTDIR/filename; if not found, try BASE_URL/filename via requests."""
    local_path = os.path.join(OUTDIR, filename)
    if os.path.exists(local_path):
        try:
            return joblib.load(local_path)
        except Exception:
            pass

    if BASE_URL:
        url = BASE_URL.rstrip("/") + "/" + filename
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return joblib.load(BytesIO(r.content))
        except Exception:
            return None
    return None


def compute_rmst_from_survival(surv_df: pd.DataFrame, horizon_months: int) -> dict:
    """
    Compute average time alive & event-free (RMST) under RT vs Chemo-RT
    up to a given horizon, using step-function approximation.
    """
    if surv_df is None or surv_df.empty:
        return {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}

    h_days = horizon_months * INTERVAL_DAYS
    s = surv_df.sort_values("days").copy()

    # clip to horizon
    s = s[s["days"] <= h_days].copy()
    if s.empty:
        return {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}

    times = s["days"].values
    S_c_end = s["S_control"].values
    S_t_end = s["S_treat"].values

    t0 = np.concatenate(([0.0], times[:-1]))
    dt = times - t0

    S_c_start = np.concatenate(([1.0], S_c_end[:-1]))
    S_t_start = np.concatenate(([1.0], S_t_end[:-1]))

    rmst_c = np.sum(S_c_start * dt)
    rmst_t = np.sum(S_t_start * dt)

    return {
        "rmst_treat": rmst_t / 30.0,     # days → months
        "rmst_control": rmst_c / 30.0,
        "delta": (rmst_t - rmst_c) / 30.0
    }


def interpret_delta_months(delta_m: float) -> str:
    """Clinician-friendly explanation of ΔRMST magnitude and direction."""
    if np.isnan(delta_m):
        return "The model could not compute a reliable difference between RT and Chemo-RT for this patient."

    mag = abs(delta_m)
    if mag < 0.5:
        size = "very small"
    elif mag < 2:
        size = "small"
    elif mag < 4:
        size = "moderate"
    else:
        size = "large"

    if delta_m > 0:
        return (f"Chemo-RT is estimated to give a **{size} gain** of about "
                f"**{delta_m:.1f} extra months alive and well** by the chosen time point.")
    elif delta_m < 0:
        return (f"Chemo-RT is estimated to be **worse overall**, with about "
                f"**{mag:.1f} fewer months alive and well** by the chosen time point.")
    else:
        return "The model predicts essentially no difference in time alive and well between RT and Chemo-RT."


def describe_cate_table(cates: dict) -> str:
    """Short narrative from per-horizon absolute risk differences (Chemo-RT − RT)."""
    vals = [(h, v["CATE"]) for h, v in cates.items()
            if v.get("CATE") is not None and not np.isnan(v.get("CATE"))]
    if not vals:
        return "The model could not provide reliable horizon-specific risk differences for this patient."

    parsed = []
    for h, v in vals:
        try:
            mh = float(h)
        except Exception:
            mh = np.nan
        parsed.append((mh, v))

    arr = np.array([v for _, v in parsed])
    mh_arr = np.array([mh for mh, _ in parsed])

    idx_benefit = np.argmin(arr)
    idx_harm = np.argmax(arr)

    best_h = mh_arr[idx_benefit]
    best_v = arr[idx_benefit]
    worst_h = mh_arr[idx_harm]
    worst_v = arr[idx_harm]

    text = []
    if best_v < 0:
        text.append(
            f"- **Largest estimated benefit**: around **{best_h:.0f} months**, "
            f"Chemo-RT is predicted to **reduce the event risk by about "
            f"{abs(best_v)*100:.1f} percentage points**."
        )
    if worst_v > 0:
        text.append(
            f"- **Largest estimated potential harm**: around **{worst_h:.0f} months**, "
            f"Chemo-RT is predicted to **increase the event risk by about "
            f"{worst_v*100:.1f} percentage points**."
        )

    if not text:
        text.append(
            "Across the modelled time points, Chemo-RT appears roughly risk-neutral "
            "(no horizon stands out as clearly better or worse)."
        )

    text.append(
        "\nNegative numbers mean **fewer events** with Chemo-RT; "
        "positive numbers mean **more events**."
    )

    return "\n".join(text)


def generate_patient_summary(patient, rmst_res, surv_df, horizon_months: int) -> str:
    """
    Short, patient-friendly paragraph (2–3 sentences).
    """
    rmst_t = rmst_res.get("rmst_treat", np.nan)
    rmst_c = rmst_res.get("rmst_control", np.nan)
    delta_m = rmst_res.get("delta", np.nan)

    p_rt, p_chemo = np.nan, np.nan
    if surv_df is not None and not surv_df.empty:
        s = surv_df.sort_values("days").copy()
        h_days = horizon_months * 30.0
        s_h = s[s["days"] <= h_days]
        if not s_h.empty:
            p_rt = s_h["S_control"].iloc[-1]
            p_chemo = s_h["S_treat"].iloc[-1]

    if np.isnan(delta_m):
        main = (
            f"For a patient similar to this one, the model could not reliably estimate the "
            f"difference between radiotherapy alone and chemoradiotherapy over {horizon_months} months."
        )
    else:
        if delta_m > 0:
            phrasing = "a small benefit"
        elif delta_m < 0:
            phrasing = "a small disadvantage"
        else:
            phrasing = "no clear difference"

        main = (
            f"For a patient like this, over about {horizon_months} months the model suggests "
            f"{phrasing} with chemoradiotherapy compared with radiotherapy alone, "
            f"equivalent to about {delta_m:+.1f} months difference in time alive and well."
        )

    surv_sentence = ""
    if not np.isnan(p_rt) and not np.isnan(p_chemo):
        surv_sentence = (
            f" By {horizon_months} months, the estimated chance of being alive and well is "
            f"around {p_rt*100:.0f}% with radiotherapy alone and {p_chemo*100:.0f}% with chemoradiotherapy."
        )

    disclaimer = (
        " These figures come from statistical models based on previous patients and are approximate. "
        "They are meant to support, not replace, a discussion between you and your care team."
    )

    return main + surv_sentence + disclaimer


def build_print_summary(patient, rmst_res, surv_df, horizon_months, cates) -> str:
    """Text summary for download / notes."""
    lines = []
    lines.append("Head & Neck Cancer: RT vs Chemo-RT Summary")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Patient snapshot (as entered):")
    for k, v in patient.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")

    rmst_t = rmst_res.get("rmst_treat", np.nan)
    rmst_c = rmst_res.get("rmst_control", np.nan)
    delta_m = rmst_res.get("delta", np.nan)

    lines.append(f"Average time alive & well (up to {horizon_months} months):")
    lines.append(f"  - RT alone:    {rmst_c:.1f} months" if not np.isnan(rmst_c) else "  - RT alone:    N/A")
    lines.append(f"  - Chemo-RT:    {rmst_t:.1f} months" if not np.isnan(rmst_t) else "  - Chemo-RT:    N/A")
    lines.append(f"  - Difference (Chemo-RT − RT): {delta_m:+.1f} months" if not np.isnan(delta_m) else "  - Difference:  N/A")
    lines.append("")

    p_rt, p_chemo = np.nan, np.nan
    if surv_df is not None and not surv_df.empty:
        s = surv_df.sort_values("days").copy()
        h_days = horizon_months * 30.0
        s_h = s[s["days"] <= h_days]
        if not s_h.empty:
            p_rt = s_h["S_control"].iloc[-1]
            p_chemo = s_h["S_treat"].iloc[-1]

    if not np.isnan(p_rt) and not np.isnan(p_chemo):
        lines.append(f"Estimated probability of being alive & well at {horizon_months} months:")
        lines.append(f"  - RT alone:    {p_rt*100:.0f}%")
        lines.append(f"  - Chemo-RT:    {p_chemo*100:.0f}%")
        lines.append("")

    if cates:
        lines.append("Change in event risk at each time point (Chemo-RT − RT):")
        for h, v in sorted(cates.items(), key=lambda kv: float(kv[0])):
            cate = v.get("CATE")
            if cate is None or np.isnan(cate):
                continue
            lines.append(f"  - {h} months: {cate*100:.1f} percentage points")
        lines.append("")

    lines.append("Patient-facing explainer:")
    lines.append("")
    lines.append(generate_patient_summary(patient, rmst_res, surv_df, horizon_months))
    lines.append("")

    lines.append("Notes:")
    lines.append("  - Estimates are model-based and derived from retrospective data.")
    lines.append("  - Interpret together with clinical judgment, comorbidities, and patient preferences.")
    lines.append("")

    return "\n".join(lines)


# --- Diagnostics helper: robust Xpatient builder + safe scaling ---
def build_and_scale_Xpatient(patient_dict, patient_columns_obj, scaler_obj=None):
    """
    Build Xpatient DataFrame from canonical patient columns artifact and the patient dict.
    Attempt safe scaling on numeric subset. Returns (Xp, status_str, debug_dict).
    """
    debug = {}
    if patient_columns_obj is None:
        return None, "patient_columns_missing", {"error": "patient_columns missing"}

    # Extract canonical columns list
    if isinstance(patient_columns_obj, (list, tuple, pd.Series, np.ndarray)):
        pcols = list(patient_columns_obj)
    elif isinstance(patient_columns_obj, dict):
        # common stored shapes: {'columns': [...]} or mapping
        if "columns" in patient_columns_obj:
            pcols = list(patient_columns_obj["columns"])
        else:
            pcols = list(patient_columns_obj.keys())
    elif isinstance(patient_columns_obj, pd.DataFrame):
        pcols = patient_columns_obj.iloc[:, 0].astype(str).tolist()
    else:
        pcols = list(patient_columns_obj)

    Xp = pd.DataFrame(np.zeros((1, len(pcols))), columns=pcols)

    # Fill direct matches and simple one-hot patterns root_value
    for c in pcols:
        if c in patient_dict:
            Xp.at[0, c] = patient_dict[c]
        else:
            # fallback: try detect 'root_value' one-hot pattern
            if "_" in c:
                root, tail = c.split("_", 1)
                if root in patient_dict and str(patient_dict[root]) == tail:
                    Xp.at[0, c] = 1.0

    debug["n_cols"] = len(pcols)

    # Safe scaling
    if scaler_obj is not None:
        scaler_feat_names = None
        if hasattr(scaler_obj, "feature_names_in_"):
            try:
                scaler_feat_names = list(scaler_obj.feature_names_in_)
            except Exception:
                scaler_feat_names = None

        # numeric candidates
        numeric_mask = Xp.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
        numeric_cols = Xp.columns[numeric_mask].tolist()
        debug["numeric_cols_candidate"] = numeric_cols
        debug["scaler_feature_names_present"] = bool(scaler_feat_names)

        if scaler_feat_names:
            num_to_scale = [c for c in numeric_cols if c in scaler_feat_names]
        else:
            # if scaler has mean_/scale_ sized like numeric_cols, assume mapping by order
            try:
                msz = len(getattr(scaler_obj, "mean_", []))
            except Exception:
                msz = 0
            if msz == len(numeric_cols) and msz > 0:
                num_to_scale = numeric_cols.copy()
            else:
                num_to_scale = numeric_cols.copy()  # fallback

        debug["numeric_cols_to_scale"] = num_to_scale

        if num_to_scale:
            try:
                Xp[num_to_scale] = scaler_obj.transform(Xp[num_to_scale])
                debug["scaled_ok"] = True
            except Exception as e:
                debug["scaled_ok"] = False
                debug["scale_exception"] = str(e)
        else:
            debug["scaled_ok"] = False
            debug["scale_exception"] = "no numeric columns matched scaler; skipping transform"
    else:
        debug["scaled_ok"] = False
        debug["scale_exception"] = "no scaler provided"

    return Xp, "built", debug


# ----------------- LAYOUT: TABS -----------------
tab_patient, tab_timecourse, tab_insights = st.tabs(
    ["👤 Patient decision aid", "⏱ Effect over time", "🧠 Patterns from data"]
)

# ==========================================================
# ---------- TAB 1: PATIENT DECISION AID ----------
# ==========================================================
with tab_patient:
    st.subheader("1. Enter patient details")

    with st.expander("What this page provides", expanded=True):
        st.markdown("""
- An estimate of **how much benefit** a patient like this may gain from **Chemo-RT vs RT alone**  
- The **extra time alive and well** (in months) and **difference in survival probability** at a chosen time  
- Optional detailed plots and subgroup summaries for deeper exploration
        """)

    # patient form
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age (years)", value=62, min_value=18, max_value=99)
            sex = st.selectbox("Sex", ["Male", "Female", "Missing"], index=1)
            ecog_ps = st.selectbox(
                "ECOG performance status",
                [0, 1, 2, 3],
                index=0,
                help="0 = fully active; 1 = restricted in strenuous activity; 2–3 = limited / often bedridden."
            )

        with c2:
            primary_site_group = st.selectbox(
                "Primary site",
                ["Oropharynx", "Nasopharynx", "Other_HNC", "Missing"],
                index=0
            )
            pathology_group = st.selectbox(
                "Histology",
                ["SCC", "Other_epithelial", "Other_rare", "Missing"],
                index=0
            )
            smoking_status_clean = st.selectbox(
                "Smoking status",
                ["Current", "Ex-Smoker", "Non-Smoker", "Unknown", "Missing"],
                index=1
            )
            smoking_py_clean = st.number_input(
                "Smoking pack-years (approx.)",
                min_value=0.0,
                max_value=500.0,
                value=20.0,
                step=1.0,
                help="Packs per day × years smoked (0 if never smoked)."
            )

        with c3:
            hpv_clean = st.selectbox(
                "HPV status",
                ["HPV_Positive", "HPV_Negative", "HPV_Unknown", "Missing"],
                index=0
            )
            stage = st.selectbox(
                "Overall stage (AJCC-like)",
                ["I", "II", "III", "IV", "Missing"],
                index=2
            )

        st.markdown("**TNM classification**")

        c4, c5, c6 = st.columns(3)
        with c4:
            t_cat = st.selectbox("T category", ["T1", "T2", "T3", "T4", "Tx"], index=1)
        with c5:
            n_cat = st.selectbox("N category", ["N0", "N1", "N2", "N3", "Nx"], index=0)
        with c6:
            m_cat = st.selectbox("M category", ["M0", "M1", "Mx"], index=0)

        treatment = st.selectbox(
            "Planned strategy (both options will be modelled)",
            options=[0, 1],
            format_func=lambda x: "RT alone" if x == 0 else "Chemo-RT",
            index=0
        )

        submitted = st.form_submit_button("Estimate outcomes")

    if submitted:
        patient = {
            "age": age,
            "sex": sex,
            "primary_site_group": primary_site_group,
            "pathology_group": pathology_group,
            "hpv_clean": hpv_clean,
            "stage": stage,
            "t": t_cat,
            "n": n_cat,
            "m": m_cat,
            "ecog_ps": ecog_ps,
            "smoking_status_clean": smoking_status_clean,
            "smoking_py_clean": smoking_py_clean,
            "treatment": treatment
        }

        with st.spinner("Running models for this patient..."):
            # Call infer with return_raw=True to get debug info
            try:
                out_dbg = infer_new_patient_fixed(
                    patient_data=patient,
                    outdir=OUTDIR,
                    base_url=BASE_URL,
                    max_period_override=int(max_period_months),
                    return_raw=True
                )
            except Exception as e:
                st.error("infer_new_patient_fixed raised an exception.")
                st.exception(e)
                out_dbg = {"errors": {"infer_call": str(e)}, "debug": {}}

            # unify out vs out_dbg
            out = {k: out_dbg.get(k) for k in ["survival_curve", "CATEs", "errors"]}
            # ensure errors always present
            out["errors"] = out.get("errors", {}) or {}

            # show technical notes (filter scaler note if desired)
            raw_errors = out.get("errors", {})
            filtered_errors = {k: v for k, v in raw_errors.items() if k not in ["scaler"]}
            if filtered_errors:
                with st.expander("Technical notes from modelling pipeline", expanded=False):
                    for k, msg in filtered_errors.items():
                        st.write(f"- **{k}**: {msg}")

            # expose debug keys
            debug_block = out_dbg.get("debug", {}) if isinstance(out_dbg, dict) else {}
            st.markdown("### Diagnostics (developer)")
            st.write("infer() debug keys:")
            try:
                st.json(list(debug_block.keys()))
            except Exception:
                st.write(list(debug_block.keys()))

            st.write("Top-level errors:")
            try:
                st.json(out.get("errors", {}))
            except Exception:
                st.write(out.get("errors", {}))

            # show artifact sources if present
            if isinstance(debug_block, dict) and debug_block.get("artifact_sources"):
                st.write("Artifact sources (from infer debug):")
                st.json(debug_block["artifact_sources"])
            else:
                st.info("No artifact sources returned in debug. infer may not have loaded artifacts or return_raw was False earlier.")

            # --- Attempt to load canonical patient_columns & patient_scaler for diagnostics ---
            # Try joblib first, then csv fallback for patient_columns
            patient_cols_art = load_joblib_with_fallback("causal_patient_columns.joblib")
            if patient_cols_art is None:
                patient_cols_art = load_csv_with_fallback("causal_patient_columns.csv")

            patient_scaler_art = load_joblib_with_fallback("causal_patient_scaler.joblib")
            # fallback: maybe scaler is named differently; try 'pp_scaler' artifact too
            if patient_scaler_art is None:
                patient_scaler_art = load_joblib_with_fallback("pp_scaler.joblib")

            st.markdown("**Detailed Xpatient / scaler diagnostics**")

            # show first canonical columns so developer can inspect
            try:
                if patient_cols_art is None:
                    st.write("causal_patient_columns artifact: MISSING")
                else:
                    if isinstance(patient_cols_art, (list, tuple, pd.Series, np.ndarray)):
                        canon = list(patient_cols_art)
                    elif isinstance(patient_cols_art, dict):
                        canon = patient_cols_art.get("columns", list(patient_cols_art.keys()))
                    elif isinstance(patient_cols_art, pd.DataFrame):
                        canon = patient_cols_art.iloc[:, 0].astype(str).tolist()
                    else:
                        canon = list(patient_cols_art)
                    st.write("First 80 canonical patient columns (expected by model):")
                    st.write(canon[:80])
            except Exception as e:
                st.write("Error showing canonical columns:", e)

            # build and scale Xpatient
            Xp, status, xdbg = build_and_scale_Xpatient(patient, patient_cols_art, patient_scaler_art)
            st.write("Xpatient build status:", status)
            try:
                st.json(xdbg)
            except Exception:
                st.write(xdbg)

            if Xp is None:
                st.error("Could not build Xpatient (patient_columns missing or incompatible).")
            else:
                st.write("Xpatient (first 80 columns):")
                try:
                    st.dataframe(Xp.iloc[:, :80].T, use_container_width=True)
                except Exception:
                    st.write(Xp.iloc[:, :80].to_dict())

                # numeric stats
                st.write("Xpatient stats: min / max / mean")
                try:
                    st.write(float(Xp.values.min()), float(Xp.values.max()), float(Xp.values.mean()))
                except Exception as e:
                    st.write("stats error:", e)

                if np.allclose(Xp.values, 0):
                    st.error("Xpatient is ALL ZEROS — identical predictions will follow.")
                    st.info("Causes: mismatch between form names and canonical one-hot columns, or canonical columns expect e.g. 'sex_Male' but your form didn't set it. Inspect the canonical columns shown above.")
                else:
                    st.success("Xpatient not all zero — OK.")

            # now continue with regular UI outputs using `out`
            surv = out.get("survival_curve")
            cates = out.get("CATEs", {})

            # ---------- SURVIVAL & SUMMARY ----------
            st.subheader("2. Summary at your chosen time point")

            if surv is None or surv.empty:
                st.warning("Survival curve could not be computed for this patient.")
                rmst_res = {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}
                p_rt = p_chemo = np.nan
            else:
                rmst_res = compute_rmst_from_survival(surv, rmst_horizon_months)
                rmst_t = rmst_res["rmst_treat"]
                rmst_c = rmst_res["rmst_control"]
                delta_m = rmst_res["delta"]

                # survival probabilities at horizon
                s = surv.sort_values("days").copy()
                h_days = rmst_horizon_months * 30.0
                s_h = s[s["days"] <= h_days]
                if not s_h.empty:
                    p_rt = s_h["S_control"].iloc[-1]
                    p_chemo = s_h["S_treat"].iloc[-1]
                else:
                    p_rt = p_chemo = np.nan

                label = categorize_benefit(delta_m, p_rt, p_chemo)

                # Summary metrics card
                c_sum1, c_sum2, c_sum3, c_sum4 = st.columns(4)
                c_sum1.metric(
                    "Average time alive & well\nRT alone",
                    f"{rmst_c:.1f} m" if not np.isnan(rmst_c) else "N/A"
                )
                c_sum2.metric(
                    "Average time alive & well\nChemo-RT",
                    f"{rmst_t:.1f} m" if not np.isnan(rmst_t) else "N/A"
                )
                c_sum3.metric(
                    "Extra time with Chemo-RT",
                    f"{delta_m:+.1f} m" if not np.isnan(delta_m) else "N/A"
                )
                if not np.isnan(p_rt) and not np.isnan(p_chemo):
                    diff_surv = (p_chemo - p_rt) * 100.0
                    c_sum4.metric(
                        f"Difference in chance alive & well\nat {rmst_horizon_months} m",
                        f"{diff_surv:+.1f} %pts"
                    )
                else:
                    c_sum4.metric(
                        f"Difference in chance alive & well\nat {rmst_horizon_months} m",
                        "N/A"
                    )

                st.markdown(
                    f"""
<div style="
  border-radius: 8px;
  padding: 10px 14px;
  border: 1px solid #e2e8f0;
  background-color: #f8fafc;">
<b>Model interpretation:</b> {label}<br>
{interpret_delta_months(delta_m)}
</div>
                    """,
                    unsafe_allow_html=True
                )

            # Simple vs advanced view
            st.markdown("---")
            simple_view = st.toggle("Show advanced model details", value=False)

            if surv is not None and not surv.empty:
                surv_plot = surv.copy()
                surv_plot["months"] = surv_plot["days"] / 30.0

                st.subheader("3. Survival over time (RT vs Chemo-RT)")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=surv_plot["months"], y=surv_plot["S_control"],
                    mode="lines+markers",
                    name="RT alone",
                    line=dict(color=COLOR_RT),
                    marker=dict(color=COLOR_RT)
                ))
                fig.add_trace(go.Scatter(
                    x=surv_plot["months"], y=surv_plot["S_treat"],
                    mode="lines+markers",
                    name="Chemo-RT",
                    line=dict(color=COLOR_CHEMO),
                    marker=dict(color=COLOR_CHEMO)
                ))
                fig.update_layout(
                    xaxis_title="Time since RT start (months)",
                    yaxis_title="Probability alive & well",
                    yaxis=dict(range=[0, 1]),
                    legend_title="Strategy"
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Table of modelled survival over time (first few rows)"):
                    st.dataframe(
                        surv_plot[["period", "months", "S_control", "S_treat"]].head(),
                        use_container_width=True
                    )

            # (rest of UI remains unchanged — CATE plots, subgroup scorecard, etc.)
            # For brevity we reuse the rest of your previously defined UI code...
            # (The rest of your original code continues unchanged from here.)
            # ---------- ADVANCED DETAILS ----------
            if simple_view:
                # still show patient-facing summary & download
                st.subheader("4. Patient-friendly paragraph")
                summary_text = generate_patient_summary(
                    patient=patient,
                    rmst_res=rmst_res,
                    surv_df=surv,
                    horizon_months=rmst_horizon_months
                )
                st.markdown(
                    f"""
<div style="
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #e2e8f0;
    background-color: #f8fafc;
    ">
<p style="margin: 0; font-size: 0.95rem;">
{summary_text}
</p>
</div>
                    """,
                    unsafe_allow_html=True
                )

                printable_summary = build_print_summary(
                    patient=patient,
                    rmst_res=rmst_res,
                    surv_df=surv,
                    horizon_months=rmst_horizon_months,
                    cates=cates
                )

                st.download_button(
                    label="📄 Download 1-page summary (text)",
                    data=printable_summary,
                    file_name="hnc_treatment_summary.txt",
                    mime="text/plain"
                )

            else:
                # Advanced: CATEs, subgroup scorecard, etc.
                st.subheader("4. Change in chance of being alive & well at different times")

                if not cates:
                    st.info("No time-specific risk differences available for this patient.")
                else:
                    rows = []
                    for h, v in cates.items():
                        cate = v.get("CATE")
                        err = v.get("error")
                        try:
                            mh = float(h)
                        except Exception:
                            mh = np.nan
                        rows.append({
                            "horizon_label": str(h),
                            "horizon_months": mh,
                            "CATE": cate,
                            "CATE_percent": cate * 100 if cate is not None and not np.isnan(cate) else np.nan,
                            "error": err
                        })
                    df_cate = pd.DataFrame(rows).sort_values("horizon_months")

                    valid = df_cate[df_cate["CATE"].notna()].copy()
                    errors_cate = df_cate[df_cate["CATE"].isna() & df_cate["error"].notna()]

                    if not valid.empty:
                        colors = [
                            COLOR_BENEFIT if x < 0 else COLOR_HARM
                            for x in valid["CATE_percent"]
                        ]

                        fig_c = go.Figure()
                        fig_c.add_trace(go.Bar(
                            x=valid["horizon_months"],
                            y=valid["CATE_percent"],
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in valid["CATE_percent"]],
                            textposition="outside"
                        ))
                        fig_c.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_c.update_layout(
                            xaxis_title="Time point (months)",
                            yaxis_title="Chemo-RT − RT (percentage-point change)",
                            title="Change in chance of being alive & well"
                        )
                        st.plotly_chart(fig_c, use_container_width=True)

                        st.markdown(describe_cate_table(cates))

                    if not errors_cate.empty:
                        with st.expander("Technical issues at some time points"):
                            st.table(errors_cate[["horizon_label", "error"]])

                st.subheader("5. Patient-friendly paragraph & download")

                summary_text = generate_patient_summary(
                    patient=patient,
                    rmst_res=rmst_res,
                    surv_df=surv,
                    horizon_months=rmst_horizon_months
                )
                st.markdown(
                    f"""
<div style="
    border-radius: 8px;
    padding: 12px 16px;
    border: 1px solid #e2e8f0;
    background-color: #f8fafc;
    ">
<p style="margin: 0; font-size: 0.95rem;">
{summary_text}
</p>
</div>
                    """,
                    unsafe_allow_html=True
                )

                printable_summary = build_print_summary(
                    patient=patient,
                    rmst_res=rmst_res,
                    surv_df=surv,
                    horizon_months=rmst_horizon_months,
                    cates=cates
                )

                st.download_button(
                    label="📄 Download 1-page summary (text)",
                    data=printable_summary,
                    file_name="hnc_treatment_summary.txt",
                    mime="text/plain"
                )

                # Subgroup scorecard
                st.subheader("6. How this patient compares with similar groups")

                subgroup_df = load_csv_with_fallback("subgroup_summary_cates.csv")
                if subgroup_df is None or subgroup_df.empty:
                    st.info(
                        "Subgroup summary `subgroup_summary_cates.csv` not found. "
                        "Run the training notebook to regenerate it."
                    )
                else:
                    scorecard_df = build_patient_scorecard_from_subgroups(
                        patient,
                        subgroup_df,
                        score_horizon_months=rmst_horizon_months,
                    )
                    if scorecard_df.empty:
                        st.info(
                            "Could not match this patient's features to subgroup summaries. "
                            "This may happen if variable names differ or some fields are missing."
                        )
                    else:
                        metric_unit = scorecard_df["metric_unit"].iloc[0]

                        if metric_unit == "days":
                            display_df = scorecard_df.assign(
                                rank_text=lambda d: d["rank_within_feature"].astype(str)
                                + " / " + d["n_levels"].astype(str)
                            )[[
                                "feature",
                                "patient_level",
                                "n_in_level",
                                "metric",
                                "metric_months",
                                "rank_text"
                            ]].rename(columns={
                                "feature": "Feature",
                                "patient_level": "Patient subgroup",
                                "n_in_level": "N in subgroup",
                                "metric": "Mean extra time with Chemo-RT (days)",
                                "metric_months": "Mean extra time (months)",
                                "rank_text": "Rank within feature\n(1 = highest gain)",
                            })

                            st.dataframe(display_df, use_container_width=True)

                        else:
                            hm = scorecard_df["horizon_months"].iloc[0]
                            display_df = scorecard_df.assign(
                                metric_percent=lambda d: d["metric"] * 100.0,
                                rank_text=lambda d: d["rank_within_feature"].astype(str)
                                + " / " + d["n_levels"].astype(str)
                            )[[
                                "feature",
                                "patient_level",
                                "n_in_level",
                                "metric_percent",
                                "rank_text"
                            ]].rename(columns={
                                "feature": "Feature",
                                "patient_level": "Patient subgroup",
                                "n_in_level": "N in subgroup",
                                "metric_percent": f"Average difference at {int(hm)}m (% points)\n(Chemo-RT − RT)",
                                "rank_text": "Rank within feature\n(1 = most favourable)",
                            })

                            st.dataframe(display_df, use_container_width=True)

# The rest of your app (tabs 2 and 3) unchanged...
# (Copy the rest of your original timecourse + patterns-from-data UI code here.)
