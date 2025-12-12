# app.py (updated) ----------------------------------------------------------
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


def build_patient_scorecard_from_subgroups(
    patient: dict,
    subgroup_df: pd.DataFrame,
    score_horizon_months: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a simple scorecard showing how this patient's subgroups
    compare to others (where Chemo-RT looked most / least beneficial).
    """
    if subgroup_df is None or subgroup_df.empty:
        return pd.DataFrame()

    df = subgroup_df.copy()

    # ----- CASE 1: RMST-style table -----
    if (
        ("feature" in df.columns or "group_var" in df.columns)
        and ("group" in df.columns or "group_level" in df.columns)
        and (
            "mean_CATE_days" in df.columns
            or "Mean_CATE_days" in df.columns
        )
    ):
        if "feature" not in df.columns and "group_var" in df.columns:
            df = df.rename(columns={"group_var": "feature"})
        if "group" not in df.columns and "group_level" in df.columns:
            df = df.rename(columns={"group_level": "group"})
        if "mean_CATE_days" not in df.columns and "Mean_CATE_days" in df.columns:
            df = df.rename(columns={"Mean_CATE_days": "mean_CATE_days"})

        if "mean_CATE_months" not in df.columns:
            df["mean_CATE_months"] = df["mean_CATE_days"] / 30.0
        if "n" not in df.columns:
            df["n"] = np.nan

        rows = []
        for feat in sorted(df["feature"].unique()):
            if feat not in patient:
                continue
            level = patient[feat]
            df_feat = df[df["feature"] == feat].copy()
            if df_feat.empty:
                continue

            df_feat = df_feat.sort_values("mean_CATE_days", ascending=False).reset_index(drop=True)
            df_feat["rank_within_feature"] = df_feat.index + 1
            df_feat["n_levels"] = len(df_feat)

            match = df_feat[df_feat["group"].astype(str) == str(level)]
            if match.empty:
                continue

            m = match.iloc[0]
            rows.append({
                "feature": feat,
                "patient_level": m["group"],
                "n_in_level": int(m["n"]) if not pd.isna(m["n"]) else np.nan,
                "metric": m["mean_CATE_days"],
                "metric_months": m["mean_CATE_months"],
                "metric_unit": "days",
                "horizon_months": np.nan,
                "rank_within_feature": int(m["rank_within_feature"]),
                "n_levels": int(m["n_levels"]),
            })

        if not rows:
            return pd.DataFrame()

        scorecard = pd.DataFrame(rows)
        scorecard = scorecard.sort_values("metric", ascending=False).reset_index(drop=True)
        return scorecard

    # ----- CASE 2: probability CATE table -----
    cate_cols = [c for c in df.columns if c.startswith("CATE_")]
    if not cate_cols or "group" not in df.columns:
        return pd.DataFrame()

    horizons = []
    for c in cate_cols:
        try:
            num = "".join(ch for ch in c.replace("CATE_", "") if ch.isdigit())
            m = float(num)
            horizons.append((c, m))
        except Exception:
            continue

    if not horizons:
        return pd.DataFrame()

    if score_horizon_months is not None:
        chosen_col, chosen_h = min(
            horizons,
            key=lambda tup: abs(tup[1] - float(score_horizon_months)),
        )
    else:
        chosen_col, chosen_h = max(horizons, key=lambda tup: tup[1])

    rows = []
    for feat in sorted(df["group"].unique()):
        if feat not in patient:
            continue

        level = patient[feat]
        df_feat = df[df["group"] == feat].copy()
        if df_feat.empty or feat not in df_feat.columns:
            continue

        df_feat = df_feat.rename(columns={feat: "group_level"})

        if "n" not in df_feat.columns:
            df_feat["n"] = np.nan

        df_feat["metric"] = df_feat[chosen_col]          # Chemo-RT − RT risk diff
        df_feat["metric_unit"] = "risk_diff"
        df_feat["horizon_months"] = chosen_h

        df_feat = df_feat.sort_values("metric", ascending=True).reset_index(drop=True)
        df_feat["rank_within_feature"] = df_feat.index + 1
        df_feat["n_levels"] = len(df_feat)

        match = df_feat[df_feat["group_level"].astype(str) == str(level)]
        if match.empty:
            continue

        m = match.iloc[0]
        rows.append({
            "feature": feat,
            "patient_level": m["group_level"],
            "n_in_level": int(m["n"]) if not pd.isna(m["n"]) else np.nan,
            "metric": m["metric"],
            "metric_months": np.nan,
            "metric_unit": "risk_diff",
            "horizon_months": m["horizon_months"],
            "rank_within_feature": int(m["rank_within_feature"]),
            "n_levels": int(m["n_levels"]),
        })

    if not rows:
        return pd.DataFrame()

    scorecard = pd.DataFrame(rows)
    scorecard = scorecard.sort_values("metric", ascending=True).reset_index(drop=True)
    return scorecard


def categorize_benefit(delta_m: float, p_rt: float, p_chemo: float) -> str:
    """
    Simple label summarising benefit level for clinicians.
    """
    if np.isnan(delta_m) or np.isnan(p_rt) or np.isnan(p_chemo):
        return "Estimate uncertain"

    diff_surv = (p_chemo - p_rt) * 100.0
    mag = abs(delta_m)

    if delta_m <= -0.5:
        return "Chemo-RT not favoured (possible net harm)"

    if mag < 0.5 and abs(diff_surv) < 3:
        return "Either option acceptable (minimal difference)"

    if delta_m > 0 and mag < 2:
        return "Chemo-RT: small benefit"
    if delta_m > 0 and mag < 4:
        return "Chemo-RT: moderate benefit"
    if delta_m > 0:
        return "Chemo-RT: large benefit"

    return "Estimate uncertain"


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
            # --- main inference (user-facing) ---
            out = infer_new_patient_fixed(
                patient_data=patient,
                outdir=OUTDIR,
                base_url=BASE_URL,
                max_period_override=int(max_period_months)
            )

            # --- DIAGNOSTICS (developer) ---
            st.markdown("### 🔍 Diagnostics (developer)")
            try:
                out_dbg = infer_new_patient_fixed(
                    patient_data=patient,
                    outdir=OUTDIR,
                    base_url=BASE_URL,
                    max_period_override=int(max_period_months),
                    return_raw=True
                )
            except Exception as e:
                st.error("infer_new_patient_fixed raised an exception during diagnostics.")
                st.exception(e)
                out_dbg = {"errors": {"infer_call": str(e)}}

            debug_block = out_dbg.get("debug", {})
            st.write("**infer() debug keys:**", list(debug_block.keys()) if isinstance(debug_block, dict) else debug_block)
            st.write("**Top-level errors:**")
            st.json(out_dbg.get("errors", {}))

            if isinstance(debug_block, dict) and debug_block.get("artifact_sources"):
                st.write("Artifact sources (from infer debug):")
                st.json(debug_block["artifact_sources"])
            else:
                st.info("No artifact sources returned in debug. infer() may not have loaded artifacts or return_raw was False.")

            # Re-run local/remote loading for quick inspection (mimic infer internals)
            import io as _io
            def _try_local_or_remote(fn):
                local_path = os.path.join(OUTDIR, fn)
                if os.path.exists(local_path):
                    try:
                        return joblib.load(local_path), f"local:{local_path}"
                    except Exception:
                        try:
                            return pd.read_csv(local_path), f"local_csv:{local_path}"
                        except Exception as e:
                            return None, f"failed_local:{local_path}:{e}"
                if BASE_URL:
                    url = BASE_URL.rstrip("/") + "/" + fn
                    try:
                        r = requests.get(url, timeout=30)
                        r.raise_for_status()
                        try:
                            return joblib.load(_io.BytesIO(r.content)), f"remote_joblib:{url}"
                        except Exception:
                            try:
                                return pd.read_csv(_io.StringIO(r.text)), f"remote_csv:{url}"
                            except Exception:
                                return None, f"remote_failed_read:{url}"
                    except Exception as e:
                        return None, f"remote_failed:{url}:{e}"
                return None, "not_found"

            patient_cols_art, pc_src = _try_local_or_remote("causal_patient_columns.joblib")
            st.write("patient_columns source:", pc_src)

            # Build Xpatient locally and display quick diagnostics
            def build_Xpatient_local(patient_dict, patient_columns_obj):
                if patient_columns_obj is None:
                    return None
                # derive pcols
                if isinstance(patient_columns_obj, (list, tuple, pd.Series, np.ndarray)):
                    pcols = list(patient_columns_obj)
                elif isinstance(patient_columns_obj, dict):
                    pcols = patient_columns_obj.get("columns", list(patient_columns_obj.keys()))
                elif isinstance(patient_columns_obj, pd.DataFrame):
                    pcols = patient_columns_obj.iloc[:, 0].astype(str).tolist()
                else:
                    pcols = list(patient_columns_obj)

                Xp = pd.DataFrame(np.zeros((1, len(pcols))), columns=pcols)
                for c in pcols:
                    if c in patient_dict:
                        Xp.at[0, c] = patient_dict[c]
                    else:
                        # attempt one-hot match like sex_Male
                        if "_" in c:
                            root, tail = c.split("_", 1)
                            if root in patient_dict and str(patient_dict[root]) == tail:
                                Xp.at[0, c] = 1.0
                return Xp

            Xpatient_local = build_Xpatient_local(patient, patient_cols_art)
            if Xpatient_local is None:
                st.error("❌ Could not build Xpatient locally — patient_columns missing or incompatible.")
            else:
                st.write("Xpatient (first 60 columns):")
                try:
                    st.dataframe(Xpatient_local.iloc[:, :60].T, use_container_width=True)
                except Exception:
                    st.write(Xpatient_local.iloc[:, :60].to_dict())

                st.write("Xpatient stats: min / max / mean")
                st.write(
                    float(Xpatient_local.values.min()),
                    float(Xpatient_local.values.max()),
                    float(Xpatient_local.values.mean())
                )

                if np.allclose(Xpatient_local.values, 0):
                    st.error("❌ Xpatient is ALL ZEROS — this will produce identical predictions for all patients.")
                    st.info("Likely causes: patient_columns expect one-hot column names (e.g., sex_Male) while app inputs are raw (sex='Male'), or the canonical column names differ from those used here. Check `causal_patient_columns` artifact and training notebook.")
                else:
                    st.success("Xpatient looks non-zero — OK.")

            # Proceed with the rest of the user-facing display (unchanged)
            raw_errors = out.get("errors", {})
            # hide 'scaler' messages if present
            filtered_errors = {k: v for k, v in raw_errors.items() if k != "scaler"}
            if filtered_errors:
                with st.expander("Technical notes from modelling pipeline", expanded=False):
                    for k, msg in filtered_errors.items():
                        st.write(f"- **{k}**: {msg}")

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

           
            # If you want the full unabridged file, the above contains the necessary diagnostics integration.
