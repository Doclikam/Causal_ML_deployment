import os
from io import BytesIO
from typing import Optional
from datetime import date

import joblib
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

- The chance of being **alive and well** (no death from any cause) over time  
- The **extra time alive and well** that Chemo-RT may offer, in **months**  
- How similar patients in the dataset have done  

> ⚠️ These are **model-based estimates from retrospective data**.  
> They support – but do not replace – clinical judgment and guidelines.
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

    lines.append("Patient report summary:")
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
            start_date = st.date_input(
                "Planned RT start date",
                value=date.today()
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
            out = infer_new_patient_fixed(
                patient_data=patient,
                outdir=OUTDIR,
                base_url=BASE_URL,
                max_period_override=int(max_period_months)
            )

            raw_errors = out.get("errors", {})
            filtered_errors = {
                k: v for k, msg in raw_errors.items()
                if k not in ["scaler"]
            }
            if filtered_errors:
                with st.expander("Technical notes from modelling pipeline", expanded=False):
                    for k, msg in filtered_errors.items():
                        st.write(f"- **{k}**: {msg}")

            surv = out.get("survival_curve")
            cates = out.get("CATEs", {})

            # ---------- SURVIVAL & SUMMARY ----------
            st.subheader("2. Summary at your chosen time point")

            rmst_res = {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}
            p_rt = p_chemo = np.nan
            label = "Estimate uncertain"

            if surv is None or surv.empty:
                st.warning("Survival curve could not be computed for this patient.")
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

                label = categorize_benefit(delta_m, p_rt, p_chemo)

                # Summary metrics card (shorter labels to avoid ellipses)
                c_sum1, c_sum2, c_sum3, c_sum4 = st.columns(4)
                c_sum1.metric(
                    "RT alone\n(avg months)",
                    f"{rmst_c:.1f}" if not np.isnan(rmst_c) else "N/A"
                )
                c_sum2.metric(
                    "Chemo-RT\n(avg months)",
                    f"{rmst_t:.1f}" if not np.isnan(rmst_t) else "N/A"
                )
                c_sum3.metric(
                    "Extra time\nChemo-RT − RT",
                    f"{delta_m:+.1f}" if not np.isnan(delta_m) else "N/A"
                )
                if not np.isnan(p_rt) and not np.isnan(p_chemo):
                    diff_surv = (p_chemo - p_rt) * 100.0
                    c_sum4.metric(
                        f"Diff in chance\nalive & well at {rmst_horizon_months}m",
                        f"{diff_surv:+.1f} %"
                    )
                else:
                    c_sum4.metric(
                        f"Diff in chance\nalive & well at {rmst_horizon_months}m",
                        "N/A"
                    )

                # Bold markdown (no HTML wrapper so ** works)
                st.markdown(f"**Model interpretation:** {label}")
                st.markdown(interpret_delta_months(delta_m))

            st.markdown("---")

            # ---------- SURVIVAL CURVES ----------
                        # ---------- SURVIVAL CURVES ----------
            if surv is not None and not surv.empty:
                surv_plot = surv.copy()

                # Make sure we have a 'days' column
                if "days" not in surv_plot.columns:
                    # fall back to period * interval length
                    if "period" in surv_plot.columns:
                        surv_plot["days"] = surv_plot["period"].astype(float) * INTERVAL_DAYS
                    else:
                        st.error("Survival data has no 'days' or 'period' column; cannot plot time axis.")
                        surv_plot = None

                if surv_plot is not None:
                    # ensure numeric
                    surv_plot["days"] = pd.to_numeric(surv_plot["days"], errors="coerce")
                    surv_plot = surv_plot.dropna(subset=["days"])

                    # months since RT start
                    surv_plot["months"] = surv_plot["days"] / 30.0

                    # calendar dates since RT start
                    # start_date comes from st.date_input, which is a datetime.date
                    surv_plot["date"] = pd.to_datetime(start_date) + pd.to_timedelta(
                        surv_plot["days"].astype(float), unit="D"
                    )

                    st.subheader("3. Survival over time (RT vs Chemo-RT)")

                    # Plot with months on x-axis (default)
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

                    # Optional calendar-date view
                    show_dates = st.checkbox(
                        "Show calendar dates on the time axis",
                        value=False
                    )
                    if show_dates:
                        fig_date = go.Figure()
                        fig_date.add_trace(go.Scatter(
                            x=surv_plot["date"], y=surv_plot["S_control"],
                            mode="lines+markers",
                            name="RT alone",
                            line=dict(color=COLOR_RT),
                            marker=dict(color=COLOR_RT)
                        ))
                        fig_date.add_trace(go.Scatter(
                            x=surv_plot["date"], y=surv_plot["S_treat"],
                            mode="lines+markers",
                            name="Chemo-RT",
                            line=dict(color=COLOR_CHEMO),
                            marker=dict(color=COLOR_CHEMO)
                        ))
                        fig_date.update_layout(
                            xaxis_title="Calendar date",
                            yaxis_title="Probability alive & well",
                            yaxis=dict(range=[0, 1]),
                            legend_title="Strategy"
                        )
                        st.plotly_chart(fig_date, use_container_width=True)

                    with st.expander("Table of modelled survival over time (first few rows)"):
                        st.dataframe(
                            surv_plot[["period", "months", "date", "S_control", "S_treat"]].head(),
                            use_container_width=True
                        )


            # ---------- PATIENT REPORT SUMMARY ----------
            st.subheader("4. Patient report summary")

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
                label="📄 Download patient report (text)",
                data=printable_summary,
                file_name="hnc_treatment_summary.txt",
                mime="text/plain"
            )

            # ---------- ADVANCED DETAILS ----------
            show_advanced = st.checkbox("Show advanced model details (risk differences & subgroup patterns)", value=False)

            if show_advanced:
                # CATEs
                st.subheader("5. Change in chance of being alive & well at different times")

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

                        # Show only top ~6 features in a simple horizontal bar chart
                        top_n = min(6, len(scorecard_df))
                        top_sc = scorecard_df.head(top_n).copy()

                        if metric_unit == "days":
                            top_sc["label"] = top_sc["feature"] + " = " + top_sc["patient_level"].astype(str)
                            top_sc["metric_months"] = top_sc["metric"] / 30.0

                            fig_sc = go.Figure()
                            fig_sc.add_trace(go.Bar(
                                x=top_sc["metric_months"],
                                y=top_sc["label"],
                                orientation="h",
                                marker_color=COLOR_BENEFIT
                            ))
                            fig_sc.update_layout(
                                xaxis_title="Average extra time with Chemo-RT (months)",
                                yaxis_title="Patient subgroup",
                                title=f"Top subgroups where similar patients gained more time by {rmst_horizon_months}m"
                            )
                            st.plotly_chart(fig_sc, use_container_width=True)

                        else:
                            hm = top_sc["horizon_months"].iloc[0]
                            top_sc["label"] = top_sc["feature"] + " = " + top_sc["patient_level"].astype(str)
                            top_sc["metric_percent"] = top_sc["metric"] * 100.0

                            fig_sc = go.Figure()
                            fig_sc.add_trace(go.Bar(
                                x=top_sc["metric_percent"],
                                y=top_sc["label"],
                                orientation="h",
                                marker_color=[
                                    COLOR_BENEFIT if v < 0 else COLOR_HARM
                                    for v in top_sc["metric_percent"]
                                ]
                            ))
                            fig_sc.update_layout(
                                xaxis_title=f"Average difference at {int(hm)} months (% points, Chemo-RT − RT)",
                                yaxis_title="Patient subgroup",
                                title="Subgroups where similar patients did better or worse with Chemo-RT"
                            )
                            st.plotly_chart(fig_sc, use_container_width=True)

                        with st.expander("See underlying subgroup table (optional)"):
                            st.dataframe(scorecard_df, use_container_width=True)

# ==========================================================
# ---------- TAB 2: POPULATION EFFECT OVER TIME ----------
# ==========================================================
with tab_timecourse:
    st.subheader("How treatment effect changes over time (population-level)")

    st.markdown("""
This page shows **overall patterns** from the study population:

- How often events occurred over time under **RT vs Chemo-RT**
- How the **relative effect** (hazard ratio) changed over time
- Which time windows contributed most to the overall **extra time alive & well**

These summaries are **not patient-specific**, but can help frame when treatment intensity
seems most influential in the underlying data.
    """)

    tv = load_csv_with_fallback("timevarying_summary_by_period.csv")
    if tv is None:
        st.info("`timevarying_summary_by_period.csv` not found locally or at BASE_URL.")
    else:
        tv = tv.copy()
        tv["months"] = tv["period"] * (INTERVAL_DAYS / 30.0)
        tv["delta_rmst_period_months"] = tv["delta_rmst_period_days"] / 30.0

        c1, c2 = st.columns(2)

        with c1:
            fig_h = go.Figure()
            if "haz_control" in tv.columns:
                fig_h.add_trace(go.Scatter(
                    x=tv["months"], y=tv["haz_control"],
                    mode="lines+markers", name="RT event rate",
                    line=dict(color=COLOR_RT),
                    marker=dict(color=COLOR_RT)
                ))
            if "haz_treated" in tv.columns:
                fig_h.add_trace(go.Scatter(
                    x=tv["months"], y=tv["haz_treated"],
                    mode="lines+markers", name="Chemo-RT event rate",
                    line=dict(color=COLOR_CHEMO),
                    marker=dict(color=COLOR_CHEMO)
                ))
            fig_h.update_layout(
                xaxis_title="Time (months)",
                yaxis_title="Event probability per interval",
                title="Event rates over time"
            )
            st.plotly_chart(fig_h, use_container_width=True)

        with c2:
            if {"hr", "hr_lo", "hr_up"}.issubset(tv.columns):
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Scatter(
                    x=tv["months"], y=tv["hr"],
                    mode="lines+markers", name="Hazard ratio",
                    line=dict(color=COLOR_CHEMO),
                    marker=dict(color=COLOR_CHEMO)
                ))
                fig_hr.add_trace(go.Scatter(
                    x=np.concatenate([tv["months"], tv["months"][::-1]]),
                    y=np.concatenate([tv["hr_lo"], tv["hr_up"][::-1]]),
                    fill="toself",
                    line=dict(width=0),
                    name="95% CI",
                    opacity=0.2
                ))
                fig_hr.add_hline(y=1.0, line_dash="dash", line_color="gray")
                fig_hr.update_layout(
                    xaxis_title="Time (months)",
                    yaxis_title="Chemo-RT / RT",
                    title="Relative effect (hazard ratio) over time"
                )
                st.plotly_chart(fig_hr, use_container_width=True)

        st.subheader("Where the extra time alive & well comes from")

        fig_dr = go.Figure()
        fig_dr.add_trace(go.Bar(
            x=tv["months"],
            y=tv["delta_rmst_period_months"],
            name="Contribution to extra time (months)",
            marker_color=COLOR_BENEFIT
        ))
        fig_dr.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dr.update_layout(
            xaxis_title="Time (months)",
            yaxis_title="Contribution to extra time (months)",
            title="Time windows contributing to overall benefit"
        )
        st.plotly_chart(fig_dr, use_container_width=True)

        if "hr" in tv.columns:
            tv_sub = tv[tv["months"] <= 60].copy()
            if not tv_sub.empty:
                min_hr_row = tv_sub.loc[tv_sub["hr"].idxmin()]
                peak_m = float(min_hr_row["months"])
                peak_hr = float(min_hr_row["hr"])
                st.markdown(f"""
**Interpretation (population-level):**

- The **strongest relative benefit** of Chemo-RT (lowest hazard ratio) appears around  
  **{peak_m:.0f} months** after starting treatment, with HR ≈ **{peak_hr:.2f}**.
- Before this time, Chemo-RT is generally associated with **fewer events** than RT alone;  
  later, the difference narrows.
                """)

        st.markdown("""
⚠️ These curves average over all patients in the dataset.
They do **not** directly capture toxicity or individual comorbidities.
Use them as background context alongside the patient-specific page and clinical judgment.
        """)

# ==========================================================
# ---------- TAB 3: PATTERNS FROM DATA ----------
# ==========================================================
with tab_insights:
    st.subheader("Patterns from the training data")

    st.markdown("""
This section gives a **quick look** at how different clinical subgroups
tended to respond in the dataset used to train the models.

It is not personalised, but can help sense-check whether your patient sits in a group
that typically showed **stronger** or **weaker** benefit from Chemo-RT.
    """)

    subgroup_df = load_csv_with_fallback("subgroup_summary_cates.csv")
    if subgroup_df is None or subgroup_df.empty:
        st.info(
            "Subgroup summary file `subgroup_summary_cates.csv` was not found. "
            "Run the training notebook to regenerate it."
        )
    else:
        st.markdown("A small snapshot of the subgroup summary table:")
        st.dataframe(subgroup_df.head(20), use_container_width=True)

        st.markdown("""
You can use this table (and the patient scorecard on the first tab) to identify:

- Subgroups where Chemo-RT looked **clearly favourable** on average  
- Subgroups where benefit was **small or uncertain**

These patterns complement the personalised estimates, especially for borderline cases.
        """)

surv.head()
