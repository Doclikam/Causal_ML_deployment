import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.special import expit
from functools import lru_cache
import os
import scipy.signal as signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import pairwise_distances_argmin_min
from io import BytesIO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from math import ceil
import plotly.graph_objects as go
from utils.infer import infer_new_patient_fixed 
import requests
from utils.infer import infer_new_patient_fixed

# ----------------- CONFIG -----------------
st.set_page_config(
    page_title="Head & Neck Cancer – Personalized Treatment Effect Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Head & Neck Cancer: RT vs ChemoRT Outcome Explorer")

st.markdown(
    """
    *Interactive tool to visualise expected survival and treatment effects  
    for patients receiving radiotherapy alone versus chemoradiotherapy.*  
    """
)

st.markdown(
    """
    *Interactive tool to visualise expected survival and treatment effects  
    for patients receiving radiotherapy alone versus chemoradiotherapy.*  
    """
)
st.markdown(
    "<small style='color: grey;'>Designed for clinicians to support risk–benefit discussions with patients. Not a substitute for clinical judgment.
    Estimates are model-based and derived from retrospective data.</small>",
    unsafe_allow_html=True
)

 
Estimates are model-based and derived from retrospective data.


# Defaults for loading artifacts
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/Doclikam/Causal_ML_deployment/main/outputs/"
DEFAULT_OUTDIR = "outputs"
INTERVAL_DAYS = 30

# ----------------- SIDEBAR: SETTINGS -----------------
st.sidebar.header("Model & data source")

BASE_URL = st.sidebar.text_input(
    "BASE_URL (raw GitHub path to outputs/)",
    value=DEFAULT_BASE_URL
)

OUTDIR = st.sidebar.text_input(
    "Local outputs folder (optional, used if present)",
    value=DEFAULT_OUTDIR
)

max_period_months = st.sidebar.number_input(
    "Max follow-up horizon (months) for survival curve",
    value=60,
    min_value=6,
    max_value=156,
    step=6
)

rmst_horizon_months = st.sidebar.number_input(
    "RMST horizon (months)",
    value=36,
    min_value=6,
    max_value=int(max_period_months),
    step=6
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Note**: The app first looks in the local `outputs/` folder.\n"
                    "If artifacts are missing, it will try to load them from the GitHub URL.")


# ----------------- HELPERS -----------------
def load_csv_with_fallback(filename: str):
    """
    Try local OUTDIR/filename; if not found, try BASE_URL/filename.
    Returns DataFrame or None.
    """
    # local
    local_path = os.path.join(OUTDIR, filename)
    if os.path.exists(local_path):
        try:
            return pd.read_csv(local_path)
        except Exception:
            pass

    # remote
    if BASE_URL:
        url = BASE_URL.rstrip("/") + "/" + filename
        try:
            return pd.read_csv(url)
        except Exception:
            return None
    return None


def load_joblib_with_fallback(filename: str):
    """
    Try local OUTDIR/filename; if not found, try BASE_URL/filename via requests.
    Returns object or None.
    """
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
    Compute RMST_treat, RMST_control and ΔRMST up to a given horizon.
    Uses step-function approximation from survival curves.
    surv_df: columns ['period','days','S_control','S_treat']
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

    # start times
    t0 = np.concatenate(([0.0], times[:-1]))
    dt = times - t0

    # survival at start of each interval (step function)
    S_c_start = np.concatenate(([1.0], S_c_end[:-1]))
    S_t_start = np.concatenate(([1.0], S_t_end[:-1]))

    rmst_c = np.sum(S_c_start * dt)
    rmst_t = np.sum(S_t_start * dt)

    return {
        "rmst_treat": rmst_t / 30.0,     # convert days -> months
        "rmst_control": rmst_c / 30.0,
        "delta": (rmst_t - rmst_c) / 30.0
    }


def interpret_delta_months(delta_m: float) -> str:
    if np.isnan(delta_m):
        return "The model could not compute an RMST difference for this patient."
    mag = abs(delta_m)
    if mag < 0.5:
        size = "very small"
    elif mag < 2:
        size = "modest"
    else:
        size = "substantial"

    if delta_m > 0:
        return (f"Chemo-RT is estimated to provide a **{size} gain** of about "
                f"**{delta_m:.1f} additional event-free months** over the chosen horizon.")
    elif delta_m < 0:
        return (f"Chemo-RT is estimated to be **worse** by about "
                f"**{mag:.1f} event-free months** over the chosen horizon (possible net harm).")
    else:
        return "The model predicts no difference in event-free months between RT and Chemo-RT."


def describe_cate_table(cates: dict) -> str:
    """Generate short narrative from per-horizon CATEs."""
    # Keep only horizons with valid CATE
    vals = [(h, v["CATE"]) for h, v in cates.items() if v["CATE"] is not None and not np.isnan(v["CATE"])]
    if not vals:
        return "The causal forest could not provide reliable horizon-specific risk differences (CATEs) for this patient."

    # convert horizons to months if possible
    parsed = []
    for h, v in vals:
        try:
            mh = float(h)
        except Exception:
            mh = np.nan
        parsed.append((mh, v))

    # find maximum benefit (most negative) and maximum harm (most positive)
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
            f"- **Largest estimated benefit**: at about **{best_h:.0f} months**, "
            f"Chemo-RT is predicted to **reduce the absolute event risk by "
            f"{abs(best_v)*100:.1f} percentage points** for this patient."
        )
    if worst_v > 0:
        text.append(
            f"- **Largest estimated harm**: at about **{worst_h:.0f} months**, "
            f"Chemo-RT is predicted to **increase the absolute event risk by "
            f"{worst_v*100:.1f} percentage points**."
        )

    if not text:
        text.append(
            "Across the modelled horizons, Chemo-RT appears roughly risk-neutral "
            "(no strongly beneficial or harmful horizon emerges)."
        )

    text.append(
        "\nRemember: these are **absolute risk differences**, not relative hazard ratios. "
        "Negative values = fewer events with Chemo-RT; positive values = more events."
    )

    return "\n".join(text)


# ----------------- LAYOUT: TABS -----------------
tab_patient, tab_timecourse = st.tabs(["👤 Single patient", "⏱ Population time-course"])

# ---------- TAB 1: SINGLE PATIENT ----------
with tab_patient:
    st.subheader("1. Enter patient baseline information")

    with st.expander("What this tool does", expanded=True):
        st.markdown("""
- Uses a **pooled logistic survival model** to estimate the patient's probability of being alive and event-free over time under:
  - **RT alone** (control)  
  - **Chemo-RT** (treated)
- Uses **causal forests** to estimate **horizon-specific risk differences (CATEs)** for this patient:
  - Negative CATE → **fewer events** with Chemo-RT (benefit)
  - Positive CATE → **more events** with Chemo-RT (potential harm)
- Summarises the gain or loss in **event-free time** as **ΔRMST (restricted mean survival time)** over a horizon you choose.
        """)

    # patient form 
with st.form("patient_form"):

    c1, c2, c3 = st.columns(3)

    # ---- COLUMN 1: demographics ----
    with c1:
        age = st.number_input("Age (years)", value=62, min_value=18, max_value=99)
        sex = st.selectbox("Sex", ["Male", "Female", "Missing"], index=1)

    # ---- COLUMN 2: tumour site & histology ----
    with c2:
        primary_site_group = st.selectbox(
            "Primary site group",
            ["Oropharynx", "Nasopharynx", "Other_HNC", "Missing"],
            index=0
        )
        pathology_group = st.selectbox(
            "Pathology group",
            ["SCC", "Other_epithelial", "Other_rare", "Missing"],
            index=0
        )

    # ---- COLUMN 3: HPV + TNM ----
    with c3:
        hpv_clean = st.selectbox(
            "HPV status (cleaned)",
            ["HPV_Positive", "HPV_Negative", "HPV_Unknown", "Missing"],
            index=0
        )
        stage = st.selectbox(
            "Overall Stage (AJCC-like)",
            ["I", "II", "III", "IV", "Missing"],
            index=2
        )

    st.markdown("### TNM classification")

    c4, c5, c6 = st.columns(3)
    with c4:
        t_cat = st.selectbox("T category", ["T1", "T2", "T3", "T4", "Tx"], index=1)
    with c5:
        n_cat = st.selectbox("N category", ["N0", "N1", "N2", "N3", "Nx"], index=0)
    with c6:
        m_cat = st.selectbox("M category", ["M0", "M1", "Mx"], index=0)

    treatment = st.selectbox(
        "Planned treatment strategy",
        options=[0, 1],
        format_func=lambda x: "RT alone" if x == 0 else "Chemo-RT",
        index=0
    )

    submitted = st.form_submit_button("Estimate personalised outcomes")


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
        "treatment": treatment
    }

    with st.spinner("Calculating personalised survival and treatment benefit..."):
        out = infer_new_patient_fixed(
            patient_data=patient,
            outdir=OUTDIR,
            base_url=BASE_URL,
            max_period_override=int(max_period_months)
        )

        # Show any technical errors
        if out.get("errors"):
            st.error("Technical notes from the modelling pipeline:")
            for k, msg in out["errors"].items():
                st.write(f"- **{k}**: {msg}")

        surv = out.get("survival_curve")
        cates = out.get("CATEs", {})

        # ---- SECTION A: SURVIVAL & RMST ----
        st.subheader("2. Survival under RT vs Chemo-RT")

        if surv is None or surv.empty:
            st.warning("Survival curve could not be computed.")
        else:
            # Convert days -> months for display
            surv_plot = surv.copy()
            surv_plot["months"] = surv_plot["days"] / 30.0

            # Plot survival curves
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=surv_plot["months"], y=surv_plot["S_control"],
                mode="lines+markers", name="RT alone"
            ))
            fig.add_trace(go.Scatter(
                x=surv_plot["months"], y=surv_plot["S_treat"],
                mode="lines+markers", name="Chemo-RT"
            ))
            fig.update_layout(
                xaxis_title="Time since RT start (months)",
                yaxis_title="Probability alive & event-free",
                yaxis=dict(range=[0, 1]),
                legend_title="Strategy"
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("**Table snapshot (first few time points):**")
                st.dataframe(
                    surv_plot[["period", "months", "S_control", "S_treat"]].head(),
                    use_container_width=True
                )
            with c2:
                st.markdown("**How to read this plot**")
                st.markdown("""
- Each point shows, for a patient like this:
  - the estimated probability of being **alive and event-free** at that time  
  - under **RT alone** vs **Chemo-RT**.
- A **higher curve** indicates **better outcomes**.
- The difference between the two curves summarizes how much Chemo-RT might help or harm.
                """)

            # Compute RMST at chosen horizon
            rmst_res = compute_rmst_from_survival(surv, rmst_horizon_months)
            rmst_t = rmst_res["rmst_treat"]
            rmst_c = rmst_res["rmst_control"]
            delta_m = rmst_res["delta"]

            st.subheader(f"3. RMST at {rmst_horizon_months} months (event-free time)")

            m1, m2, m3 = st.columns(3)
            m1.metric("RT alone: event-free months", f"{rmst_c:.1f}" if not np.isnan(rmst_c) else "N/A")
            m2.metric("Chemo-RT: event-free months", f"{rmst_t:.1f}" if not np.isnan(rmst_t) else "N/A")
            m3.metric("ΔRMST (Chemo-RT − RT)", f"{delta_m:+.1f} months" if not np.isnan(delta_m) else "N/A")

            st.markdown(interpret_delta_months(delta_m))

        # ---- SECTION B: CATE PER HORIZON ----
        st.subheader("4. Horizon-specific treatment effect (CATE)")

        if not cates:
            st.info("No CATE estimates available for this patient.")
        else:
            # Build DataFrame for plotting
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

            # split valid vs errored
            valid = df_cate[df_cate["CATE"].notna()].copy()
            errors_cate = df_cate[df_cate["CATE"].isna() & df_cate["error"].notna()]

            if not valid.empty:
                # color: benefit (negative) vs harm (positive)
                colors = [
                    "rgba(0, 120, 200, 0.8)" if x < 0 else "rgba(220, 80, 80, 0.8)"
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
                    xaxis_title="Horizon (months)",
                    yaxis_title="Absolute risk difference (Chemo-RT − RT, % points)",
                    title="Per-horizon CATE for this patient"
                )
                st.plotly_chart(fig_c, use_container_width=True)

                st.markdown(describe_cate_table(cates))

            if not errors_cate.empty:
                with st.expander("Technical issues for some horizons"):
                    st.table(errors_cate[["horizon_label", "error"]])

# ---------- TAB 2: POPULATION TIME-COURSE ----------
with tab_timecourse:
    st.subheader("Population-level time-course of treatment effect")

    st.markdown("""
This panel summarizes the **time-varying behaviour of Chemo-RT vs RT**  
across the whole dataset used to train the models.

It shows:
- **Marginal hazards** by period (how often events occur per time block)
- **Hazard ratio (HR)** over time with bootstrap CIs
- **Contribution to ΔRMST** from each period

This is **not patient-specific**, but helps you understand **when** treatment
seems to matter most in the data.
    """)

    tv = load_csv_with_fallback("timevarying_summary_by_period.csv")
    if tv is None:
        st.info("timevarying_summary_by_period.csv not found locally or at BASE_URL.")
    else:
        # assume columns: period, haz_treated, haz_control, hr, hr_lo, hr_up, delta_rmst_period_days, ...
        tv = tv.copy()
        tv["months"] = tv["period"] * (INTERVAL_DAYS / 30.0)
        tv["delta_rmst_period_months"] = tv["delta_rmst_period_days"] / 30.0

        c1, c2 = st.columns(2)

        # Hazards plot
        with c1:
            fig_h = go.Figure()
            if "haz_control" in tv.columns:
                fig_h.add_trace(go.Scatter(
                    x=tv["months"], y=tv["haz_control"],
                    mode="lines+markers", name="RT hazard"
                ))
            if "haz_treated" in tv.columns:
                fig_h.add_trace(go.Scatter(
                    x=tv["months"], y=tv["haz_treated"],
                    mode="lines+markers", name="Chemo-RT hazard"
                ))
            fig_h.update_layout(
                xaxis_title="Time (months)",
                yaxis_title="Marginal event probability per interval",
                title="Interval hazards over time"
            )
            st.plotly_chart(fig_h, use_container_width=True)

        # HR plot
        with c2:
            if {"hr", "hr_lo", "hr_up"}.issubset(tv.columns):
                fig_hr = go.Figure()
                fig_hr.add_trace(go.Scatter(
                    x=tv["months"], y=tv["hr"],
                    mode="lines+markers", name="HR (Chemo-RT / RT)"
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
                    yaxis_title="Hazard ratio",
                    title="Time-varying HR (Chemo-RT vs RT)"
                )
                st.plotly_chart(fig_hr, use_container_width=True)

        # Contribution to ΔRMST
        st.subheader("Contribution to ΔRMST by period")

        fig_dr = go.Figure()
        fig_dr.add_trace(go.Bar(
            x=tv["months"],
            y=tv["delta_rmst_period_months"],
            name="ΔRMST contribution (months)"
        ))
        fig_dr.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_dr.update_layout(
            xaxis_title="Time (months)",
            yaxis_title="ΔRMST contribution (months)",
            title="Where the gain/loss in event-free time comes from"
        )
        st.plotly_chart(fig_dr, use_container_width=True)

        # Simple narrative: when effect peaks
        if "hr" in tv.columns:
            # restrict to first, say, 60 months for narrative
            tv_sub = tv[tv["months"] <= 60].copy()
            if not tv_sub.empty:
                min_hr_row = tv_sub.loc[tv_sub["hr"].idxmin()]
                peak_m = float(min_hr_row["months"])
                peak_hr = float(min_hr_row["hr"])
                st.markdown(f"""
**Interpretation (for the population in the data):**

- The **strongest relative benefit** of Chemo-RT (lowest HR) is seen around  
  **{peak_m:.0f} months** after treatment start, with HR ≈ **{peak_hr:.2f}**.
- Before this time, Chemo-RT is generally associated with **lower event rates**;  
  beyond this point, the HR tends to move closer to 1 (less difference).
                """)
        st.markdown("""
⚠️ **Important caveats**

- These curves are **averaged over all patients** in the dataset – individuals can differ.
- They do **not** capture acute vs chronic toxicity profiles directly.
- Use this information as a **starting point** for discussions on:
  - when treatment intensity matters most,
  - when to focus on surveillance vs escalation,
  - which patients (from the single-patient panel) have clearly favourable vs unfavourable profiles.
        """)
