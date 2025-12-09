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

# ----------------- STYLE & CONFIG -----------------
# Color palette 
COLOR_RT = "#1f77b4"        # calm blue
COLOR_CHEMO = "#2ca02c"     # medical green
COLOR_BENEFIT = "#1f77b4"   # blue for benefit
COLOR_HARM = "#d62728"      # red only for harm

st.set_page_config(
    page_title="Head & Neck Cancer – Personalized Treatment Effect Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Head & Neck Cancer: RT vs ChemoRT Outcome Explorer")

st.markdown(
    """
    *A clinician-facing tool to explore expected survival and treatment effects  
    for patients receiving radiotherapy alone versus chemoradiotherapy (Chemo-RT).*  
    """
)

st.markdown(
    "<small style='color: grey;'>Designed to support risk–benefit discussions with patients. "
    "Not a substitute for clinical judgment. Estimates are model-based and derived from retrospective data.</small>",
    unsafe_allow_html=True
)

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
    vals = [(h, v["CATE"]) for h, v in cates.items()
            if v.get("CATE") is not None and not np.isnan(v.get("CATE"))]
    if not vals:
        return "The causal forest could not provide reliable horizon-specific risk differences (CATEs) for this patient."

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


def generate_patient_summary(patient, rmst_res, surv_df, horizon_months: int) -> str:
    """
    Generate a short, patient-facing summary (2–3 sentences).
    """
    rmst_t = rmst_res.get("rmst_treat", np.nan)
    rmst_c = rmst_res.get("rmst_control", np.nan)
    delta_m = rmst_res.get("delta", np.nan)

    # Try to get survival probabilities at the chosen horizon
    p_rt, p_chemo = np.nan, np.nan
    if surv_df is not None and not surv_df.empty:
        s = surv_df.sort_values("days").copy()
        h_days = horizon_months * 30.0
        s_h = s[s["days"] <= h_days]
        if not s_h.empty:
            p_rt = s_h["S_control"].iloc[-1]
            p_chemo = s_h["S_treat"].iloc[-1]

    # Core message about benefit/harm
    if np.isnan(delta_m):
        main = (
            f"For a patient with features similar to this one, "
            f"the model could not reliably estimate the difference between radiotherapy alone and "
            f"chemoradiotherapy over {horizon_months} months."
        )
    else:
        if delta_m > 0:
            phrasing = "a modest gain"
        elif delta_m < 0:
            phrasing = "a modest loss"
        else:
            phrasing = "no clear difference"

        main = (
            f"For a patient like this, over about {horizon_months} months the model predicts "
            f"{phrasing} in time spent alive and event-free with chemoradiotherapy compared with radiotherapy alone "
            f"(about {delta_m:+.1f} months difference)."
        )

    # Add survival probabilities if available
    surv_sentence = ""
    if not np.isnan(p_rt) and not np.isnan(p_chemo):
        surv_sentence = (
            f" By {horizon_months} months, the estimated chance of being alive and event-free is "
            f"around {p_rt*100:.0f}% with radiotherapy alone and {p_chemo*100:.0f}% with chemoradiotherapy."
        )

    disclaimer = (
        " These figures come from statistical models based on previous patients and are approximate. "
        "They are meant to support, not replace, a discussion between you and your treatment team."
    )

    return main + surv_sentence + disclaimer


def build_print_summary(patient, rmst_res, surv_df, horizon_months, cates) -> str:
    lines = []
    lines.append("Head & Neck Cancer: RT vs Chemo-RT Summary")
    lines.append("=" * 60)
    lines.append("")

    # Patient snapshot
    lines.append("Patient snapshot (as entered):")
    for k, v in patient.items():
        lines.append(f"  - {k}: {v}")
    lines.append("")

    # RMST
    rmst_t = rmst_res.get("rmst_treat", np.nan)
    rmst_c = rmst_res.get("rmst_control", np.nan)
    delta_m = rmst_res.get("delta", np.nan)

    lines.append(f"Restricted mean event-free survival up to {horizon_months} months:")
    lines.append(f"  - RT alone:    {rmst_c:.1f} months" if not np.isnan(rmst_c) else "  - RT alone:    N/A")
    lines.append(f"  - Chemo-RT:    {rmst_t:.1f} months" if not np.isnan(rmst_t) else "  - Chemo-RT:    N/A")
    lines.append(f"  - ΔRMST (Chemo-RT − RT): {delta_m:+.1f} months" if not np.isnan(delta_m) else "  - ΔRMST:        N/A")
    lines.append("")

    # Survival at horizon (if available)
    p_rt, p_chemo = np.nan, np.nan
    if surv_df is not None and not surv_df.empty:
        s = surv_df.sort_values("days").copy()
        h_days = horizon_months * 30.0
        s_h = s[s["days"] <= h_days]
        if not s_h.empty:
            p_rt = s_h["S_control"].iloc[-1]
            p_chemo = s_h["S_treat"].iloc[-1]

    if not np.isnan(p_rt) and not np.isnan(p_chemo):
        lines.append(f"Estimated probability of being alive & event-free at {horizon_months} months:")
        lines.append(f"  - RT alone:    {p_rt*100:.0f}%")
        lines.append(f"  - Chemo-RT:    {p_chemo*100:.0f}%")
        lines.append("")

    # CATE summary (short)
    if cates:
        lines.append("Horizon-specific absolute risk differences (CATE; Chemo-RT − RT):")
        for h, v in sorted(cates.items(), key=lambda kv: float(kv[0])):
            cate = v.get("CATE")
            if cate is None or np.isnan(cate):
                continue
            lines.append(f"  - {h} months: {cate*100:.1f} percentage points")
        lines.append("")

    # Patient-facing summary
    lines.append("Clinical summary (patient-facing):")
    lines.append("")
    lines.append(generate_patient_summary(patient, rmst_res, surv_df, horizon_months))
    lines.append("")

    lines.append("Notes:")
    lines.append("  - These estimates are model-based and derived from retrospective data.")
    lines.append("  - They should be interpreted together with clinical judgment, comorbidities, and patient preferences.")
    lines.append("")

    return "\n".join(lines)

# ----------------- LAYOUT: TABS -----------------
tab_patient, tab_timecourse, tab_insights = st.tabs(
    ["👤 Single patient", "⏱ Population time-course", "🧠 AI Insights"]
)


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
            "Planned treatment strategy (both will be modelled for comparison)",
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
                rmst_res = {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}
            else:
                # Convert days -> months for display
                surv_plot = surv.copy()
                surv_plot["months"] = surv_plot["days"] / 30.0

                # Plot survival curves
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=surv_plot["months"], y=surv_plot["S_control"],
                    mode="lines+markers", name="RT alone",
                    line=dict(color=COLOR_RT),
                    marker=dict(color=COLOR_RT)
                ))
                fig.add_trace(go.Scatter(
                    x=surv_plot["months"], y=surv_plot["S_treat"],
                    mode="lines+markers", name="Chemo-RT",
                    line=dict(color=COLOR_CHEMO),
                    marker=dict(color=COLOR_CHEMO)
                ))
                fig.update_layout(
                    xaxis_title="Time since RT start (months)",
                    yaxis_title="Probability alive & event-free",
                    yaxis=dict(range=[0, 1]),
                    legend_title="Strategy"
                )
                st.plotly_chart(fig, use_container_width=True)

                c1_, c2_ = st.columns([2, 1])
                with c1_:
                    st.markdown("**Table snapshot (first few time points):**")
                    st.dataframe(
                        surv_plot[["period", "months", "S_control", "S_treat"]].head(),
                        use_container_width=True
                    )
                with c2_:
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

                m1_, m2_, m3_ = st.columns(3)
                m1_.metric("RT alone: event-free months", f"{rmst_c:.1f}" if not np.isnan(rmst_c) else "N/A")
                m2_.metric("Chemo-RT: event-free months", f"{rmst_t:.1f}" if not np.isnan(rmst_t) else "N/A")
                m3_.metric("ΔRMST (Chemo-RT − RT)", f"{delta_m:+.1f} months" if not np.isnan(delta_m) else "N/A")

                st.markdown(interpret_delta_months(delta_m))

            # ---- SECTION B: CATE PER HORIZON ----
            st.subheader("4. Horizon-specific treatment effect (CATE: absolute risk difference)")

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
                        xaxis_title="Horizon (months)",
                        yaxis_title="Absolute risk difference (Chemo-RT − RT, % points)",
                        title="Per-horizon CATE for this patient"
                    )
                    st.plotly_chart(fig_c, use_container_width=True)

                    st.markdown(describe_cate_table(cates))

                if not errors_cate.empty:
                    with st.expander("Technical issues for some horizons"):
                        st.table(errors_cate[["horizon_label", "error"]])

            # ---- SECTION C: Clinical summary card + download ----
            st.subheader("5. Clinical summary (patient-facing text)")

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

# ---------- TAB 2: POPULATION TIME-COURSE ----------
with tab_timecourse:
    st.subheader("Population-level pattern of treatment effect over time")

    st.markdown("""
This panel summarizes how the **relative and absolute effects** of Chemo-RT vs RT  
behave over time in the study population.

It shows:
- **Interval event rates** (hazards) under each strategy  
- **Time-varying hazard ratio (HR)** with confidence intervals  
- How each time window contributes to the overall **difference in event-free time (ΔRMST)**  

These summaries are not patient-specific, but help you understand **when** treatment intensity  
seems most influential in the underlying data.
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
                    mode="lines+markers", name="RT hazard",
                    line=dict(color=COLOR_RT),
                    marker=dict(color=COLOR_RT)
                ))
            if "haz_treated" in tv.columns:
                fig_h.add_trace(go.Scatter(
                    x=tv["months"], y=tv["haz_treated"],
                    mode="lines+markers", name="Chemo-RT hazard",
                    line=dict(color=COLOR_CHEMO),
                    marker=dict(color=COLOR_CHEMO)
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
                    mode="lines+markers", name="HR (Chemo-RT / RT)",
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
            name="ΔRMST contribution (months)",
            marker_color=COLOR_BENEFIT
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
