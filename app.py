import os
from io import BytesIO
from typing import Optional

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
        "These options are mainly for developers. Defaults should work for routine use."
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

# ----------------- SMALL HELPERS (kept minimal) -----------------
def compute_rmst_from_survival(surv_df: pd.DataFrame, horizon_months: int) -> dict:
    if surv_df is None or surv_df.empty:
        return {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}
    h_days = horizon_months * INTERVAL_DAYS
    s = surv_df.sort_values("days").copy()
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
    return {"rmst_treat": rmst_t / 30.0, "rmst_control": rmst_c / 30.0, "delta": (rmst_t - rmst_c) / 30.0}


def describe_cate_table(cates: dict) -> str:
    vals = [(h, v["CATE"]) for h, v in cates.items() if v.get("CATE") is not None and not np.isnan(v.get("CATE"))]
    if not vals:
        return "No reliable horizon-specific risk differences available."
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
    best_h = mh_arr[idx_benefit]; best_v = arr[idx_benefit]
    worst_h = mh_arr[idx_harm]; worst_v = arr[idx_harm]
    text = []
    if best_v < 0:
        text.append(f"- Largest estimated benefit ~{best_h:.0f}m: Chemo-RT reduces event risk by ~{abs(best_v)*100:.1f} percentage points.")
    if worst_v > 0:
        text.append(f"- Largest potential harm ~{worst_h:.0f}m: Chemo-RT increases event risk by ~{worst_v*100:.1f} percentage points.")
    if not text:
        text.append("No horizon stands out as clearly better or worse.")
    return "\n".join(text)


# ----------------- UI LAYOUT: TABS -----------------
tab_patient, tab_timecourse, tab_insights = st.tabs(
    ["👤 Patient decision aid", "⏱ Effect over time", "🧠 Patterns from data"]
)

# ----------------- TAB: Patient -----------------
with tab_patient:
    st.subheader("1. Enter patient details")

    with st.expander("What this page provides", expanded=True):
        st.markdown("""
- Personalized estimate of time-alive-and-well (RT vs Chemo-RT)
- Per-horizon risk differences (CATEs) from causal forests (if available)
- Downloadable one-page summary
""")

    # Form (keeps TNM fields; we supply t/n/m as 't','n','m' keys)
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", value=62, min_value=18, max_value=99)
            sex = st.selectbox("Sex", ["Male", "Female", "Missing"], index=0)
            ecog_ps = st.selectbox("ECOG performance status", [0, 1, 2, 3], index=0)
        with c2:
            primary_site_group = st.selectbox("Primary site", ["Oropharynx", "Nasopharynx", "Other_HNC", "Missing"], index=0)
            pathology_group = st.selectbox("Histology", ["SCC", "Other_epithelial", "Other_rare", "Missing"], index=0)
            smoking_status_clean = st.selectbox("Smoking status", ["Current", "Ex-Smoker", "Non-Smoker", "Unknown", "Missing"], index=1)
            smoking_py_clean = st.number_input("Smoking pack-years (approx.)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
        with c3:
            hpv_clean = st.selectbox("HPV status", ["HPV_Positive", "HPV_Negative", "HPV_Unknown", "Missing"], index=0)
            stage = st.selectbox("Overall stage", ["I", "II", "III", "IV", "Missing"], index=2)
        st.markdown("**TNM classification**")
        c4, c5, c6 = st.columns(3)
        with c4:
            t_cat = st.selectbox("T category", ["T1", "T2", "T3", "T4", "Tx"], index=1)
        with c5:
            n_cat = st.selectbox("N category", ["N0", "N1", "N2", "N3", "Nx"], index=0)
        with c6:
            m_cat = st.selectbox("M category", ["M0", "M1", "Mx"], index=0)

        treatment = st.selectbox(
            "Planned strategy (both options modelled)",
            options=[0, 1],
            format_func=lambda x: "RT alone" if x == 0 else "Chemo-RT",
            index=0
        )

        submitted = st.form_submit_button("Estimate outcomes")

    if submitted:
        # Build patient dict - include TNM as 't','n','m' keys (infer will accept extras)
        patient = {
            "age": age,
            "sex": sex,
            "primary_site_group": primary_site_group,
            "pathology_group": pathology_group,
            "hpv_clean": hpv_clean,
            "stage": stage,
            "t": t_cat,   # keep TNM fields as keys that may be used by patient_columns
            "n": n_cat,
            "m": m_cat,
            "ecog_ps": ecog_ps,
            "smoking_status_clean": smoking_status_clean,
            "smoking_py_clean": smoking_py_clean,
            "treatment": int(treatment),
            "patient_id": "new"
        }

        st.info("Running models — this may take a few seconds on first run (artifacts may be loaded).")
        # Defensive inference call
        try:
            out = infer_new_patient_fixed(
                patient_data=patient,
                outdir=OUTDIR,
                base_url=BASE_URL,
                max_period_override=int(max_period_months),
                return_raw=True
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")
            out = {"survival_curve": None, "CATEs": {}, "errors": {"exception": str(e)}, "debug": {}}

        errors = out.get("errors", {}) or {}
        debug = out.get("debug", {}) or {}
        surv = out.get("survival_curve")
        cates = out.get("CATEs", {})

        # Show technical notes if any
        if errors:
            with st.expander("Technical notes / warnings (developer)", expanded=False):
                for k, v in errors.items():
                    st.write(f"- **{k}**: {v}")
        # Show artifact sources (from infer debug) for debugging
        if debug.get("artifact_sources"):
            with st.expander("Artifact sources (developer)", expanded=False):
                for k, v in debug.get("artifact_sources", {}).items():
                    st.write(f"- {k}: {v}")

        # ---------- SURVIVAL & SUMMARY ----------
        st.subheader("2. Summary at your chosen time point")
        if surv is None or (hasattr(surv, "empty") and surv.empty):
            st.warning("Survival curve could not be computed for this patient.")
            rmst_res = {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}
            p_rt = p_chemo = np.nan
        else:
            rmst_res = compute_rmst_from_survival(surv, rmst_horizon_months)
            rmst_t = rmst_res["rmst_treat"]; rmst_c = rmst_res["rmst_control"]; delta_m = rmst_res["delta"]
            # survival probs at horizon
            s = surv.sort_values("days").copy()
            h_days = rmst_horizon_months * 30.0
            s_h = s[s["days"] <= h_days]
            if not s_h.empty:
                p_rt = s_h["S_control"].iloc[-1]; p_chemo = s_h["S_treat"].iloc[-1]
            else:
                p_rt = p_chemo = np.nan

            # metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg time alive & well (RT)", f"{rmst_c:.1f} m" if not np.isnan(rmst_c) else "N/A")
            c2.metric("Avg time alive & well (Chemo-RT)", f"{rmst_t:.1f} m" if not np.isnan(rmst_t) else "N/A")
            c3.metric("Extra time with Chemo-RT", f"{delta_m:+.1f} m" if not np.isnan(delta_m) else "N/A")
            if not np.isnan(p_rt) and not np.isnan(p_chemo):
                diff_surv = (p_chemo - p_rt) * 100.0
                c4.metric(f"Difference at {rmst_horizon_months} m", f"{diff_surv:+.1f} %pts")
            else:
                c4.metric(f"Difference at {rmst_horizon_months} m", "N/A")

            # interpretation box (short)
            if not np.isnan(delta_m):
                if delta_m > 0:
                    lbl = f"Estimated small-to-moderate gain of {delta_m:.1f} months with Chemo-RT (by chosen horizon)."
                elif delta_m < 0:
                    lbl = f"Estimated disadvantage of {abs(delta_m):.1f} months with Chemo-RT (by chosen horizon)."
                else:
                    lbl = "No meaningful difference predicted between strategies."
            else:
                lbl = "Model could not compute a reliable ΔRMST."

            st.markdown(f"<div style='padding:10px;border:1px solid #eee;border-radius:8px;background:#f8fafc'>{lbl}</div>", unsafe_allow_html=True)

        # Plot survival if available
        st.markdown("---")
        if surv is not None and not (hasattr(surv, "empty") and surv.empty):
            surv_plot = surv.copy()
            surv_plot["months"] = surv_plot["days"] / 30.0

            st.subheader("3. Survival over time (RT vs Chemo-RT)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=surv_plot["months"], y=surv_plot["S_control"],
                mode="lines+markers", name="RT alone", line=dict(color=COLOR_RT)
            ))
            fig.add_trace(go.Scatter(
                x=surv_plot["months"], y=surv_plot["S_treat"],
                mode="lines+markers", name="Chemo-RT", line=dict(color=COLOR_CHEMO)
            ))
            fig.update_layout(xaxis_title="Months", yaxis_title="Prob alive & well", yaxis=dict(range=[0,1]))
            st.plotly_chart(fig, use_container_width=True)

            # survival table download
            csv_buf = surv_plot.to_csv(index=False)
            st.download_button("Download survival table (CSV)", data=csv_buf, file_name="survival_curve.csv", mime="text/csv")

        # CATEs panel
        st.markdown("---")
        st.subheader("4. Time-specific change in chance (CATEs)")
        if not cates:
            st.info("No per-horizon CATEs available for this patient.")
        else:
            rows = []
            for h, v in cates.items():
                try:
                    mh = float(h)
                except Exception:
                    mh = np.nan
                rows.append({"horizon_months": mh, "CATE": v.get("CATE"), "error": v.get("error")})
            df_cate = pd.DataFrame(rows).sort_values("horizon_months")
            if not df_cate.empty:
                # show bar chart of percent-points
                df_cate["CATE_pct"] = df_cate["CATE"] * 100.0
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(x=df_cate["horizon_months"], y=df_cate["CATE_pct"],
                                       marker_color=[COLOR_BENEFIT if x<0 else COLOR_HARM for x in df_cate["CATE_pct"]]))
                fig_c.update_layout(xaxis_title="Months", yaxis_title="Chemo-RT − RT (% points)")
                st.plotly_chart(fig_c, use_container_width=True)
                st.dataframe(df_cate, use_container_width=True)
                st.markdown(describe_cate_table(cates))
            else:
                st.info("CATE table empty.")

        # Printable summary (text)
        st.markdown("---")
        st.subheader("5. Downloadable summary")
        def build_printable(patient, rmst_res, surv_df, horizon_months, cates):
            lines = []
            lines.append("Patient summary")
            lines.append("="*40)
            lines.append("")
            for k, v in patient.items():
                lines.append(f"- {k}: {v}")
            lines.append("")
            if rmst_res:
                lines.append(f"RT avg time: {rmst_res.get('rmst_control'):.1f} m" if not np.isnan(rmst_res.get('rmst_control', np.nan)) else "RT avg time: N/A")
                lines.append(f"Chemo-RT avg time: {rmst_res.get('rmst_treat'):.1f} m" if not np.isnan(rmst_res.get('rmst_treat', np.nan)) else "Chemo-RT avg time: N/A")
                lines.append(f"ΔRMST: {rmst_res.get('delta'):+.2f} m")
            lines.append("")
            return "\n".join(lines)

        printable = build_printable(patient, rmst_res if 'rmst_res' in locals() else None, surv, rmst_horizon_months, cates)
        st.download_button("Download 1-page summary (text)", data=printable, file_name="hnc_summary.txt", mime="text/plain")

# ----------------- TAB: Population effect over time -----------------
with tab_timecourse:
    st.subheader("How treatment effect changes over time (population-level)")
    tv = None
    try:
        # try local then remote using helper pattern
        local_path = os.path.join(OUTDIR, "timevarying_summary_by_period.csv")
        if os.path.exists(local_path):
            tv = pd.read_csv(local_path)
        else:
            if BASE_URL:
                try:
                    tv = pd.read_csv(BASE_URL.rstrip("/") + "/timevarying_summary_by_period.csv")
                except Exception:
                    tv = None
    except Exception:
        tv = None

    if tv is None or tv.empty:
        st.info("`timevarying_summary_by_period.csv` not found.")
    else:
        tv = tv.copy()
        tv["months"] = tv["period"] * (INTERVAL_DAYS / 30.0)
        tv["delta_rmst_period_months"] = tv["delta_rmst_period_days"] / 30.0 if "delta_rmst_period_days" in tv.columns else np.nan

        fig_h = go.Figure()
        if "haz_control" in tv.columns:
            fig_h.add_trace(go.Scatter(x=tv["months"], y=tv["haz_control"], mode="lines+markers", name="RT", line=dict(color=COLOR_RT)))
        if "haz_treated" in tv.columns:
            fig_h.add_trace(go.Scatter(x=tv["months"], y=tv["haz_treated"], mode="lines+markers", name="Chemo-RT", line=dict(color=COLOR_CHEMO)))
        fig_h.update_layout(xaxis_title="Months", yaxis_title="Event prob per interval")
        st.plotly_chart(fig_h, use_container_width=True)

        if {"hr", "hr_lo", "hr_up"}.issubset(tv.columns):
            fig_hr = go.Figure()
            fig_hr.add_trace(go.Scatter(x=tv["months"], y=tv["hr"], mode="lines+markers", name="HR", line=dict(color=COLOR_CHEMO)))
            fig_hr.add_trace(go.Scatter(x=np.concatenate([tv["months"], tv["months"][::-1]]),
                                        y=np.concatenate([tv["hr_lo"], tv["hr_up"][::-1]]),
                                        fill="toself", line=dict(width=0), opacity=0.2, name="95% CI"))
            fig_hr.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig_hr.update_layout(xaxis_title="Months", yaxis_title="Chemo-RT / RT")
            st.plotly_chart(fig_hr, use_container_width=True)

# ----------------- TAB: Patterns from data -----------------
with tab_insights:
    st.subheader("Patterns from the training data")
    subgroup_df = None
    try:
        local_path = os.path.join(OUTDIR, "subgroup_summary_cates.csv")
        if os.path.exists(local_path):
            subgroup_df = pd.read_csv(local_path)
        else:
            if BASE_URL:
                try:
                    subgroup_df = pd.read_csv(BASE_URL.rstrip("/") + "/subgroup_summary_cates.csv")
                except Exception:
                    subgroup_df = None
    except Exception:
        subgroup_df = None

    if subgroup_df is None or subgroup_df.empty:
        st.info("`subgroup_summary_cates.csv` not found.")
    else:
        st.dataframe(subgroup_df.head(20), use_container_width=True)
        st.markdown("Use subgroup summaries together with the patient-level estimate to sense-check decisions.")
