
import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from textwrap import dedent
from utils.infer import infer_new_patient_fixed

# ----------------- CONFIG -----------------
DEFAULT_OUTDIR = "outputs"
DEFAULT_BASE_URL = ""  # if you want remote fallback (raw github path)
INTERVAL_DAYS = 30
DEFAULT_MAX_MONTHS = 60

st.set_page_config(page_title="H&N Chemo-RT Decision Aid", layout="wide")

# ----------------- HELPERS: artifact loading -----------------

def load_joblib_with_fallback_local_then_remote(path, base_url=None):
    """Try a local path first, otherwise attempt to fetch a joblib from base_url+path."""
    if os.path.exists(path):
        try:
            return joblib.load(path), f"local:{path}"
        except Exception as e:
            return None, f"local_failed:{path}:{e}"
    if base_url:
        url = base_url.rstrip('/') + '/' + os.path.basename(path)
        try:
            import requests
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return joblib.load(io.BytesIO(r.content)), f"remote:{url}"
        except Exception as e:
            return None, f"remote_failed:{url}:{e}"
    return None, None


def load_csv_with_fallback(path, base_url=None):
    if os.path.exists(path):
        try:
            return pd.read_csv(path), f"local:{path}"
        except Exception as e:
            return None, f"local_failed:{path}:{e}"
    if base_url:
        url = base_url.rstrip('/') + '/' + os.path.basename(path)
        try:
            import requests
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text)), f"remote:{url}"
        except Exception as e:
            return None, f"remote_failed:{url}:{e}"
    return None, None


# ----------------- UI: sidebar settings -----------------

st.sidebar.header("Data & model settings")
OUTDIR = st.sidebar.text_input("Local outputs folder", value=DEFAULT_OUTDIR)
BASE_URL = st.sidebar.text_input("Optional BASE_URL for remote artifacts (raw GitHub path)", value=DEFAULT_BASE_URL)

max_period_months = st.sidebar.number_input("Max follow-up (months)", value=DEFAULT_MAX_MONTHS, min_value=6, max_value=156, step=6)
rmst_horizon_months = st.sidebar.number_input("RMST/Horizon months (summary)", value=36, min_value=6, max_value=int(max_period_months), step=6)

show_dev = st.sidebar.checkbox("Developer view: show diagnostics", value=False)

# ----------------- Load artifacts -----------------
ART = {
    'model_columns': os.path.join(OUTDIR, 'pooled_logit_model_columns.csv'),
    'pooled_logit': os.path.join(OUTDIR, 'pooled_logit_logreg_saga.joblib'),
    'pp_scaler': os.path.join(OUTDIR, 'pp_scaler.joblib'),
    'pp_train_medians': os.path.join(OUTDIR, 'pp_train_medians.joblib'),
    'collapse_maps': os.path.join(OUTDIR, 'pp_collapse_maps.joblib'),
    'patient_columns': os.path.join(OUTDIR, 'causal_patient_columns.joblib'),
    'forests_bundle': os.path.join(OUTDIR, 'causal_forests_period_horizons_patient_level.joblib')
}

artifact_sources = {}
artifacts = {}

# CSV model_columns: special handling to allow header/no-header
mc_df, mc_src = load_csv_with_fallback(ART['model_columns'], base_url=BASE_URL)
artifact_sources['model_columns'] = mc_src
if mc_df is None:
    artifacts['model_columns'] = None
else:
    # read as single column strings
    try:
        cols = mc_df.iloc[:, 0].astype(str).tolist()
        if cols and cols[0].strip().lower() in ("model_columns", "column", "columns", "feature", "features"):
            cols = cols[1:]
        cols = [c.strip() for c in cols if isinstance(c, str) and c.strip()]
        artifacts['model_columns'] = cols
    except Exception:
        artifacts['model_columns'] = None

# load joblib artifacts
for name in ['pooled_logit','pp_scaler','pp_train_medians','collapse_maps','patient_columns','forests_bundle']:
    val, src = load_joblib_with_fallback_local_then_remote(ART[name], base_url=BASE_URL)
    artifact_sources[name] = src
    artifacts[name] = val

# quick dev diagnostics
if show_dev:
    st.sidebar.markdown("**Artifact sources (dev)**")
    for k, v in artifact_sources.items():
        st.sidebar.text(f"{k}: {v}")

# ----------------- small helper functions -----------------

def interpret_delta_months(delta_m: float) -> str:
    if np.isnan(delta_m):
        return "Model could not compute a reliable difference."
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
        return f"Chemo-RT estimated to give a {size} gain of about {delta_m:.1f} months."
    elif delta_m < 0:
        return f"Chemo-RT estimated to be worse by about {mag:.1f} months."
    else:
        return "No meaningful difference predicted."


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
    rmst_c = np.sum(S_c_start * dt) / 30.0
    rmst_t = np.sum(S_t_start * dt) / 30.0
    return {"rmst_treat": rmst_t, "rmst_control": rmst_c, "delta": (rmst_t - rmst_c)}


# ----------------- APP LAYOUT -----------------
st.title("Head & Neck Cancer — Chemo‑RT Decision Aid")
st.markdown(
    """
This tool compares **Radiotherapy alone (RT)** vs **Concurrent Chemo‑RT** using pooled-logit survival models
and patient-level CATEs (if forests are available). Figures are model-based estimates from retrospective data.
"""
)

# Patient form
with st.form("patient_form"):
    st.subheader("Enter patient details")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age (years)", value=62, min_value=18, max_value=100)
        sex = st.selectbox("Sex", ["Male", "Female", "Missing"], index=0)
        ecog_ps = st.selectbox("ECOG (0–3)", [0,1,2,3], index=0)
    with c2:
        primary_site_group = st.selectbox("Primary site", ["Oropharynx","Nasopharynx","Other_HNC","Missing"], index=0)
        pathology_group = st.selectbox("Histology", ["SCC","Other_epithelial","Other_rare","Missing"], index=0)
        smoking_status_clean = st.selectbox("Smoking status", ["Current","Ex-Smoker","Non-Smoker","Unknown","Missing"], index=1)
        smoking_py_clean = st.number_input("Smoking pack-years (approx)", min_value=0.0, max_value=500.0, value=20.0)
    with c3:
        hpv_clean = st.selectbox("HPV status", ["HPV_Positive","HPV_Negative","HPV_Unknown","Missing"], index=0)
        stage = st.selectbox("Overall stage", ["I","II","III","IV","Missing"], index=2)
        treatment_choice = st.selectbox("Planned strategy (we'll model both)", options=[0,1], format_func=lambda x: "RT alone" if x==0 else "Chemo-RT", index=0)
    st.markdown("**TNM (optional)**")
    c4, c5, c6 = st.columns(3)
    with c4:
        t_cat = st.text_input("T category (e.g. T2)", value="T2")
    with c5:
        n_cat = st.text_input("N category (e.g. N0)", value="N0")
    with c6:
        m_cat = st.text_input("M category (e.g. M0)", value="M0")

    submitted = st.form_submit_button("Estimate outcomes")


# ----------------- ON SUBMIT -----------------
if submitted:
    # Build patient dict; include tn,m as strings but not break pipeline
    patient = {
        "age": float(age),
        "sex": sex,
        "primary_site_group": primary_site_group,
        "pathology_group": pathology_group,
        "hpv_clean": hpv_clean,
        "stage": stage,
        "t": t_cat,  # preserve TNM strings, not used by pooled-logit dummies unless expected
        "n": n_cat,
        "m": m_cat,
        "ecog_ps": int(ecog_ps),
        "smoking_status_clean": smoking_status_clean,
        "smoking_py_clean": float(smoking_py_clean),
        "treatment": int(treatment_choice),
        "patient_id": "new"
    }

    # Run inference (pooled-logit + survival). Use return_raw to get survival df & debug
    st.info("Running models — this may take a few seconds")
    try:
        out = infer_new_patient_fixed(patient, max_months=int(max_period_months), return_raw=True)
    except TypeError:
        # fallback in case infer returns only RMST by default
        out = {"survival_curve": None}
        rmst_res = {'rmst_treat': np.nan, 'rmst_control': np.nan, 'delta': np.nan}

    surv = out.get('survival_curve')
    cates = out.get('CATEs', {})
    errors = out.get('errors', {})

    # Compute RMSTs
    if surv is not None and not surv.empty:
        rmst_res = compute_rmst_from_survival(surv, rmst_horizon_months)
    else:
        rmst_res = {'rmst_treat': np.nan, 'rmst_control': np.nan, 'delta': np.nan}

    # Display summary cards
    st.subheader("Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg time alive & well (RT)", f"{rmst_res['rmst_control']:.1f} m" if not np.isnan(rmst_res['rmst_control']) else "N/A")
    c2.metric("Avg time alive & well (Chemo-RT)", f"{rmst_res['rmst_treat']:.1f} m" if not np.isnan(rmst_res['rmst_treat']) else "N/A")
    c3.metric("Extra time with Chemo-RT", f"{rmst_res['delta']:+.1f} m" if not np.isnan(rmst_res['delta']) else "N/A")
    if surv is not None and not surv.empty:
        s_h = surv[surv['days'] <= (rmst_horizon_months * INTERVAL_DAYS)]
        if not s_h.empty:
            p_rt = s_h['S_control'].iloc[-1]
            p_chemo = s_h['S_treat'].iloc[-1]
            c4.metric(f"Chance alive & well at {rmst_horizon_months} m", f"RT {p_rt*100:.0f}% | Chemo {p_chemo*100:.0f}%")
        else:
            c4.metric(f"Chance alive & well at {rmst_horizon_months} m", "N/A")
    else:
        c4.metric(f"Chance alive & well at {rmst_horizon_months} m", "N/A")

    # Interpretation string
    st.markdown("---")
    st.markdown(f"**Interpretation:** {interpret_delta_months(rmst_res['delta'])}")

    # Show survival plot
    if surv is not None and not surv.empty:
        surv_plot = surv.copy()
        surv_plot['months'] = surv_plot['days'] / 30.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=surv_plot['months'], y=surv_plot['S_control'], name='RT alone', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=surv_plot['months'], y=surv_plot['S_treat'], name='Chemo-RT', mode='lines+markers'))
        fig.update_layout(title='Estimated survival (alive & well)', xaxis_title='Months', yaxis_title='Probability', yaxis=dict(range=[0,1]))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander('Survival table (first rows)'):
            st.dataframe(surv_plot[['period','months','S_control','S_treat']].head(20))

    else:
        st.warning('Survival curve not available for this patient (check artifacts).')

    # CATEs (from forests) if available
    st.markdown('---')
    st.subheader('Change in chance of being alive & well (Chemo-RT − RT)')
    if cates and isinstance(cates, dict):
        rows = []
        for h, v in cates.items():
            cate = v.get('CATE') if isinstance(v, dict) else v
            try:
                mh = float(h)
            except Exception:
                mh = np.nan
            rows.append({'horizon': mh, 'CATE': cate})
        df_cate = pd.DataFrame(rows).sort_values('horizon')
        df_cate['CATE_pct'] = df_cate['CATE'] * 100
        if not df_cate.empty:
            figc = go.Figure()
            figc.add_trace(go.Bar(x=df_cate['horizon'], y=df_cate['CATE_pct'], text=[f"{v:.1f}%" if not np.isnan(v) else 'NA' for v in df_cate['CATE_pct']], textposition='outside'))
            figc.update_layout(xaxis_title='Months', yaxis_title='Chemo-RT − RT (percentage points)', title='Estimated risk difference by horizon')
            st.plotly_chart(figc, use_container_width=True)
        else:
            st.info('CATEs present but empty.')
    else:
        st.info('No CATEs available (forests bundle missing or failed).')

    # Downloadable summary
    def build_print_summary(patient, rmst_res, surv_df, cates):
        lines = []
        lines.append('Head & Neck Cancer: RT vs Chemo-RT Summary')
        lines.append('='*60)
        lines.append('Patient snapshot:')
        for k,v in patient.items():
            lines.append(f" - {k}: {v}")
        lines.append('')
        lines.append(f"Avg time alive & well (up to {rmst_horizon_months} m):")
        lines.append(f" - RT alone: {rmst_res['rmst_control']:.1f} m" if not np.isnan(rmst_res['rmst_control']) else ' - RT alone: N/A')
        lines.append(f" - Chemo-RT: {rmst_res['rmst_treat']:.1f} m" if not np.isnan(rmst_res['rmst_treat']) else ' - Chemo-RT: N/A')
        lines.append(f" - Difference (Chemo-RT − RT): {rmst_res['delta']:+.1f} m" if not np.isnan(rmst_res['delta']) else ' - Difference: N/A')
        lines.append('')
        if surv_df is not None and not surv_df.empty:
            s_h = surv_df[surv_df['days'] <= (rmst_horizon_months * INTERVAL_DAYS)]
            if not s_h.empty:
                p_rt = s_h['S_control'].iloc[-1]
                p_chemo = s_h['S_treat'].iloc[-1]
                lines.append(f"Estimated probability alive & well at {rmst_horizon_months}m: RT {p_rt*100:.0f}%, Chemo {p_chemo*100:.0f}%")
        if cates:
            lines.append('\nCATEs (Chemo-RT − RT):')
            for h,v in (cates.items() if isinstance(cates, dict) else []):
                lines.append(f" - {h} m: {v.get('CATE') if isinstance(v, dict) else v}")
        lines.append('\nNote: these are model-based estimates from retrospective data.')
        return '\n'.join(lines)

    summary_txt = build_print_summary(patient, rmst_res, surv, cates)
    st.download_button('Download 1-page summary', summary_txt, file_name='hnc_summary.txt')

    # Developer diagnostics
    if show_dev:
        st.subheader('Developer diagnostics')
        st.markdown('**Artifact sources**')
        st.json(artifact_sources)
        st.markdown('**Raw errors from inference**')
        st.json(errors)

        # show X_pp if build_X_for_pp exists within utils.infer
        try:
            from utils.infer import build_X_for_pp
            df_pp = expand_patient_to_pp(patient, max_months=int(max_period_months))
            X_pp = build_X_for_pp(df_pp)
            st.markdown('Sample X_pp (first rows)')
            st.dataframe(X_pp.head(8))
        except Exception as e:
            st.text('build_X_for_pp not available or failed: ' + str(e))

# End

