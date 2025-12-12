
import os
import io
import joblib
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.infer import infer_new_patient_fixed, build_canonical_Xpatient

# ---------------- config ----------------
DEFAULT_BASE_URL = "https://raw.githubusercontent.com/Doclikam/Causal_ML_deployment/main/outputs/"
DEFAULT_OUTDIR = "outputs"
INTERVAL_DAYS = 30

st.set_page_config(page_title="H&N Chemo-RT Decision Aid", layout='wide')
st.title("Head & Neck Cancer – Chemo-RT Decision Aid")

# Sidebar
if 'show_advanced_sidebar' not in st.session_state:
    st.session_state.show_advanced_sidebar = False

st.sidebar.header("Settings")
max_period_months = st.sidebar.number_input("Max follow-up (months)", value=60, min_value=6, max_value=156, step=6)
rmst_horizon_months = st.sidebar.number_input("Key time point (months)", value=36, min_value=6, max_value=int(max_period_months), step=6)

BASE_URL = st.sidebar.text_input("BASE_URL", value=DEFAULT_BASE_URL)
OUTDIR = st.sidebar.text_input("OUTDIR", value=DEFAULT_OUTDIR)

# helpers

def load_csv_with_fallback(filename: str):
    local = os.path.join(OUTDIR, filename)
    if os.path.exists(local):
        try:
            return pd.read_csv(local)
        except Exception:
            pass
    if BASE_URL:
        url = BASE_URL.rstrip('/') + '/' + filename
        try:
            return pd.read_csv(url)
        except Exception:
            return None
    return None


def load_joblib_with_fallback(filename: str):
    local = os.path.join(OUTDIR, filename)
    if os.path.exists(local):
        try:
            return joblib.load(local)
        except Exception:
            pass
    if BASE_URL:
        url = BASE_URL.rstrip('/') + '/' + filename
        try:
            r = requests.get(url, timeout=30); r.raise_for_status()
            return joblib.load(io.BytesIO(r.content))
        except Exception:
            return None
    return None


def compute_rmst_from_survival(surv_df: pd.DataFrame, horizon_months: int):
    if surv_df is None or getattr(surv_df, 'empty', True):
        return {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}
    h_days = horizon_months * INTERVAL_DAYS
    s = surv_df.sort_values('days').copy()
    s = s[s['days'] <= h_days].copy()
    if s.empty:
        return {"rmst_treat": np.nan, "rmst_control": np.nan, "delta": np.nan}
    times = s['days'].values
    S_c_end = s['S_control'].values
    S_t_end = s['S_treat'].values
    t0 = np.concatenate(([0.0], times[:-1]))
    dt = times - t0
    S_c_start = np.concatenate(([1.0], S_c_end[:-1]))
    S_t_start = np.concatenate(([1.0], S_t_end[:-1]))
    rmst_c = np.sum(S_c_start * dt)
    rmst_t = np.sum(S_t_start * dt)
    return {"rmst_treat": rmst_t/30.0, "rmst_control": rmst_c/30.0, "delta": (rmst_t-rmst_c)/30.0}


def dev_show_inference_debug(out: dict):
    st.markdown('### ⚙️ Developer diagnostics')
    st.write('Errors:')
    st.json(out.get('errors', {}))
    debug = out.get('debug', {}) or {}
    st.write('Artifact sources:')
    st.json(debug.get('artifact_sources', {}))
    # show Xpatient if present
    xpd = debug.get('Xpatient_app') or debug.get('Xpatient_debug') or debug.get('Xpatient')
    if xpd is not None:
        try:
            xp_df = pd.DataFrame(xpd) if not isinstance(xpd, pd.DataFrame) else xpd
            st.write('Xpatient (single-row):')
            st.dataframe(xp_df.T)
        except Exception as e:
            st.write('Xpatient present but could not display:', str(e))

# ---------------- form ----------------
with st.form('patient_form'):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input('Age', value=62, min_value=18)
        sex = st.selectbox('Sex', ['Male','Female','Missing'], index=1)
        ecog_ps = st.selectbox('ECOG', [0,1,2,3], index=0)
    with c2:
        primary_site_group = st.selectbox('Primary site', ['Oropharynx','Nasopharynx','Other_HNC','Missing'], index=0)
        pathology_group = st.selectbox('Histology', ['SCC','Other_epithelial','Other_rare','Missing'], index=0)
        smoking_status_clean = st.selectbox('Smoking', ['Current','Ex-Smoker','Non-Smoker','Unknown','Missing'], index=1)
        smoking_py_clean = st.number_input('Smoking pack-years', value=20.0, min_value=0.0)
    with c3:
        hpv_clean = st.selectbox('HPV', ['HPV_Positive','HPV_Negative','HPV_Unknown','Missing'], index=0)
        stage = st.selectbox('Stage', ['I','II','III','IV','Missing'], index=2)
        t_cat = st.selectbox('T', ['T1','T2','T3','T4','Tx'], index=1)
    treatment = st.selectbox('Planned strategy', options=[0,1], format_func=lambda x: 'RT alone' if x==0 else 'Chemo-RT')
    submitted = st.form_submit_button('Estimate outcomes')

if submitted:
    patient = dict(age=age, sex=sex, primary_site_group=primary_site_group,
                   pathology_group=pathology_group, hpv_clean=hpv_clean, stage=stage,
                   t=t_cat, ecog_ps=ecog_ps, smoking_status_clean=smoking_status_clean,
                   smoking_py_clean=smoking_py_clean, treatment=treatment)

    # build canonical Xpatient in-app
    Xpatient_app, build_debug = build_canonical_Xpatient(patient, outdir=OUTDIR, base_url=BASE_URL)

    # call infer, pass Xpatient_override so infer uses same features
    out = infer_new_patient_fixed(patient, return_raw=True, outdir=OUTDIR, base_url=BASE_URL,
                                  max_period_override=int(max_period_months), Xpatient_override=Xpatient_app)
    # attach app debug
    out['debug'] = out.get('debug', {})
    out['debug']['Xpatient_app'] = Xpatient_app
    out['debug'].update(build_debug)

    dev_show_inference_debug(out)

    surv = out.get('survival_curve')
    cates = out.get('CATEs', {})

    if surv is None or getattr(surv,'empty',True):
        st.warning('Survival curve not available')
    else:
        surv_df = surv.copy()
        surv_df['months'] = surv_df['days']/30.0
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=surv_df['months'], y=surv_df['S_control'], name='RT'))
        fig.add_trace(go.Scatter(x=surv_df['months'], y=surv_df['S_treat'], name='Chemo-RT'))
        fig.update_layout(xaxis_title='Months', yaxis_title='Probability alive & well')
        st.plotly_chart(fig, use_container_width=True)

        rmst = compute_rmst_from_survival(surv_df, rmst_horizon_months)
        st.metric('Extra time with Chemo-RT (months)', f"{rmst.get('delta', np.nan):+.2f} m")

    # show CATE bar if present
    if cates:
        rows = []
        for h,v in cates.items():
            try:
                mh = float(h)
            except Exception:
                mh = None
            rows.append({'horizon_months': mh, 'CATE': v.get('CATE'), 'error': v.get('error')})
        df_c = pd.DataFrame(rows).dropna(subset=['horizon_months']).sort_values('horizon_months')
        if not df_c.empty:
            df_c['CATE_pct'] = df_c['CATE'] * 100
            figc = go.Figure()
            figc.add_trace(go.Bar(x=df_c['horizon_months'], y=df_c['CATE_pct']))
            figc.update_layout(xaxis_title='Months', yaxis_title='Chemo-RT − RT (% points)')
            st.plotly_chart(figc, use_container_width=True)

st.markdown('---')
st.caption('Developer: artifacts are taken from local OUTDIR first, then from BASE_URL if missing.')
```
