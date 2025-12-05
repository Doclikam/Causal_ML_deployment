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
from utils.infer import infer_new_patient_fixed as infer_new_patient

# Optional: set default OUTDIR (if not already set in infer.py)
DEFAULT_OUTDIR = "outputs"
os.environ.setdefault("OUTDIR", DEFAULT_OUTDIR)

st.set_page_config(page_title="RMST / CATE Clinical Dashboard", layout="wide")
st.title("RMST / CATE Clinical Dashboard")

# Sidebar controls
st.sidebar.header("Settings")
outdir = st.sidebar.text_input("Artifacts folder (OUTDIR)", value=os.environ.get("OUTDIR", DEFAULT_OUTDIR))
st.sidebar.markdown("Make sure `outputs/` contains pooled-logit, model_columns, and forests bundle.")
run_boot = st.sidebar.checkbox("Run heavy bootstraps in app (not recommended)", value=False)

# Preload artifacts once (so infer function doesn't reload every patient)
@st.cache_resource(show_spinner=False)
def _warm_artifacts(outdir_value):
    """
    Try one dummy call that triggers artifact loading inside infer module.
    We call infer with a minimal dummy patient and return artifact sources.
    """
    try:
        dummy = {'age':50, 'sex':'F', 'treatment':0}
        res = infer_new_patient(dummy, return_raw=True)
        return res.get('artifacts_sources', {}), None
    except Exception as e:
        return {}, str(e)

art_src, warm_err = _warm_artifacts(outdir)
if warm_err:
    st.sidebar.error(f"Artifact warm-up failed: {warm_err}")
else:
    if art_src:
        st.sidebar.success("Artifacts loaded (sources):")
        for k,v in art_src.items():
            st.sidebar.text(f"{k}: {v}")

# Mode selection
mode = st.sidebar.radio("Mode", ["Single patient", "Batch CSV"])

# Single-patient UI
if mode == "Single patient":
    st.subheader("Single patient inference")
    with st.form("single_form"):
        pid = st.text_input("patient_id", value="new_001")
        age = st.number_input("age", min_value=0, max_value=120, value=62)
        sex = st.selectbox("sex", ["F","M"], index=0)
        treatment = st.selectbox("treatment (0=RT, 1=Chemo+RT)", [0,1], index=0)
        primary_site_group = st.text_input("primary_site_group", value="Oropharynx")
        pathology_group = st.text_input("pathology_group", value="Squamous")
        hpv_clean = st.selectbox("hpv_clean", ["HPV_Positive","HPV_Negative"], index=0)
        submitted = st.form_submit_button("Run inference")
    if submitted:
        patient = dict(
            patient_id = pid,
            age = age,
            sex = sex,
            treatment = int(treatment),
            primary_site_group = primary_site_group,
            pathology_group = pathology_group,
            hpv_clean = hpv_clean
        )
        with st.spinner("Running inference (this loads model artifacts once)..."):
            res = infer_new_patient(patient, return_raw=True)

        # Show errors / warnings
        if res.get('errors'):
            st.warning("Inference warnings / errors (see details):")
            st.json(res['errors'])

        # Show artifact sources if available
        if 'artifacts_sources' in res:
            st.markdown("**Artifacts loaded from:**")
            st.json(res['artifacts_sources'])

        # Show CATEs nicely
        cates = res.get('CATEs', {})
        if cates:
            cate_df = pd.DataFrame([
                {"horizon_months": k, "CATE_days": v['CATE'], "CATE_months": (v['CATE']/30.0 if pd.notna(v['CATE']) else None), "error": v['error']}
                for k,v in cates.items()
            ])
            st.subheader("Predicted CATEs")
            st.dataframe(cate_df.sort_values('horizon_months'))
            st.download_button("Download CATEs CSV", cate_df.to_csv(index=False).encode('utf-8'), file_name=f"{pid}_cates.csv")

        # Survival plot (pooled-logit counterfactuals)
        surv = res.get('survival_curve')
        if surv is not None and isinstance(surv, pd.DataFrame):
            st.subheader("Counterfactual marginal survival (pooled-logit predictions)")
            # plotly step plot
            fig = go.Figure()
            # align to step plotting by adding initial point at 0 days with S=1
            days = surv['days'].values
            s_ctrl = np.concatenate(([1.0], surv['S_control'].values[:-1]))
            s_tr = np.concatenate(([1.0], surv['S_treat'].values[:-1]))
            fig.add_trace(go.Scatter(x=days, y=s_ctrl, mode='lines+markers', name='Control (RT)', hovertemplate='day: %{x}<br>S: %{y:.3f}'))
            fig.add_trace(go.Scatter(x=days, y=s_tr, mode='lines+markers', name='Treated (Chemo+RT)', hovertemplate='day: %{x}<br>S: %{y:.3f}'))
            fig.update_layout(xaxis_title='Days since RT start', yaxis_title='Survival S(t)', legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No survival curve produced (pooled-logit artifact may be missing).")

# Batch CSV mode
else:
    st.subheader("Batch inference (CSV)")
    st.markdown("CSV must have one row per patient and at least a patient_id column. Other columns like age, sex, treatment, hpv_clean help predictions.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df_batch = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_batch)} rows.")
        run_btn = st.button("Run batch inference (synchronous)")
        if run_btn:
            out_rows = []
            progress = st.progress(0)
            n = len(df_batch)
            for i, (_, r) in enumerate(df_batch.iterrows()):
                # convert row to dict and run infer
                out = infer_new_patient(r.to_dict(), return_raw=False)
                row_out = {'patient_id': r.get('patient_id', f'row_{i}')}
                # flatten CATEs
                for k,v in out.get('CATEs', {}).items():
                    row_out[f"CATE_{k}m_days"] = v.get('CATE', None)
                    row_out[f"CATE_{k}m_error"] = v.get('error', None)
                row_out['errors'] = str(out.get('errors', {}))
                out_rows.append(row_out)
                progress.progress((i+1)/n)
            df_out = pd.DataFrame(out_rows)
            st.write("Batch inference finished.")
            st.dataframe(df_out)
            st.download_button("Download results CSV", df_out.to_csv(index=False).encode('utf-8'), file_name="batch_inference_results.csv")

st.sidebar.markdown("---")
st.sidebar.caption("If artifacts cannot be found, make sure outputs/ contains:\n- pooled_logit_logreg_saga.joblib\n- pooled_logit_model_columns.csv\n- causal_forests bundle (joblib)")

