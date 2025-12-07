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



st.set_page_config(page_title="Causal Inference — Quick infer", layout="wide")

st.sidebar.header("Settings")
BASE_URL = st.sidebar.text_input("BASE_URL (raw github path)", value="https://raw.githubusercontent.com/Doclikam/Causal_ML_deployment/main/outputs/")
max_period = st.sidebar.number_input("Max periods (months)", value=60, min_value=6, max_value=360)

st.title("Individual patient inference")

# simple patient form
with st.form("patient_form"):
    age = st.number_input("Age", value=62)
    sex = st.selectbox("Sex", ["Male","Female","Missing"])
    primary_site_group = st.text_input("Primary site group", value="Oropharynx")
    pathology_group = st.text_input("Pathology group", value="Squamous")
    hpv_clean = st.selectbox("HPV", ["HPV_Positive","HPV_Negative","Missing"])
    treatment = st.selectbox("Current treatment (0=RT,1=Chemo+RT)", [0,1], index=0)
    submitted = st.form_submit_button("Run inference")

if submitted:
    patient = {
        "age": age,
        "sex": sex,
        "primary_site_group": primary_site_group,
        "pathology_group": pathology_group,
        "hpv_clean": hpv_clean,
        "treatment": treatment
    }

    with st.spinner("Running inference (loading models from GitHub)..."):
        out = infer_new_patient_fixed(
    patient,
    base_url=BASE_URL,
    max_period_override=int(max_period),  
    return_raw=True                       
)


    # show errors / warnings
    if out.get("errors"):
        st.error("Warnings / errors during inference:")
        for k, v in out["errors"].items():
            st.write(f"- **{k}**: {v}")


    # survival curve
    surv = out.get("survival_curve")
    if surv is not None:
        st.subheader("Predicted marginal survival (treated vs control)")
        st.line_chart(surv.set_index("days")[["S_control","S_treat"]])
        st.write("First rows:")
        st.dataframe(surv.head())

    # show CATEs
    cates = out.get("CATEs", {})
    if cates:
        st.subheader("Per-horizon CATEs (risk-diff, probability points)")
        rows = []
        for k, v in cates.items():
            rows.append({"horizon": str(k), "CATE": v.get("CATE"), "error": v.get("error")})
        st.table(pd.DataFrame(rows))

   
    st.expander("Debug info").write(out.get("debug", {}))
