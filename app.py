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
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
from math import ceil

st.set_page_config(page_title="H&N Causal Survival Explorer", layout="wide")

# ----------------- CONFIG: edit these paths to your artifact folder -----------------
BASE = "/content/Causal_ML_deployment/outputs"
POOLED_LOGIT = os.path.join(BASE, "pooled_logit_logreg_saga.joblib")
POOLED_COLS  = os.path.join(BASE, "pooled_logit_model_columns.csv")
PP_SCALER    = os.path.join(BASE, "pp_scaler.joblib")
PP_MEDIANS   = os.path.join(BASE, "pp_train_medians.joblib")
PP_COLLAPSE  = os.path.join(BASE, "pp_collapse_maps.joblib")
KM_CENSOR    = os.path.join(BASE, "km_censor_train.joblib")
FOREST_DIR   = os.path.join(BASE, "forests")   # expects forest_3m.joblib etc.
FOREST_HORIZONS = [3,6,12,18,36,60]
INTERVAL_DAYS = 30
# -------------------------------------------------------------------------------

st.title("📈 HNCC — Survival & Individual Treatment Effects Explorer")
st.markdown(
    """
    **What this app does**
    - Load saved models (pooled logistic + causal forests) from disk.
    - Predict adjusted survival curves (Chemo+RT vs RT-alone) for a new patient.
    - Try to predict horizon-specific CATEs (risk difference) from causal forests.
    - Compute RMST difference (days) up to a chosen horizon.
    """
)

# ----- Load artifacts (graceful)
@st.cache_resource
def safe_load(path):
    try:
        return joblib.load(path) if os.path.exists(path) else None
    except Exception as e:
        st.warning(f"Failed to load {os.path.basename(path)}: {e}")
        return None

logit = safe_load(POOLED_LOGIT)
model_cols = None
if os.path.exists(POOLED_COLS):
    try:
        model_cols = pd.read_csv(POOLED_COLS).squeeze().tolist()
    except Exception:
        model_cols = safe_load(POOLED_COLS)
pp_scaler = safe_load(PP_SCALER)
pp_medians = safe_load(PP_MEDIANS)
collapse_maps = safe_load(PP_COLLAPSE) or {}
km_censor = safe_load(KM_CENSOR)

st.sidebar.header("Artifact status")
st.sidebar.write("Pooled-logit loaded:", "✅" if logit is not None else "❌")
st.sidebar.write("Model columns present:", "✅" if model_cols is not None else "❌")
st.sidebar.write("PP scaler present:", "✅" if pp_scaler is not None else "❌")
st.sidebar.write("Forests folder exists:", "✅" if os.path.isdir(FOREST_DIR) else "❌ (create and put forest_3m.joblib etc.)")
st.sidebar.write("KM censorer:", "✅" if km_censor is not None else "❌")

# ----- Data input
st.header("1) Input patient(s)")
st.markdown("You can either upload a CSV (rows = patients), or fill the single-patient form below. Required columns for single-patient form are shown; additional baseline covariates are optional.")

uploaded = st.file_uploader("Upload patient CSV (optional)", type=["csv"])
if uploaded is not None:
    df_input = pd.read_csv(uploaded)
    st.write("Preview of uploaded data (first 5 rows):")
    st.dataframe(df_input.head())
else:
    st.markdown("**Single patient form**")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("age", min_value=18, max_value=120, value=62)
        sex = st.selectbox("sex", options=["F","M"], index=0)
        hpv_clean = st.selectbox("hpv_clean", options=["HPV_Positive","HPV_Negative","Unknown"], index=2)
    with col2:
        primary_site_group = st.text_input("primary_site_group", value="Oropharynx")
        pathology_group = st.text_input("pathology_group", value="")
        smoking_status_clean = st.selectbox("smoking_status_clean", options=["Never","Former","Current","Unknown"], index=0)
    with col3:
        ecog_ps = st.number_input("ecog_ps", min_value=0, max_value=4, value=1)
        smoking_py_clean = st.number_input("smoking_py_clean", min_value=0, max_value=500, value=0)
        treatment = st.selectbox("treatment (current)", options=[0,1], index=0)

    df_input = pd.DataFrame([{
        'age': age, 'sex': sex, 'hpv_clean': hpv_clean,
        'primary_site_group': primary_site_group, 'pathology_group': pathology_group,
        'smoking_status_clean': smoking_status_clean, 'ecog_ps': ecog_ps,
        'smoking_py_clean': smoking_py_clean, 'treatment': treatment
    }])

# ----- Helper functions (safe minimal versions)
def build_X_for_pp_min(df_pp):
    df = df_pp.copy()
    # collapse rare categories if collapse_maps available
    for c,allowed in collapse_maps.items():
        if c in df.columns:
            df[c] = df[c].astype(str).where(df[c].astype(str).isin(allowed), 'Other')
    if 'period' not in df.columns:
        df['period'] = np.arange(1, len(df)+1)
    df['period_month'] = df['period'].astype(int)
    bins = [0,3,6,12,24,60,np.inf]
    labels = ['0-3','4-6','7-12','13-24','25-60','60+']
    df['period_bin'] = pd.cut(df['period_month'], bins=bins, labels=labels, right=True)
    # categorical dummies (simple)
    cat_cols = [c for c in ['period_bin','sex','smoking_status_clean','primary_site_group','subsite_clean','stage','hpv_clean'] if c in df.columns]
    Xc = pd.get_dummies(df[cat_cols].astype(str), drop_first=True) if len(cat_cols)>0 else pd.DataFrame(index=df.index)
    # numeric
    num_cols = [c for c in ['age','ecog_ps','smoking_py_clean','time_since_rt_days'] if c in df.columns]
    Xn = df[num_cols].copy() if len(num_cols)>0 else pd.DataFrame(index=df.index)
    if not Xn.empty:
        for c in Xn.columns:
            Xn[c] = pd.to_numeric(Xn[c], errors='coerce')
        if pp_medians is not None:
            Xn = Xn.fillna(pd.Series(pp_medians))
        else:
            Xn = Xn.fillna(Xn.median())
        if pp_scaler is not None:
            try:
                Xn = pd.DataFrame(pp_scaler.transform(Xn), columns=Xn.columns, index=Xn.index)
            except Exception:
                # fallback
                Xn = Xn
    Xnew = pd.concat([Xc.reset_index(drop=True), Xn.reset_index(drop=True)], axis=1)
    if 'treatment' in df.columns:
        Xnew['treatment'] = pd.to_numeric(df['treatment'], errors='coerce').fillna(0).astype(int).values
    else:
        Xnew['treatment'] = 0
    # add interactions for period dummies if any
    period_cols = [c for c in Xnew.columns if c.startswith('period_bin')]
    for pcol in period_cols:
        Xnew[f"treat_x_{pcol}"] = Xnew['treatment'] * Xnew[pcol]
    # align to model columns
    if model_cols is not None:
        for c in model_cols:
            if c not in Xnew.columns:
                Xnew[c] = 0.0
        Xnew = Xnew[model_cols]
    return Xnew

def predict_survival_for_patient(patient_row, max_period=60):
    # build person-period rows up to max_period
    rows = []
    for p in range(1, max_period+1):
        r = patient_row.copy()
        r['period'] = p
        r['period_month'] = p
        r['time_since_rt_days'] = p * INTERVAL_DAYS if 'time_since_rt_days' not in r else r['time_since_rt_days']
        rows.append(r)
    pp = pd.DataFrame(rows)
    Xpp = build_X_for_pp_min(pp)
    # two counterfactuals
    Xctrl = Xpp.copy()
    if 'treatment' in Xctrl.columns:
        Xctrl['treatment'] = 0
    for pcol in [c for c in Xctrl.columns if c.startswith('period_bin')]:
        Xctrl[f"treat_x_{pcol}"] = 0
    Xtrt = Xpp.copy()
    Xtrt['treatment'] = 1
    for pcol in [c for c in Xtrt.columns if c.startswith('period_bin')]:
        Xtrt[f"treat_x_{pcol}"] = Xtrt.get(pcol, 0)
    # align
    if model_cols is not None:
        Xctrl = Xctrl.reindex(columns=model_cols, fill_value=0.0)
        Xtrt  = Xtrt.reindex(columns=model_cols, fill_value=0.0)
    # predict
    p0 = logit.predict_proba(Xctrl)[:,1] if logit is not None else np.zeros(Xctrl.shape[0])
    p1 = logit.predict_proba(Xtrt)[:,1] if logit is not None else np.zeros(Xtrt.shape[0])
    S0 = np.cumprod(1 - p0)
    S1 = np.cumprod(1 - p1)
    out = pd.DataFrame({'period': np.arange(1, len(S0)+1), 'S_control': S0, 'S_treat': S1})
    out['days'] = out['period'] * INTERVAL_DAYS
    return out, p0, p1

# ----- Run inference for first patient in df_input
st.header("2) Model output (single patient preview)")
patient0 = df_input.iloc[0].to_dict()
# Safety checks
if logit is None or model_cols is None:
    st.error("Pooled-logit artifacts missing. Please set BASE to a folder with pooled_logit and model columns.")
else:
    max_period = 60
    survival_df, p0, p1 = predict_survival_for_patient(patient0, max_period=max_period)
    st.subheader("Adjusted marginal survival curve (pooled-logit)")
    st.write("Table (first 10 rows):")
    st.dataframe(survival_df.head(10))

    # RMST up to N months (selectable)
    horizon_months = st.slider("Compute RMST difference up to (months)", min_value=3, max_value=60, value=36, step=3)
    tau_days = horizon_months * INTERVAL_DAYS
    # compute discrete RMST under each
    def rmst_from(hazards, interval_days, tau_days):
        surv = 1.0
        rmst = 0.0
        cum = 0
        for h in hazards:
            if cum >= tau_days:
                break
            length = min(interval_days, tau_days - cum)
            rmst += surv * length
            surv *= (1 - h)
            cum += interval_days
        return rmst
    rmst_ct = rmst_from(p0, INTERVAL_DAYS, tau_days)
    rmst_tr = rmst_from(p1, INTERVAL_DAYS, tau_days)
    st.metric("ΔRMST (treated − control) in days", f"{rmst_tr - rmst_ct:.1f} days")
    st.write(f"RMST treated: {rmst_tr:.1f} days, RMST control: {rmst_ct:.1f} days")

    # Plot survival curves
    fig, ax = plt.subplots(figsize=(6,4))
    ax.step(survival_df['days'], np.concatenate(([1.0], survival_df['S_control'][:-1])), where='post', label='RT-alone (adjusted)')
    ax.step(survival_df['days'], np.concatenate(([1.0], survival_df['S_treat'][:-1])), where='post', label='Chemo+RT (adjusted)')
    ax.set_xlabel("Days since RT start"); ax.set_ylabel("Survival probability")
    ax.set_title("Adjusted survival (marginal)")
    ax.legend(); ax.grid(alpha=0.2)
    st.pyplot(fig)

# ----- Try to get patient-level CATEs from forests
st.header("3) Horizon CATEs (causal forests)")
st.write("We will attempt to load horizon forests and predict patient-level CATEs (risk differences). If you see `NaN` values, see the troubleshooting notes below.")

cate_results = {}
for h in FOREST_HORIZONS:
    fpath = os.path.join(FOREST_DIR, f"forest_{h}m.joblib")
    if not os.path.exists(fpath):
        cate_results[h] = np.nan
        continue
    try:
        est = joblib.load(fpath)
        # build patient-level X aligned to training: we assume training used simple baseline dummies + numeric
        dfp = pd.DataFrame([patient0])
        Xc = pd.get_dummies(dfp[[c for c in ['sex','smoking_status_clean','primary_site_group','pathology_group','hpv_clean'] if c in dfp.columns]].astype(str), drop_first=True)
        # try to reindex to saved dummy template if present
        dummy_template_path = os.path.join(BASE, "train_dummy_columns.joblib")
        if os.path.exists(dummy_template_path):
            dummy_cols = joblib.load(dummy_template_path)
            Xc = Xc.reindex(columns=dummy_cols, fill_value=0)
        Xn = dfp[[c for c in ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean'] if c in dfp.columns]].copy()
        # scale if needed
        if os.path.exists(os.path.join(BASE, "causal_patient_scaler.joblib")):
            try:
                scaler = joblib.load(os.path.join(BASE, "causal_patient_scaler.joblib"))
                Xn[Xn.columns] = scaler.transform(Xn)
            except Exception:
                pass
        Xcf = pd.concat([Xc.reset_index(drop=True), Xn.reset_index(drop=True)], axis=1).fillna(0)
        arr = Xcf.values
        res = est.effect(arr)
        cate_results[h] = float(np.asarray(res).flatten()[0])
    except Exception as e:
        cate_results[h] = np.nan
        st.warning(f"Forest load/predict failed for {h}m: {e}")

st.table(pd.DataFrame.from_dict(cate_results, orient='index', columns=['CATE (prob points)']))

