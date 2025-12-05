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




import joblib, os, numpy as np, pandas as pd
from scipy.special import expit
from sklearn.preprocessing import StandardScaler

# === CONFIG: point these to your saved files ===
BASE = "/content/drive/MyDrive/outputs_hncc_project"   # adjust to folder you want
POOLED_LOGIT_PATH = os.path.join(BASE, "pooled_logit_logreg_saga.joblib")
POOLED_COLS_PATH  = os.path.join(BASE, "pooled_logit_model_columns.csv")   # or X_train_columns.joblib
PP_SCALER_PATH    = os.path.join(BASE, "pp_scaler.joblib")
PP_MEDIANS_PATH   = os.path.join(BASE, "pp_train_medians.joblib")
PP_COLLAPSE_PATH  = os.path.join(BASE, "pp_collapse_maps.joblib")
KM_CENSOR_PATH    = os.path.join(BASE, "km_censor_train.joblib")
FOREST_FOLDER     = os.path.join(BASE, "forests")   # folder with forest_3m.joblib etc.
PATIENT_SCALER    = os.path.join(BASE, "causal_patient_scaler.joblib")  # optional
FOREST_LIST       = [3,6,12,18,36,60]

interval_days = 30
period_labels = ['0-3','4-6','7-12','13-24','25-60','60+']

# === Load artifacts safely ===
def safe_load(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

logit = safe_load(POOLED_LOGIT_PATH)
model_columns = None
if os.path.exists(POOLED_COLS_PATH):
    try:
        model_columns = pd.read_csv(POOLED_COLS_PATH).squeeze().tolist()
    except Exception:
        # maybe joblib
        model_columns = safe_load(POOLED_COLS_PATH)
pp_scaler = safe_load(PP_SCALER_PATH)
pp_medians = safe_load(PP_MEDIANS_PATH)  # pandas Series or dict
collapse_maps = safe_load(PP_COLLAPSE_PATH) or {}
km_censor = safe_load(KM_CENSOR_PATH)

# patient-level design templates (from training objects)
# dummy_cols come from how you built Xtr_patient: saved earlier as 'dummy_cols' or 'train_dummy_columns'
DUMMY_COLS_PATH = os.path.join(BASE, "train_dummy_columns.joblib")
dummy_cols = safe_load(DUMMY_COLS_PATH) or []     # used for causal forest patient-level dummies

# baseline lists used when training forests (should match what you trained with)
baseline_cat = ['sex','smoking_status_clean','primary_site_group','pathology_group','hpv_clean']
baseline_num = ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean']
# if your training used scaler for patient-level features:
patient_scaler = safe_load(PATIENT_SCALER)

# helper: build pooled-logit X for person-period rows
def build_X_for_pp(df_pp):
    df = df_pp.copy()
    # apply collapse maps
    for c, keep in (collapse_maps.items() if collapse_maps else []):
        if c in df.columns:
            df[c] = df[c].astype(str).where(df[c].astype(str).isin(keep), 'Other')
    # ensure period_bin exists
    if 'period_month' not in df.columns:
        df['period_month'] = df['period'].astype(int) if 'period' in df.columns else 1
    if 'period_bin' not in df.columns:
        bins = [0,3,6,12,24,60, np.inf]
        labels = ['0-3','4-6','7-12','13-24','25-60','60+']
        df['period_bin'] = pd.cut(df['period_month'], bins=bins, labels=labels, right=True)

    # categorical dummies (use the exact categorical columns used in training)
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

        # scale
        if pp_scaler is not None:
            try:
                Xn_scaled = pd.DataFrame(pp_scaler.transform(Xn), columns=Xn.columns, index=Xn.index)
            except Exception as e:
                # feature name mismatch -> fall back to numeric-only transform without names
                Xn_scaled = pd.DataFrame(StandardScaler().fit_transform(Xn), columns=Xn.columns, index=Xn.index)
        else:
            Xn_scaled = Xn
    else:
        Xn_scaled = Xn

    Xnew = pd.concat([Xc.reset_index(drop=True), Xn_scaled.reset_index(drop=True)], axis=1)
    # add treatment column if present
    if 'treatment' in df.columns:
        Xnew['treatment'] = pd.to_numeric(df['treatment'], errors='coerce').fillna(0).astype(int).values
    else:
        Xnew['treatment'] = 0
    # add treat x period interactions for all period_bin columns present
    period_dummy_cols_local = [c for c in Xnew.columns if str(c).startswith('period_bin')]
    for pcol in period_dummy_cols_local:
        Xnew[f'treat_x_{pcol}'] = Xnew['treatment'] * Xnew[pcol]
    # align to saved model columns if available
    if model_columns is not None:
        # add missing columns then reorder
        for c in model_columns:
            if c not in Xnew.columns:
                Xnew[c] = 0.0
        Xnew = Xnew[model_columns]
    return Xnew

# helper: patient-level X_cf builder (for causal forest)
def build_X_patient(df_patient_row):
    df = pd.DataFrame([df_patient_row]) if isinstance(df_patient_row, dict) else df_patient_row.copy().reset_index(drop=True)
    # dummies using dummy_cols (from training)
    Xc = pd.get_dummies(df[baseline_cat].astype(str), drop_first=True)
    if isinstance(dummy_cols, (list, tuple)) and len(dummy_cols)>0:
        Xc = Xc.reindex(columns=dummy_cols, fill_value=0)
    # numeric
    Xn = df[baseline_num].copy() if any([c in df.columns for c in baseline_num]) else pd.DataFrame(index=df.index)
    if not Xn.empty:
        for c in Xn.columns:
            Xn[c] = pd.to_numeric(Xn[c], errors='coerce')
        if 'train_medians_pp' in globals() and train_medians_pp is not None:
            fill = train_medians_pp
            Xn = Xn.fillna(fill)
        else:
            Xn = Xn.fillna(Xn.median())
    Xfull = pd.concat([Xc.reset_index(drop=True), Xn.reset_index(drop=True)], axis=1).fillna(0)
    # scaling if patient_scaler exists (this is optional)
    if patient_scaler is not None:
        try:
            Xfull[baseline_num] = patient_scaler.transform(Xfull[baseline_num])
        except Exception:
            pass
    return Xfull

# === Main inference function ===
def infer_new_patient(patient_data, return_probs=False):
    """
    patient_data: dict or 1-row DataFrame with baseline covariates.
    returns: {'survival_curve': DataFrame(period, S_control, S_treat, days), 'CATEs': {h: val}}
    """
    # 1) pooled-logit survival: build pp rows and predict hazards under T=0 and T=1
    if logit is None or model_columns is None:
        print("Warning: pooled-logit or model_columns missing; survival cannot be computed.")
        survival_df = pd.DataFrame()
    else:
        # create pp rows up to max period from training (use pp_test max if available)
        max_period =  max( int(pp_medians.get('max_period', 60)) if isinstance(pp_medians, dict) and 'max_period' in pp_medians else 60, 60)
        # safer: try to find pp_test global
        try:
            max_period = int(pp_test['period'].max())
        except Exception:
            pass

        # expand new patient to person-period
        if isinstance(patient_data, dict):
            df_base = pd.DataFrame([patient_data])
        else:
            df_base = patient_data.copy().reset_index(drop=True)
        rows = []
        for p in range(1, max_period+1):
            r = df_base.iloc[0].to_dict()
            r['period'] = p
            r['period_month'] = p
            r['treatment'] = r.get('treatment', 0)
            # time_since_rt_days if used
            r['time_since_rt_days'] = p * interval_days
            rows.append(r)
        df_pp_new = pd.DataFrame(rows)
        X_new = build_X_for_pp(df_pp_new)
        # two counterfactuals
        X_ctrl = X_new.copy()
        X_ctrl['treatment'] = 0
        for pcol in [c for c in X_ctrl.columns if c.startswith('period_bin')]:
            X_ctrl[f'treat_x_{pcol}'] = 0
        X_trt = X_new.copy()
        X_trt['treatment'] = 1
        for pcol in [c for c in X_trt.columns if c.startswith('period_bin')]:
            X_trt[f'treat_x_{pcol}'] = X_trt.get(pcol,0)
        # align
        X_ctrl = X_ctrl.reindex(columns=model_columns, fill_value=0.0)
        X_trt  = X_trt.reindex(columns=model_columns, fill_value=0.0)
        p0 = logit.predict_proba(X_ctrl)[:,1]
        p1 = logit.predict_proba(X_trt)[:,1]
        S0 = np.cumprod(1 - p0)
        S1 = np.cumprod(1 - p1)
        survival_df = pd.DataFrame({'period': np.arange(1, len(S0)+1), 'S_control': S0, 'S_treat': S1})
        survival_df['days'] = survival_df['period'] * interval_days
        if return_probs:
            survival_df['p0'] = p0
            survival_df['p1'] = p1

    # 2) CATEs from patient-level forests
    cate_preds = {}
    for h in FOREST_LIST:
        forest_path = os.path.join(FOREST_FOLDER, f"forest_{h}m.joblib")
        if not os.path.exists(forest_path):
            cate_preds[h] = np.nan
            continue
        try:
            est = joblib.load(forest_path)
            Xcf = build_X_patient(patient_data)
            # reorder to the columns used when training forest (most forests were fit using dummy_cols + base_num)
            # if est expects a certain column order we try to match: use training Xtr_patient.columns if saved
            if hasattr(Xcf, "values"):
                arr = Xcf.values
            else:
                arr = np.asarray(Xcf)
            pred = est.effect(arr)
            if isinstance(pred, (list, np.ndarray)):
                cate_preds[h] = float(np.asarray(pred).flatten()[0])
            else:
                cate_preds[h] = float(pred)
        except Exception as e:
            print(f"Forest load/predict failed for {h} m: {e}")
            cate_preds[h] = np.nan

    return {'survival_curve': survival_df, 'CATEs': cate_preds}

# === Example ===
# new_patient = {'age':62, 'sex':'F', 'primary_site_group':'Oropharynx', 'stage':'III', 'hpv_clean':'HPV_Positive', 'treatment':0}
# out = infer_new_patient(new_patient)
# out['survival_curve'].head(), out['CATEs']
