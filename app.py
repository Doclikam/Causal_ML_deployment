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
import matplotlib.pyplot as plt
from math import ceil


st.set_page_config(page_title="H&N Causal Survival Explorer", layout="wide")

import os, joblib, pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler

# set this to the folder that contains your saved artifacts (you already confirmed this)
BASE = "/content/drive/MyDrive/outputs_hncc_project"

# candidate artifact names (common)
FILES = {
    "pooled_logit": ["pooled_logit_logreg_saga.joblib", "pooled_logit.joblib", "pooled_logit.joblib"],
    "pooled_cols_csv": "pooled_logit_model_columns.csv",
    "pp_scaler": "pp_scaler.joblib",
    "pp_train_medians": "pp_train_medians.joblib",
    "pp_collapse_maps": "pp_collapse_maps.joblib",
    "causal_patient_scaler": "causal_patient_scaler.joblib",
    "causal_forests_all": "causal_forests_period_horizons_patient_level.joblib",
    "forests_dir": os.path.join(BASE, "forests"),
    "train_dummy_columns": "train_dummy_columns.joblib",
    "X_train_columns": "X_train_columns.joblib"
}

def load_if_exists(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

# load pooled-logit artifacts (for survival predictions)
pooled_path = None
for p in FILES["pooled_logit"]:
    cand = os.path.join(BASE, p)
    if os.path.exists(cand):
        pooled_path = cand
        break

pooled_logit = joblib.load(pooled_path) if pooled_path else None
print("Loaded pooled_logit:", pooled_path)

model_columns = None
cols_csv = os.path.join(BASE, FILES["pooled_cols_csv"])
if os.path.exists(cols_csv):
    model_columns = pd.read_csv(cols_csv).squeeze().tolist()
    print("Loaded pooled-logit model_columns from CSV, n_cols:", len(model_columns))
else:
    xcols_path = os.path.join(BASE, FILES["X_train_columns"])
    if os.path.exists(xcols_path):
        model_columns = joblib.load(xcols_path)
        print("Loaded X_train_columns joblib, n_cols:", len(model_columns))
    else:
        # last fallback
        print("Warning: pooled-logit model_columns file not found in BASE. Some predictions may fail.")

pp_scaler = load_if_exists(os.path.join(BASE, FILES["pp_scaler"]))
pp_train_medians = load_if_exists(os.path.join(BASE, FILES["pp_train_medians"]))
pp_collapse_maps = load_if_exists(os.path.join(BASE, FILES["pp_collapse_maps"]))

# causal forests: try both a single joblib bundle or individual forest files in forests/
forests_bundle = load_if_exists(os.path.join(BASE, FILES["causal_forests_all"]))
forests = {}
if forests_bundle:
    forests = forests_bundle
    print("Loaded causal forests bundle.")
else:
    # try to load by horizon files inside BASE/forests
    for h in [3,6,12,18,36,60]:
        p = os.path.join(BASE, "forests", f"forest_{h}m.joblib")
        if os.path.exists(p):
            forests[h] = joblib.load(p)
            print("Loaded forest for horizon", h)

from sklearn.exceptions import NotFittedError

# Define which baseline variables the causal forest expects.
baseline_cat = [c for c in ['sex','smoking_status_clean','primary_site_group','pathology_group','hpv_clean'] if c in train_patients.columns]
baseline_num = [c for c in ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean'] if c in train_patients.columns]

# Try to load the trained scaler & the template dummy columns used to create Xtr_patient
causal_scaler = load_if_exists(os.path.join(BASE, "causal_patient_scaler.joblib"))
# If you saved the patient-level column names anywhere (recommended), load them:
patient_X_cols = None
xcols_path = os.path.join(BASE, "causal_patient_columns.joblib")
if os.path.exists(xcols_path):
    patient_X_cols = joblib.load(xcols_path)
else:
    # try X_train_columns as fallback
    if os.path.exists(os.path.join(BASE, "X_train_columns.joblib")):
        patient_X_cols = joblib.load(os.path.join(BASE, "X_train_columns.joblib"))

def make_patient_X(patient_dict):
    """
    patient_dict: dict with baseline keys (e.g. {'age':62,'sex':'F',...})
    returns: DataFrame with 1 row aligned to patient-level forest X columns
    """
    df = pd.DataFrame([patient_dict]).copy()
    # apply collapse mapping (safe)
    if pp_collapse_maps:
        for c, keep in pp_collapse_maps.items():
            if c in df.columns:
                df[c] = df[c].astype(str).where(df[c].astype(str).isin(keep), 'Other')
    # ensure baseline columns exist
    for c in baseline_cat + baseline_num:
        if c not in df.columns:
            df[c] = np.nan

    # dummies from baseline_cat
    if len(baseline_cat) > 0:
        Xc = pd.get_dummies(df[baseline_cat].astype(str), drop_first=True)
    else:
        Xc = pd.DataFrame(index=df.index)

    # numeric part: fill medians and coerce
    Xn = df[baseline_num].copy()
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors='coerce')
    if pp_train_medians is not None:
        Xn = Xn.fillna(pp_train_medians.to_dict())
    else:
        Xn = Xn.fillna(Xn.median())

    # scale numeric
    if causal_scaler is not None:
        try:
            Xn_scaled = pd.DataFrame(causal_scaler.transform(Xn), columns=Xn.columns, index=Xn.index)
        except Exception as e:
            print("Causal scaler transform failed:", e)
            # fallback: fit-transform small scaler (not ideal)
            tmp_s = StandardScaler()
            Xn_scaled = pd.DataFrame(tmp_s.fit_transform(Xn), columns=Xn.columns, index=Xn.index)
    else:
        tmp_s = StandardScaler()
        Xn_scaled = pd.DataFrame(tmp_s.fit_transform(Xn), columns=Xn.columns, index=Xn.index)

    Xnew = pd.concat([Xc.reset_index(drop=True), Xn_scaled.reset_index(drop=True)], axis=1).fillna(0)

    # align to patient_X_cols if available (best), else return what we have
    if patient_X_cols is not None:
        Xnew = Xnew.reindex(columns=patient_X_cols, fill_value=0.0)
    return Xnew
def predict_patient_cates(patient_dict, forests_dict=forests):
    # returns dict horizon->(cate_point_estimate, [interval_low, interval_high] or NaN)
    Xnew = make_patient_X(patient_dict)
    if Xnew.shape[1] == 0:
        raise RuntimeError("Empty Xnew — check baseline variables provided to make_patient_X.")
    results = {}
    for h, est in forests_dict.items():
        try:
            # econml CausalForestDML's .effect expects same n_features as training.
            cate = est.effect(Xnew.values).flatten()[0]
            # try to get uncertainty interval if available
            try:
                lo, hi = est.effect_interval(Xnew.values, alpha=0.05)
                lo = lo.flatten()[0]
                hi = hi.flatten()[0]
            except Exception:
                lo, hi = (np.nan, np.nan)
            results[h] = {"CATE": float(cate), "lo": lo, "hi": hi}
        except Exception as e:
            results[h] = {"CATE": np.nan, "lo": np.nan, "hi": np.nan, "error": str(e)}
    return results

# Example:
new_patient = {'age':62, 'sex':'F', 'primary_site_group':'Oropharynx', 'pathology_group':'Squamous', 'hpv_clean':'HPV_Positive', 'treatment':0}
cates = predict_patient_cates(new_patient)
print("Per-horizon CATEs (risk diff = treated − control, probability points):")
print(cates)



print("Forests available for horizons:", sorted(list(forests.keys())))
