
import numpy as np
import pandas as pd
import joblib
import os

# ----------------------------------------------
# Load shared artifacts
# ----------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "outputs")

# Model artifacts
MODEL_COLUMNS_PATH = os.path.join(OUT, "pooled_logit_model_columns.csv")
SCALER_PATH        = os.path.join(OUT, "pp_scaler.joblib")
MEDIANS_PATH       = os.path.join(OUT, "pp_train_medians.joblib")
COLLAPSE_MAPS_PATH = os.path.join(OUT, "pp_collapse_maps.joblib")
LOGIT_MODEL_PATH   = os.path.join(OUT, "pooled_logit_logreg_saga.joblib")

# Load once
model_columns_raw = pd.read_csv(MODEL_COLUMNS_PATH, header=None).iloc[:, 0].astype(str).tolist()

# Clean a potential bad header
if model_columns_raw and model_columns_raw[0].strip().lower() in ("model_columns", "columns", "column"):
    model_columns_raw = model_columns_raw[1:]

model_columns = [c.strip() for c in model_columns_raw if c.strip()]

scaler        = joblib.load(SCALER_PATH)
train_medians = joblib.load(MEDIANS_PATH)
collapse_maps = joblib.load(COLLAPSE_MAPS_PATH)
logit_model   = joblib.load(LOGIT_MODEL_PATH)


# ----------------------------------------------
# Configuration
# ----------------------------------------------

INTERVAL_DAYS  = 30
HORIZON_MONTHS = 36

period_bins   = [0, 3, 6, 12, 24, 60, 999]
period_labels = ['0-3', '4-6', '7-12', '13-24', '25-60', '60+']

# Which columns in patient dictionary should be included
CAT_COLS = [
    'period_bin','sex','smoking_status_clean','primary_site_group',
    'subsite_clean','stage','hpv_clean'
]

NUM_COLS = ['age','ecog_ps','smoking_py_clean','time_since_rt_days']


# ----------------------------------------------
# Expand a patient into person-period rows
# ----------------------------------------------

def expand_patient_to_pp(patient_dict, max_months=36):
    """
    Convert a single patient into person-period rows for inference.
    """
    rows = []
    for p in range(1, max_months + 1):
        row = patient_dict.copy()
        row['period'] = p
        row['time_since_rt_days'] = p * INTERVAL_DAYS
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------
# FIXED period-bin + collapse maps + dummy creation
# ----------------------------------------------

def apply_period_bins(df_pp):
    """
    Assign period_bin categories robustly.
    Uses safe upper bound instead of np.inf.
    """
    df_pp['period_month'] = df_pp['period'].astype(int)

    df_pp['period_bin'] = pd.cut(
        df_pp['period_month'],
        bins=period_bins,
        labels=period_labels,
        right=True,
        include_lowest=True
    )

    # Ensure no NA remains
    df_pp['period_bin'] = df_pp['period_bin'].cat.add_categories(period_labels)
    df_pp['period_bin'] = df_pp['period_bin'].fillna(
        pd.cut(df_pp['period_month'], bins=period_bins, labels=period_labels)
    )

    return df_pp


def apply_collapse_maps(df_pp):
    """
    Ensure all categorical columns use the same levels as training.
    """
    for col, allowed in collapse_maps.items():
        if col in df_pp.columns:
            df_pp[col] = df_pp[col].astype(str).where(
                df_pp[col].astype(str).isin(allowed), 'Other'
            )
    return df_pp


# ----------------------------------------------
# Build X_pp for pooled logit model
# ----------------------------------------------

def build_X_for_pp(df_pp):
    """
    Main preprocessing pipeline for person-period data.
    Produces a model-aligned matrix X_pp.
    """

    # 1. Apply period bins
    df_pp = df_pp.copy()
    df_pp = apply_period_bins(df_pp)

    # 2. Collapse maps
    df_pp = apply_collapse_maps(df_pp)

    # 3. Dummies for categorical vars
    Xc = pd.get_dummies(df_pp[CAT_COLS].astype(str), drop_first=True)

    # 4. Numeric variables
    Xn = df_pp[NUM_COLS].copy()

    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors='coerce').fillna(train_medians.get(c, 0))

    # Scale numerics
    Xn_scaled = pd.DataFrame(
        scaler.transform(Xn),
        columns=Xn.columns,
        index=Xn.index
    )

    # Combine
    X = pd.concat([Xc.reset_index(drop=True), Xn_scaled.reset_index(drop=True)], axis=1)

    # 5. Add treatment
    X['treatment'] = pd.to_numeric(df_pp['treatment'], errors='coerce').fillna(0).astype(int)

    # 6. Add treatment × period-bin interactions
    period_dummy_cols = [c for c in X.columns if c.startswith('period_bin')]
    for col in period_dummy_cols:
        X[f"treat_x_{col}"] = X['treatment'] * X[col]

    # 7. Align columns
    X = X.reindex(columns=model_columns, fill_value=0.0)

    # 8. Clean booleans
    for c in X.columns:
        if X[c].dtype == 'bool':
            X[c] = X[c].astype(int)
        if X[c].dtype == 'object':
            X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0.0)

    return X


# ----------------------------------------------
# Compute survival from hazards
# ----------------------------------------------

def survival_from_hazards(haz, interval_days=30):
    """
    Convert hazards into a survival curve (stepwise).
    """
    haz = np.asarray(haz)
    S = np.cumprod(1 - haz)
    days = np.arange(1, len(S) + 1) * interval_days

    df = pd.DataFrame({
        "days": days,
        "hazard": haz,
        "survival": S
    })
    return df


# ----------------------------------------------
# RMST computation
# ----------------------------------------------

def rmst_from_survival(df_surv, horizon_months=36, interval_days=30):
    """
    Compute restricted mean survival time using trapezoid (simplified discrete).
    """

    horizon_days = horizon_months * interval_days
    df = df_surv[df_surv["days"] <= horizon_days].copy()
    if df.empty:
        return np.nan

    S = df["survival"].values
    times = df["days"].values

    # Discrete RMST approx
    t0 = np.concatenate(([0], times[:-1]))
    dt = times - t0
    S_start = np.concatenate(([1.0], S[:-1]))
    rmst = np.sum(S_start * dt) / 30.0
    return rmst


# ----------------------------------------------
# MAIN FUNCTION: infer_new_patient_fixed
# ----------------------------------------------

def infer_new_patient_fixed(patient_dict, max_months=36, return_raw=False):
    """
    Full inference pipeline:

    - Expand patient into person-period
    - Build X_pp
    - Predict hazards with pooled logit
    - Convert to survival curve
    - Compute RMST for treatment group and control group
    - Compute delta-RMST
    """

    # Expand
    df_pp = expand_patient_to_pp(patient_dict, max_months=max_months)

    # Build X_pp
    X_pp = build_X_for_pp(df_pp)

    # Predict hazards
    hazards = logit_model.predict_proba(X_pp)[:, 1]

    # Survival curve
    surv_df = survival_from_hazards(hazards, interval_days=INTERVAL_DAYS)

    # RMST
    rmst = rmst_from_survival(surv_df, horizon_months=HORIZON_MONTHS)

    out = {
        "hazards": hazards,
        "survival_curve": surv_df,
        "rmst_36m": rmst
    }

    return out if return_raw else rmst
