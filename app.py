import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Causal Survival Insights – RadCure",
    layout="wide",
    page_icon="🩺"
)
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

# horizons used during training
HORIZONS = [3, 6, 12, 18, 36, 60]

# These must match the baseline features used at training time for causal forests.
# Update these lists if your training used different columns.
BASE_CAT = ['sex', 'primary_site_group', 'stage', 'hpv_clean']
BASE_NUM = ['age', 'ecog_ps', 'BED_eff', 'EQD2', 'smoking_py_clean']

FOREST_DIR = "outputs/forests"

# Cache loader so Streamlit doesn't re-load forests on every rerun
@lru_cache(maxsize=8)
def _load_forest(h):
    path = f"{FOREST_DIR}/forest_{h}m.joblib"
    return joblib.load(path)

def _build_X_cf(df):
    """
    Build the X matrix expected by the causal forests. This replicates the
    training-time construction:
      1) one-hot encode BASE_CAT with drop_first=True
      2) concat numeric BASE_NUM
      3) fill missing columns/values with zeros

    Returns DataFrame X_cf (columns are feature names).
    """
    # Ensure expected columns exist in incoming df
    missing_cols = [c for c in BASE_CAT + BASE_NUM if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input dataframe missing required baseline columns: {missing_cols}")

    # One-hot encode categorical baseline (drop_first=True to match training)
    cat_df = pd.get_dummies(df[BASE_CAT].astype(str), drop_first=True)

    # Numeric baseline (keep order)
    num_df = df[BASE_NUM].copy().reset_index(drop=True)

    # Concat
    X_cf = pd.concat([cat_df.reset_index(drop=True), num_df.reset_index(drop=True)], axis=1)

    # Ensure no NaNs
    X_cf = X_cf.fillna(0.0)

    return X_cf

def predict_CATEs(df):
    """
    df: DataFrame of 1-or-more patients with baseline features.
    Returns dict:
      - 'per_patient': DataFrame with CATE per horizon for each row in df
      - 'mean_cates': dict {horizon: mean CATE across df}
    """
    # build feature matrix
    X_cf = _build_X_cf(df)

    per_patient = pd.DataFrame(index=df.index)

    mean_results = {}

    for h in HORIZONS:
        try:
            forest = _load_forest(h)
        except FileNotFoundError:
            # forest not available for this horizon
            per_patient[f"CATE_{h}m"] = np.nan
            mean_results[h] = np.nan
            continue
        try:
            # try .feature_importances_ shape to infer n_features or forest.feature_names_in_ (sklearn >=1.0)
            train_cols = None
            if hasattr(forest, "feature_names_in_"):
                train_cols = list(forest.feature_names_in_)
            # If no feature_names_in_, fall back to X_cf.columns (best-effort)
            if train_cols is None:
                X_to_use = X_cf
            else:
                # reindex to training columns (missing -> 0, extra -> dropped)
                X_to_use = X_cf.reindex(columns=train_cols, fill_value=0)

            # compute individual CATEs
            te = forest.effect(X_to_use)     # <-- correct econml call
            te = np.asarray(te).reshape(-1,)  # ensure 1d

            per_patient[f"CATE_{h}m"] = te
            mean_results[h] = float(np.nanmean(te))

        except Exception as e:
            # last-resort attempt: pass numpy array
            try:
                te = forest.effect(X_cf.values)
                per_patient[f"CATE_{h}m"] = np.asarray(te).reshape(-1,)
                mean_results[h] = float(np.nanmean(te))
            except Exception as e2:
                # If this still fails, return NaN and print the error for debugging
                per_patient[f"CATE_{h}m"] = np.nan
                mean_results[h] = np.nan
                print(f"[predict_CATEs] Horizon {h}m: failed to compute effect: {e} || fallback error: {e2}")

    return {"per_patient": per_patient, "mean_cates": mean_results}

# -------------------------------------------------------
# Load saved artifacts from the outputs/ folder
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts["logit"] = joblib.load("outputs/pooled_logit.joblib")
    artifacts["scaler"] = joblib.load("outputs/scaler_pp_train.joblib")
    artifacts["collapse_maps"] = joblib.load("outputs/collapse_maps.joblib")
    artifacts["X_train_cols"] = joblib.load("outputs/X_train_columns.joblib")
    artifacts["period_mean"] = pd.read_csv("outputs/period_mean_hazards.csv")

    # Load forests (dynamic)
    import os
    forest_dir = "outputs/forests"
    forests = {}
    if os.path.exists(forest_dir):
        for fname in os.listdir(forest_dir):
            if fname.endswith(".joblib"):
                key = fname.replace("forest_", "").replace("m.joblib", "")
                forests[int(key)] = joblib.load(f"{forest_dir}/{fname}")
    artifacts["forests"] = forests
    return artifacts


artifacts = load_artifacts()
logit = artifacts["logit"]
scaler = artifacts["scaler"]
collapse_maps = artifacts["collapse_maps"]
X_train_cols = artifacts["X_train_cols"]
period_mean = artifacts["period_mean"]
forests = artifacts["forests"]

# -------------------------------------------------------
# Helper preprocessing functions
# -------------------------------------------------------
def collapse_categories(df, collapse_maps):
    """Ensure unseen categories → 'Other'."""
    for col, keep_vals in collapse_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].where(df[col].isin(keep_vals), "Other")
    return df


def preprocess_patient(df):
    """Apply collapse → dummies → scaling → align to model features."""
    df = collapse_categories(df, collapse_maps)

    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    num_cols = [c for c in df.columns if c not in cat_cols]

    # Dummies
    df_dum = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add missing columns
    for col in X_train_cols:
        if col not in df_dum.columns:
            df_dum[col] = 0

    # Drop extra columns
    df_dum = df_dum[X_train_cols]

    # Scale numeric
    num_cols_final = [c for c in X_train_cols if c in df.columns and df[c].dtype != 'object']
    if len(num_cols_final) > 0:
        df_dum[num_cols_final] = scaler.transform(df_dum[num_cols_final])

    return df_dum


def predict_survival_curve(df):
    """Compute patient-level survival probability."""
    X = preprocess_patient(df)
    hazards = logit.predict_proba(X)[:, 1]

    # Smooth by average period hazards
    hazard_df = period_mean.copy()
    hazard_df["hazard_patient"] = hazards.mean()

    hazard_df["S"] = np.exp(-hazard_df["hazard_patient"].cumsum())
    return hazard_df[["period", "S"]]


def predict_CATEs(df):
    """Compute heterogeneous treatment effects for horizons."""
    X = preprocess_patient(df)
    results = {}

    for h, forest in forests.items():
        tau = forest.predict(X).mean()
        results[h] = tau

    return results


# -------------------------------------------------------
# STREAMLIT USER INTERFACE
# -------------------------------------------------------
st.title("🩺 Causal Survival Insights for Radiotherapy + Chemotherapy")
st.markdown(
    """
### **AI-powered causal inference for RADCURE patients**
Upload patient data → Get personalized survival curves & treatment effect estimates (CATE).
"""
)

st.sidebar.header("Upload Patient Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
st.sidebar.download_button(
    label="Download Sample CSV",
    data=open("sample_data/sample_patient_data.csv", "rb").read(),
    file_name="sample_patient_data.csv",
    mime="text/csv"
)


# -------------------------------------------------------
# MAIN WORKFLOW
# -------------------------------------------------------
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Uploaded {df.shape[0]} patients.")

    # Predict survival
    st.subheader("📈 Adjusted Survival Curve (Pooled Logistic Model)")
    surv = predict_survival_curve(df)

    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(surv["period"], surv["S"], marker="o")
    ax.set_xlabel("Months since treatment")
    ax.set_ylabel("Survival Probability")
    ax.set_title("Adjusted Survival (Patient-Level)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # CATE predictions
    st.subheader(" Treatment Effect (CATE) Across Time Horizons")
    cates = predict_CATEs(df)
    cate_df = pd.DataFrame({"Horizon (months)": list(cates.keys()),
                            "CATE": list(cates.values())})
    st.dataframe(cate_df)

    # CATE plot
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.plot(cate_df["Horizon (months)"], cate_df["CATE"], marker="o", color="teal")
    ax2.axhline(0, color="black", linestyle="--")
    ax2.set_title("Conditional Average Treatment Effect")
    ax2.set_ylabel("CATE (ChemoRT – RT)")
    ax2.grid(alpha=0.3)
    st.pyplot(fig2)

else:
    st.info("Upload a CSV file to generate predictions.")

st.markdown("---")
st.markdown("Built with  using Streamlit • Causal ML • RADCURE Dataset")

M
