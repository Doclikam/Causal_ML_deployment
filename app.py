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
