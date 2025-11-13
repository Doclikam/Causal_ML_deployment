
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.special import expit
import os

# -------------------------------------------------------
#  PAGE CONFIGURATION (bright, clinical)
# -------------------------------------------------------
st.set_page_config(
    page_title="RadCure–AI Clinical Explorer",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<div style='text-align:center; margin-bottom:20px;'>
    <h1 style='color:#0077B6;'>RadCure–AI</h1>
    <h3>Personalized Survival & Treatment Effect (CATE) Explorer</h3>
    <p style='font-size:16px;'>AI-assisted causal inference for Head and Neck Cancer (RT vs ChemoRT)</p>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------------
#  LOAD ARTIFACTS
# -------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = {}
    artifacts["dummy_cols"] = joblib.load("train_dummy_columns.joblib")
    artifacts["scaler"] = joblib.load("scaler_pp_train.joblib")
    artifacts["train_cols"] = joblib.load("X_train_columns.joblib")
    artifacts["collapse_maps"] = joblib.load("collapse_maps.joblib")
    artifacts["logit"] = joblib.load("pooled_logit.joblib")
    artifacts["period_mean"] = pd.read_csv("period_mean_hazards.csv")

    forests = {}
    for h in [3, 6, 12, 18, 36, 60]:
        path = f"forests/forest_{h}m.joblib"
        if os.path.exists(path):
            forests[h] = joblib.load(path)
    artifacts["forests"] = forests
    return artifacts

artifacts = load_artifacts()


# -------------------------------------------------------
#  PATIENT SUMMARY CARD
# -------------------------------------------------------
def cohort_summary(df):
    with st.expander("📋 Cohort Summary (auto-generated)", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Patients", len(df))
        col2.metric("Median Age", f"{df['age'].median():.1f}")
        col3.metric("ChemoRT Rate", f"{100*df['treatment'].mean():.1f}%")

        col4, col5 = st.columns(2)
        col4.metric("HPV+ (%)", f"{100*df['hpv_clean'].mean():.1f}%")
        col5.metric("Stage Missing (%)", f"{100*df['stage_missing'].mean():.1f}%")


# -------------------------------------------------------
#  DESIGN MATRIX FOR LOGIT MODEL
# -------------------------------------------------------
def build_design_matrix(df, treat, artifacts):
    dummies = pd.get_dummies(df[cat_cols + ["period_bin"]], drop_first=True)
    dummies = dummies.reindex(columns=artifacts["dummy_cols"], fill_value=0)

    nums = df[num_for_model].fillna(0)
    nums_scaled = pd.DataFrame(
        artifacts["scaler"].transform(nums),
        columns=num_for_model
    )

    X = pd.concat([dummies.reset_index(drop=True), nums_scaled.reset_index(drop=True)], axis=1)

    # Add T × period interaction
    for lbl in period_labels:
        colname = f"treat_x_period_bin_{lbl}"
        mask = (df["period_bin"].astype(str) == lbl).astype(int).values
        X[colname] = treat * mask

    X = X.reindex(columns=artifacts["train_cols"], fill_value=0)
    return X


# -------------------------------------------------------
#  SURVIVAL CURVE PLOT
# -------------------------------------------------------
def plot_survival_curve(surv_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=surv_df["Day"], y=surv_df["RT"],
        mode="lines+markers",
        name="RT alone",
        line=dict(color="#0077B6")
    ))
    fig.add_trace(go.Scatter(
        x=surv_df["Day"], y=surv_df["ChemoRT"],
        mode="lines+markers",
        name="ChemoRT",
        line=dict(color="#FB8500")
    ))

    fig.update_layout(
        title="Adjusted Survival Curves",
        xaxis_title="Days from RT Start",
        yaxis_title="Survival Probability",
        template="plotly_white"
    )
    return fig


# -------------------------------------------------------
#  CATE CURVE PLOT (multi-horizon)
# -------------------------------------------------------
def plot_CATE_curve(cates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(cates.keys()),
        y=list(cates.values()),
        mode="lines+markers",
        line=dict(color="#0096C7"),
        name="CATE"
    ))

    fig.add_hline(y=0, line_dash="dash", opacity=0.6)

    fig.update_layout(
        title="Personalized Treatment Effect (CATE) across time horizons",
        xaxis_title="Months",
        yaxis_title="CATE (ChemoRT − RT)",
        template="plotly_white"
    )
    return fig


# -------------------------------------------------------
#  PREDICT SURVIVAL & CATE FOR ONE PATIENT
# -------------------------------------------------------
def predict_for_patient(df, artifacts):

    # === Predict Survival Curves ===
    pmean = artifacts["period_mean"]
    h0 = pmean["hazard_rt"].values
    h1 = pmean["hazard_chemo"].values

    S0 = np.cumprod(1 - h0)
    S1 = np.cumprod(1 - h1)

    days = pmean["period"] * 30
    surv_df = pd.DataFrame({"Day": days, "RT": S0, "ChemoRT": S1})

    # === Predict CATE values using causal forests ===
    X_cf = df[[c for c in cf_base_vars if c in df.columns]].copy()
    X_cf = pd.get_dummies(X_cf, drop_first=True).fillna(0)

    cates = {}
    for h, forest in artifacts["forests"].items():
        try:
            tau = forest.effect(X_cf)[0]
            cates[h] = tau
        except:
            cates[h] = np.nan

    return surv_df, cates


# -------------------------------------------------------
#  USER INPUT REGION
# -------------------------------------------------------
st.sidebar.title("🧭 Workflow")
st.sidebar.markdown("""
1. **Upload CSV**  
2. **Select Patient**  
3. **View Survival & CATE Curves**  
4. **Download Patient Report**  
""")

uploaded = st.file_uploader("📤 Upload Patient CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    cohort_summary(df)

    # Select patient for personalized prediction
    pid = st.selectbox("Select a patient ID", df["patient_id"].unique())
    patient = df[df["patient_id"] == pid].copy()

    st.markdown("### 🩺 Selected Patient Overview")
    st.dataframe(patient)

    # ---------------------------
    # PREDICTION
    # ---------------------------
    surv_df, cates = predict_for_patient(patient, artifacts)

    colA, colB = st.columns(2)
    with colA:
        st.plotly_chart(plot_survival_curve(surv_df), use_container_width=True)

    with colB:
        st.plotly_chart(plot_CATE_curve(cates), use_container_width=True)

    # Show CATE table too
    st.markdown("### 📊 Numerical CATE Table")
    st.dataframe(pd.DataFrame({
        "Months": list(cates.keys()),
        "CATE": list(cates.values())
    }))

else:
    st.info("Please upload a CSV file to begin analysis.")

