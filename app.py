import streamlit as st
import pandas as pd
import numpy as np
import joblib
from scipy.special import expit
from functools import lru_cache
import os

st.set_page_config(
    page_title="RadCure–AI: Personalized Survival & Treatment Effect Prediction",
    page_icon="🩺",
    layout="wide",
)

# -----------------------------
# GLOBAL CONSTANTS
# -----------------------------
HORIZONS = [3, 6, 12, 18, 36, 60]
FOREST_DIR = "outputs/forests"

# Must match training definitions
BASE_CAT = ['sex','primary_site_group','stage','hpv_clean']
BASE_NUM = ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean']

# For pooled logistic model
PERIOD_LABELS = ['0-3','4-6','7-12','13-24','25-60','60+']


# ============================================================
# LOAD ALL ARTIFACTS
# ============================================================
@st.cache_resource
def load_artifacts():
    artifacts = {}

    artifacts["train_dummy_columns"] = joblib.load("outputs/train_dummy_columns.joblib")
    artifacts["scaler_pp"] = joblib.load("outputs/scaler_pp_train.joblib")
    artifacts["X_train_columns"] = joblib.load("outputs/X_train_columns.joblib")
    artifacts["collapse_maps"] = joblib.load("outputs/collapse_maps.joblib")
    artifacts["logit"] = joblib.load("outputs/pooled_logit.joblib")
    artifacts["period_mean"] = pd.read_csv("outputs/period_mean_hazards.csv")

    # Load causal forests
    forests = {}
    for h in HORIZONS:
        fp = f"{FOREST_DIR}/forest_{h}m.joblib"
        if os.path.exists(fp):
            forests[h] = joblib.load(fp)
        else:
            forests[h] = None
    artifacts["forests"] = forests

    return artifacts


# ============================================================
# PREPROCESS INPUT DATA
# ============================================================
def preprocess_uploaded_data(df, artifacts):
    """
    Matches the training pipeline for:
    1) collapse maps
    2) dummy encoding
    3) numeric scaling
    4) align columns with X_train_columns
    """

    collapse_maps = artifacts["collapse_maps"]
    train_dummy_cols = artifacts["train_dummy_columns"]
    scaler_pp = artifacts["scaler_pp"]
    X_train_cols = artifacts["X_train_columns"]

    # Apply collapse maps
    for col, keep_values in collapse_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str).where(df[col].isin(keep_values), "Other")

    # Build categorical dummies (only for columns used at training)
    cat_cols = [c.split("_")[0] for c in train_dummy_cols]
    cat_cols = list(set(cat_cols) & set(df.columns))

    dummies = pd.get_dummies(df[cat_cols].astype(str), prefix_sep="_", drop_first=True)

    # Align dummy columns
    dummies = dummies.reindex(columns=train_dummy_cols, fill_value=0)

    # Numeric features
    num_cols = artifacts["period_mean"].columns  # fallback
    numeric_used = [c for c in ['age','ecog_ps','smoking_py_clean','BED_eff','EQD2'] if c in df.columns]

    nums = df[numeric_used].copy().fillna(0)
    nums_scaled = pd.DataFrame(scaler_pp.transform(nums), columns=numeric_used)

    # Combine
    X = pd.concat([dummies.reset_index(drop=True), nums_scaled.reset_index(drop=True)], axis=1)

    # Add missing interaction columns
    for lbl in PERIOD_LABELS:
        cname = f"treat_x_period_bin_{lbl}"
        if cname not in X.columns:
            X[cname] = 0

    # Align final matrix
    X = X.reindex(columns=X_train_cols, fill_value=0)

    return X


# ============================================================
# SURVIVAL PREDICTION
# ============================================================
def predict_survival(df, artifacts):
    logit = artifacts["logit"]

    # Build features
    X = preprocess_uploaded_data(df, artifacts)

    # Predict hazards for t=0, t=1
    hazards_rt = expit(logit.predict_log_proba(X)[:, 1])
    hazards_chemo = hazards_rt.copy()  # placeholder if interaction missing

    S_rt = np.cumprod(1 - hazards_rt)
    S_ch = np.cumprod(1 - hazards_chemo)

    return pd.DataFrame({
        "Period": np.arange(len(S_rt)),
        "RT_alone": S_rt,
        "ChemoRT": S_ch
    })


# ============================================================
# CAUSAL FOREST CATE PREDICTION
# ============================================================
def build_cf_features(df):
    """Build baseline features expected by causal forests."""
    cat_df = pd.get_dummies(df[BASE_CAT].astype(str), drop_first=True)
    num_df = df[BASE_NUM].copy()
    X = pd.concat([cat_df, num_df], axis=1)
    return X.fillna(0)


def predict_CATEs(df, artifacts):
    forests = artifacts["forests"]

    X_cf = build_cf_features(df)

    mean_results = {}
    per_patient = pd.DataFrame(index=df.index)

    for h, forest in forests.items():
        if forest is None:
            mean_results[h] = np.nan
            per_patient[f"CATE_{h}m"] = np.nan
            continue

        try:
            # Align columns if forest supports feature_names_in_
            if hasattr(forest, "feature_names_in_"):
                X_use = X_cf.reindex(columns=forest.feature_names_in_, fill_value=0)
            else:
                X_use = X_cf

            te = forest.effect(X_use)
            te = np.asarray(te).reshape(-1,)

            per_patient[f"CATE_{h}m"] = te
            mean_results[h] = float(np.nanmean(te))

        except Exception as e:
            per_patient[f"CATE_{h}m"] = np.nan
            mean_results[h] = np.nan
            print(f"Error horizon {h}: {e}")

    return per_patient, mean_results


# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    st.title("Causal ML – RadCure Survival & Treatment Effect Explorer")
    st.markdown("""
<div style='text-align:center;'>
    <h1 style='color:#0058A3;'>RadCure–AI</h1>
    <h3>Personalized Survival & Treatment Effect Prediction for Head & Neck Cancer</h3>
    <p>Powered by Causal Machine Learning and Real-World Radiotherapy Data</p>
</div>
""", unsafe_allow_html=True)
    artifacts = load_artifacts()

    st.sidebar.header("Upload Data")
    file = st.sidebar.file_uploader("Upload patient CSV", type=["csv"])

    if file is None:
        st.info("Please upload a CSV to begin.")
        return

    st.sidebar.header("Upload Patient Data")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    st.sidebar.title("🧭 Clinical Workflow")
    st.sidebar.markdown("""
    1. **Upload patient CSV**  
    2. **Select analysis outputs**  
    3. **View predicted survival curves**  
    4. **Review personalized treatment benefit (CATE)**  
    5. **Download report for patient file**  
    """)
    
    df = pd.read_csv(file)
    st.success("Uploaded successfully.")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ---------------------------------------------------------
    # Survival Curve
    # ---------------------------------------------------------
    st.subheader("Adjusted Survival Curves (Pooled Logistic Regression)")

    try:
        survival_df = predict_survival(df, artifacts)
        st.line_chart(survival_df.set_index("Period"))
    except Exception as e:
        st.error(f"Survival prediction failed: {e}")

    # ---------------------------------------------------------
    # CATE
    # ---------------------------------------------------------
    st.subheader("Conditional Average Treatment Effect (CATE)")

    try:
        per_patient, mean_cates = predict_CATEs(df, artifacts)

        st.write("**Mean CATEs across uploaded patients:**")
        st.dataframe(pd.DataFrame({
            "Horizon (months)": list(mean_cates.keys()),
            "CATE": list(mean_cates.values())
        }))

        st.write("**Per-patient CATE estimates:**")
        st.dataframe(per_patient)

    except Exception as e:
        st.error(f"CATE estimation failed: {e}")


if __name__ == "__main__":
    main()








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

