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




# app_shap_lightgbm.py
import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------- Config ----------
OUTDIR = "outputs"
RNG = 42
TARGET_COL = "delta_rmst_days"      # computed earlier (we will convert to months)
USE_MONTHS = True                   # True -> model months (easier to read)
MODEL_PATH = os.path.join(OUTDIR, "lgbm_delta_rmst_36m.joblib")
SHAP_OBJ_PATH = os.path.join(OUTDIR, "shap_lgbm_delta_rmst_36m.joblib")
MAX_SHAP_EXPLAIN = 3000             # max rows to compute SHAP on (sample if bigger)

# ---------- 1) Load data ----------
rmst_path = os.path.join(OUTDIR, "cf_vs_logit_rmst_36m_per_patient_fixed.csv")
tp_path = os.path.join(OUTDIR, "test_patients_with_period_cates.csv")

if not os.path.exists(rmst_path):
    raise FileNotFoundError(f"{rmst_path} not found. Run RMST computation first.")
if not os.path.exists(tp_path):
    raise FileNotFoundError(f"{tp_path} not found. Ensure test_patients saved in outputs/")

df_rmst = pd.read_csv(rmst_path)
test_patients = pd.read_csv(tp_path)

df = test_patients.merge(df_rmst, on="patient_id", how="left")
df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
st.write if "st" in globals() else print  # avoid linter noise

# ---------- 2) Features (adjust as needed) ----------
baseline_cat = [c for c in ['sex','smoking_status_clean','primary_site_group','pathology_group','hpv_clean'] if c in df.columns]
baseline_num = [c for c in ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean'] if c in df.columns]

# create X
X_cat = pd.get_dummies(df[baseline_cat].fillna("Missing").astype(str), drop_first=True)
X_num = df[baseline_num].copy()
for c in X_num.columns:
    X_num[c] = pd.to_numeric(X_num[c], errors='coerce')
X_num = X_num.fillna(X_num.median())

# optional: quadratic terms (uncomment if desired)
for c in ['age','ecog_ps','smoking_py_clean']:
    if c in X_num.columns:
        X_num[f"{c}_sq"] = X_num[c]**2

# combine, scale numeric
X = pd.concat([X_cat.reset_index(drop=True), X_num.reset_index(drop=True)], axis=1).fillna(0)
scaler_path = os.path.join(OUTDIR, "rf_delta_model_columns.csv")  # not used here but we save scaler later if needed

# Target y in months
y_days = df[TARGET_COL].astype(float).values
if USE_MONTHS:
    y = y_days / 30.0
    target_unit = "months"
else:
    y = y_days
    target_unit = "days"

# sample weights (if present)
if 'sample_w_36m' in df.columns:
    sample_w = df['sample_w_36m'].fillna(1.0).astype(float).values
elif 'sw_trunc' in df.columns:
    sample_w = df['sw_trunc'].fillna(1.0).astype(float).values
elif 'sw' in df.columns:
    sample_w = df['sw'].fillna(1.0).astype(float).values
else:
    sample_w = np.ones(len(df))

# train/hold split
X_train, X_hold, y_train, y_hold, w_train, w_hold = train_test_split(
    X, y, sample_w, test_size=0.2, random_state=RNG
)

# ---------- 3) LightGBM training ----------
# convert to LGBM dataset (sklearn API is fine for early stopping)
lgbm = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=7,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RNG,
    n_jobs=-1
)

# fit with early stopping using holdout
lgbm.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_hold, y_hold)],
    eval_sample_weight=[w_hold],
    eval_metric="rmse",
    early_stopping_rounds=50,
    verbose=50
)

# save model and column names
joblib.dump(lgbm, MODEL_PATH)
pd.Series(X.columns).to_csv(os.path.join(OUTDIR, "lgbm_delta_model_columns.csv"), index=False)
print("Saved LightGBM model to", MODEL_PATH)

# ---------- 4) SHAP explanation ----------
# Use TreeExplainer (fast for tree models)
explainer = shap.TreeExplainer(lgbm, feature_perturbation="tree_path_dependent")

# choose rows for SHAP (holdout or sample if large)
if X_hold.shape[0] <= MAX_SHAP_EXPLAIN:
    X_shap = X_hold.copy()
else:
    X_shap = X_hold.sample(MAX_SHAP_EXPLAIN, random_state=RNG)

shap_values = explainer.shap_values(X_shap)  # returns array (n_rows, n_features) for single-output
# store for reuse
joblib.dump({"explainer": explainer, "shap_values": shap_values, "X_shap": X_shap}, SHAP_OBJ_PATH)
print("Saved SHAP objects to", SHAP_OBJ_PATH)

# ---------- 5) Helper to show SHAP in Streamlit ----------
def st_shap_force_plot(explainer, shap_vals, X_row):
    """
    Render a shap.force_plot for a single row in Streamlit via HTML component.
    """
    import streamlit.components.v1 as components
    # shap.force_plot produces an HTML object; convert to html and render
    fp = shap.force_plot(explainer.expected_value, shap_vals, X_row, matplotlib=False)
    html = f"<head>{shap.getjs()}</head><body>{fp.html()}</body>"
    components.html(html, height=350)

# Interactive Plotly global plots (mean |SHAP| bar) and dependence
def plotly_shap_summary(shap_values, X_df, top_n=25):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    feat_imp = pd.DataFrame({"feature": X_df.columns, "mean_abs_shap": mean_abs})
    feat_imp = feat_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    fig = px.bar(feat_imp.head(top_n), x="mean_abs_shap", y="feature", orientation="h", title=f"Top {top_n} feature importance (mean |SHAP|)")
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    return fig, feat_imp

def plotly_shap_dependence(shap_values, X_df, feature):
    idx = list(X_df.columns).index(feature)
    sv = shap_values[:, idx]
    fig = px.scatter(x=X_df[feature], y=sv, labels={"x": feature, "y": "SHAP"}, title=f"SHAP dependence: {feature}")
    # add smoothed trend
    try:
        import numpy as np
        order = np.argsort(X_df[feature].values)
        fig.add_traces(go.Scatter(x=X_df[feature].values[order], y=pd.Series(sv).values[order].cumsum()*0+np.nan, mode="lines"))  # dummy if you want to add fit
    except Exception:
        pass
    return fig

# ---------- 6) Streamlit app UI ----------
def run_streamlit():
    st.title("ΔRMST explainability — LightGBM + SHAP (interactive)")

    st.sidebar.header("Model info")
    st.sidebar.write(f"Rows: {X.shape[0]}  Features: {X.shape[1]}")
    st.sidebar.write(f"Target unit: {target_unit}")

    # Global importance
    fig_imp, feat_imp_df = plotly_shap_summary(shap_values, X_shap, top_n=30)
    st.plotly_chart(fig_imp, use_container_width=True)

    # choose a feature to show dependence
    feat = st.selectbox("Choose feature for SHAP dependence", X.columns.tolist(), index=0)
    fig_dep = plotly_shap_dependence(shap_values, X_shap, feat)
    st.plotly_chart(fig_dep, use_container_width=True)

    # show scatter of predicted vs actual on holdout
    y_hold_pred = lgbm.predict(X_hold)
    scatter_fig = px.scatter(x=y_hold, y=y_hold_pred, labels={"x":"actual ΔRMST ("+target_unit+")", "y":"predicted ΔRMST ("+target_unit+")"}, title="Holdout: actual vs predicted")
    st.plotly_chart(scatter_fig, use_container_width=True)

    # Per-patient explain: choose patient id
    pid = st.selectbox("Choose patient_id to explain", df["patient_id"].unique().tolist())
    row = df[df["patient_id"]==pid].iloc[0]
    st.write("Patient baseline (selected):")
    st.json(row[baseline_cat + baseline_num].to_dict())

    # Build X_row (same preprocessing)
    X_row = pd.DataFrame([{
        **row[baseline_num].to_dict(),
        **{k: v for k, v in pd.get_dummies(pd.DataFrame([row[baseline_cat].to_dict()]).fillna('Missing').astype(str), drop_first=True).iloc[0].to_dict().items()}
    }])
    # align columns
    X_row = X_row.reindex(columns=X.columns, fill_value=0)

    st.subheader("Per-patient predicted ΔRMST")
    pred_val = lgbm.predict(X_row)[0]
    st.write(f"Predicted ΔRMST (in {target_unit}): **{pred_val:.3f}**")

    # compute shap for that single row (use explainer)
    explainer_local = explainer
    shap_vals_row = explainer_local.shap_values(X_row)
    st.write("Feature contributions (SHAP):")
    # convert to DataFrame
    sv_df = pd.DataFrame({"feature": X.columns, "shap": shap_vals_row.flatten(), "value": X_row.iloc[0].values})
    sv_df = sv_df.sort_values("shap", key=lambda s: np.abs(s), ascending=False)
    st.dataframe(sv_df.head(30))

    # show waterfall/force plot (render via HTML)
    st.subheader("SHAP force plot (interactive)")
    try:
        st_shap_force_plot(explainer_local, shap_vals_row, X_row)
    except Exception as e:
        st.error("SHAP force plot failed: " + str(e))

# If run directly, launch Streamlit app
if __name__ == "__main__":
    # If executed by "streamlit run app_shap_lightgbm.py", Streamlit will run this file and this block executes.
    run_streamlit()
