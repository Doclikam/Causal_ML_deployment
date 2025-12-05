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


st.set_page_config(page_title="H&N Causal Survival Explorer", layout="wide")

# ------------------ Insert near top after artifact loading ------------------
import scipy.signal as signal
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import pairwise_distances_argmin_min

# helper to safe-get columns
def safe_col(df, colnames):
    for c in colnames:
        if c in df.columns: return c
    return None

# helper: compute/return period-level summary (haz_treated,haz_control,hr,delta_rmst_by_period)
def compute_period_summary(pp_df, model_logit=None, model_columns=None, interval_days=30, weight_col='w_for_agg'):
    # expects pp_df to have period column and patient-level rows for person-period predictions
    if pp_df is None or pp_df.shape[0]==0:
        return None
    # if pooled-logit available, prefer using p_treated/p_control already in pp_df
    if {'p_treated','p_control'}.issubset(pp_df.columns):
        haz_t = pp_df.groupby('period').apply(lambda g: np.average(g['p_treated'], weights=g.get(weight_col, np.ones(len(g)))))
        haz_c = pp_df.groupby('period').apply(lambda g: np.average(g['p_control'], weights=g.get(weight_col, np.ones(len(g)))))
    elif model_logit is not None and model_columns is not None:
        X_pp = build_X_for_pp(pp_df)
        X_t = X_pp.copy(); X_t['treatment']=1
        X_c = X_pp.copy(); X_c['treatment']=0
        X_t = X_t.reindex(columns=model_columns, fill_value=0)
        X_c = X_c.reindex(columns=model_columns, fill_value=0)
        hr = None
        pt = model_logit.predict_proba(X_t)[:,1]; pc = model_logit.predict_proba(X_c)[:,1]
        pp_df = pp_df.copy()
        pp_df['p_treated'] = pt; pp_df['p_control'] = pc
        haz_t = pp_df.groupby('period').apply(lambda g: np.average(g['p_treated'], weights=g.get(weight_col, np.ones(len(g)))))
        haz_c = pp_df.groupby('period').apply(lambda g: np.average(g['p_control'], weights=g.get(weight_col, np.ones(len(g)))))
    else:
        return None

    periods = np.array(sorted(haz_t.index.astype(int)))
    haz_t = np.array([haz_t.loc[p] for p in periods])
    haz_c = np.array([haz_c.loc[p] for p in periods])

    with np.errstate(divide='ignore', invalid='ignore'):
        hr_period = haz_t / haz_c
        hr_period[~np.isfinite(hr_period)] = np.nan

    # cumulative RMST contribution per period (days)
    S_t = np.cumprod(1 - haz_t)
    S_c = np.cumprod(1 - haz_c)
    S_prev_t = np.concatenate(([1.0], S_t[:-1]))
    S_prev_c = np.concatenate(([1.0], S_c[:-1]))
    period_len = np.array([interval_days]*len(periods))
    delta_rmst_by_period = (S_prev_t - S_prev_c) * period_len
    return {
        'periods': periods,
        'haz_treated': haz_t,
        'haz_control': haz_c,
        'hr': hr_period,
        'delta_rmst_by_period': delta_rmst_by_period,
        'S_t': S_t, 'S_c': S_c
    }

# helper: detect peak period (smoothed)
def detect_peak(delta_periods, periods, frac=0.3):
    if len(delta_periods) < 3:
        idx = np.nanargmax(delta_periods) if len(delta_periods)>0 else None
        return periods[idx] if idx is not None else None, delta_periods.max() if len(delta_periods)>0 else np.nan
    lo = lowess(delta_periods, periods, frac=frac, return_sorted=False)
    # detect local maxima indices in lowess
    peaks, _ = signal.find_peaks(lo)
    if len(peaks)==0:
        imax = int(np.nanargmax(lo))
        return int(periods[imax]), float(lo[imax])
    # pick highest peak
    imax = peaks[np.argmax(lo[peaks])]
    return int(periods[imax]), float(lo[imax])

# ------------------ In "Time-varying & period-level panel" section update ------------------
st.header('Time-varying effects: hazards, HR, cumulative ΔRMST (interactive)')
tv = loaded.get('timevarying_summary') or loaded.get('bootstrap_period_results')
# attempt to build from pp if missing and pooled-logit exists
pp_for_tv = None
if tv is None and df is not None:
    # try to find person-period (pp_test) in outputs as fallback
    pp_paths = [os.path.join(OUTDIR, fn) for fn in ['pp_test.csv','pp_test.parquet','pp_test.pkl','pp_test.feather']]
    for p in pp_paths:
        if os.path.exists(p):
            try:
                pp_for_tv = pd.read_csv(p) if p.endswith('.csv') else joblib.load(p)
                break
            except:
                pass
else:
    pp_for_tv = None

# load pooled-logit model if available
pooled_logit = loaded.get('pooled_logit')
model_columns = None
pc_path = os.path.join(OUTDIR, FILES['pooled_cols'])
if os.path.exists(pc_path):
    try:
        model_columns = pd.read_csv(pc_path).squeeze().tolist()
    except:
        model_columns = None

if tv is None and pp_for_tv is None and (pooled_logit is None):
    st.info('Time-varying results / person-period table / pooled-logit not found. Provide one of these artifacts to enable interactive time-varying analysis.')
else:
    # compute period summary
    pp_df = loaded.get('pooled_pp', None) or pp_for_tv or (df if 'period' in df.columns else None)
    if pp_df is None:
        st.warning('No person-period table detected; showing saved summary if present.')
        if tv is not None:
            pp_summary = tv.copy()
        else:
            pp_summary = None
    else:
        psum = compute_period_summary(pp_df, model_logit=pooled_logit, model_columns=model_columns)
        if psum is None:
            st.warning('Could not compute period summary; check pooled-logit or pp table.')
            pp_summary = tv
        else:
            # build a simple DataFrame for display
            pp_summary = pd.DataFrame({
                'period': psum['periods'],
                'haz_treated': psum['haz_treated'],
                'haz_control': psum['haz_control'],
                'hr': psum['hr'],
                'delta_rmst_period_days': psum['delta_rmst_by_period']
            })

    if pp_summary is not None:
        # interactive slider: period or months
        max_period = int(pp_summary['period'].max())
        sel_period = st.slider('Select period (index)', min_value=int(pp_summary['period'].min()), max_value=max_period, value=int(min(3,max_period)), step=1)
        # present summary for selected period
        row = pp_summary.loc[pp_summary['period']==sel_period].iloc[0]
        st.metric("Period (index)", sel_period)
        st.metric("Hazard (treated)", f"{row['haz_treated']:.4f}")
        st.metric("Hazard (control)", f"{row['haz_control']:.4f}")
        st.metric("Hazard Ratio (treated/control)", f"{row['hr']:.3f}")

        # Plot hazards & HR full trend
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pp_summary['period'], y=pp_summary['haz_control'], name='Control hazard', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=pp_summary['period'], y=pp_summary['haz_treated'], name='Treated hazard', mode='lines+markers'))
        fig.update_layout(title='Marginal interval hazards by period', xaxis_title='Period (index)', yaxis_title='Hazard')
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pp_summary['period'], y=pp_summary['hr'], name='HR', mode='lines+markers'))
        fig2.add_trace(go.Scatter(x=np.concatenate([pp_summary['period'], pp_summary['period'][::-1]]),
                                  y=np.concatenate([pp_summary['hr'], pp_summary['hr'][::-1]]), fill='toself', name='CI (none)', opacity=0.1))
        fig2.add_hline(y=1.0, line_dash='dash')
        fig2.update_layout(title='Time-varying HR (treated / control)', xaxis_title='Period (index)', yaxis_title='HR')
        st.plotly_chart(fig2, use_container_width=True)

        # cumulative ΔRMST
        cumul_days = np.cumsum(pp_summary['delta_rmst_period_days'].fillna(0).values)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=pp_summary['period'], y=cumul_days/30.0, mode='lines+markers', name='Cumulative ΔRMST (months)'))
        fig3.update_layout(title=f'Cumulative ΔRMST up to period (months)', xaxis_title='Period', yaxis_title='Months')
        st.plotly_chart(fig3, use_container_width=True)

        # detect and show peak period
        peak_period, peak_val = detect_peak(pp_summary['delta_rmst_period_days'].fillna(0).values, pp_summary['period'].values, frac=0.25)
        st.success(f"Peak marginal ΔRMST period: **{peak_period}** (approx. ΔRMST contribution {peak_val/30.0:.2f} months at that period).")
        st.caption("Interpretation: the period shown is where the marginal, population-level, **per-period** contribution to ΔRMST is largest. This helps identify when treatment effect peaks after RT start.")

# ------------------ SHAP / Top predictors section (insert where you show SHAP) ------------------
st.header("Top predictors & SHAP explanations")
if loaded.get('rf_shap') is None and loaded.get('rf_model') is None:
    st.info("SHAP objects or surrogate RF not found in outputs/. The app can still show top features from model coefficients if available.")
else:
    shap_obj = loaded.get('rf_shap')
    # try to handle both dict and TreeExplainer objects
    try:
        if isinstance(shap_obj, dict):
            X_explain = shap_obj.get('X_explain')
            shap_vals = shap_obj.get('shap_values')
        else:
            # if user saved explainer only, user will need to recompute shap on small sample
            X_explain = None
            shap_vals = None
    except Exception:
        X_explain = None
        shap_vals = None

    if X_explain is None or shap_vals is None:
        st.warning("SHAP objects incomplete; showing saved summary image if present.")
        if os.path.exists(os.path.join(OUTDIR, FILES['shap_summary_img'])):
            st.image(os.path.join(OUTDIR, FILES['shap_summary_img']), caption="SHAP summary (global)")
    else:
        imp_df = pd.DataFrame({'feature': X_explain.columns, 'mean_abs_shap': np.abs(shap_vals).mean(axis=0)})
        imp_df = imp_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
        st.subheader("Top features (mean |SHAP|)")
        st.dataframe(imp_df.head(20))

        # interactive feature dependence
        feat_choice = st.selectbox('Select feature for SHAP dependence plot', options=imp_df['feature'].tolist()[:30])
        if feat_choice:
            # produce dependence plot via SHAP
            try:
                fig_shap = shap.dependence_plot(feat_choice, shap_vals, X_explain, show=False, interaction_index=None)
                st.pyplot(bbox_inches='tight')
            except Exception as e:
                st.warning("Could not render dependence plot in-streamlit: " + str(e))

# ------------------ Upload CSV → Batch prediction & download ------------------
st.header("Upload CSV → Batch survival & CATE prediction")
st.write("Upload a CSV of patient rows (columns should match those in your train patient table). The app will attempt to apply saved artifacts (preprocessor, pooled-logit, CF) to produce ΔRMST & CATE predictions. Files required: ps_preprocessor.joblib, pooled_logit_model_columns.csv, pooled_logit_logreg_saga.joblib, and cf_rmst_36m_patient_level.joblib (optional).")

uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    try:
        new_df = pd.read_csv(uploaded_file)
        st.write("Rows:", new_df.shape[0])
        # Apply collapse_maps and preprocessor if available
        preproc = safe_load_joblib(os.path.join(OUTDIR, 'ps_preprocessor.joblib'))
        pooled_cols = None
        if os.path.exists(os.path.join(OUTDIR, FILES['pooled_cols'])):
            pooled_cols = pd.read_csv(os.path.join(OUTDIR, FILES['pooled_cols'])).squeeze().tolist()
        pooled_model = safe_load_joblib(os.path.join(OUTDIR, 'pooled_logit')) or safe_load_joblib(os.path.join(OUTDIR, 'pooled_logit_logreg_saga.joblib'))
        # Build person-period for upload if time_os_days present, else predict patient-level only
        if 'time_os_days' in new_df.columns and 'event_os' in new_df.columns:
            # expand to person-period using existing function if available
            try:
                new_pp = expand_to_pp(new_df, interval_days=interval_days)
            except Exception:
                # fallback simple single-period prediction
                new_pp = None
        else:
            new_pp = None

        # apply pooled-logit to predict per-row hazard if model available
        if preproc is not None and pooled_model is not None and pooled_cols is not None:
            # transform features (safe subset)
            cat_cols_local = [c for c in ['sex','smoking_status_clean','primary_site_group','t','n','m','pathology_group','hpv_clean'] if c in new_df.columns]
            num_cols_local = [c for c in ['age','ecog_ps','smoking_py_clean'] if c in new_df.columns]
            try:
                Xu = preproc.transform(new_df[cat_cols_local + num_cols_local])
            except Exception:
                # try transform with available columns
                st.warning("Preprocessor transform failed on uploaded data — ensure categorical and numeric columns align.")
                Xu = None
            # if successful, compute per-row p(treated) or per-pp hazard
        else:
            st.warning("Missing artifacts for pooled-logit predictions; will skip pooled-logit steps.")

        # predict CF CATE per patient if CF model exists
        cf_model = safe_load_joblib(os.path.join(OUTDIR, FILES['cf_model']))
        if cf_model is not None:
            # to predict CATEs we need same X used to train CF (baseline features)
            # build those features from new_df (best-effort)
            baseline_cat = [c for c in ['sex','smoking_status_clean','primary_site_group','pathology_group','hpv_clean'] if c in new_df.columns]
            baseline_num = [c for c in ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean'] if c in new_df.columns]
            X_cat_new = pd.get_dummies(new_df[baseline_cat].fillna('Missing').astype(str), drop_first=True)
            # align columns with training (we saved rf_delta_model_columns or trained CF cols earlier)
            # best-effort: reindex to columns of Xtr if available in outputs (rf_delta_model_columns.csv)
            rf_cols_path = os.path.join(OUTDIR, "rf_delta_model_columns.csv")
            if os.path.exists(rf_cols_path):
                rf_cols = pd.read_csv(rf_cols_path).squeeze().tolist()
            else:
                rf_cols = None
            # numeric
            X_num_new = new_df[baseline_num].fillna(new_df[baseline_num].median()) if baseline_num else pd.DataFrame(index=new_df.index)
            Xnew = pd.concat([X_cat_new.reset_index(drop=True), X_num_new.reset_index(drop=True)], axis=1).fillna(0)
            if rf_cols is not None:
                Xnew = Xnew.reindex(columns=rf_cols, fill_value=0.0)
            try:
                cates_new = cf_model.effect(Xnew.values)
                new_df['cf_cate_36m_months'] = cates_new/30.0
            except Exception as e:
                st.warning("CF prediction failed on uploaded data: " + str(e))
        else:
            st.info("CF model missing → cannot compute CATEs. If you saved CF model file, place in outputs/ and reload app.")

        # produce a downloadable CSV
        outbuf = BytesIO()
        new_df.to_csv(outbuf, index=False)
        outbuf.seek(0)
        st.download_button("Download predictions CSV", data=outbuf, file_name="uploaded_predictions_with_CATE.csv")

    except Exception as e:
        st.error("Upload failed: " + str(e))

# ------------------ Subgroup summary section (add T,N,M,age, pathology) ------------------
st.header("Subgroup summaries (CATE & ΔRMST)")
if df is None:
    st.info("Merged data not loaded; subgroup summaries disabled.")
else:
    subs = st.multiselect("Subgroup variables", options=['hpv_clean','ecog_ps','t','n','m','stage','pathology_group','age'], default=['hpv_clean'])
    # make a small summary table
    def subgroup_table(df, group_vars, cate_col='cf_cate_months', delta_col='delta_rmst_months'):
        rows = []
        for g in group_vars:
            if g not in df.columns: continue
            if g == 'age':
                df['age_q'] = pd.qcut(df['age'].fillna(df['age'].median()), q=4, labels=['Q1','Q2','Q3','Q4'])
                groups = df['age_q'].unique()
                gcol = 'age_q'
            else:
                groups = df[g].dropna().unique()
                gcol = g
            for val in sorted([str(x) for x in groups]):
                mask = df[gcol].astype(str) == str(val)
                if mask.sum() < 5:
                    continue
                w = df.loc[mask, 'w_use'] if 'w_use' in df.columns else np.ones(mask.sum())
                cf_mean = np.average(df.loc[mask, cate_col].fillna(0), weights=w)
                log_mean = np.average(df.loc[mask, delta_col].fillna(0), weights=w)
                rows.append({'group_var': g, 'group': val, 'n': int(mask.sum()), 'cf_mean_months': cf_mean, 'delta_mean_months': log_mean})
        return pd.DataFrame(rows)
    st.dataframe(subgroup_table(use_df if filtered_df is not None else df, subs).sort_values(['group_var','n'], ascending=[True, False]).head(200))

# ----------------------------------------------------------------
# small UX tweak: explanation text blocks placed near charts are already included above
# ----------------------------------------------------------------
