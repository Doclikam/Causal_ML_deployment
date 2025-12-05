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









# ---------- Bootstrap: ensure 'loaded' and defaults exist (paste right after imports) ----------
import os, joblib, pandas as pd

# safe 'loaded' dict (prevents NameError everywhere)
if not isinstance(globals().get('loaded'), dict):
    loaded = {}
    globals()['loaded'] = loaded

# OUTDIR and FILES defaults (adjust filenames to match your project)
OUTDIR = globals().get('OUTDIR', os.path.join(os.getcwd(), "outputs"))
FILES = globals().get('FILES', {
    'pooled_cols': "pooled_logit_model_columns.csv",
    'pooled_logit': "pooled_logit_logreg_saga.joblib",
    'cf_model': "cf_rmst_36m_patient_level.joblib",
    'pooled_pp': "pp_test.csv",
    'shap_summary_img': "shap_summary.png"
})
globals()['OUTDIR'] = OUTDIR
globals()['FILES'] = FILES

# helper: safe joblib loader that won't crash app
def safe_load_joblib(path):
    try:
        if path is None:
            return None
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        # don't raise — just return None and let later code handle missing artifact
        print(f"[bootstrap] safe_load_joblib failed for {path}: {e}")
    return None

# try to preload a few common artifacts into loaded (non-fatal)
_try_paths = {
    'pooled_logit': os.path.join(OUTDIR, FILES.get('pooled_logit')),
    'pooled_cols': os.path.join(OUTDIR, FILES.get('pooled_cols')),
    'cf_model': os.path.join(OUTDIR, FILES.get('cf_model')),
    'pooled_pp': os.path.join(OUTDIR, FILES.get('pooled_pp'))
}
for k, p in _try_paths.items():
    if k in loaded:  # don't overwrite if already set upstream
        continue
    try:
        if p.endswith('.csv') or p.endswith('.txt'):
            if os.path.exists(p):
                if k == 'pooled_cols':
                    loaded[k] = pd.read_csv(p).squeeze().tolist()
                else:
                    loaded[k] = pd.read_csv(p)
        else:
            v = safe_load_joblib(p)
            if v is not None:
                loaded[k] = v
    except Exception as e:
        print(f"[bootstrap] preload failed for {p}: {e}")

# ensure loaded is available globally (helpful when streamlit re-runs)
globals()['loaded'] = loaded
print("[bootstrap] loaded dict initialized; OUTDIR:", OUTDIR)
# ---------- end bootstrap ----------



# ------------------ In "Time-varying & period-level panel" section update ------------------
st.header('Time-varying effects: hazards, HR, cumulative ΔRMST (interactive)')

# Safe access to loaded dict and df
tv = (loaded.get('timevarying_summary') if isinstance(loaded.get('timevarying_summary'), (pd.DataFrame, dict)) else None) \
     or (loaded.get('bootstrap_period_results') if isinstance(loaded.get('bootstrap_period_results'), (pd.DataFrame, dict)) else None)

# attempt to build from pp if missing and pooled-logit exists
pp_for_tv = None
pooled_logit = loaded.get('pooled_logit', None)
model_columns = None
pc_path = os.path.join(OUTDIR, FILES.get('pooled_cols', 'pooled_logit_model_columns.csv'))
if os.path.exists(pc_path):
    try:
        model_columns = pd.read_csv(pc_path).squeeze().tolist()
    except Exception as e:
        st.warning(f"Could not load pooled-logit column file {pc_path}: {e}")

# Try to find person-period table
pp_for_tv = loaded.get('pooled_pp', None) or loaded.get('pp_test', None)
if pp_for_tv is None:
    # try files in OUTDIR
    for fn in ['pp_test.csv','pp_test.parquet','pp_test.pkl','pp_test.feather']:
        p = os.path.join(OUTDIR, fn)
        if os.path.exists(p):
            try:
                if p.endswith('.csv'):
                    pp_for_tv = pd.read_csv(p)
                elif p.endswith('.parquet'):
                    pp_for_tv = pd.read_parquet(p)
                else:
                    pp_for_tv = safe_load_pkl(p)
                break
            except Exception as e:
                st.warning(f"Could not read {p}: {e}")

# If none of the inputs exist, instruct the user
if tv is None and pp_for_tv is None and (pooled_logit is None or model_columns is None):
    st.info('Time-varying results / person-period table / pooled-logit not found. Provide one of these artifacts to enable interactive time-varying analysis.')
else:
    pp_df = pp_for_tv if pp_for_tv is not None else (tv if isinstance(tv, pd.DataFrame) else None)
    psum = None
    if pp_df is not None:
        try:
            psum = compute_period_summary(pp_df, model_logit=pooled_logit, model_columns=model_columns)
        except Exception as e:
            st.warning("compute_period_summary failed: " + str(e))
            psum = None

    if psum is None:
        st.warning('Could not compute period summary; check pooled-logit or pp table.')
        pp_summary = (tv.copy() if isinstance(tv, pd.DataFrame) else None)
    else:
        pp_summary = pd.DataFrame({
            'period': psum['periods'],
            'haz_treated': psum['haz_treated'],
            'haz_control': psum['haz_control'],
            'hr': psum['hr'],
            'delta_rmst_period_days': psum['delta_rmst_by_period']
        })

    if pp_summary is not None and not pp_summary.empty:
        max_period = int(pp_summary['period'].max())
        sel_period = st.slider('Select period (index)', min_value=int(pp_summary['period'].min()), max_value=max_period, value=int(min(3,max_period)), step=1)
        row = pp_summary.loc[pp_summary['period']==sel_period].iloc[0]
        st.metric("Period (index)", sel_period)
        st.metric("Hazard (treated)", f"{row['haz_treated']:.4f}")
        st.metric("Hazard (control)", f"{row['haz_control']:.4f}")
        st.metric("Hazard Ratio (treated/control)", f"{row['hr']:.3f}")

        # plots (uses plotly go)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pp_summary['period'], y=pp_summary['haz_control'], name='Control hazard', mode='lines+markers'))
        fig.add_trace(go.Scatter(x=pp_summary['period'], y=pp_summary['haz_treated'], name='Treated hazard', mode='lines+markers'))
        fig.update_layout(title='Marginal interval hazards by period', xaxis_title='Period (index)', yaxis_title='Hazard')
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=pp_summary['period'], y=pp_summary['hr'], name='HR', mode='lines+markers'))
        fig2.add_hline(y=1.0, line_dash='dash')
        fig2.update_layout(title='Time-varying HR (treated / control)', xaxis_title='Period (index)', yaxis_title='HR')
        st.plotly_chart(fig2, use_container_width=True)

        cumul_days = np.cumsum(pp_summary['delta_rmst_period_days'].fillna(0).values)
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=pp_summary['period'], y=cumul_days/30.0, mode='lines+markers', name='Cumulative ΔRMST (months)'))
        fig3.update_layout(title=f'Cumulative ΔRMST up to period (months)', xaxis_title='Period', yaxis_title='Months')
        st.plotly_chart(fig3, use_container_width=True)

        peak_period, peak_val = detect_peak(pp_summary['delta_rmst_period_days'].fillna(0).values, pp_summary['period'].values, frac=0.25)
        st.success(f"Peak marginal ΔRMST period: **{peak_period}** (approx. ΔRMST contribution {peak_val/30.0:.2f} months at that period).")
        st.caption("Interpretation: the period shown is where the marginal, population-level, per-period contribution to ΔRMST is largest.")

# ----------------------------------------------------------------
