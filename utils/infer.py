import os
import io
import joblib
import json
import requests
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# module-level constants (defaults)
DEFAULT_OUTDIR = "outputs"
DEFAULT_INTERVAL_DAYS = 30
DEFAULT_PERIOD_LABELS = ['0-3','4-6','7-12','13-24','25-60','60+']

# ---------- Module-level lazy cache ----------
_artifact_cache: Dict[str, Any] = {
    "patient_columns": None,
    "patient_scaler": None,
    "pp_train_medians": None,
    "pooled_logit": None,
    "model_columns": None,
    "forests_bundle": None,
    "pp_scaler": None,
    "artifact_sources": {}
}

# ---------- Helper loaders ----------

def _safe_joblib_load(path_or_bytes, is_bytes: bool = False):
    try:
        if is_bytes:
            return joblib.load(io.BytesIO(path_or_bytes)), None
        return joblib.load(path_or_bytes), None
    except Exception as e:
        return None, str(e)


def _safe_csv_load(path_or_text, is_text: bool = False):
    try:
        if is_text:
            return pd.read_csv(io.StringIO(path_or_text), header=None), None
        return pd.read_csv(path_or_text, header=None), None
    except Exception as e:
        return None, str(e)


def _try_local_then_remote(candidates, outdir: str = DEFAULT_OUTDIR, base_url: Optional[str] = None):
    """Try list of candidate filenames locally, then (optionally) remote via base_url.
    Returns (value_or_None, source_string_or_None)
    """
    for fn in candidates:
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            ext = fn.split('.')[-1].lower()
            if ext in ("joblib", "pkl"):
                val, err = _safe_joblib_load(p)
            else:
                val, err = _safe_csv_load(p)
            if val is not None:
                return val, f"local:{p}"
    # remote
    if base_url:
        base = base_url.rstrip('/') + '/'
        for fn in candidates:
            url = base + fn
            try:
                r = requests.get(url, timeout=15)
                r.raise_for_status()
            except Exception:
                continue
            # try joblib bytes
            val, err = _safe_joblib_load(r.content, is_bytes=True)
            if val is not None:
                return val, f"remote_joblib:{url}"
            # fallback to csv/text
            val, err = _safe_csv_load(r.text, is_text=True)
            if val is not None:
                return val, f"remote_csv:{url}"
    return None, None


def _lazy_load_artifacts(outdir: str = DEFAULT_OUTDIR, base_url: Optional[str] = None):
    """Populate and return a copy of _artifact_cache. Safe to call repeatedly."""
    # simple guard: if model_columns already populated then assume done
    if _artifact_cache.get('model_columns') is not None:
        return dict(_artifact_cache)

    ART = {
        'patient_columns': ['causal_patient_columns.joblib', 'causal_patient_columns.pkl', 'causal_patient_columns.csv'],
        'patient_scaler': ['causal_patient_scaler.joblib', 'causal_patient_scaler.pkl'],
        'pp_train_medians': ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv'],
        'pooled_logit': ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib','pooled_logit_logreg_saga.pkl'],
        'model_columns': ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib','pooled_logit_model_columns.pkl'],
        'forests_bundle': ['causal_forests_period_horizons_patient_level.joblib','causal_forests_period_horizons.joblib','forests_bundle.joblib'],
        'pp_scaler': ['pp_scaler.joblib','pp_scaler.pkl']
    }

    for key, candidates in ART.items():
        val, src = _try_local_then_remote(candidates, outdir=outdir, base_url=base_url)
        _artifact_cache[key] = val
        _artifact_cache['artifact_sources'][key] = src

    # Normalize model_columns to a list of strings if CSV returned
    mc = _artifact_cache.get('model_columns')
    if isinstance(mc, pd.DataFrame):
        try:
            vals = mc.iloc[:, 0].astype(str).tolist()
            if vals and vals[0].strip().lower() in ("model_columns", "columns", "feature", "features"):
                vals = vals[1:]
            vals = [v.strip() for v in vals if v and v.strip()]
            _artifact_cache['model_columns'] = vals
        except Exception:
            pass

    # If pooled_logit loaded as DataFrame (odd), leave it - downstream will detect type
    return dict(_artifact_cache)


# ---------- Minimal fallback build_X_for_pp (if notebook function absent) ----------
def _build_X_for_pp_fallback(df_pp: pd.DataFrame, model_columns, collapse_maps=None, train_medians=None, scaler=None, cat_cols=None, num_cols=None, period_bins=None, period_labels=None):
    """
    Build X (person-period) for pooled-logit when the notebook helper isn't available.
    This is intentionally conservative: one-hot columns are attempted when present in model_columns,
    numeric columns are filled with train medians and scaled if scaler provided.
    """
    # prepare df_pp copy
    df = df_pp.copy()
    # ensure period_bin exists
    if 'period' in df.columns and 'period_bin' not in df.columns and period_bins is not None and period_labels is not None:
        df['period_month'] = df['period'].astype(int)
        df['period_bin'] = pd.cut(df['period_month'], bins=period_bins, labels=period_labels, right=True)

    # start with an empty DataFrame and then fill columns required by model_columns
    cols_req = list(model_columns) if isinstance(model_columns, (list, tuple, pd.Series, np.ndarray)) else list(model_columns)
    Xnew = pd.DataFrame(index=df.index)

    # handle categorical dummies present in model_columns
    for c in cols_req:
        if c in df.columns and not c.startswith('treat_x_'):
            # if present as numeric/categorical, copy
            Xnew[c] = df[c]
        else:
            # try one-hot form like 'sex_Male' -> root 'sex', level 'Male'
            if '_' in c:
                root, tail = c.split('_', 1)
                if root in df.columns:
                    Xnew[c] = (df[root].astype(str) == tail).astype(int)
                else:
                    Xnew[c] = 0
            else:
                # numeric candidate: try df[c] else fill with train median or 0
                if c in df.columns:
                    Xnew[c] = pd.to_numeric(df[c], errors='coerce')
                else:
                    fill = 0.0
                    if train_medians is not None and c in train_medians:
                        try:
                            fill = float(train_medians[c])
                        except Exception:
                            fill = 0.0
                    Xnew[c] = fill

    # ensure types numeric where possible
    for col in Xnew.columns:
        if Xnew[col].dtype == 'bool':
            Xnew[col] = Xnew[col].astype(int)
        # attempt numeric coercion for columns that should be numeric
        if col in (num_cols or []):
            Xnew[col] = pd.to_numeric(Xnew[col], errors='coerce').fillna(train_medians.get(col, 0.0) if train_medians is not None else 0.0)

    # apply scaler to numeric columns if provided and scaler has feature names
    if scaler is not None:
        try:
            if hasattr(scaler, 'feature_names_in_'):
                scols = list(scaler.feature_names_in_)
            else:
                scols = [c for c in ['age','ecog_ps','smoking_py_clean','time_since_rt_days'] if c in Xnew.columns]
            # fill missing scaler cols
            for sc in scols:
                if sc not in Xnew.columns:
                    Xnew[sc] = train_medians.get(sc, 0.0) if train_medians is not None else 0.0
            Xnum = Xnew[scols].astype(float)
            Xnum_scaled = pd.DataFrame(scaler.transform(Xnum), columns=scols, index=Xnew.index)
            for c in scols:
                Xnew[c] = Xnum_scaled[c]
        except Exception:
            # scaling failed -> proceed without scaling
            pass

    # ensure all required columns present and in order
    Xnew = Xnew.reindex(columns=cols_req, fill_value=0.0)
    return Xnew


# ---------- Main inference function ----------
def infer_new_patient_fixed(
    patient_data,
    return_raw: bool = False,
    outdir: str = DEFAULT_OUTDIR,
    base_url: Optional[str] = None,
    max_period_override: Optional[int] = None,
    interval_days: int = DEFAULT_INTERVAL_DAYS,
    period_labels: Optional[list] = None,
    horizon_map: Optional[dict] = None
) -> Dict[str, Any]:
    """Robust inference for a single new patient.

    Returns dict:
      {
        'survival_curve': pd.DataFrame or None,
        'CATEs': dict,
        'errors': dict,
        'debug': dict (only if return_raw True)
      }
    """
    errors = {}
    debug = {}

    if period_labels is None:
        period_labels = DEFAULT_PERIOD_LABELS

    # make patient DataFrame
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy().reset_index(drop=True)
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'new'
    if 'treatment' not in df.columns:
        df['treatment'] = int(df.get('treatment', 0))

    # load artifacts lazily
    artifacts = _lazy_load_artifacts(outdir=outdir, base_url=base_url)
    patient_columns = artifacts.get('patient_columns')
    patient_scaler = artifacts.get('patient_scaler')
    pp_train_medians = artifacts.get('pp_train_medians')
    pooled_logit = artifacts.get('pooled_logit')
    model_columns = artifacts.get('model_columns')
    forests_bundle = artifacts.get('forests_bundle')
    pp_scaler = artifacts.get('pp_scaler')

    debug['artifact_sources'] = artifacts.get('artifact_sources', {})

    # determine max_period (in months) for person-period expansion
    max_period = None
    if max_period_override is not None:
        try:
            max_period = int(max_period_override)
        except Exception:
            max_period = None

    if max_period is None:
        # try to infer from a local pp_test (if available at module-level) else default 12
        try:
            if 'pp_test' in globals() and hasattr(globals()['pp_test'], 'period'):
                max_period = int(globals()['pp_test']['period'].max())
        except Exception:
            max_period = None
    if max_period is None:
        max_period = 12

    # build person-period rows for the new patient
    rows = []
    for p in range(1, max_period + 1):
        row = df.iloc[0].to_dict()
        row['period'] = p
        row['patient_id'] = df.iloc[0].get('patient_id', 'new')
        row['treatment'] = int(row.get('treatment', 0))
        rows.append(row)
    df_pp_new = pd.DataFrame(rows)

    # ---------- pooled-logit survival ----------
    survival_df = None
    if pooled_logit is None or model_columns is None:
        errors['pooled_logit'] = 'pooled_logit or model_columns missing; cannot compute survival.'
    else:
        try:
            # prefer a build_X_for_pp function if present in globals (from notebook)
            if 'build_X_for_pp' in globals() and callable(globals()['build_X_for_pp']):
                X_pp = globals()['build_X_for_pp'](df_pp_new.copy())
            else:
                # use fallback builder
                X_pp = _build_X_for_pp_fallback(
                    df_pp_new.copy(), model_columns,
                    collapse_maps=None,
                    train_medians=(pp_train_medians if isinstance(pp_train_medians, dict) else ({k: v for k, v in (pp_train_medians.items() if hasattr(pp_train_medians, 'items') else [])} if pp_train_medians is not None else {})),
                    scaler=pp_scaler,
                    cat_cols=None,
                    num_cols=None,
                    period_bins=None,
                    period_labels=period_labels
                )

            # ensure model_columns ordering (model_columns should be list)
            if isinstance(model_columns, pd.DataFrame):
                cols_req = model_columns.iloc[:, 0].astype(str).tolist()
            else:
                cols_req = list(model_columns)
            X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)

            # build treated & control counterfactuals
            X_t = X_pp.copy(); X_t['treatment'] = 1
            X_c = X_pp.copy(); X_c['treatment'] = 0

            # recompute any treat_x_ interactions conservatively
            for c in [c for c in cols_req if str(c).startswith('treat_x_')]:
                root = c.replace('treat_x_', '')
                X_t[c] = X_t['treatment'] * X_t.get(root, 0)
                X_c[c] = X_c['treatment'] * X_c.get(root, 0)

            # make predictions
            # pooled_logit might be sklearn logistic or statsmodels wrapper; support both
            if hasattr(pooled_logit, 'predict_proba'):
                probs_t = np.asarray(pooled_logit.predict_proba(X_t)[:, 1]).flatten()
                probs_c = np.asarray(pooled_logit.predict_proba(X_c)[:, 1]).flatten()
            else:
                # try joblib-loaded statsmodels results (result object with model)
                try:
                    probs_t = np.asarray(pooled_logit.predict(X_t)).flatten()
                    probs_c = np.asarray(pooled_logit.predict(X_c)).flatten()
                except Exception as e:
                    raise RuntimeError(f'pooled_logit predict failed: {e}')

            S_t = np.cumprod(1 - probs_t)
            S_c = np.cumprod(1 - probs_c)
            survival_df = pd.DataFrame({'period': np.arange(1, len(S_t) + 1), 'S_control': S_c, 'S_treat': S_t})
            survival_df['days'] = survival_df['period'] * interval_days
            debug['X_pp_sample'] = X_pp.head(5).to_dict(orient='records')
        except Exception as e:
            errors['pooled_logit'] = f'pipelined survival predict failed: {e}'

    # ---------- build Xpatient (for CF) ----------
    Xpatient = None
    if patient_columns is None:
        errors['patient_columns'] = 'patient_columns artifact missing. CF prediction will be limited.'
    else:
        try:
            # normalize patient_columns
            if isinstance(patient_columns, (list, tuple, pd.Series, np.ndarray)):
                pcols = list(patient_columns)
            elif isinstance(patient_columns, dict):
                if 'columns' in patient_columns:
                    pcols = list(patient_columns['columns'])
                else:
                    pcols = list(patient_columns.keys())
            elif isinstance(patient_columns, pd.DataFrame):
                pcols = patient_columns.iloc[:, 0].astype(str).tolist()
            else:
                pcols = list(patient_columns)

            Xpatient = pd.DataFrame(np.zeros((1, len(pcols))), columns=pcols)

            # Fill from df row values or infer one-hot
            for c in pcols:
                if c in df.columns:
                    Xpatient.at[0, c] = df.at[0, c]
                else:
                    if '_' in c:
                        root, tail = c.split('_', 1)
                        if root in df.columns and str(df.at[0, root]) == tail:
                            Xpatient.at[0, c] = 1.0
            Xpatient = Xpatient.reindex(columns=pcols, fill_value=0.0)
        except Exception as e:
            errors['Xpatient'] = f'Failed to construct Xpatient: {e}'
            Xpatient = None

    # scale numeric parts of Xpatient using patient_scaler if possible
    if Xpatient is not None and patient_scaler is not None:
        try:
            if hasattr(patient_scaler, 'feature_names_in_'):
                scaler_cols = list(patient_scaler.feature_names_in_)
            else:
                scaler_cols = [c for c in Xpatient.columns if c in ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean','time_since_rt_days']]

            for sc_col in scaler_cols:
                if sc_col not in Xpatient.columns:
                    fill_val = 0.0
                    try:
                        if isinstance(pp_train_medians, dict) and sc_col in pp_train_medians:
                            fill_val = pp_train_medians[sc_col]
                        elif hasattr(pp_train_medians, 'get'):
                            fill_val = pp_train_medians.get(sc_col, 0.0)
                    except Exception:
                        fill_val = 0.0
                    Xpatient[sc_col] = float(fill_val)

            Xnum = Xpatient[scaler_cols].astype(float).copy()
            Xnum_scaled = patient_scaler.transform(Xnum)
            if isinstance(Xnum_scaled, np.ndarray):
                Xnum_scaled = pd.DataFrame(Xnum_scaled, columns=scaler_cols, index=Xpatient.index)
            for c in scaler_cols:
                Xpatient[c] = Xnum_scaled[c].values
        except Exception as e:
            errors['scaler'] = f'scaler exists but failed to transform Xpatient; proceeding without scaling: {e}'

    debug['Xpatient'] = Xpatient.copy() if Xpatient is not None else None

    # ---------- CF CATE predictions ----------
    cate_results = {}
    if forests_bundle is None:
        errors['forests_bundle'] = 'forests bundle not found in outputs or via base_url.'
        for lab in (period_labels or DEFAULT_PERIOD_LABELS):
            try:
                months = int(str(lab).replace('+','').split('-')[-1])
            except Exception:
                months = lab
            cate_results[months] = {'CATE': np.nan, 'error': errors['forests_bundle']}
    else:
        # forests_bundle may be dict mapping label->estimator
        bundle_items = forests_bundle.items() if isinstance(forests_bundle, dict) else [(str(k), v) for k, v in enumerate([forests_bundle])]
        for lab, est in bundle_items:
            if horizon_map and lab in horizon_map:
                months = horizon_map[lab]
            else:
                try:
                    months = int(str(lab).replace('+','').split('-')[-1])
                except Exception:
                    months = lab

            if Xpatient is None:
                cate_results[months] = {'CATE': np.nan, 'error': 'Xpatient not built'}
                continue

            try:
                candidate = est
                if isinstance(est, dict):
                    # try to find nested estimator with effect method
                    for v in est.values():
                        if hasattr(v, 'effect'):
                            candidate = v
                            break

                # prepare input array aligned to candidate expectations
                if hasattr(candidate, 'feature_names_in_'):
                    req = list(candidate.feature_names_in_)
                    for c in req:
                        if c not in Xpatient.columns:
                            Xpatient[c] = 0.0
                    Xfor_in = Xpatient[req].values
                else:
                    # fallback to model_columns
                    if isinstance(model_columns, (list, pd.Series, pd.DataFrame, np.ndarray)):
                        if isinstance(model_columns, pd.DataFrame):
                            req = model_columns.iloc[:,0].astype(str).tolist()
                        else:
                            req = list(model_columns)
                        for c in req:
                            if c not in Xpatient.columns:
                                Xpatient[c] = 0.0
                        Xfor_in = Xpatient[req].values
                    else:
                        Xfor_in = Xpatient.values

                # effect method expected to return Chemo-RT − RT in days (or probability), keep as-is
                eff = np.asarray(candidate.effect(Xfor_in)).flatten()
                val = float(eff[0]) if eff.size > 0 else np.nan
                cate_results[months] = {'CATE': val, 'error': None}
            except Exception as e:
                cate_results[months] = {'CATE': np.nan, 'error': str(e)}

    # sort cate_results by numeric horizon where possible
    try:
        cate_results = dict(sorted(cate_results.items(), key=lambda kv: (float(kv[0]) if isinstance(kv[0], (int,float,str)) and str(kv[0]).replace('.','',1).isdigit() else 1e9)))
    except Exception:
        pass

    out = {'survival_curve': survival_df, 'CATEs': cate_results, 'errors': errors}
    if return_raw:
        out['debug'] = debug
    return out


# convenience alias
infer = infer_new_patient_fixed
