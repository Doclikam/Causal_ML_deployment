"""
utils/infer.py

Robust inference utilities for single-patient predictions:
 - pooled-logit survival prediction (counterfactual treated/control)
 - patient-level CATE predictions from a forests bundle
Design goals:
 - import-time safe (no crashes if artifacts missing)
 - lazy-load artifacts from outputs/ (local) or optional base_url (raw files)
 - return structured output with errors + debug info
"""

import os
import io
import joblib
import json
import math
import requests
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any

# -------------------------
# Configuration / defaults
# -------------------------
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_OUTDIR = os.path.join(_REPO_ROOT, "outputs")
_DEFAULT_INTERVAL_DAYS = 30
_DEFAULT_PERIOD_LABELS = ['0-3', '4-6', '7-12', '13-24', '25-60', '60+']

# Candidate filenames (prefer joblib / csv pairs)
_ARTIFACT_CANDIDATES = {
    "model_columns": ["pooled_logit_model_columns.csv", "pooled_logit_model_columns.joblib"],
    "pooled_logit": ["pooled_logit_logreg_saga.joblib", "pooled_logit.joblib"],
    "pp_scaler": ["pp_scaler.joblib", "pp_scaler.pkl"],
    "pp_train_medians": ["pp_train_medians.joblib", "pp_train_medians.pkl"],
    "collapse_maps": ["pp_collapse_maps.joblib", "pp_collapse_maps.pkl"],
    "patient_scaler": ["causal_patient_scaler.joblib","causal_patient_scaler.pkl"],
    "patient_columns": ["causal_patient_columns.joblib","causal_patient_columns.pkl","causal_patient_columns.csv"],
    "forests_bundle": ["causal_forests_period_horizons_patient_level.joblib","causal_forests_period_horizons.joblib","forests_bundle.joblib"]
}

# Lazy-loaded globals
_ARTS = {
    "model_columns": None,
    "pooled_logit": None,
    "pp_scaler": None,
    "pp_train_medians": None,
    "collapse_maps": None,
    "patient_scaler": None,
    "patient_columns": None,
    "forests_bundle": None,
}
_ART_SOURCES = {}  # tracks where each artifact came from

# -------------------------
# Helpers: artifact loading
# -------------------------
def _local_path(outdir: str, name: str) -> Optional[str]:
    cand = _ARTIFACT_CANDIDATES.get(name, [])
    for fn in cand:
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            return p
    return None

def _read_csv_or_joblib_from_url(url: str):
    """
    Try to download and read either joblib (binary) or csv content from raw url.
    Returns (obj, source_str) or (None, None)
    """
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        # try joblib first
        try:
            return joblib.load(io.BytesIO(r.content)), f"remote_joblib:{url}"
        except Exception:
            try:
                # treat as csv text
                txt = r.text
                df = pd.read_csv(io.StringIO(txt), header=None)
                return df, f"remote_csv:{url}"
            except Exception:
                return None, None
    except Exception:
        return None, None

def _try_load_local_or_remote(name: str, outdir: str, base_url: Optional[str] = None):
    """
    Load artifact by checking local candidate files first, then remote (base_url + filename).
    Sets _ARTS[name] and _ART_SOURCES[name] if loaded.
    """
    # local
    p = _local_path(outdir, name)
    if p:
        try:
            # if it's a CSV list of one column we interpret differently
            if p.endswith(".csv"):
                df = pd.read_csv(p, header=None)
                _ARTS[name] = df
                _ART_SOURCES[name] = f"local_csv:{p}"
                return
            else:
                val = joblib.load(p)
                _ARTS[name] = val
                _ART_SOURCES[name] = f"local_joblib:{p}"
                return
        except Exception as e:
            _ARTS[name] = None
            _ART_SOURCES[name] = f"local_failed:{p}:{e}"

    # remote
    if base_url:
        base = base_url.rstrip("/") + "/"
        for fn in _ARTIFACT_CANDIDATES.get(name, []):
            url = base + fn
            val, src = _read_csv_or_joblib_from_url(url)
            if val is not None:
                _ARTS[name] = val
                _ART_SOURCES[name] = src
                return

    # not found
    _ARTS[name] = None
    _ART_SOURCES[name] = None

def _ensure_artifacts_loaded(outdir: str, base_url: Optional[str]):
    """
    Load artifacts into _ARTS as needed. Safe to call repeatedly.
    """
    for key in _ARTIFACT_CANDIDATES.keys():
        if _ARTS.get(key) is None:
            _try_load_local_or_remote(key, outdir, base_url)

# -------------------------
# Small utilities
# -------------------------
def _clean_model_columns_from_df(df):
    """
    Given a DataFrame read from pooled_logit_model_columns.csv (single-column),
    return a clean list of column names (drop header-like first row).
    """
    if df is None:
        return None
    try:
        lst = df.iloc[:, 0].astype(str).tolist()
        if lst and lst[0].strip().lower() in ("model_columns", "column", "columns", "feature", "features"):
            lst = lst[1:]
        lst = [s.strip() for s in lst if isinstance(s, str) and s.strip()]
        return lst
    except Exception:
        # if df already joblib list-like
        try:
            if isinstance(df, (list, tuple, np.ndarray, pd.Series)):
                return [str(x).strip() for x in list(df) if str(x).strip()]
        except Exception:
            return None

def _ensure_numeric_array(x):
    try:
        return np.asarray(x, dtype=float)
    except Exception:
        return np.asarray(x, dtype=float, order='C')

# -------------------------
# Public accessor helpers
# -------------------------
def get_model_columns(outdir: Optional[str] = None, base_url: Optional[str] = None):
    """
    Return the list of model column names expected by pooled-logit model.
    Loads artifacts lazily if needed.
    """
    if outdir is None:
        outdir = _DEFAULT_OUTDIR
    _ensure_artifacts_loaded(outdir, base_url)
    mc = _ARTS.get("model_columns", None)
    if isinstance(mc, pd.DataFrame):
        mc_clean = _clean_model_columns_from_df(mc)
        return mc_clean
    elif isinstance(mc, (list, tuple, np.ndarray, pd.Series)):
        return [str(x).strip() for x in list(mc)]
    else:
        # try to parse if joblib stored a dataframe inside
        return None

# -------------------------
# Core function: inference
# -------------------------
def infer_new_patient_fixed(
    patient_data,
    return_raw: bool = False,
    outdir: Optional[str] = None,
    base_url: Optional[str] = None,
    max_period_override: Optional[int] = None,
    interval_days: int = _DEFAULT_INTERVAL_DAYS,
    period_labels: Optional[list] = None,
    horizon_map: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """
    Robust inference for a single new patient.

    Returns dict with keys:
      - 'survival_curve': DataFrame with columns ['period','S_control','S_treat','days'] or None
      - 'CATEs': dict mapping horizon-months -> {'CATE': float or nan, 'error': str or None}
      - 'errors': dict of named errors (artifact load failures, predict failures)
      - 'debug': optional debug info if return_raw True

    patient_data: dict or 1-row DataFrame
    """
    outdir = outdir or _DEFAULT_OUTDIR
    period_labels = period_labels or _DEFAULT_PERIOD_LABELS
    errors = {}
    debug = {"artifact_sources": {}}

    # Accept dict or DataFrame
    if isinstance(patient_data, dict):
        df_patient = pd.DataFrame([patient_data.copy()])
    elif isinstance(patient_data, pd.DataFrame):
        df_patient = patient_data.copy().reset_index(drop=True)
    else:
        raise ValueError("patient_data must be a dict or single-row DataFrame")

    if 'patient_id' not in df_patient.columns:
        df_patient['patient_id'] = 'new'
    if 'treatment' not in df_patient.columns:
        df_patient['treatment'] = int(df_patient.get('treatment', 0))

    # -------------------------
    # Load artifacts 
    # -------------------------
    _ensure_artifacts_loaded(outdir, base_url)
    for k, v in _ART_SOURCES.items():
        debug['artifact_sources'][k] = v

    # model columns
    model_columns = get_model_columns(outdir=outdir, base_url=base_url)
    if model_columns is None:
        errors['model_columns'] = "pooled_logit_model_columns not found or unreadable in outputs/"
    else:
        debug['model_columns_count'] = len(model_columns)

    # pooled-logit model
    pooled_logit = _ARTS.get("pooled_logit", None)
    if pooled_logit is None:
        errors['pooled_logit'] = "pooled_logit model not found in outputs/"

    # pp scaler & medians
    pp_scaler = _ARTS.get("pp_scaler", None)
    pp_train_medians = _ARTS.get("pp_train_medians", None)

    # patient-level scaler and columns (for causal forests input)
    patient_scaler = _ARTS.get("patient_scaler", None)
    patient_columns = _ARTS.get("patient_columns", None)
    forests_bundle = _ARTS.get("forests_bundle", None)

    # -------------------------
    # determine max_period
    # -------------------------
    max_period = None
    if max_period_override:
        max_period = int(max_period_override)
    else:
        # try pp_test global if available (best-effort)
        if 'pp_test' in globals() and hasattr(globals()['pp_test'], 'period'):
            try:
                max_period = int(globals()['pp_test']['period'].max())
            except Exception:
                max_period = None
    if max_period is None:
        max_period = 12  # safe default

    # build person-period toy rows for the new patient
    rows = []
    for p in range(1, max_period + 1):
        row = df_patient.iloc[0].to_dict()
        row['period'] = p
        row['patient_id'] = df_patient.iloc[0].get('patient_id', 'new')
        row['treatment'] = int(row.get('treatment', 0))
        rows.append(row)
    df_pp_new = pd.DataFrame(rows)

    # -------------------------
    # pooled-logit survival
    # -------------------------
    survival_df = None
    if pooled_logit is None or model_columns is None:
        # cannot compute survival; make sure errors already populated
        pass
    else:
        try:
            # Build X_pp: try to use external build_X_for_pp if available in globals
            if 'build_X_for_pp' in globals() and callable(globals()['build_X_for_pp']):
                X_pp = globals()['build_X_for_pp'](df_pp_new.copy())
            else:
                # Minimal fallback: create DataFrame with required model_columns using medians/defaults
                X_pp = pd.DataFrame(index=df_pp_new.index)
                # attempt to fill numeric cols from pp_train_medians (if Series/dict)
                if pp_train_medians is not None:
                    try:
                        # accept Series or dict
                        if isinstance(pp_train_medians, pd.Series):
                            med = pp_train_medians.to_dict()
                        elif isinstance(pp_train_medians, dict):
                            med = pp_train_medians
                        else:
                            med = {}
                    except Exception:
                        med = {}
                else:
                    med = {}

                # For each expected column, try to populate:
                for c in model_columns:
                    # categorical dummies: if column name contains '_' and root exists in df_pp_new, try one-hot logic
                    if c in df_pp_new.columns:
                        X_pp[c] = df_pp_new[c]
                    else:
                        # try one-hot like 'sex_Male' pattern
                        if '_' in c:
                            root, tail = c.split('_', 1)
                            if root in df_pp_new.columns and any([True]):
                                # set 1 where equal, else 0
                                X_pp[c] = (df_pp_new[root].astype(str) == tail).astype(int)
                                continue
                        # numeric fallback from medians
                        if c in med:
                            X_pp[c] = float(med[c])
                        else:
                            X_pp[c] = 0.0

                # ensure ordering
                X_pp = X_pp.reindex(columns=model_columns, fill_value=0.0)

            # Make treated/control counterfactuals
            X_t = X_pp.copy()
            X_t['treatment'] = 1
            X_c = X_pp.copy()
            X_c['treatment'] = 0

            # recompute treatment x period interactions if period dummies exist
            for pcol in [col for col in X_t.columns if str(col).startswith('period_bin')]:
                # ensure interaction columns exist in model_columns
                inter = f"treat_x_{pcol}"
                if inter in X_t.columns:
                    X_t[inter] = X_t['treatment'] * X_t.get(pcol, 0)
                    X_c[inter] = X_c['treatment'] * X_c.get(pcol, 0)

            # predict per-interval event probability
            # scikit-learn logistic: predict_proba expects feature order matching training; ensure that
            # pooled_logit may be either an sklearn estimator or a wrapper; attempt predict_proba
            try:
                probs_t = np.asarray(pooled_logit.predict_proba(X_t)[:, 1], dtype=float)
                probs_c = np.asarray(pooled_logit.predict_proba(X_c)[:, 1], dtype=float)
            except Exception:
                # try fallback: if pooled_logit exposes coef_ and intercept_ (manual linear logits)
                try:
                    coefs = np.asarray(pooled_logit.coef_).flatten()
                    intercept = float(getattr(pooled_logit, "intercept_", 0.0).ravel()[0])
                    # ensure X_t columns align with coefficient length
                    Xt_mat = np.asarray(X_t)  # shape (n_periods, n_features)
                    logits_t = Xt_mat.dot(coefs) + intercept
                    probs_t = 1.0 / (1.0 + np.exp(-logits_t))
                    Xc_mat = np.asarray(X_c)
                    logits_c = Xc_mat.dot(coefs) + intercept
                    probs_c = 1.0 / (1.0 + np.exp(-logits_c))
                except Exception as e2:
                    raise RuntimeError(f"pooled_logit predict failed: {e2}")

            # cumulative survival (period-end)
            S_t = np.cumprod(1.0 - probs_t)
            S_c = np.cumprod(1.0 - probs_c)
            survival_df = pd.DataFrame({
                "period": np.arange(1, len(S_t) + 1),
                "S_control": S_c,
                "S_treat": S_t
            })
            survival_df['days'] = survival_df['period'] * interval_days

        except Exception as e:
            errors['pooled_logit_predict'] = str(e)
            survival_df = None

    # -------------------------
    # Build Xpatient for CATE prediction (canonical patient vector)
    # -------------------------
    Xpatient = None
    if patient_columns is None:
        errors['patient_columns'] = errors.get('patient_columns', "patient_columns artifact missing")
    else:
        try:
            # Normalize patient_columns into list of strings
            if isinstance(patient_columns, pd.DataFrame):
                pcols = patient_columns.iloc[:, 0].astype(str).tolist()
            elif isinstance(patient_columns, (list, tuple, np.ndarray, pd.Series)):
                pcols = [str(x) for x in list(patient_columns)]
            elif isinstance(patient_columns, dict):
                # mapping -> keys or 'columns'
                if 'columns' in patient_columns:
                    pcols = list(patient_columns['columns'])
                else:
                    pcols = list(patient_columns.keys())
            else:
                pcols = list(patient_columns)

            # create zero-vector DataFrame
            Xpatient = pd.DataFrame(np.zeros((1, len(pcols))), columns=pcols)

            # fill from df_patient where possible
            for c in pcols:
                if c in df_patient.columns:
                    Xpatient.at[0, c] = df_patient.at[0, c]
                else:
                    # try one-hot inference: root_tail pattern
                    if '_' in c:
                        root, tail = c.split('_', 1)
                        if root in df_patient.columns and str(df_patient.at[0, root]) == tail:
                            Xpatient.at[0, c] = 1.0
            # reindex to ensure column order
            Xpatient = Xpatient.reindex(columns=pcols, fill_value=0.0)
        except Exception as e:
            errors['Xpatient_build'] = str(e)
            Xpatient = None

    # apply patient_scaler if present
    if Xpatient is not None and patient_scaler is not None:
        try:
            # prefer feature_names_in_ if available
            if hasattr(patient_scaler, "feature_names_in_"):
                scaler_cols = list(patient_scaler.feature_names_in_)
            else:
                # safe default numeric candidates
                scaler_cols = [c for c in Xpatient.columns if c in ('age','ecog_ps','BED_eff','EQD2','smoking_py_clean','time_since_rt_days')]
            # ensure those cols exist
            for sc_col in scaler_cols:
                if sc_col not in Xpatient.columns:
                    # try to fill from medians or with zero
                    fill_val = 0.0
                    try:
                        if isinstance(pp_train_medians, dict) and sc_col in pp_train_medians:
                            fill_val = pp_train_medians[sc_col]
                        elif isinstance(pp_train_medians, pd.Series) and sc_col in pp_train_medians.index:
                            fill_val = float(pp_train_medians.loc[sc_col])
                    except Exception:
                        fill_val = 0.0
                    Xpatient[sc_col] = float(fill_val)
            Xnum = Xpatient[scaler_cols].astype(float).copy()
            Xnum_scaled = patient_scaler.transform(Xnum)
            Xnum_scaled = pd.DataFrame(Xnum_scaled, columns=scaler_cols, index=Xpatient.index)
            for c in scaler_cols:
                Xpatient[c] = Xnum_scaled[c].values
        except Exception as e:
            errors['patient_scaler'] = f"patient_scaler exists but failed to transform Xpatient; proceeding without scaling: {e}"

    debug['Xpatient'] = None
    try:
        debug['Xpatient'] = Xpatient.copy() if Xpatient is not None else None
    except Exception:
        try:
            debug['Xpatient'] = str(Xpatient)
        except Exception:
            debug['Xpatient'] = None

    # -------------------------
    # CATE: predict with forests bundle
    # -------------------------
    cate_results = {}
    if forests_bundle is None:
        # fill placeholder NaNs for expected period_labels
        for lab in period_labels:
            try:
                months = int(str(lab).replace('+', '').split('-')[-1])
            except Exception:
                months = lab
            cate_results[months] = {'CATE': np.nan, 'error': 'forests_bundle missing'}
        errors['forests_bundle'] = errors.get('forests_bundle', "forests_bundle artifact missing")
    else:
        # forests_bundle may be a dict mapping label -> estimator or a single estimator
        try:
            # Normalize bundle items to (label, estimator)
            if isinstance(forests_bundle, dict):
                items = list(forests_bundle.items())
            else:
                # single object -> try to map to period_labels
                items = [(lab, forests_bundle) for lab in period_labels]

            for lab, est in items:
                # map label -> months int if possible
                if horizon_map and lab in horizon_map:
                    months = horizon_map[lab]
                else:
                    try:
                        months = int(str(lab).replace('+', '').split('-')[-1])
                    except Exception:
                        months = lab

                if Xpatient is None:
                    cate_results[months] = {'CATE': np.nan, 'error': 'Xpatient not built'}
                    continue

                try:
                    candidate = est
                    # if nested dict of estimators, choose first with effect method
                    if isinstance(est, dict):
                        candidate = None
                        for v in est.values():
                            if hasattr(v, 'effect'):
                                candidate = v
                                break
                        if candidate is None:
                            # as fallback pick first value
                            candidate = list(est.values())[0]

                    # Align Xpatient to estimator's expected features
                    if hasattr(candidate, "feature_names_in_"):
                        req = list(candidate.feature_names_in_)
                        Xfor_in = Xpatient.reindex(columns=req, fill_value=0.0).values
                    else:
                        # fallback: try using patient_columns or model_columns as ordering
                        if patient_columns is not None:
                            if isinstance(patient_columns, pd.DataFrame):
                                req = patient_columns.iloc[:, 0].astype(str).tolist()
                            else:
                                req = list(patient_columns)
                            Xfor_in = Xpatient.reindex(columns=req, fill_value=0.0).values
                        else:
                            Xfor_in = Xpatient.values

                    # call effect (many causal forest libs use .effect(X) signature)
                    if hasattr(candidate, "effect"):
                        eff = np.asarray(candidate.effect(Xfor_in)).flatten()
                        val = float(eff[0]) if eff.size > 0 else np.nan
                        cate_results[months] = {'CATE': val, 'error': None}
                    else:
                        cate_results[months] = {'CATE': np.nan, 'error': 'estimator has no effect() method'}
                except Exception as e:
                    cate_results[months] = {'CATE': np.nan, 'error': str(e)}
        except Exception as e:
            errors['forests_bundle_eval'] = str(e)
            # best-effort: fill NaNs
            for lab in period_labels:
                try:
                    months = int(str(lab).replace('+', '').split('-')[-1])
                except Exception:
                    months = lab
                cate_results[months] = {'CATE': np.nan, 'error': 'bundle evaluation failed'}

    # sort cate_results by numeric month keys if possible
    try:
        cate_results = dict(sorted(
            cate_results.items(),
            key=lambda kv: (float(kv[0]) if (isinstance(kv[0], (int, float)) or str(kv[0]).replace('.', '', 1).isdigit()) else 1e9)
        ))
    except Exception:
        pass

    out = {
        "survival_curve": survival_df,
        "CATEs": cate_results,
        "errors": errors,
    }
    if return_raw:
        out["debug"] = debug

    return out

# convenience alias
infer = infer_new_patient_fixed

# run to see if okay
if __name__ == "__main__":
    print("Running utils/infer.py smoke test (no artifacts required).")
    demo_patient = {
        "age": 62, "sex": "Male", "ecog_ps": 1,
        "smoking_status_clean": "Ex-Smoker", "smoking_py_clean": 20,
        "primary_site_group": "Oropharynx", "subsite_clean": "Tonsil",
        "stage": "III", "hpv_clean": "HPV_Positive", "treatment": 0,
        "patient_id": "demo_local"
    }
    res = infer_new_patient_fixed(demo_patient, return_raw=True, max_period_override=6)
    print("Keys returned:", list(res.keys()))
    if res.get("errors"):
        print("Errors (smoke test):", res["errors"])
    if res.get("survival_curve") is not None:
        print("Survival shape:", res["survival_curve"].shape)
