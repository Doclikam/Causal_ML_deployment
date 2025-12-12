import os
import io
import joblib
import requests
import traceback
from typing import Optional, Iterable, Tuple, Dict, Any
import numpy as np
import pandas as pd

DEFAULT_OUTDIR = "outputs"
DEFAULT_INTERVAL_DAYS = 30
DEFAULT_PERIOD_LABELS = ['0-3','4-6','7-12','13-24','25-60','60+']

# ------------------ I/O helpers ------------------
def _load_local_art(path: str):
    """Try to load a joblib or csv from local path; return (value, source_str)."""
    if not os.path.exists(path):
        return None, None
    try:
        v = joblib.load(path)
        return v, f"local:{path}"
    except Exception:
        # try csv
        try:
            v = pd.read_csv(path)
            return v, f"local_csv:{path}"
        except Exception as e:
            return None, f"failed_local:{path}:{e}"

def _load_remote_joblib(url: str):
    """Load joblib from raw url using requests; return (value, source_str) or (None, err)."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return joblib.load(io.BytesIO(r.content)), f"remote:{url}"
    except Exception as e:
        return None, f"remote_failed:{url}:{e}"

def _try_load_artifact(name: str, candidates: Iterable[str], outdir: str=DEFAULT_OUTDIR, base_url: Optional[str]=None):
    """
    Try: globals[name] -> local files under outdir -> remote via base_url.
    Returns (value_or_None, source_str_or_None)
    """
    # 1) globals (importing code can attach objects)
    if name in globals() and globals()[name] is not None:
        return globals()[name], f"globals:{name}"

    # 2) local
    for fn in candidates:
        p = os.path.join(outdir, fn)
        val, src = _load_local_art(p)
        if val is not None:
            return val, src

    # 3) remote
    if base_url:
        base = base_url.rstrip("/") + "/"
        for fn in candidates:
            url = base + fn
            val, src = _load_remote_joblib(url)
            if val is not None:
                return val, src
            # try CSV fallback
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                df = pd.read_csv(io.StringIO(r.text))
                return df, f"remote_csv:{url}"
            except Exception:
                continue
    return None, None

# ------------------ sanitization helpers ------------------
def _sanitize_colname(c: Any) -> str:
    """Return canonical column name used across pipelines."""
    if not isinstance(c, str):
        c = str(c)
    s = c.strip()
    # common sanitizations to stable tokens
    s = s.replace(" ", "_").replace("-", "_").replace("+","plus").replace("/","_").replace(".","_")
    return s

def _align_and_numeric_df(df: pd.DataFrame, req_cols: Iterable[str]) -> pd.DataFrame:
    """
    Ensure df has req_cols in that order, creating missing ones as zeros.
    Coerce to numeric where possible and fillna with 0.
    req_cols should be already sanitized or will be sanitized here.
    """
    out = df.copy()
    out.columns = [_sanitize_colname(c) for c in out.columns]
    req_cols_s = [_sanitize_colname(c) for c in req_cols]
    for c in req_cols_s:
        if c not in out.columns:
            out[c] = 0.0
    out = out.reindex(columns=req_cols_s, fill_value=0.0)
    # numeric conversion
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0)
    return out

def _reshape_to_expected(Xarr: np.ndarray, expected_d_x):
    """
    econml compares candidate._d_x == X.shape[1:].
    Xarr normally has shape (n_samples, n_features). This function tries to
    return an array with shape (n_samples,)+expected_d_x by adding/removing
    trailing singleton dimensions where sensible.
    """
    if expected_d_x is None:
        return Xarr
    # canonicalize expected_d_x
    if isinstance(expected_d_x, (int, np.integer)):
        expected = (int(expected_d_x),)
    else:
        try:
            expected = tuple(int(x) for x in expected_d_x)
        except Exception:
            expected = None

    if expected is None:
        return Xarr

    cur = Xarr.shape[1:]
    # exact match
    if tuple(cur) == tuple(expected):
        return Xarr

    n = Xarr.shape[0]
    # handle expected like (k,1)
    if len(expected) == 2 and expected[1] == 1 and len(cur) == 1 and cur[0] == expected[0]:
        return Xarr.reshape((n, expected[0], 1))
    # if expected has extra singleton dims at end: (k,1,1) etc
    if len(expected) > 1 and len(cur) == 1 and cur[0] == expected[0]:
        shape = (n, expected[0]) + tuple(1 for _ in expected[1:])
        return Xarr.reshape(shape)
    # if expected shorter, attempt to squeeze trailing singleton axes
    if len(expected) < len(cur):
        arr = Xarr
        while arr.ndim > 1 + len(expected) and arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
        if arr.shape[1:] == expected:
            return arr
    # last resort: try to reshape
    try:
        return Xarr.reshape((n,) + expected)
    except Exception:
        return Xarr

# ------------------ estimator introspection helpers ------------------
def _estimator_get_candidate_feature_names(est) -> Optional[list]:
    """
    Try to discover feature names an estimator was trained on.
    Returns list[str] or None.
    """
    # sklearn-style
    if hasattr(est, "feature_names_in_"):
        try:
            return list(est.feature_names_in_)
        except Exception:
            pass
    # econml stores cate_feature_names or cate_feature_names_
    if hasattr(est, "cate_feature_names"):
        try:
            return list(est.cate_feature_names)
        except Exception:
            pass
    if hasattr(est, "cate_feature_names_"):
        try:
            return list(est.cate_feature_names_)
        except Exception:
            pass
    # Some wrappers expose feature names in attribute
    if hasattr(est, "feature_names"):
        try:
            return list(est.feature_names)
        except Exception:
            pass
    return None

def _estimator_get_feature_importances(est) -> Optional[np.ndarray]:
    """
    Return feature_importances_ or feature_importances or None
    """
    if hasattr(est, "feature_importances_"):
        try:
            arr = np.asarray(est.feature_importances_)
            return arr
        except Exception:
            pass
    if hasattr(est, "feature_importances"):
        try:
            arr = np.asarray(est.feature_importances)
            return arr
        except Exception:
            pass
    return None

def infer_top_k_from_importances(est, candidate_columns: Iterable[str], k: int) -> Optional[list]:
    """
    Aggressive heuristic: use estimator.feature_importances_ (if present)
    to pick top-k features from candidate_columns. Returns sanitized column list.
    If importances length matches candidate_columns length, map them directly.
    Otherwise return None.
    """
    fi = _estimator_get_feature_importances(est)
    if fi is None:
        return None
    # sanitize candidate_columns
    cand = [_sanitize_colname(c) for c in list(candidate_columns)]
    # if lengths match, pair them
    if fi.size == len(cand):
        idx = np.argsort(-fi)[:k]
        chosen = [cand[i] for i in idx]
        return chosen
    # if importances length smaller but we can still use top entries as a heuristic:
    if fi.size < len(cand) and fi.size >= k:
        idx = np.argsort(-fi)[:k]
        # we don't know mapping - assume candidate_columns were provided in the same order as fi
        chosen = [cand[i] for i in idx if i < len(cand)]
        return chosen[:k] if len(chosen) >= k else None
    return None

# ------------------ safe CATE prediction (with aggressive fallback) ------------------
def safe_predict_cates(
    forests_bundle,
    Xpatient_df: pd.DataFrame,
    model_columns=None,
    patient_columns=None,
    aggressive_infer: bool = True,
    debug_level: int = 1
) -> Tuple[Dict[Any, Dict[str, Any]], Dict[Any, Dict[str, Any]]]:
    """
    Predict per-horizon CATEs safely.

    Parameters:
      - forests_bundle: dict mapping horizon label -> estimator or nested dict
      - Xpatient_df: DataFrame with a single row (or N rows) representing canonical patient features
      - model_columns: fallback list of possible model columns (pooled_logit_model_columns)
      - patient_columns: fallback to build Xpatient if needed
      - aggressive_infer: if True, try to infer top-k features from feature_importances_
      - debug_level: verbosity of debug info

    Returns:
      - out: dict mapping horizon (e.g., 3,6,12...) -> {'CATE': float or np.nan, 'error': None or info}
      - debug_local: dict mapping horizon -> diagnostics (selected columns, shapes, attributes used)
    """
    out = {}
    debug_local = {}

    if forests_bundle is None:
        return out, debug_local

    items = forests_bundle.items() if isinstance(forests_bundle, dict) else [(0, forests_bundle)]
    Xpatient = Xpatient_df.copy() if Xpatient_df is not None else pd.DataFrame()
    Xpatient.columns = [_sanitize_colname(c) for c in Xpatient.columns]

    for lab, est in items:
        # normalize horizon label to int if possible
        try:
            months = int(str(lab).replace("+", "").split("-")[-1])
        except Exception:
            months = lab

        candidate = est
        # if nested mapping, find first estimator with .effect
        if isinstance(est, dict):
            for v in est.values():
                if hasattr(v, "effect"):
                    candidate = v
                    break

        # start debug record
        rec = {"requested_label": lab, "months": months, "ok": False, "why": None}

        # try to gather required names from estimator
        req_names = _estimator_get_candidate_feature_names(candidate)
        rec['est_feature_names_source'] = None
        if req_names:
            rec['est_feature_names_source'] = 'estimator_direct'
            req_s = [_sanitize_colname(c) for c in req_names]
        else:
            # try model_columns fallback
            if model_columns is not None:
                req_s = [_sanitize_colname(c) for c in list(model_columns)]
                rec['est_feature_names_source'] = 'model_columns_fallback'
            else:
                req_s = list(Xpatient.columns)
                rec['est_feature_names_source'] = 'xpatient_cols'

        # align Xpatient to req_s (create zeros for missing)
        Xaligned = _align_and_numeric_df(Xpatient.copy(), req_s)

        # check estimator expected dimensionality (econml _d_x)
        expected = getattr(candidate, "_d_x", None)
        rec['_d_x'] = expected

        # prepare X array
        Xarr = Xaligned.values  # (n, n_features)
        X_for_effect = _reshape_to_expected(Xarr, expected)

        rec['aligned_cols_before_aggressive'] = list(Xaligned.columns)
        rec['Xaligned_shape_before'] = Xaligned.shape
        rec['X_for_effect_shape_before'] = getattr(X_for_effect, "shape", None)

        # If a mismatch (assertion) occurs, try aggressive inference if allowed
        attempt_aggressive = False
        # Decide whether shapes likely mismatch: if expected is tuple and doesn't equal Xarr.shape[1:]
        if expected is not None:
            try:
                expected_tuple = expected if isinstance(expected, tuple) else (int(expected),)
                if tuple(Xarr.shape[1:]) != tuple(expected_tuple):
                    attempt_aggressive = True
            except Exception:
                attempt_aggressive = False

        if attempt_aggressive and aggressive_infer:
            # aggressive strategy: try to infer which k features the estimator used
            try:
                # infer desired k
                k = expected_tuple[0] if isinstance(expected_tuple, tuple) and len(expected_tuple) >= 1 else None
                chosen = None
                # First try: estimator feature names (if present) and intersect with model_columns/patient_columns
                cand_names = _estimator_get_candidate_feature_names(candidate)
                if cand_names:
                    cand_names_s = [_sanitize_colname(c) for c in cand_names]
                    # intersect with Xpatient columns
                    chosen = [c for c in cand_names_s if c in Xpatient.columns]
                    if k is not None and len(chosen) > k:
                        chosen = chosen[:k]
                # Second try: use feature_importances_ mapped to model_columns
                if chosen is None or (k is not None and len(chosen) < k):
                    # candidate_columns to map importances to: prefer model_columns, else patient_columns, else Xpatient.columns
                    candidate_columns = None
                    if model_columns is not None:
                        candidate_columns = [_sanitize_colname(c) for c in model_columns]
                    elif patient_columns is not None:
                        candidate_columns = [_sanitize_colname(c) for c in patient_columns]
                    else:
                        candidate_columns = list(Xpatient.columns)
                    if k is None:
                        # choose reasonable fallback k (min of 18 or len(candidate_columns))
                        k = min(18, max(1, len(candidate_columns)))
                    inferred = infer_top_k_from_importances(candidate, candidate_columns, k)
                    if inferred:
                        chosen = inferred
                # Final pruning: if chosen list present but longer than k, trim
                if chosen:
                    if k is not None and len(chosen) > k:
                        chosen = chosen[:k]
                    rec['aggressive_chosen_cols'] = chosen
                    # build Xaligned from chosen cols
                    Xaligned = _align_and_numeric_df(Xpatient.copy(), chosen)
                    Xarr = Xaligned.values
                    X_for_effect = _reshape_to_expected(Xarr, expected)
                    rec['aligned_cols_after_aggressive'] = list(Xaligned.columns)
                    rec['Xaligned_shape_after'] = Xaligned.shape
                    rec['X_for_effect_shape_after'] = getattr(X_for_effect, "shape", None)
                else:
                    rec['aggressive_failure'] = "could_not_infer_top_k"
            except Exception as e:
                rec['aggressive_failure'] = str(e)
                rec['aggressive_tb'] = traceback.format_exc().splitlines()[-6:]

        # Finally, attempt effect()
        try:
            if hasattr(candidate, "effect"):
                eff = np.asarray(candidate.effect(X_for_effect)).flatten()
                val = float(eff[0]) if eff.size > 0 else np.nan
                out[months] = {"CATE": val, "error": None}
                rec['ok'] = True
                rec['why'] = 'effect_ok'
            elif hasattr(candidate, "const_marginal_effect"):
                eff = np.asarray(candidate.const_marginal_effect(X_for_effect)).flatten()
                val = float(eff[0]) if eff.size > 0 else np.nan
                out[months] = {"CATE": val, "error": None}
                rec['ok'] = True
                rec['why'] = 'const_marginal_effect_ok'
            elif hasattr(candidate, "predict"):
                pred = np.asarray(candidate.predict(X_for_effect)).flatten()
                out[months] = {"CATE": float(pred[0]) if pred.size > 0 else np.nan, "error": None}
                rec['ok'] = True
                rec['why'] = 'predict_ok'
            else:
                out[months] = {"CATE": np.nan, "error": "estimator missing effect/const_marginal_effect/predict"}
                rec['ok'] = False
                rec['why'] = 'missing_api'
        except AssertionError as ae:
            # econml raises an AssertionError for dimension mismatch; capture detail
            tb = traceback.format_exc()
            err_info = {
                "msg": "AssertionError during effect() (likely dimension mismatch)",
                "exception": str(ae),
                "_d_x": getattr(candidate, "_d_x", None),
                "candidate_feature_names": _estimator_get_candidate_feature_names(candidate),
                "Xaligned_cols": list(Xaligned.columns),
                "Xaligned_shape": Xaligned.shape,
                "X_for_effect_shape": getattr(X_for_effect, "shape", None),
                "trace_tail": tb.splitlines()[-6:]
            }
            out[months] = {"CATE": np.nan, "error": err_info}
            rec['ok'] = False
            rec['why'] = 'assertion_error'
            rec['assertion_info'] = err_info
        except Exception as e:
            tb = traceback.format_exc()
            err_info = {
                "msg": str(e),
                "exception": repr(e),
                "_d_x": getattr(candidate, "_d_x", None),
                "candidate_feature_names": _estimator_get_candidate_feature_names(candidate),
                "Xaligned_cols": list(Xaligned.columns),
                "Xaligned_shape": Xaligned.shape,
                "X_for_effect_shape": getattr(X_for_effect, "shape", None),
                "trace_tail": tb.splitlines()[-6:]
            }
            out[months] = {"CATE": np.nan, "error": err_info}
            rec['ok'] = False
            rec['why'] = 'other_exception'
            rec['exception_info'] = err_info

        debug_local[months] = rec

    # try to sort numeric keys
    try:
        out = dict(sorted(out.items(), key=lambda kv: float(kv[0]) if isinstance(kv[0], (int,float,str)) and str(kv[0]).replace('.','',1).isdigit() else 1e9))
        debug_local = dict(sorted(debug_local.items(), key=lambda kv: float(kv[0]) if isinstance(kv[0], (int,float,str)) and str(kv[0]).replace('.','',1).isdigit() else 1e9))
    except Exception:
        pass

    return out, debug_local

# ------------------ main inference function ------------------
def infer_new_patient_fixed(
    patient_data,
    return_raw: bool=False,
    outdir: str=DEFAULT_OUTDIR,
    base_url: Optional[str]=None,
    max_period_override: Optional[int]=None,
    interval_days: int = DEFAULT_INTERVAL_DAYS,
    period_labels = DEFAULT_PERIOD_LABELS,
    horizon_map=None,
    aggressive_infer: bool = True
):
    """
    Robust inference for a single new patient.
    Returns dict: {'survival_curve','CATEs','errors','debug'}
    """
    errors = {}
    debug = {}

    # patient -> df
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy().reset_index(drop=True)
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'new'
    if 'treatment' not in df.columns:
        # ensure integer 0/1 present
        df['treatment'] = int(df.get('treatment', 0))

    # artifact candidate filenames
    ART = {
        'patient_columns': ['causal_patient_columns.joblib', 'causal_patient_columns.pkl', 'causal_patient_columns.npy', 'causal_patient_columns.csv'],
        'patient_scaler': ['causal_patient_scaler.joblib', 'causal_patient_scaler.pkl'],
        'pp_train_medians': ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv'],
        'pooled_logit': ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib','pooled_logit_logreg_saga.pkl'],
        'model_columns': ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib'],
        'forests_bundle': ['causal_forests_period_horizons_patient_level.joblib','causal_forests_period_horizons.joblib','forests_bundle.joblib'],
        'pp_scaler': ['pp_scaler.joblib','pp_scaler.pkl']
    }

    # load artifacts (local then base_url)
    patient_columns, pc_src = _try_load_artifact('patient_columns', ART['patient_columns'], outdir=outdir, base_url=base_url)
    patient_scaler, sc_src = _try_load_artifact('patient_scaler', ART['patient_scaler'], outdir=outdir, base_url=base_url)
    pp_train_medians, pm_src = _try_load_artifact('pp_train_medians', ART['pp_train_medians'], outdir=outdir, base_url=base_url)
    pooled_logit, lp_src = _try_load_artifact('pooled_logit', ART['pooled_logit'], outdir=outdir, base_url=base_url)
    model_columns, mc_src = _try_load_artifact('model_columns', ART['model_columns'], outdir=outdir, base_url=base_url)
    forests_bundle, fb_src = _try_load_artifact('forests_bundle', ART['forests_bundle'], outdir=outdir, base_url=base_url)
    pp_scaler, pps_src = _try_load_artifact('pp_scaler', ART['pp_scaler'], outdir=outdir, base_url=base_url)

    debug['artifact_sources'] = {
        'patient_columns': pc_src, 'patient_scaler': sc_src, 'pp_train_medians': pm_src,
        'pooled_logit': lp_src, 'model_columns': mc_src, 'forests_bundle': fb_src, 'pp_scaler': pps_src
    }

    # decide max_period
    max_period = None
    if max_period_override:
        max_period = int(max_period_override)
    else:
        if 'pp_test' in globals() and hasattr(globals()['pp_test'], 'period'):
            try:
                max_period = int(globals()['pp_test']['period'].max())
            except Exception:
                max_period = None
    if max_period is None:
        max_period = 12

    # create person-period toy rows for new patient
    rows = []
    for p in range(1, max_period+1):
        row = df.iloc[0].to_dict()
        row['period'] = p
        row['patient_id'] = df.iloc[0].get('patient_id', 'new')
        row['treatment'] = int(row.get('treatment', 0))
        rows.append(row)
    df_pp_new = pd.DataFrame(rows)

    # ---------- pooled-logit survival ----------
    survival_df = None
    if pooled_logit is None or model_columns is None:
        errors['pooled_logit'] = "pooled-logit or model_columns missing; cannot compute survival."
    else:
        try:
            # Try to use build_X_for_pp from the notebook (if present globally)
            if 'build_X_for_pp' in globals():
                X_pp = build_X_for_pp(df_pp_new.copy())
            else:
                # fallback: try to create minimal X_pp using pp_train_medians & model_columns
                X_pp = pd.DataFrame(index=df_pp_new.index)
                # model_columns may be a DataFrame or list or series; normalize to list
                if isinstance(model_columns, pd.DataFrame):
                    cols_req = model_columns.iloc[:,0].astype(str).tolist()
                else:
                    cols_req = list(model_columns)
                # fill numeric cols with patient values or medians
                for c in cols_req:
                    sc = _sanitize_colname(c)
                    if sc in df_pp_new.columns:
                        X_pp[c] = pd.to_numeric(df_pp_new[sc], errors='coerce').fillna(np.nan)
                    else:
                        # try medians dict/series
                        fill = 0.0
                        try:
                            if hasattr(pp_train_medians, "get"):
                                fill = pp_train_medians.get(c, 0.0)
                            elif isinstance(pp_train_medians, (pd.Series, dict)):
                                fill = pp_train_medians[c]
                        except Exception:
                            fill = 0.0
                        X_pp[c] = float(fill)
                X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)

            # ensure order and fill missing
            if isinstance(model_columns, pd.DataFrame):
                cols_req = model_columns.iloc[:,0].astype(str).tolist()
            else:
                cols_req = list(model_columns)
            X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)

            # create counterfactual treated vs control
            X_t = X_pp.copy(); X_t['treatment'] = 1
            X_c = X_pp.copy(); X_c['treatment'] = 0
            # recompute interaction columns if present (treat_x_...)
            for pcol in [c for c in X_t.columns if str(c).startswith('period_bin')]:
                X_t[f'treat_x_{pcol}'] = X_t['treatment'] * X_t.get(pcol, 0)
                X_c[f'treat_x_{pcol}'] = X_c['treatment'] * X_c.get(pcol, 0)

            probs_t = pooled_logit.predict_proba(X_t)[:,1]
            probs_c = pooled_logit.predict_proba(X_c)[:,1]
            S_t = np.cumprod(1 - probs_t)
            S_c = np.cumprod(1 - probs_c)
            survival_df = pd.DataFrame({'period': np.arange(1, len(S_t)+1), 'S_control': S_c, 'S_treat': S_t})
            survival_df['days'] = survival_df['period'] * interval_days
        except Exception as e:
            errors['pooled_logit'] = f"pipelined survival predict failed: {e}"
            debug['pooled_logit_tb'] = traceback.format_exc().splitlines()[-8:]

    # ---------- build canonical Xpatient (for CF) ----------
    Xpatient = None
    if patient_columns is None:
        # patient_columns not supplied as artifact - attempt to synthesize from model_columns
        if model_columns is not None:
            try:
                patient_columns = list(model_columns)
            except Exception:
                patient_columns = None

    if patient_columns is None:
        errors['patient_columns'] = "patient_columns artifact missing; cannot build canonical patient vector (fallback attempted)."
    else:
        try:
            # normalize patient_columns into a list
            if isinstance(patient_columns, (pd.Series, list, np.ndarray)):
                pcols = list(patient_columns)
            elif isinstance(patient_columns, dict):
                if 'columns' in patient_columns:
                    pcols = list(patient_columns['columns'])
                else:
                    pcols = list(patient_columns.keys())
            else:
                pcols = list(patient_columns)

            pcols_s = [_sanitize_colname(c) for c in pcols]
            Xpatient = pd.DataFrame(np.zeros((1, len(pcols_s))), columns=pcols_s)

            # fill numeric columns from df row where possible or 0 otherwise
            for c in pcols_s:
                if c in df.columns:
                    Xpatient.at[0, c] = df.at[0, c]
                else:
                    # Try to support one-hot variables: sex_Male -> check df.sex == 'Male'
                    if "_" in c:
                        root0, tail = c.split("_", 1)
                        if root0 in df.columns and str(df.at[0, root0]) == tail:
                            Xpatient.at[0, c] = 1.0
            Xpatient = Xpatient.reindex(columns=pcols_s, fill_value=0.0)
        except Exception as e:
            errors['Xpatient'] = f"Failed to construct Xpatient: {e}"
            debug['Xpatient_tb'] = traceback.format_exc().splitlines()[-8:]
            Xpatient = None

    # apply patient_scaler if available (safe)
    if Xpatient is not None and patient_scaler is not None:
        try:
            if hasattr(patient_scaler, "feature_names_in_"):
                scaler_cols = list(patient_scaler.feature_names_in_)
            else:
                scaler_cols = [c for c in Xpatient.columns if c in ['age','ecog_ps','BED_eff','EQD2','smoking_py_clean','time_since_rt_days']]
            for sc_col in scaler_cols:
                if sc_col not in Xpatient.columns:
                    fill_val = 0.0
                    try:
                        if isinstance(pp_train_medians, dict) and sc_col in pp_train_medians:
                            fill_val = pp_train_medians[sc_col]
                        elif hasattr(pp_train_medians, "get"):
                            fill_val = pp_train_medians.get(sc_col, 0.0)
                    except Exception:
                        fill_val = 0.0
                    Xpatient[sc_col] = float(fill_val)
            Xnum = Xpatient[scaler_cols].astype(float).copy()
            Xnum_scaled = patient_scaler.transform(Xnum)
            Xnum_scaled = pd.DataFrame(Xnum_scaled, columns=scaler_cols, index=Xpatient.index)
            for c in scaler_cols:
                Xpatient[c] = Xnum_scaled[c].values
        except Exception as e:
            errors['scaler'] = f"scaler exists but failed to transform Xpatient; proceeding without scaling: {e}"
            debug['scaler_tb'] = traceback.format_exc().splitlines()[-6:]

    # debug store Xpatient sample
    try:
        debug['Xpatient_sample_cols'] = list(Xpatient.columns) if Xpatient is not None else None
        debug['Xpatient_shape'] = (Xpatient.shape if Xpatient is not None else None)
    except Exception:
        pass

    # ---------- CF CATE predictions (safe + aggressive) ----------
    cate_results = {}
    cate_debug = {}
    if forests_bundle is None:
        errors['forests_bundle'] = "forests bundle not found in outputs or via base_url."
        for lab in period_labels:
            try:
                months = int(str(lab).replace("+","").split("-")[-1])
            except Exception:
                months = lab
            cate_results[months] = {'CATE': np.nan, 'error': errors.get('forests_bundle')}
    else:
        try:
            cates_out, debug_local = safe_predict_cates(
                forests_bundle,
                Xpatient if Xpatient is not None else pd.DataFrame(),
                model_columns=model_columns,
                patient_columns=patient_columns,
                aggressive_infer=aggressive_infer
            )
            for k, v in cates_out.items():
                if horizon_map and k in horizon_map:
                    kk = horizon_map[k]
                else:
                    kk = k
                if isinstance(v, dict):
                    cate_results[kk] = {'CATE': v.get('CATE', np.nan), 'error': v.get('error')}
                else:
                    cate_results[kk] = {'CATE': np.nan, 'error': 'unexpected_return_value'}
            cate_debug = debug_local
        except Exception as e:
            errors['forests_bundle'] = f"safe_predict_cates failed: {e}"
            debug['forests_bundle_tb'] = traceback.format_exc().splitlines()[-8:]

    # try to sort cate_results by numeric key
    try:
        cate_results = dict(sorted(cate_results.items(), key=lambda kv: float(kv[0]) if isinstance(kv[0], (int,float,str)) and str(kv[0]).replace('.','',1).isdigit() else 1e9))
    except Exception:
        pass

    out = {'survival_curve': survival_df, 'CATEs': cate_results, 'errors': errors}
    if return_raw:
        debug['cate_debug'] = cate_debug
        debug['Xpatient_preview'] = (Xpatient.head(1).to_dict() if Xpatient is not None and not Xpatient.empty else None)
        out['debug'] = debug
    return out

# shorthand alias
infer = infer_new_patient_fixed
