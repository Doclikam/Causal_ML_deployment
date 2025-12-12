import io
import os
import re
import math
import json
import traceback
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd

# -------------------------
# Config / defaults
# -------------------------
DEFAULT_OUTDIR = "outputs"
MODEL_COLUMNS_PATH = os.path.join(DEFAULT_OUTDIR, "pooled_logit_model_columns.csv")
POOLED_LOGIT_PATH = os.path.join(DEFAULT_OUTDIR, "pooled_logit_logreg_saga.joblib")
FORESTS_BUNDLE_PATH = os.path.join(DEFAULT_OUTDIR, "causal_forests_period_horizons_patient_level.joblib")
SCALER_PATH = os.path.join(DEFAULT_OUTDIR, "pp_scaler.joblib")
PATIENT_SCALER_PATH = os.path.join(DEFAULT_OUTDIR, "causal_patient_scaler.joblib")
PP_MEDIANS_PATH = os.path.join(DEFAULT_OUTDIR, "pp_train_medians.joblib")

# -------------------------
# Utilities
# -------------------------
def norm_col(c: str) -> str:
    """Normalise column names from file and from estimator expectations.
    - convert to str
    - strip
    - replace spaces, '/', '-', '+', '.' with underscores
    - replace multiple underscores with single
    - keep case consistent (lower)
    """
    if c is None:
        return ""
    c = str(c).strip()
    # canonical replacements
    c = c.replace("+", "_plus_")
    c = re.sub(r"[ \-\/\.\:\,]+", "_", c)
    c = re.sub(r"_{2,}", "_", c)
    c = c.strip("_")
    return c

def read_model_columns(path: str) -> List[str]:
    """Load model columns CSV (one-per-line) and normalise names."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"model_columns file not found: {path}")
    raw = pd.read_csv(path, header=None).iloc[:, 0].astype(str).tolist()
    cols = [norm_col(c) for c in raw if isinstance(c, str) and c.strip() != ""]
    return cols

def safe_load_joblib(path_or_url: str):
    """Try to joblib.load a local path; if it fails attempt to read raw bytes (already handled by caller)."""
    if path_or_url is None:
        return None
    if os.path.exists(path_or_url):
        try:
            return joblib.load(path_or_url)
        except Exception:
            # try to open as bytes
            with open(path_or_url, "rb") as fh:
                try:
                    return joblib.load(fh)
                except Exception:
                    raise
    # if not local file, caller may pass bytes or file-like - not handled here
    return None

def ensure_df_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return df with exactly cols (in that order). Add missing columns filled with zeros."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = 0
    # drop extra columns not in cols
    out = out.reindex(columns=cols)
    return out

def guess_estimator_feature_names(estimator) -> Optional[List[str]]:
    """Try to deduce the feature names the estimator was trained with.
    Returns a list of normalized names or None.
    Order should be the order expected by estimator.effect(X).
    """
    # 1. feature_names_in_ (sklearn-style)
    try:
        if hasattr(estimator, "feature_names_in_"):
            fns = getattr(estimator, "feature_names_in_")
            return [norm_col(x) for x in list(fns)]
    except Exception:
        pass

    # 2. cate_feature_names (econml)
    try:
        if hasattr(estimator, "cate_feature_names") and estimator.cate_feature_names is not None:
            # can be array-like or list
            fns = estimator.cate_feature_names
            # sometimes it's a nested list — flatten shallow
            if isinstance(fns, (list, tuple)) and any(isinstance(i, (list, tuple)) for i in fns):
                # flatten one level
                flat = []
                for el in fns:
                    if isinstance(el, (list, tuple)):
                        flat.extend(el)
                    else:
                        flat.append(el)
                fns = flat
            return [norm_col(str(x)) for x in fns]
    except Exception:
        pass

    # 3. feature_importances_ -> we cannot get names from importances alone
    #    but if estimator has 'feature_names_in_' in an inner model, attempt
    try:
        # inspect wrapped learners (e.g., base_estimator or model_final)
        for attr in ("model_final_", "models", "estimator_", "base_estimator_"):
            if hasattr(estimator, attr):
                try:
                    sub = getattr(estimator, attr)
                    if hasattr(sub, "feature_names_in_"):
                        return [norm_col(x) for x in list(sub.feature_names_in_)]
                except Exception:
                    continue
    except Exception:
        pass

    # 4. _d_x: econml stores shape of X used at fit as _d_x (tuple)
    #    we can't recover names but can learn expected dimension.
    return None

def estimator_expected_dim(estimator) -> Optional[int]:
    """Return expected X dimension (int) if available from estimator internals, else None."""
    try:
        if hasattr(estimator, "_d_x"):
            dx = estimator._d_x
            # _d_x may be tuple like (18,) or (18,) etc.
            if isinstance(dx, (tuple, list)) and len(dx) >= 1:
                try:
                    return int(dx[0])
                except Exception:
                    return None
            elif isinstance(dx, int):
                return dx
    except Exception:
        pass
    # fallback: if feature_names_in_ exists
    try:
        if hasattr(estimator, "feature_names_in_"):
            return len(getattr(estimator, "feature_names_in_"))
    except Exception:
        pass
    return None

def select_topk_by_importance(estimator, available_cols: List[str], k: int) -> List[str]:
    """If estimator has feature_importances_, pick top-k corresponding names from available_cols.
    If unavailable, pick top-k by variance (across available data if provided) or simply first k.
    """
    # try cate_feature_names first
    try:
        if hasattr(estimator, "cate_feature_names") and estimator.cate_feature_names:
            names = estimator.cate_feature_names
            if isinstance(names, (list, tuple)):
                flat = []
                for el in names:
                    if isinstance(el, (list, tuple)):
                        flat.extend(el)
                    else:
                        flat.append(el)
                cand = [norm_col(str(x)) for x in flat]
                # keep only those present in available_cols
                cand = [c for c in cand if c in available_cols]
                if len(cand) >= k:
                    return cand[:k]
                return cand + [c for c in available_cols if c not in cand][: (k - len(cand))]
    except Exception:
        pass

    # try feature_importances_ attribute (length must equal len(available_cols))
    try:
        if hasattr(estimator, "feature_importances_"):
            imps = np.asarray(getattr(estimator, "feature_importances_"))
            # if length matches available_cols, map directly
            if imps.shape[0] == len(available_cols):
                idx = np.argsort(imps)[::-1]
                chosen = [available_cols[i] for i in idx[:k]]
                return chosen
    except Exception:
        pass

    # fallback: take first k available_cols
    return available_cols[:k]

def align_cols_for_estimator(X_patient: pd.DataFrame, estimator, model_columns_norm: List[str], debug: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Align X_patient to the exact columns the estimator expects.
    Returns X_for_effect (np.ndarray) and the list of columns used (in order).
    """
    # standardize X_patient columns to normalized names
    Xp = X_patient.copy()
    Xp.columns = [norm_col(c) for c in Xp.columns]

    # try to retrieve estimator feature names
    feat_names = None
    try:
        feat_names = guess_estimator_feature_names(estimator)
    except Exception:
        feat_names = None

    if feat_names:
        # Keep only known columns that are present in Xp or model_columns_norm
        chosen = []
        for fn in feat_names:
            if fn in Xp.columns:
                chosen.append(fn)
            elif fn in model_columns_norm and fn not in chosen:
                # create column if present in model_columns (but missing in X_patient)
                chosen.append(fn)
        # ensure uniqueness and fill missing columns with zeros
        chosen = [c for c in chosen]
        if len(chosen) == 0:
            # fallback below
            feat_names = None
        else:
            # build Xfor
            for c in chosen:
                if c not in Xp.columns:
                    Xp[c] = 0
            Xfor = Xp[chosen].to_numpy(dtype=float)
            debug["aligned_cols"] = chosen
            debug["aligned_shape"] = list(Xfor.shape)
            return Xfor, chosen

    # No explicit names -> try expected dim
    expected_dim = estimator_expected_dim(estimator)
    if expected_dim is not None:
        # try to select expected_dim columns from model_columns_norm (with presence in Xp if possible)
        avail = [c for c in model_columns_norm if c in Xp.columns] + [c for c in model_columns_norm if c not in Xp.columns]
        chosen = select_topk_by_importance(estimator, avail, expected_dim)
        # ensure chosen present in Xp; if missing create
        for c in chosen:
            if c not in Xp.columns:
                Xp[c] = 0
        Xfor = Xp[chosen].to_numpy(dtype=float)
        debug["aligned_cols"] = chosen
        debug["aligned_shape"] = list(Xfor.shape)
        debug["expected_dim"] = expected_dim
        return Xfor, chosen

    # Last resort: use all model_columns_norm in order
    chosen = [c for c in model_columns_norm]
    for c in chosen:
        if c not in Xp.columns:
            Xp[c] = 0
    Xfor = Xp[chosen].to_numpy(dtype=float)
    debug["aligned_cols"] = chosen
    debug["aligned_shape"] = list(Xfor.shape)
    return Xfor, chosen

# -------------------------
# Main inference function
# -------------------------
def infer_new_patient_fixed(
    patient_data: Dict[str, Any],
    outdir: str = DEFAULT_OUTDIR,
    base_url: Optional[str] = None,
    max_period_override: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run the inference pipeline for a single patient input (patient_data).
    Returns dict containing survival_curve (DataFrame), CATEs dict, errors, debug.
    """
    errors: Dict[str, str] = {}
    debug: Dict[str, Any] = {}
    result = {"survival_curve": None, "CATEs": {}, "errors": errors, "debug": debug}

    # ---------- Load model columns ----------
    try:
        model_columns_norm = read_model_columns(os.path.join(outdir, "pooled_logit_model_columns.csv"))
        debug["model_columns_norm"] = model_columns_norm
    except Exception as e:
        tb = traceback.format_exc()
        errors["model_columns"] = f"Failed to load model columns: {e}\n{tb}"
        return result

    # ---------- Load artifacts ----------
    try:
        pooled_logit = safe_load_joblib(os.path.join(outdir, "pooled_logit_logreg_saga.joblib"))
        debug["pooled_logit_loaded"] = pooled_logit is not None
    except Exception as e:
        errors["pooled_logit"] = f"Failed to load pooled_logit joblib: {e}\n{traceback.format_exc()}"

    try:
        forests_bundle = safe_load_joblib(os.path.join(outdir, "causal_forests_period_horizons_patient_level.joblib"))
        debug["forests_bundle_loaded"] = isinstance(forests_bundle, dict)
    except Exception as e:
        errors["forests_bundle"] = f"Failed to load forests bundle: {e}\n{traceback.format_exc()}"
        forests_bundle = None

    try:
        pp_scaler = safe_load_joblib(os.path.join(outdir, "pp_scaler.joblib"))
        debug["pp_scaler_loaded"] = pp_scaler is not None
    except Exception:
        pp_scaler = None

    try:
        patient_scaler = safe_load_joblib(os.path.join(outdir, "causal_patient_scaler.joblib"))
        debug["patient_scaler_loaded"] = patient_scaler is not None
    except Exception:
        patient_scaler = None

    try:
        pp_medians = safe_load_joblib(os.path.join(outdir, "pp_train_medians.joblib"))
        debug["pp_medians_loaded"] = pp_medians is not None
    except Exception:
        pp_medians = None

    # ---------- Build X_patient skeleton ----------
    # Start with zeroed DataFrame for model_columns_norm
    try:
        Xp = pd.DataFrame([0], columns=model_columns_norm)
    except Exception as e:
        errors["build_Xp"] = f"Failed building base Xpatient: {e}\n{traceback.format_exc()}"
        return result

    debug["initial_Xp_cols"] = list(Xp.columns)[:20]

    # Map patient_data keys to normalized columns heuristically
    # Common mappings: sex -> sex_Male (or sex_Female), smoking_status -> smoking_status_clean_...
    try:
        def set_flag(colname):
            nc = norm_col(colname)
            if nc in Xp.columns:
                Xp[nc] = 1
                return True
            return False

        # sex
        sex_val = str(patient_data.get("sex", "")).strip().lower()
        if sex_val:
            if "male" in sex_val:
                set_flag("sex_Male")
            elif "female" in sex_val:
                # some apps use sex_Female; we'll set if present
                set_flag("sex_Female")
        # treatment
        tr = patient_data.get("treatment", patient_data.get("t", patient_data.get("treatment_planned", None)))
        try:
            # treatment encoded as 0/1 in your app
            Xp[norm_col("treatment")] = float(tr) if tr is not None else 0.0
        except Exception:
            # attempt mapping by name
            if isinstance(tr, str) and "chemo" in tr.lower():
                Xp[norm_col("treatment")] = 1.0
            else:
                Xp[norm_col("treatment")] = 0.0

        # age, ecog_ps, smoking_py_clean, time_since_rt_days
        for num_field in ("age", "ecog_ps", "smoking_py_clean", "time_since_rt_days"):
            val = patient_data.get(num_field, None)
            nc = norm_col(num_field)
            if nc in Xp.columns:
                try:
                    Xp[nc] = float(val) if val is not None else 0.0
                except Exception:
                    Xp[nc] = 0.0

        # smoking status - map to available one-hot columns
        smoking_val = str(patient_data.get("smoking_status_clean", patient_data.get("smoking_status", ""))).strip()
        if smoking_val:
            # candidate column patterns in model: smoking_status_clean_Ex-Smoker etc -> normalise
            candidates = [c for c in Xp.columns if c.startswith("smoking_status_clean")]
            smoking_norm = norm_col(smoking_val)
            # try match exact or partial
            matched = False
            for cand in candidates:
                if smoking_norm in cand or smoking_val.lower() in cand.lower():
                    Xp[cand] = 1
                    matched = True
                    break
            if not matched and candidates:
                # fallback: mark first candidate
                Xp[candidates[0]] = 1

        # hpv
        hpv_val = str(patient_data.get("hpv_clean", patient_data.get("hpv", ""))).strip()
        if hpv_val:
            candidates = [c for c in Xp.columns if c.startswith("hpv_clean")]
            for cand in candidates:
                if norm_col(hpv_val) in cand or hpv_val.lower() in cand.lower():
                    Xp[cand] = 1
                    break

        # primary_site_group
        prim = patient_data.get("primary_site_group", "")
        if prim:
            candidates = [c for c in Xp.columns if c.startswith("primary_site_group")]
            for cand in candidates:
                if norm_col(str(prim)) in cand or str(prim).lower() in cand.lower():
                    Xp[cand] = 1
                    break

        # stage mapping (I, II, III, IVA etc.)
        stg = str(patient_data.get("stage", patient_data.get("stage_overall", ""))).strip()
        if stg:
            # possible model columns: stage_I, stage_II, stage_IVA etc.
            candidates = [c for c in Xp.columns if c.startswith("stage_")]
            for cand in candidates:
                # try to match roman numeral or substage tokens
                if re.search(re.escape(stg), cand, flags=re.IGNORECASE):
                    Xp[cand] = 1
                    break

        # TNM — if user provided, keep them for debug but not required for model_columns
        # t, n, m
        for key in ("t", "n", "m"):
            val = patient_data.get(key, None)
            if val is not None:
                Xp[norm_col(key)] = str(val)

    except Exception as e:
        errors["map_patient_to_Xp"] = f"Failed mapping patient fields: {e}\n{traceback.format_exc()}"

    debug["Xp_sample_head"] = Xp.iloc[:1, :50].to_dict(orient="records")

    # ---------- Scale numeric features if scaler present ----------
    numeric_cols = [c for c in Xp.columns if c in ("age", "ecog_ps", "smoking_py_clean", "time_since_rt_days")]
    if pp_scaler is not None:
        try:
            # scaler expects columns in a particular order; try to align by names in scaler if available
            if hasattr(pp_scaler, "feature_names_in_"):
                scols = [norm_col(x) for x in list(pp_scaler.feature_names_in_)]
            else:
                scols = numeric_cols
            # create array and transform where possible
            arr = []
            for c in scols:
                if c in Xp.columns:
                    arr.append(float(Xp[c].iloc[0]))
                else:
                    arr.append(0.0)
            arr = np.asarray(arr, dtype=float).reshape(1, -1)
            try:
                arr_t = pp_scaler.transform(arr)
                # put transformed back into Xp for the corresponding columns
                for i, c in enumerate(scols):
                    if c in Xp.columns:
                        Xp.loc[0, c] = float(arr_t[0, i])
            except Exception:
                # leave unscaled
                debug["scaler_transform_error"] = "pp_scaler.transform failed; left raw values"
        except Exception:
            debug["scaler_error"] = "pp_scaler introspection failed"

    # ---------- Pooled logistic survival predict ----------
    survival_df = None
    if pooled_logit is not None:
        try:
            # pooled_logit may be a pipeline; try to detect expected columns
            expected_cols = None
            try:
                if hasattr(pooled_logit, "feature_names_in_"):
                    expected_cols = [norm_col(x) for x in pooled_logit.feature_names_in_]
                else:
                    # try pipeline step names: many sklearn pipelines have named_steps
                    if hasattr(pooled_logit, "named_steps"):
                        # find a final estimator with feature_names_in_
                        for name, step in pooled_logit.named_steps.items():
                            if hasattr(step, "feature_names_in_"):
                                expected_cols = [norm_col(x) for x in step.feature_names_in_]
                                break
            except Exception:
                expected_cols = None

            if expected_cols:
                # align Xp to expected_cols
                for c in expected_cols:
                    if c not in Xp.columns:
                        Xp[c] = 0
                Xp_for = Xp[expected_cols]
            else:
                # fallback: use model_columns_norm
                Xp_for = Xp.reindex(columns=model_columns_norm, fill_value=0)

            # convert to numeric matrix for pipeline
            # Some pipelines expect 2D arrays and may have custom transformers.
            # We'll call predict_proba OR predict depending on availability
            try:
                # if pipeline has a predict_proba -> maybe gives event probability per interval
                if hasattr(pooled_logit, "predict_proba"):
                    # many pooled logistic pipelines were trained to predict hazard per interval;
                    # we still attempt a predict_proba on Xp_for
                    proba = pooled_logit.predict_proba(Xp_for)
                    # shape may be (n_samples, 2) -> probability class 1
                    if proba is not None and proba.shape[1] >= 2:
                        p_event = float(proba[0, 1])
                    else:
                        p_event = float(proba.ravel()[0])
                    # construct minimal survival curve: user expects a DataFrame of rows by period
                    # If the pipeline is actually a time-varying model, a better approach is to have
                    # a precomputed 'timevarying_summary_by_period.csv' — we fallback to simple output
                    survival_df = pd.DataFrame({
                        "period": [0],
                        "days": [0.0],
                        "S_control": [1.0 - p_event],
                        "S_treat": [1.0 - p_event],
                    })
                else:
                    # try simple predict -> assume it's risk (0/1), set survival accordingly
                    pred = pooled_logit.predict(Xp_for)
                    p_event = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
                    survival_df = pd.DataFrame({
                        "period": [0],
                        "days": [0.0],
                        "S_control": [1.0 - p_event],
                        "S_treat": [1.0 - p_event],
                    })
            except Exception as e:
                errors["pooled_logit_predict"] = f"pipelined survival predict failed: {e}\n{traceback.format_exc()}"
                # Don't abort; leave survival_df as None
        except Exception as e:
            errors["pooled_logit_wrap"] = f"Error preparing input for pooled_logit: {e}\n{traceback.format_exc()}"

    # ---------- CATEs via causal forests ----------
    cate_results: Dict[str, Dict[str, Any]] = {}
    if forests_bundle is not None and isinstance(forests_bundle, dict):
        # forests_bundle expected structure: horizons keys -> estimator object
        # We'll iterate keys and try to call estimator.effect with properly-shaped X.
        for horizon_label, estimator in forests_bundle.items():
            info = {"estimator_type": type(estimator).__name__ if estimator is not None else None}
            try:
                if estimator is None:
                    raise ValueError("estimator is None")

                # Align Xp to estimator expectations
                debug_sub = {}
                Xfor, aligned_cols = align_cols_for_estimator(Xp, estimator, model_columns_norm, debug_sub)
                info.update(debug_sub)

                # effect() expects shape (n_samples, d_features) OR (n_samples, ) depending.
                # Call effect safely, catch assertion errors.
                try:
                    eff = None
                    # econml estimators sometimes expect 2D X. Ensure shape is correct.
                    # Xfor is already np.ndarray 2D.
                    eff_arr = None
                    try:
                        eff_arr = np.asarray(estimator.effect(Xfor)).flatten()
                    except AssertionError as ae:
                        # dimension mis-match: record and try to reshape or select subset columns
                        raise
                    except Exception as e:
                        # other errors calling effect
                        raise

                    # take the first element (single patient)
                    if eff_arr is not None and len(eff_arr) >= 1:
                        eff_val = float(eff_arr[0])
                    else:
                        eff_val = None

                    cate_results[str(horizon_label)] = {"CATE": eff_val, "error": None, "debug": info}
                except AssertionError as ae:
                    # dimension mismatch
                    msg = f"AssertionError during effect() (likely dimension mismatch): {ae}"
                    tb = traceback.format_exc().splitlines()[-8:]
                    info["trace_tail"] = tb
                    info["_d_x"] = getattr(estimator, "_d_x", None)
                    info["X_for_effect_shape"] = list(np.asarray(Xfor).shape)
                    info["Xaligned_cols"] = aligned_cols
                    cate_results[str(horizon_label)] = {"CATE": None, "error": msg, "debug": info}
                    errors.setdefault("cate_dimension_mismatch", []).append({horizon_label: msg})
                except Exception as e:
                    tb = traceback.format_exc()
                    info["exception"] = str(e)
                    info["trace"] = tb
                    cate_results[str(horizon_label)] = {"CATE": None, "error": str(e), "debug": info}
                    errors.setdefault("cate_exception", []).append({horizon_label: str(e)})
            except Exception as e:
                tb = traceback.format_exc()
                info["exception_load"] = str(e)
                info["trace"] = tb
                cate_results[str(horizon_label)] = {"CATE": None, "error": str(e), "debug": info}
                errors.setdefault("cate_outer_exception", []).append({horizon_label: str(e)})

    else:
        debug["forests_bundle_info"] = "no forests bundle loaded or not a dict"

    # ---------- Prepare survival output (if more advanced time-varying predictions exist, prefer them) ----------
    # If there exists a precomputed timevarying_summary_by_period.csv in outdir, load and use it as baseline
    try:
        tv_path = os.path.join(outdir, "timevarying_summary_by_period.csv")
        if os.path.exists(tv_path):
            try:
                tv = pd.read_csv(tv_path)
                # ensure columns exist
                if {"period", "days", "S_control", "S_treat"}.issubset(tv.columns):
                    survival_df = tv[["period", "days", "S_control", "S_treat"]].copy()
                else:
                    # try to compute S_control/S_treat from hazards if present: S = cumprod(1 - hazard)
                    if {"haz_control", "haz_treated"}.issubset(tv.columns):
                        tvc = tv.copy()
                        tvc["S_control"] = (1 - tvc["haz_control"]).cumprod()
                        tvc["S_treat"] = (1 - tvc["haz_treated"]).cumprod()
                        if "days" not in tvc.columns:
                            tvc["days"] = tvc["period"] * 30.0
                        survival_df = tvc[["period", "days", "S_control", "S_treat"]].copy()
            except Exception:
                debug["timevarying_load_error"] = traceback.format_exc()
        # if we still don't have survival_df, use whatever we created earlier (possibly point estimate)
    except Exception:
        debug["timevarying_error2"] = traceback.format_exc()

    # fill result
    result["survival_curve"] = survival_df
    result["CATEs"] = cate_results
    result["errors"] = errors
    result["debug"] = debug
    return result

# Allow direct testing when run as script
if __name__ == "__main__":
    # quick smoke test if run from repo root
    import json
    sample_patient = {
        "age": 62,
        "sex": "Male",
        "ecog_ps": 1,
        "smoking_status_clean": "Ex-Smoker",
        "smoking_py_clean": 20,
        "hpv_clean": "HPV_Positive",
        "primary_site_group": "Oropharynx",
        "stage": "III",
        "t": "T2",
        "n": "N0",
        "m": "M0",
        "treatment": 0
    }
    print("Running smoke inference...")
    out = infer_new_patient_fixed(sample_patient, outdir=DEFAULT_OUTDIR)
    print("Errors:", json.dumps(out.get("errors", {}), indent=2))
    print("Debug keys:", list(out.get("debug", {}).keys()))
    if out.get("survival_curve") is not None:
        print("Survival rows:", len(out["survival_curve"]))
    print("CATE keys:", list(out.get("CATEs", {}).keys()))
