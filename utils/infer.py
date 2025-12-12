# utils/infer.py
import os
import joblib
import pandas as pd
import numpy as np
import requests
import io
from tqdm import tqdm

# defaults used when not provided by caller
DEFAULT_OUTDIR = "outputs"
DEFAULT_INTERVAL_DAYS = 30
DEFAULT_PERIOD_LABELS = ['0-3','4-6','7-12','13-24','25-60','60+']

def _load_local_art(path):
    """Try local file with joblib or csv fallback."""
    if not os.path.exists(path):
        return None, None
    try:
        val = joblib.load(path)
        return val, f"local:{path}"
    except Exception:
        try:
            val = pd.read_csv(path)
            return val, f"local_csv:{path}"
        except Exception as e:
            return None, f"failed_local:{path}:{e}"

def _load_remote_joblib(url):
    """Load joblib (or pickled) artifact from a raw URL using requests."""
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return joblib.load(io.BytesIO(r.content)), f"remote:{url}"
    except Exception as e:
        return None, f"remote_failed:{url}:{e}"

def _try_load_artifact(name, candidates, outdir=DEFAULT_OUTDIR, base_url=None):
    """
    Try (in order):
      - global variable with `name` (if set in globals of importing module)
      - local files in outdir with candidate filenames
      - remote files via base_url + candidate (if base_url provided)
    Returns (value_or_None, source_str_or_None)
    """
    #  globals (importing code may have set)
    if name in globals() and globals()[name] is not None:
        return globals()[name], f"globals:{name}"

    # local files (outdir)
    for fn in candidates:
        p = os.path.join(outdir, fn)
        val, src = _load_local_art(p)
        if val is not None:
            return val, src

    #  remote via base_url (if available)
    if base_url:
        base = base_url.rstrip("/") + "/"
        for fn in candidates:
            url = base + fn
            # try joblib read first (binary)
            val, src = _load_remote_joblib(url)
            if val is not None:
                return val, src
            # if not joblib, try csv
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                # attempt read csv/text
                val = pd.read_csv(io.StringIO(r.text))
                return val, f"remote_csv:{url}"
            except Exception:
                continue

    return None, None


def infer_new_patient_fixed(patient_data, return_raw=False, outdir=DEFAULT_OUTDIR,
                            base_url=None, max_period_override=None,
                            interval_days=DEFAULT_INTERVAL_DAYS,
                            period_labels=DEFAULT_PERIOD_LABELS,
                            horizon_map=None):
    """
    Robust inference for a single new patient.
    - patient_data: dict or single-row DataFrame
    - base_url: optional raw github base url (ends with /outputs/) used to fetch artifacts remotely
    - max_period_override: int months to create person-period rows (if provided)
    Returns dict: {'survival_curve', 'CATEs', 'errors', 'debug'(opt)}
    """
    errors = {}
    debug = {}

    # input -> df
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy().reset_index(drop=True)
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'new'
    if 'treatment' not in df.columns:
        df['treatment'] = int(df.get('treatment', 0))

    # helpers: candidate filenames for artifacts
    ART = {
        'patient_columns': ['causal_patient_columns.joblib', 'causal_patient_columns.pkl', 'causal_patient_columns.npy', 'causal_patient_columns.csv'],
        'patient_scaler': ['causal_patient_scaler.joblib', 'causal_patient_scaler.pkl'],
        'pp_train_medians': ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv'],
        'pooled_logit': ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib','pooled_logit_logreg_saga.pkl'],
        'model_columns': ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib'],
        'forests_bundle': ['causal_forests_period_horizons_patient_level.joblib','causal_forests_period_horizons.joblib','forests_bundle.joblib'],
        'pp_scaler': ['pp_scaler.joblib','pp_scaler.pkl']
    }

    # Try load artifacts (local then remote if base_url provided)
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

    # determine max_period from pp_test if available, else override, else default 12
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

    # build person-period toy rows for the new patient
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
            # if build_X_for_pp exists in globals (from notebook), use it
            if 'build_X_for_pp' in globals():
                X_pp = build_X_for_pp(df_pp_new.copy())
            else:
                # Minimal fallback: use pp_train_medians and create numeric columns; keep categorical zeroed
                X_pp = pd.DataFrame(index=df_pp_new.index)
                if pp_train_medians is not None:
                    for c, v in (pp_train_medians.items() if isinstance(pp_train_medians, dict) else ([])):
                        X_pp[c] = pd.to_numeric(df_pp_new.get(c, pd.Series([np.nan]*len(df_pp_new))), errors='coerce').fillna(v)
                # ensure model_columns exist
                if isinstance(model_columns, (pd.Series, list, np.ndarray)):
                    cols_req = list(model_columns)
                elif isinstance(model_columns, pd.DataFrame):
                    cols_req = model_columns.iloc[:,0].astype(str).tolist()
                else:
                    cols_req = list(model_columns)
                X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)
            # ensure correct ordering
            if isinstance(model_columns, (pd.Series, list, np.ndarray)):
                cols_req = list(model_columns)
            elif isinstance(model_columns, pd.DataFrame):
                cols_req = model_columns.iloc[:,0].astype(str).tolist()
            else:
                cols_req = list(model_columns)
            X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)

            # create treated/control counterfactual rows
            X_t = X_pp.copy(); X_t['treatment'] = 1
            X_c = X_pp.copy(); X_c['treatment'] = 0
            # recompute interaction columns if present
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

    # ---------- build Xpatient (for CF) ----------
    Xpatient = None
    if patient_columns is None:
        errors['patient_columns'] = "patient_columns artifact missing. CF prediction will be impossible without canonical patient feature names."
    else:
        try:
            # normalize representation of patient_columns
            if isinstance(patient_columns, (pd.Series, list, np.ndarray)):
                pcols = list(patient_columns)
            elif isinstance(patient_columns, dict):
                # if stored as {'columns': [...]} or mapping
                if 'columns' in patient_columns:
                    pcols = list(patient_columns['columns'])
                else:
                    pcols = list(patient_columns.keys())
            else:
                pcols = list(patient_columns)

            Xpatient = pd.DataFrame(np.zeros((1, len(pcols))), columns=pcols)
            # Fill numeric or one-hot style columns
            for c in pcols:
                if c in df.columns:
                    Xpatient.at[0, c] = df.at[0, c]
                else:
                    # try to infer one-hot: e.g., sex_Male when df.sex == 'Male'
                    if '_' in c:
                        root, tail = c.split('_', 1)
                        if root in df.columns and str(df.at[0, root]) == tail:
                            Xpatient.at[0, c] = 1.0
            # apply scaler if present (safe)
            if patient_scaler is not None:
                try:
                    numeric_cols = Xpatient.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0:
                        Xpatient[numeric_cols] = patient_scaler.transform(Xpatient[numeric_cols])
                except Exception:
                    errors['scaler'] = "scaler exists but failed to transform Xpatient; proceeding without scaling"
            Xpatient = Xpatient.reindex(columns=pcols, fill_value=0.0)
        except Exception as e:
            errors['Xpatient'] = f"Failed to construct Xpatient: {e}"
            Xpatient = None

    # ---------- CF CATE predictions ----------
    cate_results = {}
    if forests_bundle is None:
        errors['forests_bundle'] = "forests bundle not found in outputs or via base_url."
        # fill placeholder NaNs
        for lab in period_labels:
            # try to infer month integer
            try:
                months = int(lab.replace('+','').split('-')[-1])
            except Exception:
                months = lab
            cate_results[months] = {'CATE': np.nan, 'error': errors['forests_bundle']}
    else:
        # forests_bundle can be a dict mapping label->estimator
        for lab, est in forests_bundle.items():
            # map label -> months
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
                    # try to find nested estimator
                    for v in est.values():
                        if hasattr(v, 'effect'):
                            candidate = v
                            break
                # if estimator exposes feature_names_in_, reindex accordingly
                if hasattr(candidate, "feature_names_in_"):
                    req = list(candidate.feature_names_in_)
                    Xfor = Xpatient.reindex(columns=req, fill_value=0.0)
                    Xfor_in = Xfor.values
                else:
                    Xfor_in = Xpatient.values
                eff = np.asarray(candidate.effect(Xfor_in)).flatten()
                val = float(eff[0]) if eff.size > 0 else np.nan
                cate_results[months] = {'CATE': val, 'error': None}
            except Exception as e:
                cate_results[months] = {'CATE': np.nan, 'error': str(e)}

    # sort results by numeric horizon when possible
    try:
        cate_results = dict(sorted(cate_results.items(), key=lambda kv: (float(kv[0]) if isinstance(kv[0], (int,float,str)) and str(kv[0]).replace('.','',1).isdigit() else 1e9)))
    except Exception:
        pass

    out = {'survival_curve': survival_df, 'CATEs': cate_results, 'errors': errors}
    if return_raw:
        out['debug'] = debug
    return out

# infer patient
infer = infer_new_patient_fixed
