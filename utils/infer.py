
import os
import joblib
import pandas as pd
import numpy as np
import requests
import io
from typing import Optional, Tuple

# simple artifact helpers
DEFAULT_OUTDIR = "outputs"
DEFAULT_INTERVAL_DAYS = 30
DEFAULT_PERIOD_LABELS = ['0-3','4-6','7-12','13-24','25-60','60+']


def _load_local(path: str):
    if not os.path.exists(path):
        return None, None
    try:
        return joblib.load(path), f"local:{path}"
    except Exception:
        try:
            return pd.read_csv(path), f"local_csv:{path}"
        except Exception as e:
            return None, f"failed_local:{path}:{e}"


def _load_remote_joblib(url: str):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return joblib.load(io.BytesIO(r.content)), f"remote:{url}"
    except Exception as e:
        return None, f"remote_failed:{url}:{e}"


def _try_load_artifact(name: str, candidates: list, outdir: str = DEFAULT_OUTDIR, base_url: Optional[str] = None):
    # globals
    if name in globals() and globals()[name] is not None:
        return globals()[name], f"globals:{name}"
    # local
    for fn in candidates:
        p = os.path.join(outdir, fn)
        val, src = _load_local(p)
        if val is not None:
            return val, src
    # remote
    if base_url:
        base = base_url.rstrip('/') + '/'
        for fn in candidates:
            url = base + fn
            val, src = _load_remote_joblib(url)
            if val is not None:
                return val, src
            # try csv
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                return pd.read_csv(io.StringIO(r.text)), f"remote_csv:{url}"
            except Exception:
                pass
    return None, None


def build_canonical_Xpatient(patient: dict, outdir: str = DEFAULT_OUTDIR, base_url: Optional[str] = None):
    """
    Build a 1-row DataFrame aligned to pooled-logit `model_columns` if present.
    Returns (Xpatient_df, debug_dict)
    """
    debug = {}
    # load model_columns
    mc_candidates = ["pooled_logit_model_columns.csv","pooled_logit_model_columns.joblib","pooled_logit_model_columns.pkl"]
    model_columns, mc_src = _try_load_artifact('model_columns', mc_candidates, outdir=outdir, base_url=base_url)
    debug['model_columns_src'] = mc_src

    if isinstance(model_columns, (pd.Series, list, tuple, np.ndarray)):
        cols = list(model_columns)
    elif isinstance(model_columns, pd.DataFrame):
        cols = model_columns.iloc[:,0].astype(str).tolist()
    else:
        cols = None

    # fallback columns minimal
    if not cols:
        cols = ['age','ecog_ps','smoking_py_clean','time_since_rt_days','treatment']
        debug['model_columns'] = 'fallback_minimal'

    # load pp_train_medians for numeric fallback
    med_candidates = ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv']
    pp_train_medians, pm_src = _try_load_artifact('pp_train_medians', med_candidates, outdir=outdir, base_url=base_url)
    debug['pp_train_medians_src'] = pm_src

    # load patient scaler (optional)
    scaler_candidates = ['causal_patient_scaler.joblib','causal_patient_scaler.pkl']
    patient_scaler, sc_src = _try_load_artifact('patient_scaler', scaler_candidates, outdir=outdir, base_url=base_url)
    debug['patient_scaler_src'] = sc_src

    # create Xpatient
    Xpatient = pd.DataFrame(0.0, index=[0], columns=[str(c) for c in cols])

    # fill direct columns
    for k, v in patient.items():
        if k in Xpatient.columns:
            Xpatient.at[0, k] = v

    # heuristic: one-hot where column like 'sex_Male'
    for c in Xpatient.columns:
        if c in patient:
            continue
        if '_' in c:
            root, tail = c.split('_',1)
            if root in patient and str(patient[root]) == tail:
                Xpatient.at[0, c] = 1.0

    # fill numeric medians when column present but zero (or NaN)
    try:
        if isinstance(pp_train_medians, (dict, pd.Series)):
            med_map = dict(pp_train_medians)
            for c in Xpatient.columns:
                if Xpatient.at[0,c] == 0.0 and c in med_map:
                    Xpatient.at[0,c] = med_map[c]
    except Exception:
        pass

    # ensure treatment
    if 'treatment' in Xpatient.columns and 'treatment' not in patient:
        Xpatient.at[0,'treatment'] = int(patient.get('treatment', 0))

    # apply scaler if present and feasible
    if patient_scaler is not None:
        try:
            if hasattr(patient_scaler, 'feature_names_in_'):
                scaler_cols = list(patient_scaler.feature_names_in_)
            else:
                # use numeric intersection
                scaler_cols = [c for c in ['age','ecog_ps','smoking_py_clean','time_since_rt_days'] if c in Xpatient.columns]
            scaler_cols = [c for c in scaler_cols if c in Xpatient.columns]
            if scaler_cols:
                Xpatient[scaler_cols] = patient_scaler.transform(Xpatient[scaler_cols].astype(float))
                debug['scaler_applied_to'] = scaler_cols
        except Exception as e:
            debug['scaler_error'] = str(e)

    # final cleaning
    Xpatient = Xpatient.fillna(0.0)
    return Xpatient, debug


def infer_new_patient_fixed(patient_data, return_raw=False, outdir: str = DEFAULT_OUTDIR,
                            base_url: Optional[str] = None, max_period_override: Optional[int] = None,
                            interval_days: int = DEFAULT_INTERVAL_DAYS, period_labels: list = DEFAULT_PERIOD_LABELS,
                            horizon_map: Optional[dict] = None, Xpatient_override: Optional[pd.DataFrame] = None):
    """
    Inference entrypoint. If Xpatient_override is provided (1-row DataFrame) it will be used for CATE predictions
    and for diagnostics. The function still computes pooled-logit survival using artifacts.
    Returns dict with keys: survival_curve, CATEs, errors, debug
    """
    errors = {}
    debug = {}

    # normalize patient_data -> df
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy().reset_index(drop=True)
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'new'
    if 'treatment' not in df.columns:
        df['treatment'] = int(df.get('treatment', 0))

    # artifact candidates
    ART = {
        'patient_columns': ['causal_patient_columns.joblib','causal_patient_columns.pkl','causal_patient_columns.csv'],
        'patient_scaler': ['causal_patient_scaler.joblib','causal_patient_scaler.pkl'],
        'pp_train_medians': ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv'],
        'pooled_logit': ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib','pooled_logit_logreg_saga.pkl'],
        'model_columns': ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib','pooled_logit_model_columns.pkl'],
        'forests_bundle': ['causal_forests_period_horizons_patient_level.joblib','causal_forests_period_horizons.joblib','forests_bundle.joblib'],
        'pp_scaler': ['pp_scaler.joblib','pp_scaler.pkl']
    }

    # load artifacts with fallback
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

    # determine max_period
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

    # build pp rows
    rows = []
    for p in range(1, max_period+1):
        row = df.iloc[0].to_dict()
        row['period'] = p
        row['patient_id'] = df.iloc[0].get('patient_id','new')
        row['treatment'] = int(row.get('treatment',0))
        rows.append(row)
    df_pp_new = pd.DataFrame(rows)

    # pooled-logit survival
    survival_df = None
    if pooled_logit is None or model_columns is None:
        errors['pooled_logit'] = 'pooled-logit or model_columns missing; cannot compute survival.'
    else:
        try:
            # if build_X_for_pp in globals use it, otherwise minimal fallback mapping
            if 'build_X_for_pp' in globals():
                X_pp = build_X_for_pp(df_pp_new.copy())
            else:
                X_pp = pd.DataFrame(index=df_pp_new.index)
                if isinstance(pp_train_medians, (dict, pd.Series)):
                    for c, v in dict(pp_train_medians).items():
                        X_pp[c] = pd.to_numeric(df_pp_new.get(c, pd.Series([np.nan]*len(df_pp_new))), errors='coerce').fillna(v)
                # ensure model_columns list
                if isinstance(model_columns, (pd.Series, list, np.ndarray)):
                    cols_req = list(model_columns)
                elif isinstance(model_columns, pd.DataFrame):
                    cols_req = model_columns.iloc[:,0].astype(str).tolist()
                else:
                    cols_req = list(model_columns) if model_columns is not None else []
                X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)

            # align ordering
            if isinstance(model_columns, (pd.Series, list, np.ndarray)):
                cols_req = list(model_columns)
            elif isinstance(model_columns, pd.DataFrame):
                cols_req = model_columns.iloc[:,0].astype(str).tolist()
            else:
                cols_req = list(model_columns) if model_columns is not None else []
            X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)

            X_t = X_pp.copy(); X_t['treatment'] = 1
            X_c = X_pp.copy(); X_c['treatment'] = 0
            for pcol in [c for c in X_pp.columns if str(c).startswith('period_bin')]:
                X_t[f'treat_x_{pcol}'] = X_t['treatment'] * X_t.get(pcol, 0)
                X_c[f'treat_x_{pcol}'] = X_c['treatment'] * X_c.get(pcol, 0)

            probs_t = pooled_logit.predict_proba(X_t)[:,1]
            probs_c = pooled_logit.predict_proba(X_c)[:,1]
            S_t = np.cumprod(1 - probs_t)
            S_c = np.cumprod(1 - probs_c)
            survival_df = pd.DataFrame({'period': np.arange(1, len(S_t)+1), 'S_control': S_c, 'S_treat': S_t})
            survival_df['days'] = survival_df['period'] * interval_days
        except Exception as e:
            errors['pooled_logit'] = f'pipelined survival predict failed: {e}'

    # build Xpatient (either override or construct)
    Xpatient = None
    Xpatient_debug = {}
    if Xpatient_override is not None:
        # ensure DataFrame and 1-row
        if isinstance(Xpatient_override, pd.DataFrame):
            Xpatient = Xpatient_override.copy()
            if Xpatient.shape[0] != 1:
                Xpatient = Xpatient.head(1)
            Xpatient_debug['provided'] = True
        else:
            # try to coerce
            try:
                Xpatient = pd.DataFrame(Xpatient_override)
                if Xpatient.shape[0] != 1:
                    Xpatient = Xpatient.head(1)
                Xpatient_debug['provided'] = True
            except Exception:
                Xpatient = None
                Xpatient_debug['provided'] = False

    if Xpatient is None:
        try:
            Xpatient, xb = build_canonical_Xpatient(df.iloc[0].to_dict(), outdir=outdir, base_url=base_url)
            Xpatient_debug.update(xb)
        except Exception as e:
            errors['Xpatient'] = f'Failed to build Xpatient: {e}'
            Xpatient = None

    debug['Xpatient_debug'] = Xpatient_debug

    # CF CATE predictions
    cate_results = {}
    if forests_bundle is None:
        errors['forests_bundle'] = 'forests bundle not found in outputs or via base_url.'
        for lab in period_labels:
            try:
                months = int(lab.replace('+','').split('-')[-1])
            except Exception:
                months = lab
            cate_results[months] = {'CATE': np.nan, 'error': errors['forests_bundle']}
    else:
        for lab, est in forests_bundle.items():
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
                    for v in est.values():
                        if hasattr(v, 'effect'):
                            candidate = v
                            break
                if hasattr(candidate, 'feature_names_in_'):
                    req = list(candidate.feature_names_in_)
                    Xfor = Xpatient.reindex(columns=req, fill_value=0.0)
                    Xfor_in = Xfor.values
                else:
                    Xfor_in = Xpatient.values
                eff = np.asarray(candidate.effect(Xfor_in)).flatten()
                val = float(eff[0]) if eff.size>0 else np.nan
                cate_results[months] = {'CATE': val, 'error': None}
            except Exception as e:
                cate_results[months] = {'CATE': np.nan, 'error': str(e)}

    try:
        cate_results = dict(sorted(cate_results.items(), key=lambda kv: (float(kv[0]) if isinstance(kv[0], (int,float,str)) and str(kv[0]).replace('.','',1).isdigit() else 1e9)))
    except Exception:
        pass

    out = {'survival_curve': survival_df, 'CATEs': cate_results, 'errors': errors, 'debug': debug}
    if return_raw:
        return out
    # else minimally return same keys
    return out


# convenience alias
infer_new_patient = infer_new_patient_fixed
```

