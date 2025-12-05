# inference.py
import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

# Defaults (overridable by setting globals before import)
OUTDIR = globals().get("OUTDIR", "outputs")
interval_days = globals().get("interval_days", 30)
period_labels = globals().get("period_labels", ['0-3','4-6','7-12','13-24','25-60','60+'])
horizon_map = globals().get("horizon_map", None)

def _load_art(name, filenames, outdir=OUTDIR):
    """helper to prefer globals() then files in OUTDIR"""
    if name in globals() and globals()[name] is not None:
        return globals()[name], f"from globals('{name}')"
    for fn in filenames:
        p = os.path.join(outdir, fn)
        if os.path.exists(p):
            # try joblib then csv
            try:
                val = joblib.load(p)
                return val, f"from {p}"
            except Exception:
                try:
                    val = pd.read_csv(p)
                    return val, f"from {p}"
                except Exception:
                    pass
    return None, None

def infer_new_patient_fixed(patient_data, return_raw=False, outdir=OUTDIR):
    """
    Robust inference for a single new patient.
    Returns: dict with keys:
      - survival_curve: pd.DataFrame or None
      - CATEs: dict{horizon_months: {'CATE': float or np.nan, 'error': None or str}}
      - errors: dict of error messages
      - artifacts_sources (if return_raw=True)
    """
    # ---- prepare input df ----
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy().reset_index(drop=True)
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'new'
    if 'treatment' not in df.columns:
        df['treatment'] = int(df.get('treatment', 0))

    # ---- load artifacts ----
    patient_columns, pc_src = _load_art('patient_columns', ['causal_patient_columns.joblib', 'causal_patient_columns.pkl', 'causal_patient_columns.npy', 'causal_patient_columns.csv'], outdir)
    patient_scaler, sc_src = _load_art('patient_scaler', ['causal_patient_scaler.joblib', 'causal_patient_scaler.pkl', 'causal_patient_scaler.npy'], outdir)
    pp_train_medians, pm_src = _load_art('pp_train_medians', ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv'], outdir)
    pooled_logit, lp_src = _load_art('logit', ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib','pooled_logit_logreg_saga.pkl'], outdir)
    model_columns, mc_src = _load_art('model_columns', ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib','pooled_logit_model_columns.pkl'], outdir)
    forests_bundle, fb_src = _load_art('FORESTS_BUNDLE', ['causal_forests_period_horizons_patient_level.joblib','forests_bundle.joblib','causal_forests_period_horizons.joblib'], outdir)

    # fallback from globals
    if forests_bundle is None:
        for candidate in ['forests','forests_bundle','FORESTS_BUNDLE','causal_forests']:
            if candidate in globals():
                forests_bundle = globals()[candidate]
                fb_src = f"from globals('{candidate}')"
                break

    cate_results = {}
    survival_df = None
    errors = {}

    # ---- build person-period rows ----
    max_period = None
    if 'pp_test' in globals() and hasattr(globals()['pp_test'], 'period'):
        try:
            max_period = int(globals()['pp_test']['period'].max())
        except Exception:
            max_period = None
    if max_period is None:
        max_period = 12

    rows = []
    for p in range(1, max_period+1):
        row = df.iloc[0].to_dict()
        row['period'] = p
        row['patient_id'] = df.iloc[0].get('patient_id', 'new')
        row['treatment'] = int(row.get('treatment', 0))
        rows.append(row)
    df_pp_new = pd.DataFrame(rows)

    # ---- pooled-logit survival ----
    if pooled_logit is None or model_columns is None:
        errors['pooled_logit'] = "pooled-logit model or model_columns missing; cannot compute survival."
    else:
        try:
            if 'build_X_for_pp' in globals():
                X_pp = build_X_for_pp(df_pp_new.copy())
            else:
                # minimal fallback: use pp_train_medians and patient_columns if available
                X_pp = pd.DataFrame(index=df_pp_new.index)
                if pp_train_medians is not None:
                    for c in pp_train_medians.keys():
                        X_pp[c] = np.where(c in df_pp_new.columns, pd.to_numeric(df_pp_new[c], errors='coerce').fillna(pp_train_medians[c]), pp_train_medians[c])
                if patient_columns is not None:
                    for c in patient_columns:
                        if c not in X_pp.columns:
                            X_pp[c] = 0.0
                X_pp['treatment'] = pd.to_numeric(df_pp_new['treatment'], errors='coerce').fillna(0).astype(int).values

            # align to model_columns
            if isinstance(model_columns, (pd.Series, np.ndarray, list)):
                cols_req = list(model_columns)
            elif isinstance(model_columns, pd.DataFrame):
                cols_req = model_columns.iloc[:,0].astype(str).tolist()
            else:
                cols_req = list(model_columns)
            X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)

            X_t = X_pp.copy(); X_t['treatment'] = 1
            X_c = X_pp.copy(); X_c['treatment'] = 0
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

    # ---- build Xpatient for CF ----
    Xpatient = None
    if patient_columns is None:
        errors['patient_columns'] = "patient_columns artifact missing. CF prediction will be impossible without canonical patient feature names."
    else:
        try:
            if isinstance(patient_columns, (pd.Series, np.ndarray)):
                pcols = list(patient_columns)
            elif isinstance(patient_columns, dict):
                pcols = list(patient_columns.get('columns', patient_columns.keys()))
            else:
                pcols = list(patient_columns)
            Xpatient = pd.DataFrame(np.zeros((1, len(pcols))), columns=pcols)
            for c in pcols:
                if c in df.columns:
                    Xpatient.at[0, c] = df.at[0, c]
                else:
                    if '_' in c:
                        root, tail = c.split('_', 1)
                        if root in df.columns and str(df.at[0, root]) == tail:
                            Xpatient.at[0, c] = 1.0
            if patient_scaler is not None:
                try:
                    num_cols = Xpatient.select_dtypes(include=[np.number]).columns
                    if len(num_cols)>0:
                        Xpatient[num_cols] = patient_scaler.transform(Xpatient[num_cols])
                except Exception:
                    errors['scaler'] = "scaler exists but failed to transform Xpatient; proceeding without scaling"
            Xpatient = Xpatient.reindex(columns=pcols, fill_value=0.0)
        except Exception as e:
            errors['Xpatient'] = f"Failed to construct Xpatient: {e}"
            Xpatient = None

    # ---- Predict CATEs using forests bundle ----
    if forests_bundle is None:
        errors['forests_bundle'] = "forests bundle not found in outputs or globals."
        for lab in period_labels:
            months = horizon_map.get(lab) if (horizon_map and lab in horizon_map) else (int(lab.split('-')[-1].replace('+','')) if isinstance(lab, str) and any(ch.isdigit() for ch in lab) else lab)
            cate_results[months] = {'CATE': np.nan, 'error': errors['forests_bundle']}
    else:
        for lab, est in forests_bundle.items():
            if horizon_map and lab in horizon_map:
                months = horizon_map[lab]
            else:
                try:
                    months = int(lab.replace('+','').split('-')[-1])
                except Exception:
                    months = lab
            if Xpatient is None:
                cate_results[months] = {'CATE': np.nan, 'error': 'Xpatient not built'}
                continue
            try:
                candidate_est = est
                if isinstance(est, dict):
                    for v in est.values():
                        if hasattr(v, 'effect'):
                            candidate_est = v
                            break
                if hasattr(candidate_est, "feature_names_in_"):
                    req = list(candidate_est.feature_names_in_)
                    Xfor = Xpatient.reindex(columns=req, fill_value=0.0)
                    Xfor_in = Xfor.values
                else:
                    Xfor_in = Xpatient.values
                eff = np.asarray(candidate_est.effect(Xfor_in)).flatten()
                val = float(eff[0]) if eff.size>0 else np.nan
                cate_results[months] = {'CATE': val, 'error': None}
            except Exception as e:
                cate_results[months] = {'CATE': np.nan, 'error': str(e)}

    try:
        cate_results = dict(sorted(cate_results.items(), key=lambda kv: (float(kv[0]) if isinstance(kv[0], (int,float,str)) and str(kv[0]).replace('.','',1).isdigit() else 1e9)))
    except Exception:
        pass

    out = {'survival_curve': survival_df, 'CATEs': cate_results, 'errors': errors}
    if return_raw:
        out['artifacts_sources'] = dict(patient_columns=pc_src, patient_scaler=sc_src, pp_train_medians=pm_src, pooled_logit=lp_src, model_columns=mc_src, forests_bundle=fb_src)
    return out

# for import
infer = infer_new_patient_fixed

