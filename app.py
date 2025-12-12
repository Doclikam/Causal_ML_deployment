# utils/infer.py
import os
import joblib
import pandas as pd
import numpy as np
import requests
import io
from typing import Optional

DEFAULT_OUTDIR = "outputs"
DEFAULT_INTERVAL_DAYS = 30
DEFAULT_PERIOD_LABELS = ['0-3','4-6','7-12','13-24','25-60','60+']

# -------------------- artifact loading helpers --------------------
def _load_local_art(path):
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
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return joblib.load(io.BytesIO(r.content)), f"remote:{url}"
    except Exception as e:
        return None, f"remote_failed:{url}:{e}"

def _try_load_artifact(name, candidates, outdir=DEFAULT_OUTDIR, base_url: Optional[str]=None):
    # 1) globals
    if name in globals() and globals()[name] is not None:
        return globals()[name], f"globals:{name}"

    # 2) local
    for fn in candidates:
        p = os.path.join(outdir, fn)
        val, src = _load_local_art(p)
        if val is not None:
            return val, src

    # 3) remote via base_url
    if base_url:
        base = base_url.rstrip("/") + "/"
        for fn in candidates:
            url = base + fn
            val, src = _load_remote_joblib(url)
            if val is not None:
                return val, src
            try:
                r = requests.get(url, timeout=30)
                r.raise_for_status()
                val = pd.read_csv(io.StringIO(r.text))
                return val, f"remote_csv:{url}"
            except Exception:
                continue

    return None, None

# -------------------- robust X builder --------------------
def _build_X_pp_for_model(df_pp_new,
                          model_columns,
                          scaler=None,
                          train_medians=None,
                          collapse_maps=None,
                          cat_cols=None,
                          num_cols=None,
                          period_bins=None,
                          period_labels=None):
    """
    Build person-period features to match model_columns exactly.
    """
    # canonicalize model_columns into list
    if isinstance(model_columns, (pd.Series, np.ndarray)):
        cols_req = [str(x).strip() for x in list(model_columns)]
    elif isinstance(model_columns, pd.DataFrame):
        cols_req = [str(x).strip() for x in model_columns.iloc[:,0].astype(str).tolist()]
    else:
        cols_req = [str(x).strip() for x in list(model_columns)]

    X_pp = pd.DataFrame(index=df_pp_new.index)

    # build period_bin if needed
    if 'period_bin' not in df_pp_new.columns and 'period' in df_pp_new.columns and period_bins is not None and period_labels is not None:
        df_pp_new = df_pp_new.copy()
        df_pp_new['period_month'] = df_pp_new['period'].astype(int)
        df_pp_new['period_bin'] = pd.cut(df_pp_new['period_month'], bins=period_bins, labels=period_labels, right=True)

    # find categorical roots: include provided cat_cols and derive roots from cols_req
    cat_roots = set(cat_cols or [])
    for c in cols_req:
        if '_' in c:
            root = c.rsplit('_', 1)[0]  # keep 'period_bin' rather than 'period'
            cat_roots.add(root)

    # create dummies (prefix is exact root so names match model_columns)
    for root in sorted(cat_roots):
        if root in df_pp_new.columns:
            ser = df_pp_new[root].astype(str)
            if collapse_maps and root in collapse_maps:
                allowed = set([str(x) for x in collapse_maps[root]])
                ser = ser.where(ser.isin(allowed), 'Other')
        else:
            ser = pd.Series([np.nan]*len(df_pp_new), index=df_pp_new.index)
        dummies = pd.get_dummies(ser.astype(str), prefix=root, drop_first=False)
        for col in dummies.columns:
            X_pp[col] = dummies[col].values

    # numeric columns
    if num_cols:
        for c in num_cols:
            if c in df_pp_new.columns:
                X_pp[c] = pd.to_numeric(df_pp_new[c], errors='coerce')
            else:
                X_pp[c] = np.nan
        # fill numeric missing using train medians if available
        if train_medians is not None:
            for c in num_cols:
                try:
                    if hasattr(train_medians, 'get'):
                        fillval = train_medians.get(c, np.nan)
                    elif c in getattr(train_medians, 'index', []):
                        fillval = train_medians[c]
                    else:
                        fillval = np.nan
                except Exception:
                    fillval = np.nan
                X_pp[c] = X_pp[c].fillna(fillval)
        else:
            X_pp[num_cols] = X_pp[num_cols].fillna(0.0)
        # scale if scaler provided
        if scaler is not None:
            try:
                X_pp[num_cols] = scaler.transform(X_pp[num_cols])
            except Exception:
                X_pp[num_cols] = X_pp[num_cols].fillna(0.0)

    # ensure treatment
    if 'treatment' not in X_pp.columns:
        if 'treatment' in df_pp_new.columns:
            X_pp['treatment'] = pd.to_numeric(df_pp_new['treatment'], errors='coerce').fillna(0).astype(int).values
        else:
            X_pp['treatment'] = 0

    # defensive: ensure all period_bin labels present
    if period_labels is not None:
        for lab in period_labels:
            col = f"period_bin_{lab}"
            if col not in X_pp.columns:
                X_pp[col] = 0.0

    # ensure every required column exists
    for c in cols_req:
        c = str(c).strip()
        if c not in X_pp.columns:
            X_pp[c] = 0.0

    # create interaction columns treat_x_<period_dummy>
    period_dummy_cols_req = [c for c in cols_req if str(c).startswith('period_bin')]
    for pcol in period_dummy_cols_req:
        inter = f"treat_x_{pcol}"
        if inter not in X_pp.columns:
            X_pp[inter] = X_pp['treatment'] * X_pp.get(pcol, 0.0)

    # final reorder
    X_pp = X_pp.reindex(columns=cols_req, fill_value=0.0)
    return X_pp

# -------------------- main inference function --------------------
def infer_new_patient_fixed(patient_data,
                            return_raw: bool = False,
                            outdir: str = DEFAULT_OUTDIR,
                            base_url: Optional[str] = None,
                            max_period_override: Optional[int] = None,
                            interval_days: int = DEFAULT_INTERVAL_DAYS,
                            period_labels: list = DEFAULT_PERIOD_LABELS,
                            period_bins: Optional[list] = None,
                            horizon_map: Optional[dict] = None):
    """
    Robust single-patient inference. Returns dict with survival, CATEs, errors, debug.
    """
    errors = {}
    debug = {}

    # normalize patient input
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
        'patient_columns': ['causal_patient_columns.joblib', 'causal_patient_columns.pkl', 'causal_patient_columns.npy', 'causal_patient_columns.csv'],
        'patient_scaler': ['causal_patient_scaler.joblib', 'causal_patient_scaler.pkl'],
        'pp_train_medians': ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv'],
        'pooled_logit': ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib','pooled_logit_logreg_saga.pkl'],
        'model_columns': ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib','pooled_logit_model_columns.pkl'],
        'forests_bundle': ['causal_forests_period_horizons_patient_level.joblib','causal_forests_period_horizons.joblib','forests_bundle.joblib'],
        'pp_scaler': ['pp_scaler.joblib','pp_scaler.pkl']
    }

    # load artifacts
    patient_columns, pc_src = _try_load_artifact('patient_columns', ART['patient_columns'], outdir=outdir, base_url=base_url)
    patient_scaler, sc_src = _try_load_artifact('patient_scaler', ART['patient_scaler'], outdir=outdir, base_url=base_url)
    pp_train_medians, pm_src = _try_load_artifact('pp_train_medians', ART['pp_train_medians'], outdir=outdir, base_url=base_url)
    pooled_logit, lp_src = _try_load_artifact('pooled_logit', ART['pooled_logit'], outdir=outdir, base_url=base_url)
    model_columns_art, mc_src = _try_load_artifact('model_columns', ART['model_columns'], outdir=outdir, base_url=base_url)
    forests_bundle, fb_src = _try_load_artifact('forests_bundle', ART['forests_bundle'], outdir=outdir, base_url=base_url)
    pp_scaler, pps_src = _try_load_artifact('pp_scaler', ART['pp_scaler'], outdir=outdir, base_url=base_url)

    debug['artifact_sources'] = {
        'patient_columns': pc_src, 'patient_scaler': sc_src, 'pp_train_medians': pm_src,
        'pooled_logit': lp_src, 'model_columns': mc_src, 'forests_bundle': fb_src, 'pp_scaler': pps_src
    }

    # determine max_period
    if max_period_override:
        max_period = int(max_period_override)
    else:
        max_period = 12

    # build person-period rows for new patient
    rows = []
    for p in range(1, max_period+1):
        row = df.iloc[0].to_dict()
        row['period'] = p
        row['patient_id'] = df.iloc[0].get('patient_id', 'new')
        row['treatment'] = int(row.get('treatment', 0))
        rows.append(row)
    df_pp_new = pd.DataFrame(rows)

    # pooled-logit survival
    survival_df = None
    if pooled_logit is None:
        errors['pooled_logit'] = "pooled-logit artifact missing; cannot compute survival."
    else:
        try:
            # authoritative model columns
            if hasattr(pooled_logit, "feature_names_in_"):
                model_columns_list = [str(x).strip() for x in list(pooled_logit.feature_names_in_)]
            else:
                if isinstance(model_columns_art, pd.DataFrame):
                    model_columns_list = model_columns_art.iloc[:,0].astype(str).tolist()
                elif isinstance(model_columns_art, (list, tuple, pd.Series, np.ndarray)):
                    model_columns_list = [str(x).strip() for x in list(model_columns_art)]
                else:
                    try:
                        model_columns_list = [str(x).strip() for x in list(model_columns_art)]
                    except Exception:
                        model_columns_list = []

            # default cat/num columns used to build X_pp
            cat_cols = ['period_bin','sex','smoking_status_clean','primary_site_group','subsite_clean','stage','hpv_clean']
            num_cols = ['age','ecog_ps','smoking_py_clean','time_since_rt_days']

            X_pp = _build_X_pp_for_model(
                df_pp_new.copy(),
                model_columns=model_columns_list,
                scaler=pp_scaler,
                train_medians=pp_train_medians,
                collapse_maps=globals().get('collapse_maps', None),
                cat_cols=cat_cols,
                num_cols=num_cols,
                period_bins=period_bins if period_bins is not None else [0,3,6,12,24,60,np.inf],
                period_labels=period_labels
            )

            debug['X_pp_columns'] = list(X_pp.columns)
            missing_after = [c for c in model_columns_list if c not in X_pp.columns]
            if missing_after:
                errors['pooled_logit_build'] = f"Missing cols after build: {missing_after}"

            # build treated & control counterfactuals
            X_t = X_pp.copy(); X_t['treatment'] = 1
            X_c = X_pp.copy(); X_c['treatment'] = 0

            # ensure interaction columns exist
            for pcol in [c for c in model_columns_list if str(c).startswith('period_bin')]:
                X_t[f'treat_x_{pcol}'] = X_t['treatment'] * X_t.get(pcol, 0)
                X_c[f'treat_x_{pcol}'] = X_c['treatment'] * X_c.get(pcol, 0)

            X_t = X_t.reindex(columns=model_columns_list, fill_value=0.0)
            X_c = X_c.reindex(columns=model_columns_list, fill_value=0.0)

            probs_t = pooled_logit.predict_proba(X_t)[:,1]
            probs_c = pooled_logit.predict_proba(X_c)[:,1]
            S_t = np.cumprod(1 - probs_t)
            S_c = np.cumprod(1 - probs_c)
            survival_df = pd.DataFrame({'period': np.arange(1, len(S_t)+1), 'S_control': S_c, 'S_treat': S_t})
            survival_df['days'] = survival_df['period'] * interval_days

        except Exception as e:
            errors['pooled_logit'] = f"pipelined survival predict failed: {e}"

    # ---------- build Xpatient (patient-level features) ----------
    Xpatient = None
    if patient_columns is None:
        debug.setdefault('notes', []).append("patient_columns artifact missing; Xpatient not built")
    else:
        try:
            if isinstance(patient_columns, (pd.Series, list, tuple, np.ndarray)):
                pcols = list(patient_columns)
            elif isinstance(patient_columns, dict):
                if 'columns' in patient_columns:
                    pcols = list(patient_columns['columns'])
                else:
                    pcols = list(patient_columns.keys())
            elif isinstance(patient_columns, pd.DataFrame):
                pcols = patient_columns.iloc[:,0].astype(str).tolist()
            else:
                pcols = list(patient_columns)

            Xpatient = pd.DataFrame(np.zeros((1, len(pcols))), columns=pcols)

            # fill from input df or infer one-hot
            for c in pcols:
                if c in df.columns:
                    Xpatient.at[0, c] = df.at[0, c]
                else:
                    if isinstance(c, str) and '_' in c:
                        root, tail = c.split('_', 1)
                        if root in df.columns and str(df.at[0, root]) == tail:
                            Xpatient.at[0, c] = 1.0

            # apply patient_scaler if present
            if patient_scaler is not None:
                try:
                    numeric_cols = Xpatient.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) > 0:
                        Xpatient[numeric_cols] = patient_scaler.transform(Xpatient[numeric_cols])
                except Exception as e:
                    debug.setdefault('notes', []).append(f"patient_scaler failed to transform Xpatient: {e}")

            Xpatient = Xpatient.reindex(columns=pcols, fill_value=0.0)

        except Exception as e:
            Xpatient = None
            debug.setdefault('notes', []).append(f"Failed to construct Xpatient: {e}")

    # ---------- CF CATE predictions ----------
    cate_results = {}
    if forests_bundle is None:
        for lab in period_labels:
            try:
                months = int(str(lab).replace('+','').split('-')[-1])
            except Exception:
                months = lab
            cate_results[months] = {'CATE': np.nan, 'error': 'forests_bundle missing'}
    else:
        for lab, est in forests_bundle.items():
            # map label -> months using horizon_map if provided
            if horizon_map is not None and lab in horizon_map:
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

    # finalize output
    out = {'survival_curve': survival_df, 'CATEs': cate_results, 'errors': errors}
    if return_raw:
        out['debug'] = debug
    return out
