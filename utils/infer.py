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
    # 1) globals (importing code may have set)
    if name in globals() and globals()[name] is not None:
        return globals()[name], f"globals:{name}"

    # 2) local files (outdir)
    for fn in candidates:
        p = os.path.join(outdir, fn)
        val, src = _load_local_art(p)
        if val is not None:
            return val, src

    # 3) remote via base_url (if available)
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


def infer_new_patient_fixed(
    patient_data,
    return_raw: bool = False,
    outdir: str = DEFAULT_OUTDIR,
    base_url: str = None,
    max_period_override: int = None,
    interval_days: int = DEFAULT_INTERVAL_DAYS,
    period_labels=DEFAULT_PERIOD_LABELS,
    horizon_map=None,
    outcome_type: str = "os"  # NEW: "os" (default) or "pfs"
):
    """
    Robust inference for a single new patient.

    Parameters
    ----------
    patient_data : dict or single-row DataFrame
    return_raw   : if True, include debug info
    outdir       : local outputs folder
    base_url     : optional raw github base url (ends with /outputs/) used to fetch artifacts remotely
    max_period_override : int months to create person-period rows (if provided)
    interval_days: length of each period in days
    period_labels: labels for CF horizons (used ONLY for OS CF; PFS currently returns survival only)
    horizon_map  : optional mapping from forest labels -> months
    outcome_type : "os" (overall survival, default) or "pfs" (progression-free survival)

    Returns
    -------
    dict with keys:
      - 'survival_curve' : DataFrame(period, S_control, S_treat, days)
      - 'CATEs'          : dict of horizon -> {CATE, error} (OS only; empty for PFS)
      - 'errors'         : dict of components -> message
      - 'debug'          : (optional) artifact source info
    """
    errors = {}
    debug = {}

    outcome_type = (outcome_type or "os").lower()
    if outcome_type not in ("os", "pfs"):
        outcome_type = "os"  # fall back safely

    # ------------------ INPUT → DF ------------------
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy().reset_index(drop=True)
    if 'patient_id' not in df.columns:
        df['patient_id'] = 'new'
    if 'treatment' not in df.columns:
        df['treatment'] = int(df.get('treatment', 0))

    # ------------------ ARTIFACT CANDIDATES ------------------
    # Base (OS) artifact names
    ART = {
        'patient_columns': ['causal_patient_columns.joblib', 'causal_patient_columns.pkl',
                            'causal_patient_columns.npy', 'causal_patient_columns.csv'],
        'patient_scaler': ['causal_patient_scaler.joblib', 'causal_patient_scaler.pkl'],
        'pp_train_medians': ['pp_train_medians.joblib','pp_train_medians.pkl','pp_train_medians.csv'],
        'pooled_logit': ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib','pooled_logit_logreg_saga.pkl'],
        'model_columns': ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib'],
        'forests_bundle': ['causal_forests_period_horizons_patient_level.joblib',
                           'causal_forests_period_horizons.joblib',
                           'forests_bundle.joblib'],
        'pp_scaler': ['pp_scaler.joblib','pp_scaler.pkl']
    }

    # Override some artifact names for PFS
