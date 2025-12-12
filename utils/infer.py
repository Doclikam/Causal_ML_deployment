
+import os
+import io
+import joblib
+import requests
+import numpy as np
+import pandas as pd
+from typing import Optional, Dict, Any
+
+# Simple, robust inference utilities for the Streamlit app.
+# Provides:
+#  - _try_load_artifact: local -> remote fallback loader
+#  - build_canonical_Xpatient: build 1-row feature vector aligned to saved model columns & scaler
+#  - infer_new_patient_fixed: produce survival curve (from pooled logit) and CATEs (from forests bundle)
+
+DEFAULT_OUTDIR = "outputs"
+DEFAULT_BASE_URL = None
+DEFAULT_INTERVAL_DAYS = 30
+
+def _load_local(path: str):
+    if not os.path.exists(path):
+        return None, None
+    try:
+        val = joblib.load(path)
+        return val, f"local:{path}"
+    except Exception:
+        try:
+            val = pd.read_csv(path)
+            return val, f"local_csv:{path}"
+        except Exception:
+            return None, f"failed_local:{path}"
+
+def _load_remote_joblib(url: str):
+    try:
+        r = requests.get(url, timeout=30)
+        r.raise_for_status()
+        return joblib.load(io.BytesIO(r.content)), f"remote:{url}"
+    except Exception:
+        return None, f"remote_failed:{url}"
+
+def _try_load_artifact(candidates, outdir=DEFAULT_OUTDIR, base_url: Optional[str]=DEFAULT_BASE_URL):
+    # try local files in order, then remote via base_url if provided
+    for fn in candidates:
+        p = os.path.join(outdir, fn)
+        val, src = _load_local(p)
+        if val is not None:
+            return val, src
+    if base_url:
+        base = base_url.rstrip("/") + "/"
+        for fn in candidates:
+            url = base + fn
+            val, src = _load_remote_joblib(url)
+            if val is not None:
+                return val, src
+            # try csv fallback
+            try:
+                r = requests.get(url, timeout=30)
+                r.raise_for_status()
+                df = pd.read_csv(io.StringIO(r.text))
+                return df, f"remote_csv:{url}"
+            except Exception:
+                continue
+    return None, None
+
+
+def build_canonical_Xpatient(
+    patient: Dict[str, Any],
+    patient_columns,
+    patient_scaler=None,
+    pp_train_medians=None
+) -> pd.DataFrame:
+    """
+    Build a 1-row DataFrame aligned to patient_columns:
+      - patient: dict of raw patient fields (age, sex, stage, etc.)
+      - patient_columns: list-like expected feature names (one-hot & numeric)
+      - patient_scaler: optional scaler to transform numeric columns (must expose feature_names_in_)
+      - pp_train_medians: optional dict/Series of medians for numeric fills
+    Returns single-row DataFrame Xpatient with columns = patient_columns
+    """
+    if patient_columns is None:
+        raise ValueError("patient_columns required")
+    # Normalize patient_columns into list
+    if isinstance(patient_columns, (pd.Series, np.ndarray, list)):
+        cols = list(patient_columns)
+    elif isinstance(patient_columns, pd.DataFrame):
+        cols = patient_columns.iloc[:, 0].astype(str).tolist()
+    else:
+        cols = list(patient_columns)
+
+    X = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
+
+    # fill from exact column names matching patient dict
+    for c in cols:
+        # direct numeric / categorical keys
+        if c in patient:
+            X.at[0, c] = patient[c]
+        else:
+            # attempt to set one-hot style names like "sex_Male" if patient['sex']=='Male'
+            if "_" in c:
+                root, tail = c.split("_", 1)
+                if root in patient and str(patient[root]) == tail:
+                    X.at[0, c] = 1.0
+
+    # Fill missing numeric features from pp_train_medians when available
+    if pp_train_medians is not None:
+        try:
+            for c in cols:
+                if pd.isna(X.at[0, c]) or X.at[0, c] == 0:
+                    # only fill numeric-like columns
+                    if c in pp_train_medians:
+                        X.at[0, c] = pp_train_medians[c]
+        except Exception:
+            pass
+
+    # Apply scaler if provided and compatible
+    if patient_scaler is not None:
+        try:
+            # derive scaler column order
+            if hasattr(patient_scaler, "feature_names_in_"):
+                scaler_cols = list(patient_scaler.feature_names_in_)
+            else:
+                # fallback: take numeric intersection
+                scaler_cols = [c for c in cols if c in ['age', 'ecog_ps', 'smoking_py_clean', 'time_since_rt_days', 'BED_eff', 'EQD2']]
+
+            # add missing scaler cols as 0 / medians
+            for sc in scaler_cols:
+                if sc not in X.columns:
+                    fill = 0.0
+                    if pp_train_medians is not None and sc in pp_train_medians:
+                        fill = pp_train_medians[sc]
+                    X[sc] = fill
+
+            Xnum = X[scaler_cols].astype(float)
+            Xnum_scaled = patient_scaler.transform(Xnum)
+            X.loc[:, scaler_cols] = pd.DataFrame(Xnum_scaled, columns=scaler_cols, index=X.index)
+        except Exception:
+            # do not fail: return unscaled X but warn via returned DataFrame
+            pass
+
+    # ensure column order stable
+    X = X.reindex(columns=cols, fill_value=0.0)
+    return X
+
+
+def infer_new_patient_fixed(
+    patient_data: Dict[str, Any],
+    outdir: str = DEFAULT_OUTDIR,
+    base_url: Optional[str] = None,
+    max_period_override: Optional[int] = None,
+    interval_days: int = DEFAULT_INTERVAL_DAYS,
+    return_raw: bool = False,
+    Xpatient_override: Optional[pd.DataFrame] = None
+) -> Dict[str, Any]:
+    """
+    Produce survival curve and CATEs for a single patient.
+    - If Xpatient_override is provided, it will be used directly (bypassing internal builder).
+    - Loads artifacts from outdir or base_url if needed.
+    Returns dict with keys: survival_curve (DataFrame), CATEs (dict), errors (dict), debug (dict)
+    """
+    errors = {}
+    debug = {}
+
+    # ensure patient dict -> DataFrame for convenience
+    if isinstance(patient_data, dict):
+        patient_df = pd.DataFrame([patient_data])
+    else:
+        patient_df = patient_data.copy().reset_index(drop=True)
+
+    # Artifact names to try
+    ART = {
+        "patient_columns": ['causal_patient_columns.joblib', 'causal_patient_columns.pkl', 'causal_patient_columns.csv'],
+        "patient_scaler": ['causal_patient_scaler.joblib', 'causal_patient_scaler.pkl'],
+        "pp_train_medians": ['pp_train_medians.joblib', 'pp_train_medians.csv'],
+        "pooled_logit": ['pooled_logit_logreg_saga.joblib','pooled_logit.joblib'],
+        "model_columns": ['pooled_logit_model_columns.csv','pooled_logit_model_columns.joblib'],
+        "forests_bundle": ['causal_forests_period_horizons_patient_level.joblib', 'forests_bundle.joblib']
+    }
+
+    # load artifacts (robust)
+    patient_columns, pc_src = _try_load_artifact(ART["patient_columns"], outdir=outdir, base_url=base_url)
+    patient_scaler, sc_src = _try_load_artifact(ART["patient_scaler"], outdir=outdir, base_url=base_url)
+    pp_train_medians, pm_src = _try_load_artifact(ART["pp_train_medians"], outdir=outdir, base_url=base_url)
+    pooled_logit, pl_src = _try_load_artifact(ART["pooled_logit"], outdir=outdir, base_url=base_url)
+    model_columns, mc_src = _try_load_artifact(ART["model_columns"], outdir=outdir, base_url=base_url)
+    forests_bundle, fb_src = _try_load_artifact(ART["forests_bundle"], outdir=outdir, base_url=base_url)
+
+    debug["artifact_sources"] = {
+        "patient_columns": pc_src,
+        "patient_scaler": sc_src,
+        "pp_train_medians": pm_src,
+        "pooled_logit": pl_src,
+        "model_columns": mc_src,
+        "forests_bundle": fb_src
+    }
+
+    # Build Xpatient (either override or canonical builder)
+    Xpatient = None
+    if Xpatient_override is not None:
+        Xpatient = Xpatient_override.copy()
+    else:
+        try:
+            Xpatient = build_canonical_Xpatient(
+                patient=patient_df.iloc[0].to_dict(),
+                patient_columns=patient_columns,
+                patient_scaler=patient_scaler,
+                pp_train_medians=(pp_train_medians if isinstance(pp_train_medians, dict) or pp_train_medians is None else pp_train_medians.to_dict())
+            )
+        except Exception as e:
+            errors["Xpatient"] = f"failed to build Xpatient: {e}"
+            Xpatient = None
+
+    debug["Xpatient"] = Xpatient
+
+    # pooled-logit survival (if available)
+    survival_df = None
+    if pooled_logit is None or model_columns is None:
+        errors["pooled_logit"] = "pooled_logit or model_columns missing"
+    else:
+        try:
+            # Build person-period rows: determine max_period
+            max_period = max_period_override if max_period_override is not None else 12
+            rows = []
+            pdata = patient_df.iloc[0].to_dict()
+            for p in range(1, int(max_period) + 1):
+                r = pdata.copy()
+                r["period"] = p
+                r["patient_id"] = r.get("patient_id", "new")
+                rows.append(r)
+            df_pp = pd.DataFrame(rows)
+
+            # If calling code provides a build_X_for_pp in globals, prefer it
+            if "build_X_for_pp" in globals():
+                Xpp = build_X_for_pp(df_pp)
+            else:
+                # minimal alignment: try to fill numeric medians and zeros
+                if isinstance(model_columns, (pd.Series, list, tuple, np.ndarray)):
+                    cols_req = list(model_columns)
+                elif isinstance(model_columns, pd.DataFrame):
+                    cols_req = model_columns.iloc[:, 0].astype(str).tolist()
+                else:
+                    cols_req = list(model_columns)
+                Xpp = pd.DataFrame(index=df_pp.index, columns=cols_req).fillna(0.0)
+                # fill numeric medians where column name matches
+                try:
+                    med = pp_train_medians if isinstance(pp_train_medians, dict) else (pp_train_medians.to_dict() if pp_train_medians is not None else {})
+                    for c in cols_req:
+                        if c in df_pp.columns:
+                            Xpp[c] = pd.to_numeric(df_pp[c], errors="coerce").fillna(med.get(c, 0.0))
+                        elif c in med:
+                            Xpp[c] = med[c]
+                except Exception:
+                    pass
+
+            # make counterfactuals and predict
+            Xt = Xpp.copy(); Xt["treatment"] = 1
+            Xc = Xpp.copy(); Xc["treatment"] = 0
+            for pcol in [c for c in Xt.columns if str(c).startswith("period_bin")]:
+                Xt[f"treat_x_{pcol}"] = Xt["treatment"] * Xt.get(pcol, 0)
+                Xc[f"treat_x_{pcol}"] = Xc["treatment"] * Xc.get(pcol, 0)
+
+            probs_t = pooled_logit.predict_proba(Xt)[:, 1]
+            probs_c = pooled_logit.predict_proba(Xc)[:, 1]
+            S_t = np.cumprod(1 - probs_t)
+            S_c = np.cumprod(1 - probs_c)
+            survival_df = pd.DataFrame({
+                "period": np.arange(1, len(S_t) + 1),
+                "S_control": S_c,
+                "S_treat": S_t,
+                "days": np.arange(1, len(S_t) + 1) * interval_days
+            })
+        except Exception as e:
+            errors["pooled_logit_predict"] = str(e)
+
+    # CATE predictions from forests bundle (if present)
+    cates = {}
+    if forests_bundle is None:
+        errors["forests_bundle"] = "forests bundle missing"
+    else:
+        try:
+            # forests_bundle expected mapping label -> estimator (or estimator-like)
+            for lab, est in forests_bundle.items():
+                # resolve estimator object
+                candidate = est
+                if isinstance(est, dict):
+                    # take first object that has .effect
+                    for v in est.values():
+                        if hasattr(v, "effect"):
+                            candidate = v
+                            break
+                if Xpatient is None:
+                    cates[lab] = {"CATE": np.nan, "error": "Xpatient missing"}
+                    continue
+                if hasattr(candidate, "feature_names_in_"):
+                    req = list(candidate.feature_names_in_)
+                    Xfor = Xpatient.reindex(columns=req, fill_value=0.0)
+                    Xin = Xfor.values
+                else:
+                    Xin = Xpatient.values
+                eff = np.asarray(candidate.effect(Xin)).flatten()
+                cates[lab] = {"CATE": float(eff[0]) if eff.size > 0 else np.nan, "error": None}
+        except Exception as e:
+            errors["forests_predict"] = str(e)
+
+    out = {"survival_curve": survival_df, "CATEs": cates, "errors": errors, "debug": debug}
+    if return_raw:
+        return out
+    else:
+        # drop debug in minimal mode
+        out.pop("debug", None)
+        return out
+
*** End Patch
