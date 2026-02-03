import os, gc, json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List
from datetime import datetime, timezone

import joblib
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore

from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor

from cassandra_utils.models.scada_real_time_predictions import (
    ScadaRealTimePredictions, _to_utc as _to_utc_pid, _sf as _sf_pid
)
from cassandra_utils.models.scada_real_time_prediction_summary import (
    ScadaRealTimePredictionSummary
)

# ----------------- constants -----------------
MODELS_DIR = "./models"
LOOKBACK   = 20
EPOCHS     = 3
MIN_TRAIN_POINTS = 120
BATCH_SIZE = 64

os.makedirs(MODELS_DIR, exist_ok=True)

# -------- retrain policy ----------
MODEL_STALE_SECONDS = 6 * 3600
RETRAIN_BAD_STREAK  = 3
RETRAIN_Q_MSE       = 0.02
RETRAIN_Q_MAPE      = 0.35
RETRAIN_METRIC      = "mse"
RETRAIN_MIN_POINTS  = 60
RETRAIN_BLOCK_MAX   = 100000

# ----------- Trend Analyses --------------
from statsmodels.nonparametric.smoothers_lowess import lowess

# TREND ANALYSIS ON NON-LINEAR TIME SERIES DATA
def lowess_last(y: np.ndarray, frac: float = 0.2):
    """
    Apply LOWESS smoothing and return the last smoothed value.
    
    Args:
        y: Time series data
        frac: LOWESS fraction (smoothing parameter, 0-1)
    
    Returns:
        Smoothed last value
    """
    n = len(y)
    if n < 5:
        return float(y[-1]) if n else np.nan
    
    # Remove NaN/Inf only
    y_clean = y[np.isfinite(y)]
    
    if len(y_clean) < 5:
        return float(y_clean[-1]) if len(y_clean) else np.nan
    
    # Apply LOWESS
    x = np.arange(len(y_clean), dtype=float)
    sm = lowess(y_clean, x, frac=min(max(frac, 0.05), 0.8), it=0, return_sorted=False)
    
    return float(sm[-1])

def lowess_last3(y: np.ndarray, frac: float = 0.2, outlier_std: float = 3.0): # actual = mean oluyor 
    """
    UPDATED: Filter outliers before LOWESS to prevent crazy interpolation.
    
    Args:
        y: Time series data
        frac: LOWESS fraction (smoothing)
        outlier_std: Remove points beyond this many std deviations
    
    Returns:
        Smoothed last value
    """
    n = len(y)
    if n < 5:
        return float(y[-1]) if n else np.nan
    
    # ===== OUTLIER FILTERING (NEW) =====
    y_clean = y.copy()
    
    # Remove NaN/Inf
    y_clean = y_clean[np.isfinite(y_clean)]
    
    if len(y_clean) < 5:
        return float(y_clean[-1]) if len(y_clean) else np.nan
    
    # Filter beyond 3 standard deviations
    mean_val = np.nanmean(y_clean)
    std_val = np.nanstd(y_clean)
    
    if std_val > 0:  # Only filter if there's variance
        lower_bound = mean_val - outlier_std * std_val
        upper_bound = mean_val + outlier_std * std_val
        
        mask = (y_clean >= lower_bound) & (y_clean <= upper_bound)
        y_filtered = y_clean[mask]
        
        # Only use filtered if we kept >50% of data
        if len(y_filtered) > len(y_clean) * 0.5:
            y_clean = y_filtered
    
    # ===== LOWESS ON CLEAN DATA =====
    n_clean = len(y_clean)
    if n_clean < 5:
        return float(y_clean[-1]) if n_clean else np.nan
    
    x = np.arange(n_clean, dtype=float)
    sm = lowess(y_clean, x, frac=min(max(frac, 0.05), 0.8), it=0, return_sorted=False)
    
    return float(sm[-1])

def lowess_last2(y: np.ndarray, frac: float = 0.2):
    n = len(y)
    if n < 5:
        return float(y[-1]) if n else np.nan
    x = np.arange(n, dtype=float)
    sm = lowess(y, x, frac=min(max(frac, 0.05), 0.8), it=0, return_sorted=False)
    return float(sm[-1])

# ----------------- buffers -------------------
class SeriesBuffer:
    def __init__(self, maxlen: int = 5000):
        self.df = pd.DataFrame()
        self.maxlen = maxlen
        self.first_ts = None

    def append_row(self, ts: datetime, values: Dict[str, float]):
        """Append one observation keyed by timestamp.

        We keep a datetime index so we can resample (30s / 1min) later.
        If multiple points arrive for the same timestamp we keep the last one.
        """
        if ts is None:
            ts = datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        self.df = pd.concat([self.df, pd.DataFrame([values], index=[ts])], axis=0)
        # If same timestamp appears multiple times, keep the last message.
        self.df = self.df[~self.df.index.duplicated(keep="last")]
        self.df = self.df.sort_index()
        if len(self.df) > self.maxlen:
            self.df = self.df.tail(self.maxlen)
        if self.first_ts is None:
            self.first_ts = ts


_buffers: Dict[str, SeriesBuffer] = defaultdict(lambda: SeriesBuffer(maxlen=5000))

# ============================================================
# NEW: Helper functions for separate models
# ============================================================

def _split_input_output_vars(flat_vals: dict) -> Tuple[dict, dict]:
    """Split flat_vals into separate input and output dicts."""
    inputs = {k: v for k, v in (flat_vals or {}).items() if k.startswith("in_")}
    outputs = {k: v for k, v in (flat_vals or {}).items() if k.startswith("out_")}
    return inputs, outputs


def _has_real_values(var_dict: dict) -> bool:
    """Check if dictionary has any non-zero, non-null values."""
    if not var_dict:
        return False
    
    for val in var_dict.values():
        try:
            val_float = float(val) if val is not None else 0.0
            if val_float != 0.0 and not np.isnan(val_float):
                return True
        except (ValueError, TypeError):
            continue
    
    return False

# ----------------- utils ---------------------
def _algo_tag(algo: str) -> str:
    a = (algo or "LSTM").strip().upper()
    return a.replace(" ", "_").replace("/", "_")

def _model_paths(key: str, algorithm: str):
    safe_key = key.replace("/", "_")
    a = _algo_tag(algorithm)
    base = os.path.join(MODELS_DIR, f"{safe_key}__ALG_{a}")
    
    if a == "LSTM":
        model_path = base + ".keras"
    else:
        model_path = base + ".pkl"
    
    scaler_path = base + "_scaler.pkl"
    meta_path   = base + "_meta.json"
    return model_path, scaler_path, meta_path

def _norm_str(x, default="UNKNOWN"):
    if x is None:
        return default
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return default
    return s.replace(" ", "_").replace("/", "_")

def _norm_key(name: str) -> str:
    if name is None:
        return ""
    return str(name).strip().replace(" ", "_").replace("/", "_")

def _get_stock_no(message: dict) -> str:
    st = message.get("output_stock_no")
    st = _norm_str(st, default="UNKNOWN")
    if st != "UNKNOWN":
        return st
    
    pl = message.get("prodList")
    if isinstance(pl, list) and len(pl) > 0 and isinstance(pl[0], dict):
        st = pl[0].get("stNo") or pl[0].get("stockNo")
        return _norm_str(st, default="UNKNOWN")
    
    return "UNKNOWN"

def _get_op_tc(message: dict) -> str:
    op = message.get("operationtaskcode") or message.get("opTc")
    return _norm_str(op, default="UNKNOWN")

def _realtime_model_key_with_type(scope: str, scope_id, message: dict, 
                                   group_by_stock: bool, model_type: str) -> str:
    """
    Generate model key with type suffix (_INPUT or _OUTPUT).
    
    Args:
        model_type: "INPUT" or "OUTPUT"
    """
    # Build base key
    if scope == "pid":
        op_tc = _get_op_tc(message)
        ws_id = _norm_str(message.get("wsId"), default="UNKNOWN")
        
        if op_tc == "UNKNOWN":
            base_key = f"PID_{scope_id}" if scope_id is not None else extract_realtime_key(message)
        elif group_by_stock:
            st_no = _get_stock_no(message)
            base_key = f"OPTC_{op_tc}_WS_{ws_id}_ST_{st_no}"
        else:
            base_key = f"OPTC_{op_tc}_WS_{ws_id}"
    
    elif scope == "ws":
        if scope_id is None:
            base_key = extract_realtime_key(message)
        elif group_by_stock:
            st_no = _get_stock_no(message)
            base_key = f"WS_{scope_id}_ST_{st_no}"
        else:
            base_key = f"WS_{scope_id}"
    else:
        base_key = extract_realtime_key(message)
    
    # Add type suffix
    return f"{base_key}_{model_type}"

def _build_model(n_steps: int, n_features: int) -> tf.keras.Model:
    m = models.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(n_features, activation='relu') #layers.Dense(n_features),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m

def _make_sequences(X: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i])
        ys.append(X[i])
    return np.asarray(xs), np.asarray(ys)

def json_dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def json_load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_meta(meta_path: str) -> dict:
    if os.path.exists(meta_path):
        try:
            return json_load(meta_path) or {}
        except Exception:
            return {}
    return {}

def _save_meta(meta_path: str, patch: dict):
    base = _load_meta(meta_path)
    if not isinstance(base, dict):
        base = {}
    if not isinstance(patch, dict):
        patch = {}
    base.update(patch)
    try:
        json_dump(meta_path, base)
    except Exception:
        pass

def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def _parse_iso_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None

def safe_float(x):
    try:
        if x is None:
            return np.nan
        s = str(x).replace(",", ".")
        return float(s)
    except Exception:
        return np.nan

def _metric_mse(a: np.ndarray, p: np.ndarray) -> float:
    d = (a - p)
    return float(np.nanmean(d * d))

def _metric_mape(a: np.ndarray, p: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(a), eps)
    return float(np.nanmean(np.abs(a - p) / denom))

def extract_realtime_key(message: dict) -> str:
    return (message.get("process_no")
            or str(message.get("joOpId"))
            or message.get("wsId")
            or message.get("wsNo")
            or "")

def extract_numeric_io(message: dict) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    FIXED: List source (inputVariableList vs outputValueList) is PRIMARY.
    equipment_type is only used as secondary signal.
    """
    inputs_map: Dict[str, float] = {}
    outputs_map: Dict[str, float] = {}

    # ===== INPUT LIST - ONLY ADD TO INPUTS =====
    for iv in (message.get("inputVariableList") or message.get("inVars") or []):
        if not isinstance(iv, dict):
            continue

        name = (
            iv.get("equipment_name")
            or iv.get("eqNm")
            or iv.get("varNm")
            or iv.get("param")
            or iv.get("eqNo")
            or iv.get("varNo")
            or iv.get("varId")
        )
        k = _norm_key(name)
        if not k:
            continue

        # ===== TRUST LIST SOURCE: This came from INPUT list =====
        val = (
            iv.get("gen_read_val")
            or iv.get("genReadVal")
            or iv.get("actVal")
            or iv.get("value")
            or iv.get("cntRead")
            or iv.get("counter_reading")
        )
        
        # Always add to inputs (came from input list)
        inputs_map[k] = safe_float(val)

    # ===== OUTPUT LIST - ONLY ADD TO OUTPUTS =====
    for ov in (message.get("outputValueList") or message.get("outVals") or []):
        if not isinstance(ov, dict):
            continue

        name = (
            ov.get("equipment_name")
            or ov.get("eqNm")
            or ov.get("parameter")
            or ov.get("param")
            or ov.get("eqNo")
        )
        k = _norm_key(name)
        if not k:
            continue

        # ===== TRUST LIST SOURCE: This came from OUTPUT list =====
        val = (
            ov.get("counter_reading")
            or ov.get("cntRead")
            or ov.get("value")
            or ov.get("genReadVal")
        )
        
        # Always add to outputs (came from output list)
        outputs_map[k] = safe_float(val)

    return inputs_map, outputs_map

def extract_prediction_metadata(message: dict) -> Dict[str, str]:
    return {
        "start_date":        message.get("crDt"),
        "customer":          (message.get("outVals",[{}])[0].get("cust") 
                              if isinstance(message.get("outVals"), list) and message.get("outVals") else message.get("cust")),
        "plant_id":          str(message.get("plId") or ""),
        "workcenter_name":   message.get("wcNm") or "",
        "workcenter_no":     message.get("wcNo") or "",
        "workstation_name":  message.get("wsNm") or "",
        "workstation_no":    message.get("wsNo") or "",
        "operator_name":     message.get("opNm") or "",
        "operator_no":       message.get("opNo") or "",
        "output_stock_name": (message.get("prodList",[{}])[0].get("stNm") 
                              if isinstance(message.get("prodList"), list) and message.get("prodList") else message.get("stNm")),
        "output_stock_no":   (message.get("prodList",[{}])[0].get("stNo") 
                              if isinstance(message.get("prodList"), list) and message.get("prodList") else message.get("stNo")),
        
        # "job_order_reference_no": str(message.get("joRef") or message.get("job_order_reference_no") or ""),
        # "prod_order_reference_no": str(message.get("refNo") or message.get("prod_order_reference_no") or ""),

        "job_order_reference_no": str(message.get("job_order_reference_no") or message.get("refNo") or message.get("joRef") or ""),
        "prod_order_reference_no": str(message.get("prod_order_reference_no") or message.get("joRef") or message.get("refNo") or ""),

        "operationname":     message.get("operationname") or message.get("opNm") or "",
        "operationno":       message.get("operationno") or message.get("opNo") or "",
        "operationtaskcode": message.get("operationtaskcode") or message.get("opTc") or "",
        "process_no":      str(message.get("joOpId") or message.get("job_operation_id") or "")
    }

def _sf(x: float) -> float:
    """Safe float (no NaN/Inf) for payloads."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not np.isfinite(v):
        return 0.0
    return v

def _vectors_for_write(
    df_actual,
    cols,
    y_hat,
    df_for_mean=None,
    mean_mode: str = "lowess",
    lowess_frac: float = 0.25,
    lowess_it: int = 1,
    lowess_window: int = 300
):
    if df_for_mean is None:
        df_for_mean = df_actual

    last = df_actual.iloc[-1] if len(df_actual) else pd.Series(dtype="float64")
    actual = {c: float(last.get(c, np.nan)) for c in cols}
    predicted = {c: float(v) for c, v in zip(cols, y_hat)}

    means = {}
    df_num = df_for_mean.apply(pd.to_numeric, errors="coerce")

    for c in cols:
        s = df_num[c].dropna().values
        if len(s) == 0:
            means[c] = np.nan
            continue

        # Window limiting
        if lowess_window and len(s) > lowess_window:
            s = s[-lowess_window:]

        if mean_mode == "mean":
            means[c] = float(np.nanmean(s))
        else:
            # ===== USE FIXED lowess_last (NEW) =====
            means[c] = lowess_last(s, frac=lowess_frac)

    return actual, predicted, means

def history_from_fetch(dates, sensor_values):
    """
    FIXED: Strict equipment_type checking.
    """
    hist_in = []
    hist_out = []

    for ts, bundle in zip(dates, sensor_values):
        row_in = {}
        row_out = {}

        if not bundle or len(bundle) < 2:
            continue

        for s in bundle[1:]:
            if not isinstance(s, dict):
                continue

            name = (
                s.get("equipment_name")
                or s.get("eqNm")
                or s.get("equipment_no")
                or s.get("eqNo")
                or s.get("varNm")
                or s.get("varNo")
            )
            k = _norm_key(name)
            if not k:
                continue

            # ===== STRICT equipment_type check =====
            et = s.get("equipment_type")
            
            if et is True:
                # INPUT
                val = (
                    s.get("gen_read_val")
                    or s.get("genReadVal")
                    or s.get("value")
                    or s.get("counter_reading")
                    or s.get("cntRead")
                )
                row_in[k] = safe_float(val)
            
            elif et is False:
                # OUTPUT
                val = (
                    s.get("counter_reading")
                    or s.get("cntRead")
                    or s.get("value")
                )
                row_out[k] = safe_float(val)
            
            # else: et is None or other -> SKIP

        if row_in:
            hist_in.append((ts, row_in))
        if row_out:
            hist_out.append((ts, row_out))

    return hist_in, hist_out

def _expected_flat_features(algo: str, lookback: int, n_features: int) -> int:
    a = _algo_tag(algo)
    if a == "LSTM":
        return n_features
    return lookback * n_features

def _model_expected_n_features(model, algorithm: str):
    a = _algo_tag(algorithm)
    if a == "LSTM":
        try:
            shp = getattr(model, "input_shape", None)
            if shp and len(shp) == 3:
                return int(shp[2])
        except Exception:
            pass
        return None

    # sklearn
    try:
        if hasattr(model, "n_features_in_"):
            return int(model.n_features_in_)
    except Exception:
        pass
    try:
        ests = getattr(model, "estimators_", None)
        if ests and hasattr(ests[0], "n_features_in_"):
            return int(ests[0].n_features_in_)
    except Exception:
        pass
    return None

def _load_model_any(model_path: str, algorithm: str, p3_1_log=None):
    a = _algo_tag(algorithm)
    if p3_1_log:
        p3_1_log.info(f"[rt_pred] model load try algo={a} path={model_path}")

    if not os.path.exists(model_path):
        return None

    if a == "LSTM":
        return tf.keras.models.load_model(model_path)
    else:
        return joblib.load(model_path)

def _save_model_any(model, model_path: str, algorithm: str, p3_1_log=None):
    a = _algo_tag(algorithm)
    if a == "LSTM":
        model.save(model_path)
    else:
        joblib.dump(model, model_path)
    if p3_1_log:
        p3_1_log.info(f"[rt_pred] model saved algo={a} path={model_path}")

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor

def _build_model_any(algorithm: str, lookback: int, n_features: int):
    a = _algo_tag(algorithm)

    if a == "LSTM":
        return _build_model(lookback, n_features)

    if a == "RANDOM_FOREST":
        base = RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1,
            #max_features='sqrt',      # Overfitting'i azaltır
            oob_score=True ,         # Out-of-bag error
            min_samples_leaf=2      # Leaf node kontrolü

            # SVR için
            #gamma='scale',           # Auto-tuned gamma
            #cache_size=500,          # Hızlı training

            # SGD için
            #penalty='elasticnet'    # L1 + L2
            #early_stopping=True     # Overfitting önleme
            )
        return MultiOutputRegressor(base, n_jobs=-1)

    if a == "SUPPORT_VECTOR_REGRESSOR":
        base = SVR(#C=10.0, epsilon=0.01, kernel="rbf",
                    C = 0.1,
                    epsilon = 0.2,
                    kernel = "linear",  # Start simple
                    gamma = "scale",
                    max_iter = 1000
                   )
        return MultiOutputRegressor(base, n_jobs=-1)

    if a == "DYNAMIC_VECTOR_MACHINE":
        base = SGDRegressor(loss="squared_error", tol=1e-3, random_state=42, #, alpha=1e-4, max_iter=2000
                            alpha = 1e-3,
                            l1_ratio = 0.0,  # Pure L2
                            penalty = "l2",
                            learning_rate = "constant",
                            max_iter = 500
                            )
        return MultiOutputRegressor(base, n_jobs=-1)
    
    if a == "CATBOOST":
        base = CatBoostRegressor(
            iterations=100,
            depth=5,
            learning_rate=0.1,
            loss_function='RMSE',
            verbose=False,
            random_seed=42,
            # CRITICAL for small data:
            l2_leaf_reg=3,              # Regularization
            bootstrap_type='Bernoulli',  # Prevents overfitting
            subsample=0.8,              # Use 80% of data per tree
            # Speed optimizations:
            thread_count=-1
        )
        return MultiOutputRegressor(base, n_jobs=-1)
    
    if a == "XGBOOST":
        base = XGBRegressor(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            # Regularization:
            reg_alpha=0.1,         # L1
            reg_lambda=1.0,        # L2
            gamma=0.1,             # Minimum loss reduction
            # Prevent overfitting:
            subsample=0.8,
            colsample_bytree=0.8,
            # Speed:
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )
        return MultiOutputRegressor(base, n_jobs=-1)
    
    if a == "LIGHTGBM":
        base = LGBMRegressor(
            n_estimators=50,
            max_depth=5,
            learning_rate=0.1,
            num_leaves=15,
            # Regularization:
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=5,   # Min samples in leaf
            # Prevent overfitting:
            subsample=0.8,
            colsample_bytree=0.8,
            # Speed:
            n_jobs=-1,
            random_state=42,
            verbosity=-1
        )
        return MultiOutputRegressor(base, n_jobs=-1)
    
    if a == "KNEIGHBOURS":
        base = KNeighborsRegressor(
            n_neighbors=5,
            weights='distance',     # Closer neighbors = more weight
            algorithm='auto',       # Choose best algorithm
            leaf_size=30,
            p=2,                   # Euclidean distance
            n_jobs=-1
        )
        return MultiOutputRegressor(base, n_jobs=-1)

    raise ValueError(f"Unknown algorithm={algorithm}")

def _make_sequences_flat(X: np.ndarray, lookback: int):
    xs, ys = [], []
    for i in range(lookback, len(X)):
        xs.append(X[i-lookback:i].reshape(-1))
        ys.append(X[i])
    return np.asarray(xs), np.asarray(ys)

def _train_model_any(Xs, cols, lookback, epochs, min_train_points,
                     model_path, meta_path, algorithm,
                     force_retrain, p3_1_log, info):

    min_needed = max(RETRAIN_MIN_POINTS, min_train_points, lookback + 20)

    if p3_1_log:
        p3_1_log.info(
            f"[rt_pred] train check algo={_algo_tag(algorithm)} Xs_len={len(Xs)} "
            f"min_needed={min_needed} lookback={lookback} force_retrain={force_retrain}"
        )

    if len(Xs) < min_needed:
        if p3_1_log:
            p3_1_log.info("[rt_pred] train skipped: not enough points")
        return None, False

    take_n = min(len(Xs), max(min_train_points, lookback + 200))
    take_n = min(take_n, RETRAIN_BLOCK_MAX)
    train_block = Xs[-take_n:]

    a = _algo_tag(algorithm)

    model = _build_model_any(algorithm, lookback, Xs.shape[1])

    if a == "LSTM":
        X_seq, y_seq = _make_sequences(train_block, lookback)
        if p3_1_log:
            p3_1_log.info(f"[rt_pred] LSTM seqs: X_seq.shape={X_seq.shape} y_seq.shape={y_seq.shape} epochs={epochs}")
        model.fit(X_seq, y_seq, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
    else:
        Xf, y = _make_sequences_flat(train_block, lookback)
        if p3_1_log:
            p3_1_log.info(f"[rt_pred] SKLEARN seqs: Xf.shape={Xf.shape} y.shape={y.shape} algo={a}")
        if len(Xf) < 5:
            if p3_1_log:
                p3_1_log.info("[rt_pred] sklearn train skipped: too few sequences")
            return None, False
        model.fit(Xf, y)

    _save_model_any(model, model_path, algorithm, p3_1_log=p3_1_log)
    _save_meta(meta_path, {"cols": cols, "timesteps": int(lookback)})

    info(f"[rt_pred] trained model algo={a} steps={lookback} feats={Xs.shape[1]}")

    return model, True

def _predict_next_any(model, Xs, scaler, lookback, algorithm, p3_1_log=None):
    a = _algo_tag(algorithm)

    if a == "LSTM":
        if len(Xs) >= lookback + 1:
            last_window = Xs[-lookback-1:-1]
        else:
            core = Xs[:-1] if len(Xs) > 1 else Xs
            pad = lookback - len(core)
            seed = core[:1] if len(core) else np.zeros((1, Xs.shape[1]), dtype="float32")
            last_window = np.vstack([np.repeat(seed, pad, axis=0), core])

        y_hat_scaled = model.predict(last_window[None, ...], verbose=0)[0]
        y_hat_scaled = np.clip(y_hat_scaled, 0.0, 1.0)
        y_hat = scaler.inverse_transform(y_hat_scaled[None, ...])[0]
        return y_hat

    # sklearn
    n_feat = Xs.shape[1]
    if len(Xs) >= lookback:
        last_window = Xs[-lookback:]
    else:
        pad = lookback - len(Xs)
        seed = Xs[:1] if len(Xs) else np.zeros((1, n_feat), dtype="float32")
        last_window = np.vstack([np.repeat(seed, pad, axis=0), Xs])

    x = last_window.reshape(1, -1)
    y_hat_scaled = model.predict(x)[0].astype("float32")
    y_hat_scaled = np.clip(y_hat_scaled, 0.0, 1.0)
    y_hat = scaler.inverse_transform(y_hat_scaled.reshape(1, -1))[0]

    return y_hat

# =========================
#  Core prediction helpers
# =========================
def _rt_load_state_and_stale(meta_path, p3_1_log=None):
    meta_state = _load_meta(meta_path)
    now_utc = datetime.now(timezone.utc)
    last_trained = _parse_iso_dt(meta_state.get("last_trained_ts", ""))
    is_stale = False
    age = None

    if last_trained is not None:
        age = (now_utc - last_trained).total_seconds()
        is_stale = age > MODEL_STALE_SECONDS

    bad_streak = int(meta_state.get("bad_streak", 0) or 0)
    last_pred_vec = meta_state.get("last_pred_vec")

    return meta_state, is_stale, bad_streak, last_pred_vec

def _rt_load_cols_and_update_meta(meta_path, lookback, seed_history, flat_vals, p3_1_log=None):
    cols = None
    if os.path.exists(meta_path):
        try:
            mj = json_load(meta_path)
            cols = list(mj.get("cols", [])) or None
        except Exception:
            cols = None

    before = list(cols) if cols else None

    if cols is None:
        candidates = set()
        if seed_history:
            for _, r in seed_history:
                candidates.update(r.keys())
        candidates.update((flat_vals or {}).keys())
        cols = list(sorted(candidates))
    else:
        for c in (flat_vals or {}).keys():
            if c not in cols:
                cols.append(c)
        if seed_history:
            for _, r in seed_history:
                for c in r.keys():
                    if c not in cols:
                        cols.append(c)

    _save_meta(meta_path, {"cols": cols, "timesteps": int(lookback)})

    return cols

def _rt_seed_and_append_buffer(key, cols, seed_history, flat_vals, message, p3_1_log):
    key_buf = _buffers[key]

    if seed_history and key_buf.df.empty:
        for ts, row in seed_history:
            nr = {c: safe_float(row.get(c)) for c in cols}
            key_buf.append_row(_to_utc_pid(ts) or datetime.now(timezone.utc), nr)

    if flat_vals:
        numeric_row = {c: safe_float(flat_vals.get(c)) for c in cols}
        key_buf.append_row(_to_utc_pid(message.get("crDt")) or datetime.now(timezone.utc), numeric_row)

    return key_buf.df.copy()

#UPDATECODE
def _rt_prepare_df(df_raw, p3_1_log=None, key=None, resample_seconds: int = None):
    """Prepare buffer DF for modeling.

    - ensures numeric
    - optional resampling to a fixed cadence
    - forward/back fill then 0.0 for remaining gaps
    """
    df = df_raw.copy()

    # Ensure datetime index for resampling
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass

    if resample_seconds and isinstance(df.index, pd.DatetimeIndex):
        try:
            # Use LAST value in the bin. For SCADA/counter readings, "last" preserves
            # step changes better than mean.
            df = df.sort_index().resample(f"{int(resample_seconds)}S").last()
        except Exception as e:
            if p3_1_log:
                p3_1_log.warning(f"[rt_pred] resample failed for key={key}: {e}")

    df = df.apply(pd.to_numeric, errors="coerce") \
           .ffill() \
           .bfill() \
           .fillna(0.0)
    return df


def _rt_quality_check(cols, df, scaler_path, last_pred_vec, bad_streak, p3_1_log):
    retrain_due_to_quality = False
    metric_val = None

    if not (isinstance(last_pred_vec, dict) and len(df) >= 1):
        return retrain_due_to_quality, metric_val, bad_streak

    a = df.iloc[-1].values.astype("float32")
    p = np.array([safe_float(last_pred_vec.get(c)) for c in cols], dtype="float32")

    bad = False
    if os.path.exists(scaler_path):
        try:
            sc = joblib.load(scaler_path)
            a_s = np.clip(sc.transform(a.reshape(1, -1))[0], 0.0, 1.0)
            p_s = np.clip(sc.transform(p.reshape(1, -1))[0], 0.0, 1.0)

            if RETRAIN_METRIC == "mape":
                metric_val = _metric_mape(a_s, p_s)
                bad = metric_val > RETRAIN_Q_MAPE
            else:
                metric_val = _metric_mse(a_s, p_s)
                bad = metric_val > RETRAIN_Q_MSE
        except Exception:
            pass

    if metric_val is not None:
        bad_streak = (bad_streak + 1) if bad else 0
        retrain_due_to_quality = (bad_streak >= RETRAIN_BAD_STREAK)

    return retrain_due_to_quality, metric_val, bad_streak

def _rt_fit_or_load_scaler(X, scaler_path, info, p3_1_log=None):
    if os.path.exists(scaler_path):
        try:
            scaler: MinMaxScaler = joblib.load(scaler_path)
            if getattr(scaler, "scale_", None) is None or scaler.scale_.shape[0] != X.shape[1]:
                scaler = MinMaxScaler().fit(X)
                joblib.dump(scaler, scaler_path)
        except Exception:
            scaler = MinMaxScaler().fit(X)
            joblib.dump(scaler, scaler_path)
    else:
        scaler = MinMaxScaler().fit(X)
        joblib.dump(scaler, scaler_path)

    return scaler

def _rt_update_meta_after_train(meta_state, bad_streak_reset=True):
    meta_state["last_trained_ts"] = _utc_now_iso()
    if bad_streak_reset:
        meta_state["bad_streak"] = 0
    return meta_state

def _rt_update_meta_after_pred(meta_state, cols, y_hat, bad_streak):
    meta_state["last_pred_ts"] = _utc_now_iso()
    meta_state["last_pred_vec"] = {c: float(v) for c, v in zip(cols, y_hat.tolist())}
    meta_state["bad_streak"] = int(bad_streak)
    return meta_state

# ============================================================
# PART 3: Prediction function for single model type
# ============================================================

def _predict_single_type(
    model_type: str,  # "INPUT" or "OUTPUT"
    var_dict: dict,   # Only variables for this type
    scope: str,
    scope_id,
    message: dict,
    group_by_stock: bool,
    lookback: int,
    epochs: int,
    min_train_points: int,
    algorithm: str,
    seed_history,
    p3_1_log,
    resample_seconds: int = None,
) -> Tuple[dict, bool]:
    """
    Run prediction for a single model type (INPUT or OUTPUT).
    
    Returns:
        (payload_dict, success)
    """
    log = (p3_1_log.debug if p3_1_log else print)
    warn = (p3_1_log.warning if p3_1_log else print)
    info = (p3_1_log.info if p3_1_log else print)
    
    algo_name = algorithm or "LSTM"
    
    # Check if we have data
    if not var_dict and not seed_history:
        if p3_1_log:
            p3_1_log.debug(f"[rt_pred_{model_type}] no data, skipping")
        return {}, False
    
    # Generate key for this type
    key = _realtime_model_key_with_type(scope, scope_id, message, group_by_stock, model_type)
    now_ts = _to_utc_pid(message.get("crDt")) or datetime.now(timezone.utc)
    
    if p3_1_log:
        p3_1_log.info(f"[rt_pred_{model_type}] key={key} vars={len(var_dict)}")
    
    # Model paths
    model_path, scaler_path, meta_path = _model_paths(key, algo_name)
    
    # Load state
    meta_state, is_stale, bad_streak, last_pred_vec = _rt_load_state_and_stale(meta_path, p3_1_log)
    
    # Load/update cols
    cols = _rt_load_cols_and_update_meta(meta_path, lookback, seed_history, var_dict, p3_1_log)
    
    # Seed and append buffer
    df_raw = _rt_seed_and_append_buffer(key, cols, seed_history, var_dict, message, p3_1_log)
    df = _rt_prepare_df(df_raw, p3_1_log=p3_1_log, key=key, resample_seconds=resample_seconds)
    df_for_mean = df_raw
    
    # Check if enough data
    min_needed = lookback + 1
    if len(df) < min_needed:
        if p3_1_log:
            p3_1_log.info(f"[rt_pred_{model_type}] EMA: df_len={len(df)} < {min_needed}")
        
        # EMA prediction
        ema_vec = (df.tail(min(len(df), max(5, lookback))).mean(axis=0).values
                   if len(df) else np.zeros(len(cols), dtype="float32"))
        
        actual_d, pred_d, mean_d = _vectors_for_write(df, cols, ema_vec, df_for_mean)
        
        payload = {}
        for var in cols:
            payload[var] = {
                "actual": _sf(actual_d.get(var, 0.0)),
                "predicted": _sf(pred_d.get(var, 0.0)),
                "mean": _sf(mean_d.get(var, 0.0))
            }
        
        return payload, True
    
    # Quality check
    retrain_due_to_quality, metric_val, bad_streak = _rt_quality_check(
        cols, df, scaler_path, last_pred_vec, bad_streak, p3_1_log
    )
    
    # Scaler
    X = df.values.astype("float32")
    scaler = _rt_fit_or_load_scaler(X, scaler_path, info, p3_1_log=p3_1_log)
    Xs = np.clip(scaler.transform(X), 0.0, 1.0)
    
    # Force retrain check
    force_retrain = bool(is_stale or retrain_due_to_quality)
    
    # Load model
    model = None
    need_train = True
    
    try:
        model = _load_model_any(model_path, algo_name, p3_1_log=p3_1_log)
        if model is not None and not force_retrain:
            exp = _expected_flat_features(algo_name, lookback, len(cols))
            got = _model_expected_n_features(model, algo_name)
            if got is not None and got != exp:
                force_retrain = True
                need_train = True
                if p3_1_log:
                    p3_1_log.warning(
                        f"[rt_pred_{model_type}] feature mismatch -> force retrain "
                        f"(model expects={got}, current expects={exp}, cols={len(cols)}, lookback={lookback})"
                    )
            else:
                need_train = False
    except Exception as e:
        if p3_1_log:
            p3_1_log.warning(f"[rt_pred_{model_type}] model load failed: {e}")
        model = None
        need_train = True
    
    # Train if needed
    if need_train:
        model, trained = _train_model_any(
            Xs, cols, lookback, epochs, min_train_points,
            model_path, meta_path, algo_name,
            force_retrain, p3_1_log, info
        )
        
        if not trained or model is None:
            # Fall back to EMA
            ema_vec = df.tail(min(len(df), lookback)).mean(axis=0).values
            actual_d, pred_d, mean_d = _vectors_for_write(df, cols, ema_vec, df_for_mean)
            
            payload = {}
            for var in cols:
                payload[var] = {
                    "actual": _sf(actual_d.get(var, 0.0)),
                    "predicted": _sf(pred_d.get(var, 0.0)),
                    "mean": _sf(mean_d.get(var, 0.0))
                }
            
            return payload, True
        
        # Update meta after training
        meta_state = _rt_update_meta_after_train(meta_state, bad_streak_reset=True)
        _save_meta(meta_path, meta_state)
    
    # Predict
    try:
        y_hat = _predict_next_any(model, Xs, scaler, lookback, algo_name, p3_1_log=p3_1_log)
    except ValueError as e:
        msg = str(e)
        mismatch = ("n_features" in msg) or ("features" in msg and "expecting" in msg)
        if mismatch:
            if p3_1_log:
                p3_1_log.warning(f"[rt_pred_{model_type}] predict mismatch -> retraining now: {e}")

            model, trained = _train_model_any(
                Xs, cols, lookback, epochs, min_train_points,
                model_path, meta_path, algo_name,
                force_retrain=True, p3_1_log=p3_1_log, info=info
            )
            if model is None:
                raise

            meta_state = _rt_update_meta_after_train(meta_state, bad_streak_reset=True)
            _save_meta(meta_path, meta_state)

            y_hat = _predict_next_any(model, Xs, scaler, lookback, algo_name, p3_1_log=p3_1_log)
        else:
            raise
    
    # Build payload
    actual_d, pred_d, mean_d = _vectors_for_write(df, cols, y_hat, df_for_mean)
    
    payload = {}
    for var in cols:
        payload[var] = {
            "actual": _sf(actual_d.get(var, 0.0)),
            "predicted": _sf(pred_d.get(var, 0.0)),
            "mean": _sf(mean_d.get(var, 0.0))
        }
    
    # Update meta after prediction
    meta_state = _rt_update_meta_after_pred(meta_state, cols, y_hat, bad_streak)
    _save_meta(meta_path, meta_state)
    
    # Cleanup
    del df, X, Xs, model
    gc.collect()
    
    return payload, True


# ============================================================
# MAIN FUNCTION - Orchestrates both INPUT and OUTPUT models
# ============================================================

#UPDATECODE   
def handle_realtime_prediction(message: dict,
                               p3_1_log=None,
                               lookback: int = LOOKBACK,
                               epochs: int = EPOCHS,
                               min_train_points: int = MIN_TRAIN_POINTS,
                               algorithm: str = "LSTM",
                               seed_history=None,
                               scope: str = "pid",
                               scope_id=None,
                               group_by_stock: bool = False,
                               retrain: bool = False,
                               resample_seconds: int = None,
                               dry_run: bool = False) -> dict:

    log = (p3_1_log.debug if p3_1_log else print)
    warn = (p3_1_log.warning if p3_1_log else print)
    info = (p3_1_log.info if p3_1_log else print)
    
    status = {"ok": False, "reason": "", "wrote": False, "scope": scope, "scope_id": scope_id}
    algo_name = algorithm or "LSTM"
    
    if p3_1_log:
        p3_1_log.info(
            f"[rt_pred] START scope={scope} scope_id={scope_id} group_by_stock={group_by_stock} "
            f"algo={algo_name} SEPARATE_MODELS mode"
        )
    
    # Extract ALL variables
    #inputs_map, outputs_map, flat_vals = extract_numeric_io(message)
    
    # Split into input and output variables
    #input_vars, output_vars = _split_input_output_vars(flat_vals)
    
    input_vars, output_vars = extract_numeric_io(message)

    # Check what we have
    has_inputs = _has_real_values(input_vars)
    has_outputs = _has_real_values(output_vars)
    
    if p3_1_log:
        p3_1_log.info(
            f"[rt_pred] vars: inputs={len(input_vars)} (real={has_inputs}), "
            f"outputs={len(output_vars)} (real={has_outputs})"
        )
    
    if not has_outputs and not seed_history:
        status["reason"] = "no_outputs"
        warn(f"[rt_pred] skip: no output variables")
        return status
    
    now_ts = _to_utc_pid(message.get("crDt")) or datetime.now(timezone.utc)
    meta = extract_prediction_metadata(message)
    
    # ==================== PREDICT OUTPUTS (always) ====================
    output_payload, output_success = _predict_single_type(
        model_type="OUTPUT",
        var_dict=output_vars,
        scope=scope,
        scope_id=scope_id,
        message=message,
        group_by_stock=group_by_stock,
        lookback=lookback,
        epochs=epochs,
        min_train_points=min_train_points,
        algorithm=algo_name,
        seed_history=seed_history,  # Historical outputs
        p3_1_log=p3_1_log
    )
    
    # ==================== PREDICT INPUTS (only if present) ====================
    input_payload = {}
    input_success = False
    
    if has_inputs:
        input_payload, input_success = _predict_single_type(
            model_type="INPUT",
            var_dict=input_vars,
            scope=scope,
            scope_id=scope_id,
            message=message,
            group_by_stock=group_by_stock,
            lookback=lookback,
            epochs=epochs,
            min_train_points=min_train_points,
            algorithm=algo_name,
            seed_history=None,  # No historical inputs
            p3_1_log=p3_1_log
        )
    else:
        if p3_1_log:
            p3_1_log.info("[rt_pred] skipping INPUT model (no input variables)")
    
    # ==================== WRITE TO CASSANDRA ====================
    try:
        if scope == "pid":
            ref_key = _realtime_model_key_with_type(scope, scope_id, message, group_by_stock, "OUTPUT")
            
            ScadaRealTimePredictions.saveData(
                key=ref_key,
                now_ts=now_ts,
                algorithm=algo_name,
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=p3_1_log
            )
        
        elif scope == "ws":
            ScadaRealTimePredictionSummary.saveData(
                now_ts=now_ts,
                algorithm=algo_name,
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=p3_1_log
            )
        
        elif scope == "batch":
            # Batch-scoped prediction: store like PID-level (partitioned by prod_order_reference_no)
            ref_key = _realtime_model_key_with_type(scope, scope_id, message, group_by_stock, "OUTPUT")

            ScadaRealTimePredictions.saveData(
                key=ref_key,
                now_ts=now_ts,
                algorithm=algo_name,
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=p3_1_log
            )

        status.update({
            "ok": True,
            "wrote": True,
            "reason": f"{algo_name}_separate",
            "input_predicted": has_inputs and input_success,
            "output_predicted": output_success
        })
        
        if p3_1_log:
            p3_1_log.info(
                f"[rt_pred] END {status['reason']} "
                f"(in={len(input_payload)}, out={len(output_payload)})"
            )
    
    except Exception as e:
        if p3_1_log:
            p3_1_log.error(f"[rt_pred] saveData FAILED: {e}", exc_info=True)
        raise
    
    return status