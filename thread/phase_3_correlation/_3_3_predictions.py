import os, gc, json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List
from datetime import datetime, timezone

import joblib
from sklearn.preprocessing import MinMaxScaler

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

from typing import Any, Optional, Tuple

_tf = None
_keras_layers = None
_keras_models = None

def _lazy_tf():
    """
    Import tensorflow/keras only when LSTM is actually needed.
    Returns (tf, layers, models).
    """
    global _tf, _keras_layers, _keras_models
    if _tf is None:
        import tensorflow as tf  # type: ignore
        from tensorflow.keras import layers, models  # type: ignore
        _tf = tf
        _keras_layers = layers
        _keras_models = models
    return _tf, _keras_layers, _keras_models



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

# ------------------ MEAN / BASELINE CONFIG ------------------
MEAN_BASELINE_MODE = str(os.getenv("P3_MEAN_BASELINE_MODE", "ema")).strip().lower()
MEAN_EMA_SPAN = int(os.getenv("P3_MEAN_EMA_SPAN", 30) or 30)                 # points
MEAN_ROLLING_WINDOW = int(os.getenv("P3_MEAN_ROLLING_WINDOW", 30) or 30)     # points


def _baseline_last_from_series(s: np.ndarray, mode: str) -> float:
    """
    Compute a baseline value from a 1D series.
    This is for the UI 'mean' line (baseline curve), not for model features.
    """
    if s is None or len(s) == 0:
        return np.nan

    mode = (mode or "").strip().lower()

    if mode == "ema":
        span = max(2, min(int(MEAN_EMA_SPAN or 30), len(s)))
        return float(pd.Series(s).ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1])

    if mode in ("rolling", "rolling_mean"):
        w = max(2, min(int(MEAN_ROLLING_WINDOW or 30), len(s)))
        return float(pd.Series(s).rolling(window=w, min_periods=1).mean().iloc[-1])

    if mode in ("cummean", "expanding", "cumulative"):
        return float(pd.Series(s).expanding(min_periods=1).mean().iloc[-1])

    if mode == "median":
        w = max(1, min(int(MEAN_ROLLING_WINDOW or 30), len(s)))
        return float(np.nanmedian(s[-w:]))

    if mode == "mean":
        w = max(1, min(int(MEAN_ROLLING_WINDOW or 30), len(s)))
        return float(np.nanmean(s[-w:]))

    # fallback: keep existing behavior (lowess_last)
    return float(lowess_last(s, frac=0.25))


def _baseline_row_from_df(df: pd.DataFrame, cols: List[str]) -> Dict[str, float]:
    """
    Compute baseline values for multiple columns from a DF (last baseline value per col).
    """
    if df is None or len(df) == 0 or not cols:
        return {}

    df_num = df[cols].apply(pd.to_numeric, errors="coerce")
    mode = MEAN_BASELINE_MODE

    try:
        if mode == "ema":
            span = max(2, min(int(MEAN_EMA_SPAN or 30), len(df_num)))
            row = df_num.ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1]
        elif mode in ("rolling", "rolling_mean"):
            w = max(2, min(int(MEAN_ROLLING_WINDOW or 30), len(df_num)))
            row = df_num.rolling(window=w, min_periods=1).mean().iloc[-1]
        elif mode in ("cummean", "expanding", "cumulative"):
            row = df_num.expanding(min_periods=1).mean().iloc[-1]
        elif mode == "median":
            w = max(1, min(int(MEAN_ROLLING_WINDOW or 30), len(df_num)))
            row = df_num.tail(w).median()
        elif mode == "mean":
            w = max(1, min(int(MEAN_ROLLING_WINDOW or 30), len(df_num)))
            row = df_num.tail(w).mean()
        else:
            # per-column fallback (e.g., lowess)
            row = pd.Series({c: _baseline_last_from_series(df_num[c].dropna().values, mode) for c in cols})
    except Exception:
        row = pd.Series({c: _baseline_last_from_series(df_num[c].dropna().values, mode) for c in cols})

    out = {}
    for c in cols:
        try:
            v = float(row.get(c, np.nan))
            out[c] = v
        except Exception:
            out[c] = np.nan
    return out


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
from collections import deque

class SeriesBuffer:
    """
    Fast in-memory time series buffer.

    Key differences vs old version:
    - NO pd.concat per append (avoids O(n^2) growth).
    - Stores rows in deques; builds DataFrame only when requested.
    - Supports get_df_tail(tail_n) so RF can resample only a tail window.
    """

    def __init__(self, maxlen: int = 5000):
        self.maxlen = int(maxlen)
        self._ts = deque(maxlen=self.maxlen)     # pandas.Timestamp
        self._rows = deque(maxlen=self.maxlen)   # dict of {feature: value}
        self.first_ts = None

        self._df_cache = None
        self._cache_dirty = True

    def __len__(self):
        return len(self._ts)

    @property
    def df(self):
        """Full DataFrame view (built lazily)."""
        import pandas as pd

        if (self._df_cache is None) or self._cache_dirty:
            if not self._ts:
                self._df_cache = pd.DataFrame()
            else:
                self._df_cache = pd.DataFrame(list(self._rows), index=pd.DatetimeIndex(list(self._ts)))
            self._cache_dirty = False

        return self._df_cache

    def append_row(self, ts, values: dict):
        import pandas as pd
        from datetime import datetime, timezone

        if ts is None:
            ts = datetime.now(timezone.utc)

        ts = pd.Timestamp(ts)

        # ---- Normalize to tz-aware UTC (prevents tz-naive vs tz-aware compare errors) ----
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        # Ensure strictly increasing timestamps
        if self._ts:
            last_ts = pd.Timestamp(self._ts[-1])
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            else:
                last_ts = last_ts.tz_convert("UTC")

            if ts <= last_ts:
                ts = last_ts + pd.Timedelta(microseconds=1)

        self._ts.append(ts)
        self._rows.append(values)

        if self.first_ts is None:
            self.first_ts = ts.to_pydatetime()

        self._cache_dirty = True


    def get_df_tail(self, tail_n: int):
        """Tail DataFrame view (built from last tail_n rows only)."""
        import pandas as pd

        n = int(tail_n) if tail_n else len(self._ts)
        if n <= 0 or not self._ts:
            return pd.DataFrame()

        ts_list = list(self._ts)[-n:]
        rows_list = list(self._rows)[-n:]
        return pd.DataFrame(rows_list, index=pd.DatetimeIndex(ts_list))




_buffers: Dict[str, SeriesBuffer] = defaultdict(lambda: SeriesBuffer(maxlen=5000))

# ============================================================
# NEW: Helper functions for separate models
# ============================================================

# RF caches (avoid re-loading for every message)
_RF_MODEL_CACHE = {}   # key: base_path -> {"model":..., "scaler":..., "meta":...}
_BATCH_START_MAP = {}  # key: (drug_key, batch_id) -> datetime
_RF_BUFFERS = defaultdict(lambda: SeriesBuffer(maxlen=20000))  # key: buffer_key -> SeriesBuffer
_RF_BUFFERS = defaultdict(lambda: SeriesBuffer(maxlen=20000))  # key: buffer_key -> SeriesBuffer

# Tracks whether RF buffer has been seeded for a given (buffer_key, resample_seconds)
_RF_SEEDED = set()

def _rf_seed_buffer_once(*, message: dict, seed_history, resample_seconds: int, p3_1_log=None) -> int:
    """
    Seed RF internal buffer ONCE using seed_history from history_from_fetch().
    seed_history format: list[(ts, {sensor_key: value, ...}), ...]

    Must use the SAME buffer_key scheme as _rf_predict_from_message():
        buffer_key = f"OPTC_{op_tc}_WS_{ws_id}_ST_{st_no}" then normalized via _norm_str_rf()

    Returns: number of points inserted.
    """
    if not seed_history:
        return 0

    op_tc = message.get("opTc") or message.get("operationtaskcode") or "UNKNOWN"
    ws_id = message.get("wsId") or "UNKNOWN"
    st_no = _get_stock_no_from_message(message)

    buffer_key = f"OPTC_{op_tc}_WS_{ws_id}_ST_{st_no}"
    buffer_key = _norm_str_rf(buffer_key)

    seed_key = (buffer_key, int(resample_seconds or 60))
    if seed_key in _RF_SEEDED:
        return 0

    buf = _RF_BUFFERS[buffer_key]
    inserted = 0

    for ts, row in seed_history:
        if not row or not isinstance(row, dict):
            continue
        try:
            buf.append_row(pd.Timestamp(ts), row)
            inserted += 1
        except Exception:
            continue

    _RF_SEEDED.add(seed_key)
    if p3_1_log:
        p3_1_log.info(f"[rt_pred] RF seeded buffer_key={buffer_key} inserted={inserted}")
    return inserted


def _norm_str_rf(x: str) -> str:
    x = str(x or "").strip()
    x = x.replace(" ", "_").replace("-", "_").replace("/", "_")
    return "".join(ch for ch in x if ch.isalnum() or ch == "_")

def _get_drug_name_from_message(message: dict) -> str:
    # Prefer normalized metadata already created in phase3
    if message.get("output_stock_name"):
        return str(message["output_stock_name"])
    # Fallbacks
    prod_list = message.get("prodList") or []
    if prod_list and isinstance(prod_list, list):
        st_nm = prod_list[0].get("stNm") or prod_list[0].get("stName")
        if st_nm:
            return str(st_nm)
    # Worst-case
    return "UNKNOWN"

def _get_stock_no_from_message(message: dict) -> str:
    if message.get("output_stock_no"):
        return str(message["output_stock_no"])
    prod_list = message.get("prodList") or []
    if prod_list and isinstance(prod_list, list):
        st_no = prod_list[0].get("stNo")
        if st_no:
            return str(st_no)
    return "UNKNOWN"

def _get_batch_id_from_message(message: dict) -> str:
    # Your canonical batch id
    b = message.get("prod_order_reference_no") or message.get("job_order_reference_no")
    if b not in (None, "", "None", 0, "0"):
        return str(b)
    # Fallbacks from kafka
    if message.get("refNo") not in (None, "", "None", 0, "0"):
        return str(message["refNo"])
    if message.get("joRef") not in (None, "", "None", 0, "0"):
        return str(message["joRef"])
    return "UNKNOWN_BATCH"

def _rf_model_basepath_for_drug(drug_name: str, model_type: str = "OUTPUT") -> str:
    safe_drug = _norm_str_rf(drug_name)
    # matches your artifact names:
    # models/DRUG_Antares_PV_OUTPUT__RANDOM_FOREST.pkl
    return os.path.join(MODELS_DIR, f"DRUG_{safe_drug}_{model_type}__RANDOM_FOREST")

def _rf_load_bundle(drug_name: str, model_type: str = "OUTPUT"):
    base = _rf_model_basepath_for_drug(drug_name, model_type=model_type)

    if base in _RF_MODEL_CACHE:
        return _RF_MODEL_CACHE[base]["model"], _RF_MODEL_CACHE[base]["scaler"], _RF_MODEL_CACHE[base]["meta"], base

    model_path = f"{base}.pkl"
    scaler_path = f"{base}_scaler.pkl"
    meta_path = f"{base}_meta.json"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(meta_path):
        return None, None, None, base

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    _RF_MODEL_CACHE[base] = {"model": model, "scaler": scaler, "meta": meta}
    return model, scaler, meta, base

def _rf_prepare_df_from_buffer(
    df_buf: pd.DataFrame,
    cols: list,
    batch_start: datetime,
    use_time_elapsed: bool,
    resample_seconds: int,
    resample_method: str = "mean",
    inactive_strategy: str = "FFILL",
) -> pd.DataFrame:
    """
    FAST preparation:
    - Assumes caller already provided a tail DF (not full history).
    - Aligns column names to training meta cols using normalization so "actual" doesn't become 0.
    """
    if df_buf is None or df_buf.empty:
        return pd.DataFrame()

    df = df_buf

    # Ensure time order only if needed
    try:
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
    except Exception:
        df = df.sort_index()

    # Drop duplicate timestamps (keep last)
    df = df[~df.index.duplicated(keep="last")]

    rule = f"{int(resample_seconds)}s"  # lowercase 's' avoids pandas FutureWarning

    if str(resample_method).lower() == "last":
        df_rs = df.resample(rule).last()
    else:
        df_rs = df.resample(rule).mean()

    # Fill strategy on existing columns (do NOT reindex-fill to 0 yet)
    if str(inactive_strategy).upper() == "ZERO":
        df_rs = df_rs.fillna(0.0)
    else:
        df_rs = df_rs.ffill().bfill()

    # --------- Column alignment: map df_rs columns -> expected cols by normalization ---------
    # Example: "Total Air Flow" vs "Total_Air_Flow" should match.
    expected = list(cols or [])
    if not expected:
        # fall back to whatever exists
        df_ready = df_rs.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    else:
        exp_norm = { _norm_key(c): c for c in expected }
        rename_map = {}

        matched = 0
        for c in list(df_rs.columns):
            cn = _norm_key(c)
            tgt = exp_norm.get(cn)
            if tgt is not None:
                rename_map[c] = tgt
                matched += 1

        if rename_map:
            df_rs = df_rs.rename(columns=rename_map)

        # Now reindex to expected cols (missing stay NaN initially)
        df_ready = df_rs.reindex(columns=expected)

        # Fill remaining missing values
        if str(inactive_strategy).upper() == "ZERO":
            df_ready = df_ready.fillna(0.0)
        else:
            df_ready = df_ready.ffill().bfill().fillna(0.0)

        # Debug info (optional, but super useful)
        # (Only log if p3_1_log exists in your callers — if not, ignore.)
        # You can add a logger param to this func if you want; I kept signature unchanged.

    # Add meta_time_elapsed if requested
    if use_time_elapsed and len(df_ready) > 0:
        if batch_start is None:
            batch_start = df_ready.index[0].to_pydatetime()

        idx = df_ready.index
        start_ts = batch_start

        # timezone align
        if getattr(idx, "tz", None) is not None and start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=idx.tz)
        elif getattr(idx, "tz", None) is None and start_ts.tzinfo is not None:
            idx = idx.tz_localize(start_ts.tzinfo)
            df_ready.index = idx

        elapsed_min = (df_ready.index - pd.Timestamp(start_ts)).total_seconds() / 60.0
        df_ready["meta_time_elapsed"] = np.clip(elapsed_min, 0, None)

    df_ready = df_ready.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df_ready




def _rf_predict_from_message(
    message: dict,
    p3_1_log=None,
    resample_seconds: int = 60,
    model_type: str = "OUTPUT"
):
    """
    Returns: (payload_dict, success_bool, status_reason)

    Improvement:
    - Allows early predictions BEFORE lookback is full by padding the window.
      (prevents "no prediction" on newly-started batches)
    """
    try:
        drug_name = _get_drug_name_from_message(message)
        batch_id = _get_batch_id_from_message(message)

        model, scaler, meta, base = _rf_load_bundle(drug_name, model_type=model_type)
        if model is None:
            return {}, False, f"rf_model_missing:{base}"

        cols = meta.get("cols", []) or []
        lookback = int(meta.get("timesteps", meta.get("lookback", 20)) or 20)

        cfg = meta.get("config", {}) if isinstance(meta.get("config", {}), dict) else {}
        use_time_elapsed = bool(cfg.get("USE_TIME_ELAPSED", True))
        inactive_strategy = str(cfg.get("INACTIVE_STRATEGY", "FFILL"))
        resample_method = "mean"

        # Key = opTc + wsId + stock to avoid mixing contexts
        op_tc = message.get("opTc") or message.get("operationtaskcode") or "UNKNOWN"
        ws_id = message.get("wsId") or "UNKNOWN"
        st_no = _get_stock_no_from_message(message)
        buffer_key = f"OPTC_{op_tc}_WS_{ws_id}_ST_{st_no}"
        buffer_key = _norm_str_rf(buffer_key)

        # Timestamp
        ts_str = message.get("crDt")
        try:
            if not ts_str:
                ts = datetime.now(timezone.utc)
            elif str(ts_str).endswith("Z"):
                ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
            else:
                ts = datetime.fromisoformat(str(ts_str))
        except Exception:
            ts = datetime.now(timezone.utc)

        # Extract realtime values
        row_vals = {}

        out_vals = message.get("outVals") or []
        for outv in out_vals:
            try:
                eq = outv.get("eqNm")
                val = outv.get("cntRead")
                if eq is None:
                    continue
                # IMPORTANT: normalize keys to match training/meta cols
                k = _norm_key(eq)
                row_vals[k] = float(val) if val not in (None, "", "None") else 0.0
            except Exception:
                continue

        if not row_vals and isinstance(message.get("out_vars_dict"), dict):
            for k, v in message["out_vars_dict"].items():
                try:
                    row_vals[_norm_key(k)] = float(v)
                except Exception:
                    pass

        if not row_vals:
            return {}, False, "rf_no_values"

        # Buffer update
        buf = _RF_BUFFERS[buffer_key]
        buf.append_row(ts, row_vals)

        # Tail view for speed
        margin = 50
        tail_rows = int(max(600, min(6000, (lookback + margin) * 10)))
        df_buf_tail = buf.get_df_tail(tail_rows)

        # Batch start tracking (for meta_time_elapsed)
        drug_key = _norm_str_rf(drug_name)
        batch_key = (drug_key, str(batch_id))
        if batch_key not in _BATCH_START_MAP:
            _BATCH_START_MAP[batch_key] = ts
        batch_start = _BATCH_START_MAP[batch_key]

        df_ready = _rf_prepare_df_from_buffer(
            df_buf=df_buf_tail,
            cols=cols,
            batch_start=batch_start,
            use_time_elapsed=use_time_elapsed,
            resample_seconds=int(resample_seconds or 60),
            resample_method=resample_method,
            inactive_strategy=inactive_strategy,
        )

        n = len(df_ready)
        if n <= 0:
            return {}, False, "rf_no_ready_rows"

        # ------------------ EARLY PREDICTION SUPPORT ------------------
        # If we have fewer than lookback points, pad by repeating the earliest row.
        early = False
        if n < lookback:
            early = True
            pad_rows = lookback - n
            first_row = df_ready.iloc[[0]]
            pads = [first_row] * pad_rows
            df_window = pd.concat(pads + [df_ready], axis=0)
            df_window = df_window.iloc[-lookback:]
        else:
            df_window = df_ready.iloc[-lookback:]

        X_window = df_window.values  # (T,F)

        # Scale per-timestep features
        X_scaled = scaler.transform(X_window)
        X_scaled = np.clip(X_scaled, -10.0, 10.0)

        X_flat = X_scaled.reshape(1, -1)         # (1, T*F)
        y_pred_scaled = model.predict(X_flat)    # (1, F)
        y_pred = scaler.inverse_transform(y_pred_scaled)[0]  # (F,)

        # Actuals from latest observed row
        current_actuals = df_ready.iloc[-1].values.astype(float)
        col_names = list(df_ready.columns)

        # Compute baseline ("mean") for UI once per call
        mean_cols = [c for c in col_names if c != "meta_time_elapsed"]
        mean_map = _baseline_row_from_df(df_ready, mean_cols)

        payload = {}
        for i, col in enumerate(col_names):
            if col == "meta_time_elapsed":
                continue
            act = float(current_actuals[i])
            pred = float(y_pred[i])
            mval = float(mean_map.get(col, act))  # fallback to act if something is missing
            payload[col] = {"actual": act, "predicted": pred, "mean": mval}


        if early:
            return payload, True, f"rf_ok_early_padded:{n}/{lookback}"
        return payload, True, "rf_ok"

    except Exception as e:
        if p3_1_log:
            p3_1_log.error(f"[rf_pred] error: {e}", exc_info=True)
        return {}, False, f"rf_error:{e}"


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

def _build_model(n_steps: int, n_features: int):
    tf, layers, models = _lazy_tf()
    
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
    
    mm = (mean_mode or "").strip().lower()

    for c in cols:
        s = df_num[c].dropna().values
        if len(s) == 0:
            means[c] = np.nan
            continue

        # Window limiting (keep for stability)
        if lowess_window and len(s) > lowess_window:
            s = s[-lowess_window:]

        if mm in ("ema", "rolling", "rolling_mean", "cummean", "expanding", "cumulative", "median", "mean"):
            means[c] = _baseline_last_from_series(s, mm)
        else:
            means[c] = lowess_last(s, frac=lowess_frac)


    return actual, predicted, means

def history_from_fetch(dates, sensor_values):
    """
    Tolerant history extraction.

    Goal: produce hist_out even if equipment_type is missing.
    Output format:
      hist_out = [(ts, {sensor_key: value, ...}), ...]
      hist_in  = [(ts, {sensor_key: value, ...}), ...] (optional)

    Rules:
    - If equipment_type is True -> INPUT
    - If equipment_type is False -> OUTPUT
    - If equipment_type missing/None -> treat as OUTPUT if a numeric reading exists
      (because DW/RAW bundles often don't carry equipment_type reliably)
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

            et = s.get("equipment_type")

            # Try to read values (support multiple field names)
            val_out = (
                s.get("counter_reading")
                or s.get("cntRead")
                or s.get("value")
                or s.get("gen_read_val")
                or s.get("genReadVal")
            )

            # If no value at all, skip
            if val_out in (None, "", "None"):
                continue

            fv = safe_float(val_out)

            if et is True:
                row_in[k] = fv
            elif et is False:
                row_out[k] = fv
            else:
                # equipment_type missing -> treat as OUTPUT for seeding RF
                row_out[k] = fv

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

    algo = (algorithm or "LSTM").strip().upper()
    if p3_1_log:
        p3_1_log.info(f"[rt_pred] model load try algo={algo} path={model_path}")

    if not os.path.exists(model_path):
        return None

    if algo == "LSTM":
        tf, _, _ = _lazy_tf()
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

def _dump_realtime_results_csv(*, base_dir: str, meta: dict, output_payload: dict, message: dict, p3_1_log=None):
    """
    Append per-sensor rows to a local CSV for debugging:
      ts,batch,drug,opTc,wsId,sensor,actual,predicted,raw_from_kafka
    """
    try:
        os.makedirs(base_dir, exist_ok=True)
        out_path = os.path.join(base_dir, "results.csv")

        # Raw kafka map (best-effort)
        raw_map = {}
        for o in (message.get("outVals") or []):
            try:
                k = _norm_key(o.get("eqNm"))
                raw_map[k] = o.get("cntRead")
            except Exception:
                continue

        ts = meta.get("start_date") or meta.get("ts") or message.get("crDt") or ""
        batch = meta.get("prod_order_reference_no") or message.get("prod_order_reference_no") or message.get("refNo") or ""
        drug = meta.get("output_stock_name") or message.get("output_stock_name") or ""
        opTc = meta.get("operationtaskcode") or message.get("opTc") or ""
        wsId = meta.get("workstation_no") or message.get("wsId") or ""

        header_needed = not os.path.exists(out_path)
        with open(out_path, "a", encoding="utf-8") as f:
            if header_needed:
                f.write("ts,batch,drug,opTc,wsId,sensor,actual,predicted,raw_from_kafka\n")
            for sensor, d in (output_payload or {}).items():
                act = d.get("actual", "")
                pred = d.get("predicted", "")
                raw = raw_map.get(_norm_key(sensor), "")
                f.write(f"{ts},{batch},{drug},{opTc},{wsId},{sensor},{act},{pred},{raw}\n")

        if p3_1_log:
            p3_1_log.info(f"[rt_pred][debug] Wrote results.csv -> {out_path}")
    except Exception as e:
        if p3_1_log:
            p3_1_log.warning(f"[rt_pred][debug] CSV dump failed: {e}")


#UPDATECODE
def _rt_prepare_df(df_raw, p3_1_log=None, key=None, resample_seconds: int = 60, resample_method: str = "last"):
    """
    Clean + resample to fixed cadence (default 60s) using last().

    Safe if resample_seconds is None.
    """
    import pandas as pd

    if resample_seconds is None:
        resample_seconds = 60
    resample_seconds = int(resample_seconds)

    if df_raw is None or len(df_raw) == 0:
        return pd.DataFrame()

    df = df_raw.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            return df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    rule = f"{int(resample_seconds)}s"

    try:
        if str(resample_method).lower() == "mean":
            df = df.resample(rule).mean()
        else:
            df = df.resample(rule).last()
    except Exception as e:
        if p3_1_log:
            p3_1_log.warning(f"[rt_pred] resample failed key={key}: {e}")

    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
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

def _predict_single_type_lstm_existing(
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
        
        actual_d, pred_d, mean_d = _vectors_for_write(df, cols, y_hat, df_for_mean, mean_mode=MEAN_BASELINE_MODE)

        
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
            actual_d, pred_d, mean_d = _vectors_for_write(df, cols, y_hat, df_for_mean, mean_mode=MEAN_BASELINE_MODE)

            
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
    actual_d, pred_d, mean_d = _vectors_for_write(df, cols, y_hat, df_for_mean, mean_mode=MEAN_BASELINE_MODE)

    
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


def _predict_single_type(
    model_type: str,
    var_dict: dict,
    scope: str,
    scope_id,
    message: dict,
    group_by_stock: bool,
    lookback: int,
    epochs: int,
    min_train_points: int,
    algorithm: str,
    seed_history,
    p3_1_log=None,
    resample_seconds: int = 60,
):
    """
    Returns: (payload_dict, success_bool)

    Supports:
    - LSTM (existing logic)
    - RANDOM_FOREST (drug-based artifacts in /models)

    FIXED:
    - RF now consumes seed_history (from history_from_fetch) to avoid rf_gathering_data:1/20
    """
    algo = (algorithm or "LSTM").upper().strip()

    # ---------------- RANDOM_FOREST ----------------
    # We only enable RF for OUTPUT right now.
    if algo in ("RANDOM_FOREST", "RF"):
        if model_type.upper() != "OUTPUT":
            if p3_1_log:
                p3_1_log.info("[rt_pred] RF: INPUT model_type not enabled; skipping")
            return {}, False

        rs = int(resample_seconds or 60)

        # ✅ NEW: seed RF buffer once using history provided by _run_3_tasks_and_wait()
        try:
            _rf_seed_buffer_once(
                message=message,
                seed_history=seed_history,
                resample_seconds=rs,
                p3_1_log=p3_1_log,
            )
        except Exception as e:
            if p3_1_log:
                p3_1_log.warning(f"[rt_pred] RF seeding failed (continuing): {e}")

        payload, ok, reason = _rf_predict_from_message(
            message=message,
            p3_1_log=p3_1_log,
            resample_seconds=rs,
            model_type="OUTPUT",
        )
        if p3_1_log:
            p3_1_log.info(f"[rt_pred] RF status={reason} payload_items={len(payload)}")
        return payload, bool(ok)

    # ---------------- LSTM / others ----------------
    return _predict_single_type_lstm_existing(
        model_type=model_type,
        var_dict=var_dict,
        scope=scope,
        scope_id=scope_id,
        message=message,
        group_by_stock=group_by_stock,
        lookback=lookback,
        epochs=epochs,
        min_train_points=min_train_points,
        algorithm=algorithm,
        seed_history=seed_history,
        p3_1_log=p3_1_log,
        resample_seconds=resample_seconds,
    )



# ============================================================
# MAIN FUNCTION - Orchestrates both INPUT and OUTPUT models
# ============================================================

#UPDATECODE   
def handle_realtime_prediction(
    message: dict,
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
    dry_run: bool = False
) -> dict:
    """
    Realtime prediction entry point.

    Guarantees:
    - resample_seconds defaults safely to config.json (fallback 60)
    - dry_run=True returns predictions WITHOUT any Cassandra writes
    - NON-dry_run: writes ONLY if a non-empty payload is produced (prevents empty rows)
    """
    algo_name = (algorithm or "LSTM").strip()

    def _log_info(msg):
        if p3_1_log:
            p3_1_log.info(msg)

    def _log_warn(msg):
        if p3_1_log:
            p3_1_log.warning(msg)

    def _log_err(msg):
        if p3_1_log:
            p3_1_log.error(msg)

    # -------------------- resample_seconds default --------------------
    if resample_seconds is None:
        try:
            from utils.config_reader import ConfigReader
            cfg = ConfigReader()
            # some repos store it as top-level; fallback to 60
            resample_seconds = int(getattr(cfg, "resample_seconds", 60) or 60)
        except Exception:
            resample_seconds = 60
    else:
        resample_seconds = int(resample_seconds)

    # -------------------- extract vars --------------------
    input_vars, output_vars = extract_numeric_io(message)
    has_inputs = _has_real_values(input_vars)
    has_outputs = _has_real_values(output_vars)

    if not has_outputs and not seed_history:
        return {
            "ok": False,
            "wrote": False,
            "reason": "no_outputs",
            "scope": scope,
            "scope_id": scope_id,
        }

    now_ts = _to_utc_pid(message.get("crDt")) or datetime.now(timezone.utc)
    meta = extract_prediction_metadata(message)

    _log_info(
        f"[rt_pred] START scope={scope} scope_id={scope_id} "
        f"group_by_stock={group_by_stock} algo={algo_name} resample_seconds={resample_seconds}"
    )

    # -------------------- OUTPUT prediction --------------------
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
        seed_history=seed_history,  # IMPORTANT: RF must use this to seed
        p3_1_log=p3_1_log,
        resample_seconds=resample_seconds,
    )

    # -------------------- INPUT prediction (optional) --------------------
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
            seed_history=None,
            p3_1_log=p3_1_log,
            resample_seconds=resample_seconds,
        )
    else:
        _log_info("[rt_pred] skipping INPUT model (no input variables)")

    output_predicted = bool(output_payload) and bool(output_success)
    input_predicted = bool(input_payload) and bool(input_success)

    # -------------------- DRY RUN (NO CASSANDRA) --------------------
    if dry_run:
        _log_info(
            f"[rt_pred] END {algo_name}_dry_run "
            f"(out={len(output_payload)} in={len(input_payload)})"
        )
        return {
            "ok": True,
            "wrote": False,
            "reason": f"{algo_name}_dry_run",
            "scope": scope,
            "scope_id": scope_id,
            "resample_seconds": resample_seconds,
            "output_predicted": output_predicted,
            "input_predicted": input_predicted,
            "output_payload": output_payload,
            "input_payload": input_payload,
            "meta": meta,
        }

    # -------------------- DO NOT WRITE EMPTY PREDICTIONS --------------------
    if not output_predicted and not input_predicted:
        _log_info(
            f"[rt_pred] END {algo_name}_no_prediction "
            f"(out={len(output_payload)} in={len(input_payload)}) -> Cassandra write skipped"
        )
        return {
            "ok": True,
            "wrote": False,
            "reason": f"{algo_name}_no_prediction",
            "scope": scope,
            "scope_id": scope_id,
            "resample_seconds": resample_seconds,
            "output_predicted": output_predicted,
            "input_predicted": input_predicted,
        }

    # DEBUG: local dump (set config flag or env var)
    try:
        if os.environ.get("ABDML_RT_DUMP", "0") == "1":
            dump_dir = os.path.join("models", "realtime_debug", str(meta.get("output_stock_name") or "UNKNOWN").replace(" ", "_"))
            _dump_realtime_results_csv(base_dir=dump_dir, meta=meta, output_payload=output_payload, message=message, p3_1_log=p3_1_log)
    except Exception:
        pass

    # -------------------- WRITE TO CASSANDRA --------------------
    try:
        if scope in ("pid", "batch"):
            ref_key = _realtime_model_key_with_type(
                scope, scope_id, message, group_by_stock, "OUTPUT"
            )

            ScadaRealTimePredictions.saveData(
                key=ref_key,
                now_ts=now_ts,
                algorithm=algo_name,
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=p3_1_log,
            )

        elif scope == "ws":
            ScadaRealTimePredictionSummary.saveData(
                now_ts=now_ts,
                algorithm=algo_name,
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=p3_1_log,
            )
        else:
            _log_warn(f"[rt_pred] Unknown scope='{scope}', skipping Cassandra writes.")
            return {
                "ok": True,
                "wrote": False,
                "reason": f"{algo_name}_unknown_scope",
                "scope": scope,
                "scope_id": scope_id,
            }

        _log_info(
            f"[rt_pred] END {algo_name}_saved "
            f"(out={len(output_payload)} in={len(input_payload)})"
        )

        return {
            "ok": True,
            "wrote": True,
            "reason": f"{algo_name}_saved",
            "scope": scope,
            "scope_id": scope_id,
            "resample_seconds": resample_seconds,
            "output_predicted": output_predicted,
            "input_predicted": input_predicted,
        }

    except Exception as e:
        _log_err(f"[rt_pred] saveData FAILED: {e}")
        if p3_1_log:
            p3_1_log.error("[rt_pred] Exception details:", exc_info=True)
        raise