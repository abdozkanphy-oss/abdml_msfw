import os, gc, json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Tuple, List
from datetime import datetime, timezone

import joblib
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import layers, models

from cassandra_utils.models.scada_real_time_predictions import (
    ScadaRealTimePredictions, _to_utc as _to_utc_pid, _sf as _sf_pid
)
from cassandra_utils.models.scada_real_time_prediction_summary import (
    ScadaRealTimePredictionSummary
)

# ----------------- constants -----------------
MODELS_DIR = "./models"
LOOKBACK   = 20            # default window length
EPOCHS     = 3             # quick training
MIN_TRAIN_POINTS = 120     # min points before training
BATCH_SIZE = 64

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------- buffers -------------------
class SeriesBuffer:
    def __init__(self, maxlen: int = 5000):
        self.df = pd.DataFrame()
        self.maxlen = maxlen
        self.first_ts = None

    def append_row(self, ts: datetime, values: Dict[str, float]):
        idx = len(self.df)
        self.df = pd.concat([self.df, pd.DataFrame([values], index=[idx])], axis=0)
        if len(self.df) > self.maxlen:
            self.df = self.df.tail(self.maxlen)
        if self.first_ts is None:
            self.first_ts = ts

_buffers: Dict[str, SeriesBuffer] = defaultdict(lambda: SeriesBuffer(maxlen=5000))

# ----------------- utils ---------------------

def _realtime_model_key(scope: str, scope_id, message: dict) -> str:
    """
    Stable key for model/buffer per scope:
      - pid: 'PID_<pid>'
      - ws:  'WS_<wsId or wsNo>'
    Fallback to old logic if scope/scope_id missing.
    """
    if scope == "pid" and scope_id is not None:
        return f"PID_{scope_id}"
    if scope == "ws" and scope_id is not None:
        return f"WS_{scope_id}"

    # fallback: old behavior
    return extract_realtime_key(message)


def _model_paths(key: str):
    base = os.path.join(MODELS_DIR, key.replace("/", "_"))
    return base + ".keras", base + "_scaler.pkl", base + "_meta.json"


def _build_model(n_steps: int, n_features: int) -> tf.keras.Model:
    m = models.Sequential([
        layers.Input(shape=(n_steps, n_features)),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(n_features)
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


def safe_float(x):
    try:
        if x is None:
            return np.nan
        s = str(x).replace(",", ".")
        return float(s)
    except Exception:
        return np.nan


def extract_realtime_key(message: dict) -> str:
    """Old key logic; used only as fallback."""
    return (message.get("process_no")
            or str(message.get("joOpId"))
            or message.get("wsNo")
            or message.get("wsId")
            or "")


def extract_numeric_io(message: dict) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Build inputs/outputs as in_* / out_*; flattened dict used by the model.

    - inputVariableList / inVars:
        name:  varNm / name / param / eqNo / id
        value: genReadVal / actVal / value / cntRead
    - outputValueList / outVals:
        name:  eqNm / parameter / eqNo / param
        value: cntRead / genReadVal / value / counter_reading
    """
    inputs_map: Dict[str, float] = {}
    outputs_map: Dict[str, float] = {}

    # ---------- INPUTS ----------
    for iv in (message.get("inputVariableList") or message.get("inVars") or []):
        if not isinstance(iv, dict):
            continue

        nm = (
            iv.get("varNm")
            or iv.get("varNo")
            or iv.get("param")
            or iv.get("eqNo")
            or iv.get("varId")
        )
        key = f"in_{nm}".replace(" ", "_") if nm else None

        if "genReadVal" in iv:
            val = iv.get("genReadVal")
        elif "actVal" in iv:
            val = iv.get("actVal")
        elif "value" in iv:
            val = iv.get("value")
        elif "cntRead" in iv:
            val = iv.get("cntRead")
        else:
            val = None

        if key:
            inputs_map[key] = safe_float(val)

    # ---------- OUTPUTS ----------
    for ov in (message.get("outputValueList") or message.get("outVals") or []):
        if not isinstance(ov, dict):
            continue

        nm = (
            ov.get("eqNo")
            or ov.get("parameter")
            or ov.get("eqNm")
            or ov.get("param")
        )

        key = f"out_{nm}".replace(" ", "_") if nm else None

        if "cntRead" in ov:
            val = ov.get("cntRead")
        elif "genReadVal" in ov:
            val = ov.get("genReadVal")
        elif "value" in ov:
            val = ov.get("value")
        elif "counter_reading" in ov:
            val = ov.get("counter_reading")
        else:
            val = None

        if key:
            outputs_map[key] = safe_float(val)

    flat = {**inputs_map, **outputs_map}
    return inputs_map, outputs_map, flat


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


def _vectors_for_write(df: pd.DataFrame, cols: List[str], y_hat: np.ndarray):
    """
    Build dicts per column:
      - actual:   last observed value in df
      - predicted: from model/EMA (y_hat)
      - mean:     mean over the ENTIRE buffer df
    """
    # actuals (last row)
    actual = {}
    if len(df):
        last = df.iloc[-1]
        for c in cols:
            actual[c] = float(last.get(c, np.nan))
    else:
        for c in cols:
            actual[c] = float("nan")

    # predicted
    predicted = {c: float(v) for c, v in zip(cols, y_hat)}

    # buffer means
    df_means = (df.mean(axis=0) if len(df) else pd.Series({c: np.nan for c in cols}))
    means = {c: float(df_means.get(c, np.nan)) for c in cols}

    return actual, predicted, means


def _build_payloads(cols: List[str],
                    actual_vec: dict,
                    pred_vec: dict,
                    mean_vec: dict,
                    inputs_map: dict,
                    outputs_map: dict):
    """
    Build Cassandra payloads:
      input_predicted_values  = { var: {"actual": a, "predicted": p, "mean": m}, ... }
      output_predicted_values = same structure for output vars.
    """
    def _pack(var_names):
        out = {}
        for var in var_names:
            a = _sf(actual_vec.get(var))
            p = _sf(pred_vec.get(var))
            m = _sf(mean_vec.get(var))
            out[var] = {"actual": a, "predicted": p, "mean": m}
        return out

    input_vars  = list(inputs_map.keys())
    output_vars = list(outputs_map.keys())

    input_payload  = _pack(input_vars)
    output_payload = _pack(output_vars)

    return input_payload, output_payload


def history_from_fetch(dates, sensor_values):
    """
    Convert (dates, sensor_values) to a list[(ts, row_dict)] in model format.
    Each row_dict uses keys like 'out_<sensor-name>' with float values.

    sensor_values: list of bundles:
      [ meta, sensor1, sensor2, ... ]
    where sensorX has keys: parameter, counter_reading, equipment_name.
    """
    out = []
    for ts, bundle in zip(dates, sensor_values):
        row = {}
        if not bundle or len(bundle) < 2:
            continue

        for s in bundle[1:]:
            if not isinstance(s, dict):
                continue
            p = (s.get("parameter") or "").strip()
            eq_name = s.get("equipment_name") or ""
            # Korelasyondaki gibi isim: "<parameter>-<equipment_name>" veya sadece "<parameter>"
            if p:
                base = p
                if eq_name:
                    base = f"{p}-{eq_name}"
                key = f"out_{base}".replace(" ", "_")
                v = safe_float(s.get("counter_reading"))
                row[key] = v

        if row:
            out.append((ts, row))
    return out



# ----------------- main entry -----------------
def handle_realtime_prediction(message: dict,
                               p3_1_log=None,
                               lookback: int = LOOKBACK,
                               epochs: int = EPOCHS,
                               min_train_points: int = MIN_TRAIN_POINTS,
                               algorithm: str = "LSTM",
                               seed_history=None,
                               scope: str = "pid",
                               scope_id=None) -> dict:
    """
    Online LSTM with PID/WS scopes.

    - scope="pid": models and predictions are process-based (per joOpId).
      * LSTM key:  'PID_<pid>'
      * writes to: ScadaRealTimePredictions

    - scope="ws": models and predictions are workstation-based (aggregated).
      * LSTM key:  'WS_<wsId or wsNo>'
      * writes to: ScadaRealTimePredictionSummary

    If model exists for that key → just predict.
    If not, train from corresponding buffer (seed_history + streaming rows)
    when enough points; otherwise EMA bootstrap.
    """
    log  = (p3_1_log.debug if p3_1_log else print)
    warn = (p3_1_log.warning if p3_1_log else print)
    info = (p3_1_log.info if p3_1_log else print)

    status = {"ok": False, "reason": "", "wrote": False, "scope": scope, "scope_id": scope_id}

    # --- 1) Key & time ---
    key = _realtime_model_key(scope, scope_id, message)
    if not key:
        status["reason"] = "no_key"
        warn(f"[rt_pred] skip: no key (scope={scope}, scope_id={scope_id})")
        return status

    now_ts = _to_utc_pid(message.get("crDt")) \
             or _to_utc_pid(message.get("createDate")) \
             or datetime.now(timezone.utc)

    # --- 2) IO vectors ---
    inputs_map, outputs_map, flat_vals = extract_numeric_io(message)
    if not flat_vals and not seed_history:   # allow seed-only calls
        status["reason"] = "no_numeric"
        warn(f"[rt_pred] skip: no numeric IO for key={key} (scope={scope})")
        return status

    model_path, scaler_path, meta_path = _model_paths(key)

    # --- 3) Columns order (persist in meta) ---
    cols = None
    if os.path.exists(meta_path):
        try:
            meta_json = json_load(meta_path)
            cols = list(meta_json.get("cols", [])) or None
        except Exception:
            cols = None

    # if we still don't have cols, infer from seed_history or current row
    if cols is None:
        candidates = set()
        if seed_history:
            for _, r in seed_history:
                candidates.update(r.keys())
        candidates.update(flat_vals.keys())
        cols = list(sorted(candidates))  # deterministic
        json_dump(meta_path, {"cols": cols, "timesteps": int(lookback)})
    else:
        # extend with any new cols but keep order stable
        for c in flat_vals.keys():
            if c not in cols:
                cols.append(c)
        if seed_history:
            for _, r in seed_history:
                for c in r.keys():
                    if c not in cols:
                        cols.append(c)
        json_dump(meta_path, {"cols": cols, "timesteps": int(lookback)})

    # --- 4) Seed history into buffer ---
    key_buf = _buffers[key]

    if seed_history and key_buf.df.empty:
        for ts, row in seed_history:
            nr = {c: safe_float(row.get(c)) for c in cols}
            key_buf.append_row(_to_utc_pid(ts) or datetime.now(timezone.utc), nr)
        info(f"[rt_pred] seeded history for key={key} with {len(seed_history)} points (scope={scope})")
    elif seed_history:
        # sadece debug için: history daha önce doldurulmuş
        log(f"[rt_pred] history already present for key={key} (len={len(key_buf.df)})")

    # append the "current" row from message (if provided)
    if flat_vals:
        numeric_row = {c: safe_float(flat_vals.get(c)) for c in cols}
        key_buf.append_row(_to_utc_pid(message.get("crDt")) or datetime.now(timezone.utc), numeric_row)
    
    df = key_buf.df.copy()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    info(f"[rt_pred] key={key} scope={scope} buffer_len={len(df)} n_features={len(cols)}")
    
    # --- 5) Frame & fill ---
    df = key_buf.df.copy()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    algo_name = algorithm or "LSTM"

    # Not enough points even to form 1 window → EMA bootstrap
    if len(df) < lookback + 1:
        ema_vec = (df.tail(min(len(df), max(5, lookback))).mean(axis=0).values
                   if len(df) else np.zeros(len(cols), dtype="float32"))

        actual_d, pred_d, mean_d = _vectors_for_write(df, cols, ema_vec)
        meta = extract_prediction_metadata(message)

        input_payload, output_payload = _build_payloads(
            cols, actual_d, pred_d, mean_d, inputs_map, outputs_map
        )

        # PID vs WS save
        """if scope == "pid":
            ScadaRealTimePredictions.saveData(
                key=key,
                now_ts=now_ts,
                algorithm=f"{algo_name}/EMA",
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=p3_1_log
            )
        elif scope == "ws":
            ScadaRealTimePredictionSummary.saveData(
                now_ts=now_ts,
                algorithm=f"{algo_name}/EMA",
                input_payload=input_payload,
                output_payload=output_payload,
                meta=meta,
                p3_1_log=p3_1_log
            )"""

        log(f"[rt_pred] EMA bootstrap wrote (len={len(df)}) key={key}, scope={scope}")
        status.update({"ok": True, "wrote": True, "reason": "ema_bootstrap"})
        gc.collect()
        return status

    X = df.values.astype("float32")

    # --- 6) Scaler ---
    if os.path.exists(scaler_path):
        try:
            scaler: MinMaxScaler = joblib.load(scaler_path)
            if scaler.scale_.shape[0] != X.shape[1]:
                scaler = MinMaxScaler().fit(X)
                joblib.dump(scaler, scaler_path)
                info(f"[rt_pred] refit scaler (dim change) key={key}")
        except Exception:
            scaler = MinMaxScaler().fit(X)
            joblib.dump(scaler, scaler_path)
            info(f"[rt_pred] rebuilt scaler key={key}")
    else:
        scaler = MinMaxScaler().fit(X)
        joblib.dump(scaler, scaler_path)
        info(f"[rt_pred] fitted new scaler key={key}")

    Xs = scaler.transform(X)

    # --- 7) Model load / retrain policy ---
    need_train = True
    model = None

    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            in_shape = getattr(model, "input_shape", None)
            model_steps = (in_shape[1] if isinstance(in_shape, (list, tuple)) else None)
            feat_ok = (model.output_shape[-1] == Xs.shape[1])
            steps_ok = (model_steps == lookback)

            if feat_ok and steps_ok:
                need_train = False
                log(f"[rt_pred] loaded model key={key} steps={model_steps} feats={Xs.shape[1]}")
            else:
                info(f"[rt_pred] retrain required (feat_ok={feat_ok}, steps_ok={steps_ok}) key={key}")
        except Exception as e:
            info(f"[rt_pred] model load failed; will train. key={key} err={e}")

    if need_train:
        if len(Xs) < max(min_train_points, lookback + 20):
            # Not enough points yet → EMA while waiting for training
            ema_vec = df.tail(lookback).mean(axis=0).values

            actual_d, pred_d, mean_d = _vectors_for_write(df, cols, ema_vec)
            meta = extract_prediction_metadata(message)
            input_payload, output_payload = _build_payloads(
                cols, actual_d, pred_d, mean_d, inputs_map, outputs_map
            )

            """if scope == "pid":
                ScadaRealTimePredictions.saveData(
                    key=key,
                    now_ts=now_ts,
                    algorithm=f"{algo_name}/EMA",
                    input_payload=input_payload,
                    output_payload=output_payload,
                    meta=meta,
                    p3_1_log=p3_1_log
                )
            elif scope == "ws":
                ScadaRealTimePredictionSummary.saveData(
                    now_ts=now_ts,
                    algorithm=f"{algo_name}/EMA",
                    input_payload=input_payload,
                    output_payload=output_payload,
                    meta=meta,
                    p3_1_log=p3_1_log
                )"""

            status.update({"ok": True, "wrote": True, "reason": "ema_waiting_train"})
            log(f"[rt_pred] EMA while waiting for train len={len(Xs)} key={key}, scope={scope}")
            gc.collect()
            return status

        train_block = Xs[-min(len(Xs), max(min_train_points, lookback + 200)):]
        X_seq, y_seq = _make_sequences(train_block, lookback)
        model = _build_model(lookback, Xs.shape[1])
        model.fit(X_seq, y_seq, epochs=epochs, batch_size=BATCH_SIZE, verbose=0)
        model.save(model_path)
        json_dump(meta_path, {"cols": cols, "timesteps": int(lookback)})
        info(f"[rt_pred] trained model key={key} seqs={len(X_seq)} steps={lookback} feats={Xs.shape[1]}")

    # --- 8) Build last window & predict ---
    if len(Xs) >= lookback:
        last_window = Xs[-lookback:]
    else:
        pad = lookback - len(Xs)
        seed = Xs[:1] if len(Xs) else np.zeros((1, Xs.shape[1]), dtype="float32")
        last_window = np.vstack([np.repeat(seed, pad, axis=0), Xs])

    y_hat_scaled = model.predict(last_window[None, ...], verbose=0)[0]
    y_hat = scaler.inverse_transform(y_hat_scaled[None, ...])[0]

    actual_d, pred_d, mean_d = _vectors_for_write(df, cols, y_hat)
    meta = extract_prediction_metadata(message)

    input_payload, output_payload = _build_payloads(
        cols, actual_d, pred_d, mean_d, inputs_map, outputs_map
    )

    # PID vs WS save
    if scope == "pid":
        ScadaRealTimePredictions.saveData(
            key=key,
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

    log(f"[rt_pred] wrote LSTM pred key={key} scope={scope} steps={lookback} feats={len(cols)}")
    status.update({"ok": True, "wrote": True, "reason": "lstm_pred"})
    del df, X, Xs, model
    gc.collect()
    return status
