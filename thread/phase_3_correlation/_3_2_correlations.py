import json
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from modules.kafka_modules import kafka_consumer3

from cassandra_utils.models.dw_single_data import dw_tbl_raw_data
from cassandra_utils.models.scada_correlation_matrix import ScadaCorrelationMatrix
from cassandra_utils.models.scada_correlation_matrix_summary import ScadaCorrelationMatrixSummary
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy, TokenAwarePolicy
from cassandra import ConsistencyLevel

from utils.config_reader import ConfigReader
from utils.logger_2 import setup_logger
# from utils.logger import get_p3_1_logger
# p3_1_log = get_p3_1_logger()

# Helper functions for correlation matrix
from thread.phase_3_correlation._3_1_helper_functions import map_to_text, compute_correlation, _is_input_type, aggregate_correlation_data

# Real-time LSTM prediction imports
from thread.phase_3_correlation._3_3_predictions import handle_realtime_prediction, history_from_fetch
from thread.phase_3_correlation._3_5_feature_importance import compute_and_save_feature_importance


consumer3 = None  # Kafka_init consumer
NONE_LIMIT = 100   # 30 defa üst üste None → ~30 saniye

cfg = ConfigReader()
cassandra_config = cfg["cassandra"]

CASSANDRA_HOST = cassandra_config["host"]
USERNAME = cassandra_config["username"]
PASSWORD = cassandra_config["password"]
KEYSPACE = cassandra_config["keyspace"]

auth_provider = PlainTextAuthProvider(username=USERNAME, password=PASSWORD)

profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy(local_dc=None)),
    request_timeout=20.0,
    consistency_level=ConsistencyLevel.LOCAL_ONE,
    retry_policy=RetryPolicy(),
)

cluster = Cluster(
    [CASSANDRA_HOST],
    auth_provider=auth_provider,
    execution_profiles={EXEC_PROFILE_DEFAULT: profile},
)
session = cluster.connect(KEYSPACE)
session.set_keyspace(KEYSPACE)

# Lock mechanism to ensure one message is completely processed at a time
processing_lock = threading.Lock()

# PID-level buffer (per process)
# pid -> {
#   "count": int,
#   "first_seen": float,
#   "last_time": float,
#   "first_crdt": datetime|None,
#   "messages": [message, ...]
# }
buffer_for_process = {}

# WS-level buffer (per workstation)
# ws_id -> same structure as above
buffer_for_ws = {}

COUNT_THRESHOLD = 20#20          # ilk batch için kaç nokta
TIME_THRESHOLD = 900         # 10 dakika (fallback time threshold)

# Thresholds can be tuned separately if you want
PID_COUNT_THRESHOLD = COUNT_THRESHOLD #5     # or set directly, e.g. 10
PID_TIME_THRESHOLD  = TIME_THRESHOLD      # seconds

WS_COUNT_THRESHOLD  = COUNT_THRESHOLD     # or maybe bigger, e.g. 50
WS_TIME_THRESHOLD   = TIME_THRESHOLD      # seconds

# ----------------- ABDML cadence -----------------
# We start with 1-minute resampling (stable baseline). You can later try 30s by
# adding "resample_seconds": 30 in utils/config.json.
RESAMPLE_SECONDS = int(getattr(cfg, "resample_seconds", 60) or 60)

# Prediction scope mode:
# - "pid": current behavior (operation/joOpId focused)
# - "batch": key prediction models by full batch id (prod_order_reference_no)
PREDICTION_SCOPE_MODE = str(getattr(cfg, "prediction_scope_mode", "batch") or "batch").strip().lower()

_FINALIZED_BATCHES = set()
_LAST_PROD_MESSAGE_BY_BATCH = {}
_BATCH_MESSAGE_BUFFER = {}


def finalize_batch_prediction(*, batch_id: str, last_prod_message: dict, batch_messages: list, p3_1_log):
    """
    Final, end-of-batch prediction.

    - Builds hist_out = [(ts, {sensor_key: value, ...}), ...] from buffered Kafka messages.
    - Calls handle_realtime_prediction() with algorithm=RANDOM_FOREST in scope='batch'
      so it SAVES to Cassandra (via handle_realtime_prediction).
    - Returns handle_realtime_prediction() result dict.
    """

    # ---------- local safe helpers (avoid undefined funcs) ----------
    import re
    from datetime import datetime, timezone

    def _safe_float(v):
        try:
            if v is None:
                return None
            if isinstance(v, bool):
                return float(int(v))
            if isinstance(v, (int, float)):
                return float(v)
            s = str(v).strip().replace(",", ".")
            if s in ("", "None", "nan", "NaN"):
                return None
            return float(s)
        except Exception:
            return None

    def _norm_key_local(name):
        if name is None:
            return None
        s = str(name).strip()
        if not s or s in ("None", "nan", "NaN"):
            return None
        # normalize: spaces/dots/slashes -> underscore, collapse repeats
        s = s.replace("\u00a0", " ")
        s = re.sub(r"[^\w]+", "_", s, flags=re.UNICODE)
        s = re.sub(r"_+", "_", s).strip("_")
        return s or None

    def _to_dt_local(v):
        # prefer your module-level _to_dt if present
        try:
            return _to_dt(v)  # noqa: F821  (exists in this module)
        except Exception:
            pass

        if v is None:
            return None
        if isinstance(v, datetime):
            return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

        # epoch s/ms
        try:
            s = int(v)
            if s > 10**12:
                return datetime.fromtimestamp(s / 1000.0, tz=timezone.utc)
            return datetime.fromtimestamp(s, tz=timezone.utc)
        except Exception:
            pass

        # iso
        try:
            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        except Exception:
            return None

    # ---------- guardrails ----------
    if not batch_id or str(batch_id) in ("", "None", "0"):
        return {"ok": False, "wrote": False, "reason": "finalize_no_batch_id"}

    batch_id = str(batch_id)

    if batch_id in _FINALIZED_BATCHES:
        return {"ok": True, "wrote": False, "reason": "finalize_already_done", "batch_id": batch_id}

    if not last_prod_message or not isinstance(last_prod_message, dict):
        return {"ok": False, "wrote": False, "reason": "finalize_no_last_prod_message", "batch_id": batch_id}

    if not batch_messages:
        p3_1_log.warning(f"[finalize_batch_prediction] batch_id={batch_id} -> batch_messages empty")
        return {"ok": False, "wrote": False, "reason": "finalize_no_batch_messages", "batch_id": batch_id}

    # ---------- build hist_out ----------
    hist_out = []
    bad_rows = 0
    for m in batch_messages:
        try:
            if not isinstance(m, dict):
                bad_rows += 1
                continue

            ts = _to_dt_local(m.get("crDt"))
            if ts is None:
                bad_rows += 1
                continue

            # ensure tz-aware
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            row = {}
            out_vals = m.get("outVals") or []
            for o in out_vals:
                if not isinstance(o, dict):
                    continue
                eq = o.get("eqNm") or o.get("equipment_name") or o.get("varNm")
                k = _norm_key_local(eq)
                if not k:
                    continue
                val = o.get("cntRead")
                fv = _safe_float(val)
                if fv is None:
                    continue
                row[k] = fv

            if row:
                hist_out.append((ts, row))
        except Exception:
            bad_rows += 1
            continue

    hist_out.sort(key=lambda x: x[0])

    if not hist_out:
        p3_1_log.warning(
            f"[finalize_batch_prediction] batch_id={batch_id} -> no hist_out built "
            f"(msgs={len(batch_messages)}, bad={bad_rows})"
        )
        return {"ok": False, "wrote": False, "reason": "finalize_no_history", "batch_id": batch_id}

    p3_1_log.info(
        f"[finalize_batch_prediction] batch_id={batch_id} "
        f"msgs={len(batch_messages)} bad={bad_rows} hist_points={len(hist_out)} -> predicting+saving"
    )

    # ---------- read prediction params safely ----------
    # Use config if present, else fallback.
    lookback = 20
    epochs = 50
    min_train_points = 250
    try:
        from utils.config_reader import ConfigReader
        cfg_obj = ConfigReader()
        lookback = int(getattr(cfg_obj, "lookback", lookback) or lookback)
        epochs = int(getattr(cfg_obj, "epochs", epochs) or epochs)
        min_train_points = int(getattr(cfg_obj, "min_train_points", min_train_points) or min_train_points)
    except Exception:
        pass

    # ---------- call prediction (this writes to Cassandra) ----------
    res = handle_realtime_prediction(
        message=last_prod_message,
        lookback=lookback,
        epochs=epochs,
        min_train_points=min_train_points,
        p3_1_log=p3_1_log,
        algorithm="RANDOM_FOREST",
        seed_history=hist_out,
        scope="batch",
        scope_id=batch_id,
        group_by_stock=True,
        resample_seconds=RESAMPLE_SECONDS,
        dry_run=False,
    )

    _FINALIZED_BATCHES.add(batch_id)
    return res


# FIXCODE
def _should_fire(buffer_entry, min_points, max_interval_sec, now_epoch=None):
    import time

    if now_epoch is None:
        now_epoch = time.time()

    cnt = int(buffer_entry.get("count", 0) or 0)
    first_seen = buffer_entry.get("first_seen")

    if first_seen is None:
        first_seen = now_epoch
        buffer_entry["first_seen"] = first_seen

    first_seen = float(first_seen)
    time_lapsed = max(0.0, float(now_epoch) - first_seen)

    fire = (cnt >= int(min_points)) or (time_lapsed >= float(max_interval_sec))
    return fire, cnt, time_lapsed


p3_1_log = setup_logger(
    "p3_1_logger", "logs/p3_1.log"
)

RAW_TABLE = cfg["cassandra_props"]["raw_data_table"]  # "dw_tbl_raw_data"

def _normalize_batch_id(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s or s.lower() == "none":
        return None
    if s.isdigit():
        try:
            return str(int(s))   # 010330 → 10330
        except Exception:
            return s.lstrip("0") or "0"
    return s


# -------- Utilities -------- #
def _to_dt(v):
    """Convert to aware UTC datetime.

    Destekler:
    - datetime objesi
    - ISO string (2025-11-28T13:00:00Z, 2025-11-28 13:00:00, vs.)
    - epoch saniye (int/float)
    - epoch milisaniye (int/float veya numeric string)
    """
    if v is None:
        return None

    # Zaten datetime ise
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)

    # Sayısal epoch (ms veya s)
    try:
        s = int(v)
        # çok kaba ama pratik: 10^12'den büyükse ms kabul et
        if s > 10**12:
            return datetime.fromtimestamp(s / 1000.0, tz=timezone.utc)
        else:
            return datetime.fromtimestamp(s, tz=timezone.utc)
    except Exception:
        pass

    # ISO string formatı
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return None


def _to_epoch_ms_safe(dt):
    if not dt:
        return 0
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _num_text(v):
    if v is None:
        return "0"
    s = str(v).strip().replace(",", ".")
    try:
        float(s)
        return s
    except Exception:
        return "0"


def _map_to_text(d):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool):
            out[k] = "true" if v else "false"
        elif isinstance(v, (int, float)):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).isoformat()
        else:
            out[k] = "" if v is None else str(v)
    return out

def _combine_input_output_lists(input_list, output_list):
    """
    Combine input_list and output_list into wrapped format for compatibility.
    
    Args:
        input_list: List of input variable dicts from fetchData()
        output_list: List of [output_dict] from fetchData()
    
    Returns:
        combined_list: List of [combined_dict] where each dict has:
            - equipment_type: True for inputs, False for outputs
            - All relevant fields (eqNo/varNm, cntRead/genReadVal, etc.)
    """
    combined = []
    
    # Add outputs (already in [[dict]] format)
    for ov_wrapped in output_list:
        if not ov_wrapped or not isinstance(ov_wrapped, list):
            continue
        ov = ov_wrapped[0] if len(ov_wrapped) > 0 else {}
        if not isinstance(ov, dict):
            continue
        
        # Mark as OUTPUT
        ov_copy = dict(ov)
        ov_copy["equipment_type"] = False
        combined.append([ov_copy])
    
    # Add inputs (need to wrap in [])
    for iv in input_list:
        if not isinstance(iv, dict):
            continue
        
        # Convert input format to match output format
        iv_wrapped = {
            "equipment_type": True,  # Mark as INPUT
            "eqNm": iv.get("varNm"),  # varNm → eqNm
            "eqNo": iv.get("varNm") or iv.get("varNo"),  # varNo → eqNo
            "gen_read_val": iv.get("genReadVal"),  # genReadVal
            "joOpId": iv.get("joOpId"),
            "wsId": iv.get("wsId"),
            "good": iv.get("good"),
        }
        combined.append([iv_wrapped])
    
    return combined
# -------- DW'den fetch (PID-level, snapshot üzerinden) -------- #
# ---------- STOCK HELPERS (NEW) ----------


def _extract_stock_from_prodlist_message(prod_list):
    """
    prod_list: message['prodList'] (list of dict)
    Returns: (output_stock_no, output_stock_name)
    """
    if not prod_list or not isinstance(prod_list, list):
        return None, None

    for item in prod_list:
        if not isinstance(item, dict):
            continue
        st_no = item.get("stNo") or item.get("stockNo") or item.get("st_no")
        st_nm = item.get("stNm") or item.get("stockName") or item.get("st_name")
        if st_no not in (None, ""):
            return str(st_no), (str(st_nm) if st_nm is not None else None)

    return None, None


# -------- DW'den fetch (PID-level, snapshot üzerinden) -------- #
def fetch_latest_for_pid_via_dw(pid: int, rows, input_list, output_list, _batch, max_rows: int = 20000):
    """
    DW snapshot içinden job_order_operation_id == pid olan son ölçümleri al.
    Her zaman dilimi (ts) için:
      - sensörler: parameter / counter_reading / equipment_name
      - label/meta: good (boolean), prSt (workstation_state), crDt, stock info
    """
    # ts -> list of sensor dicts
    bucket = defaultdict(list)
    # ts -> meta (good, prSt, refs, stock vs.)
    label_meta = {}

    # Combine input and output lists
    combined_list = _combine_input_output_lists(input_list, output_list)

    for r, wrapped_ov in zip(rows, combined_list):
        if getattr(r, "job_order_operation_id", None) != pid:
            continue

        ts = getattr(r, "measurement_date", None)
        if ts is None:
            p3_1_log.debug(f"[fetch_latest_for_pid_via_dw] pid={pid}: skipping row with null measurement_date")
            continue

        jo_ref_val = getattr(r, "job_order_reference_no", None)
        prod_ref_val = getattr(r, "prod_order_reference_no", None)

        # stok bilgisi DW'deki producelist kolonundan
        #produce_list = getattr(r, "producelist", None)
        #st_no, st_nm = _extract_stock_from_producelist(produce_list)
        st_no, st_nm = getattr(r, "produced_stock_no", None), getattr(r, "produced_stock_name", None)

        # ----------------- LABEL / META KISMI -----------------
        good_val = getattr(r, "good", None)
        prst_val = getattr(r, "workstation_state", None)

        # Aynı ts için meta bir kez set edilsin; sonra tekrar yazmasın
        if ts not in label_meta:
            label_meta[ts] = {
                "good": good_val,
                "prSt": prst_val,
                "job_order_reference_no": jo_ref_val,
                "prod_order_reference_no": prod_ref_val,
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                # ---- OP FIELDS (NEW) ----
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # ----------------- SENSÖR KISMI -----------------
        ov = wrapped_ov[0] if wrapped_ov else {}
        if ov.get("equipment_type", True):  # INPUT ise atla
            pname = ov.get("varNo")
            cval = ov.get("genReadVal")
            eq_name = ov.get("varNm")
            eq_type = ov.get("equipment_type", True)

            bucket[ts].append({
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })
        else:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            eq_type = ov.get("equipment_type", False)

            bucket[ts].append({
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })

    if not bucket:
        p3_1_log.info(f"[fetch_latest_for_pid_via_dw] pid={pid}: 0 rows after filtering")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        # Bu ts için daha önce doldurduğumuz label/meta
        lm = label_meta.get(ts, {}) or {}
        good_val = lm.get("good")
        prst_val = lm.get("PrSt") if "PrSt" in lm else lm.get("prSt")  # güvenlik
        prst_val = lm.get("prSt") if prst_val is None else prst_val

        meta = {
            "job_order_operation_id": pid,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            # ---- label alanları: feature importance için lazım ----
            "good": good_val,          # dw_tbl_raw_data.good
            "prSt": prst_val,          # dw_tbl_raw_data.workstation_state (PRODUCTION vs)
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            # stok meta
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            # ---- OP FIELDS (NEW) ----
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(pid)
        dates.append(ts)
        ids.append(uuid.uuid4())
        # meta + sensörler -> hepsi text’e çevrilerek sensor_values’a gidiyor
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    try:
        sample_meta = sensor_values[0][0] if sensor_values and len(sensor_values[0]) > 0 else None
        sample_sensor = sensor_values[0][1] if sensor_values and len(sensor_values[0]) > 1 else None
        p3_1_log.debug(
            f"[fetch_latest_for_pid_via_dw] sample meta keys="
            f"{list(sample_meta.keys()) if isinstance(sample_meta, dict) else type(sample_meta)}, "
            f"sample sensor keys="
            f"{list(sample_sensor.keys()) if isinstance(sample_sensor, dict) else type(sample_sensor)}"
        )
    except Exception:
        pass

    p3_1_log.info(f"[fetch_latest_for_pid_via_dw] pid={pid} bundles={len(sensor_values)} (timestamps)")
    return job_ids, dates, ids, sensor_values


# -------- DW'den fetch (WS-level, snapshot üzerinden) -------- #
def fetch_latest_for_ws_via_dw(ws_id: int, rows, input_list, output_list, _batch, max_rows: int = 20000):
    bucket = defaultdict(list)   # key = (ts, stock_no) -> sensors
    label_meta = {}              # key = (ts, stock_no) -> meta

    combined_list = _combine_input_output_lists(input_list, output_list)

    for r, wrapped_ov in zip(rows, combined_list):
        ws_attr = (
            getattr(r, "work_station_id", None)
            or getattr(r, "workstation_id", None)
            or getattr(r, "ws_id", None)
            or getattr(r, "wsid", None)
        )
        if ws_attr != ws_id:
            continue

        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        # IMPORTANT: stock for this row
        st_no = getattr(r, "produced_stock_no", None)
        st_nm = getattr(r, "produced_stock_name", None)

        stock_key = str(st_no) if st_no not in (None, "") else None
        key = (ts, stock_key)

        # label/meta per (ts, stock)
        if key not in label_meta:
            label_meta[key] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": stock_key,
                "output_stock_name": (str(st_nm) if st_nm not in (None, "") else None),
                # ---- OP FIELDS (NEW) ----
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # sensor part
        ov = wrapped_ov[0] if wrapped_ov else {}
        if ov.get("equipment_type", True):  # INPUT ise atla
            pname = ov.get("varNo")
            cval = ov.get("genReadVal")
            eq_name = ov.get("varNm")
            eq_type = ov.get("equipment_type", True)

            bucket[key].append({  # ← FIXED: Use key (ts, stock_key)
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })
        else:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            eq_type = ov.get("equipment_type", False)

            bucket[key].append({  # ← FIXED: Use key (ts, stock_key)
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": eq_type
            })

    if not bucket:
        p3_1_log.info(f"[fetch_latest_for_ws_via_dw] ws_id={ws_id}: 0 rows after filtering")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    # sort by ts (and stock_key for deterministic)
    for (ts, stock_key) in sorted(bucket.keys(), key=lambda x: (x[0], str(x[1]))):
        sensors = bucket[(ts, stock_key)]
        if not sensors:
            continue

        lm = label_meta.get((ts, stock_key), {}) or {}
        meta = {
            "workstation_id": ws_id,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            # ---- OP FIELDS (NEW) ----
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(f"[fetch_latest_for_ws_via_dw] ws_id={ws_id} bundles={len(sensor_values)} (ts,stock)")
    return job_ids, dates, ids, sensor_values


# -------- RAW_TABLE'dan direkt fetch (PID-level, eski timestamp fallback) -------- #
def fetch_latest_for_pid_via_raw_table(pid: int, max_rows: int = 20000):
    """
    UPDATED: Handles BOTH inputs and outputs from RAW_TABLE.
    """
    p3_1_log.info(
        f"[fetch_latest_for_pid_via_raw_table] Fallback querying RAW_TABLE={RAW_TABLE} for pid={pid}"
    )

    query = (
        f"SELECT job_order_operation_id, work_station_id, measurement_date, "
        f"equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type, "
        f"good, workstation_state, job_order_reference_no, prod_order_reference_no, "
        f"produced_stock_no, produced_stock_name, operationname, operationno, operationtaskcode "
        f"FROM {RAW_TABLE} "
        f"WHERE job_order_operation_id = %s "
        f"LIMIT %s ALLOW FILTERING"
    )

    try:
        rows = session.execute(query, (pid, max_rows))
    except Exception as e:
        p3_1_log.warning(
            f"[fetch_latest_for_pid_via_raw_table] CQL query failed for pid={pid}: {e}"
        )
        return [], [], [], []

    bucket = defaultdict(list)
    label_meta = {}

    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        st_no, st_nm = getattr(r, "produced_stock_no", None), getattr(r, "produced_stock_name", None)

        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # ===== NEW: Check equipment_type =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type) #(equipment_type is True)
        
        if is_input:
            # INPUT
            var_name = getattr(r, "equipment_name", None) or getattr(r, "parameter", None)
            var_value = getattr(r, "gen_read_val", None)
            
            if not var_name or var_value is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(var_name)}",
                "counter_reading": _num_text(var_value),
                "equipment_name": str(var_name),
                "equipment_type": True,
            })
        else:
            # OUTPUT
            pname = getattr(r, "parameter", None) or getattr(r, "equipment_no", None)
            cval = getattr(r, "counter_reading", None)
            eq_name = getattr(r, "equipment_name", None)
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })

    if not bucket:
        p3_1_log.info(
            f"[fetch_latest_for_pid_via_raw_table] pid={pid}: 0 rows found in RAW_TABLE fallback"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "job_order_operation_id": pid,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(pid)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[fetch_latest_for_pid_via_raw_table] pid={pid}: built {len(sensor_values)} bundles from RAW_TABLE"
    )
    return job_ids, dates, ids, sensor_values


# -------- RAW_TABLE'dan direkt fetch (WS-level, eski timestamp fallback) -------- #
def fetch_latest_for_ws_via_raw_table(ws_id: int, max_rows: int = 20000):
    """
    UPDATED: Handles BOTH inputs and outputs from RAW_TABLE.
    """
    p3_1_log.info(
        f"[fetch_latest_for_ws_via_raw_table] Fallback querying RAW_TABLE={RAW_TABLE} for ws_id={ws_id}"
    )

    query = (
        f"SELECT job_order_operation_id, work_station_id, measurement_date, "
        f"equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type, "
        f"good, workstation_state, job_order_reference_no, prod_order_reference_no, "
        f"produced_stock_no, produced_stock_name, operationname, operationno, operationtaskcode "
        f"FROM {RAW_TABLE} "
        f"WHERE work_station_id = %s "
        f"LIMIT %s ALLOW FILTERING"
    )

    try:
        rows = session.execute(query, (ws_id, max_rows))
    except Exception as e:
        p3_1_log.warning(
            f"[fetch_latest_for_ws_via_raw_table] CQL query failed for ws_id={ws_id}: {e}"
        )
        return [], [], [], []

    bucket = defaultdict(list)
    label_meta = {}

    for r in rows:
        ws_attr = getattr(r, "work_station_id", None)
        if ws_attr != ws_id:
            continue

        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        st_no, st_nm = getattr(r, "produced_stock_no", None), getattr(r, "produced_stock_name", None)

        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None)
            }

        # ===== NEW: Check equipment_type =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type) #(equipment_type is True)
        
        if is_input:
            var_name = getattr(r, "equipment_name", None) or getattr(r, "parameter", None)
            var_value = getattr(r, "gen_read_val", None)
            
            if not var_name or var_value is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(var_name)}",
                "counter_reading": _num_text(var_value),
                "equipment_name": str(var_name),
                "equipment_type": True,
            })
        else:
            pname = getattr(r, "parameter", None) or getattr(r, "equipment_no", None)
            cval = getattr(r, "counter_reading", None)
            eq_name = getattr(r, "equipment_name", None)
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })

    if not bucket:
        p3_1_log.info(
            f"[fetch_latest_for_ws_via_raw_table] ws_id={ws_id}: 0 rows found in RAW_TABLE fallback"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "workstation_id": ws_id,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[fetch_latest_for_ws_via_raw_table] ws_id={ws_id}: built {len(sensor_values)} bundles from RAW_TABLE"
    )
    return job_ids, dates, ids, sensor_values

# -------- Buffer'dan sensor_values üret (PID-level) -------- #
# ADD THIS HELPER FIRST (after line ~120)
def _extract_invars_from_message(message: dict):
    """
    Extract input variables from message's inVars or inputVariableList.
    
    Returns list of dicts with:
        - var_name: variable name (priority: varNm > varId)
        - var_value: variable value (genReadVal)
    """
    invars = message.get("inVars") or message.get("inputVariableList") or []
    if not isinstance(invars, list):
        return []
    
    result = []
    for iv in invars:
        if not isinstance(iv, dict):
            continue
        
        var_name = iv.get("varNm") or iv.get("varId")
        if not var_name or var_name in (None, "", "None"):
            continue
        
        var_value = iv.get("genReadVal")
        if var_value is None:
            continue
        
        result.append({
            "var_name": str(var_name),
            "var_value": var_value,
        })
    
    return result

# THEN UPDATE THE FUNCTION:
def build_sensor_values_from_pid_buffer(pid: int, messages):
    """
    UPDATED: Builds sensor_values from Kafka buffer with BOTH inputs and outputs.
    """
    p3_1_log.info(
        f"[build_sensor_values_from_pid_buffer] pid={pid}: received {len(messages or [])} buffered messages"
    )

    bucket = defaultdict(list)
    label_meta = {}

    for msg in messages or []:
        # ===== OUTPUTS (original) =====
        out_vals = msg.get("outVals") or []
        
        # ===== INPUTS (NEW) =====
        in_vars = _extract_invars_from_message(msg)
        
        if not out_vals and not in_vars:
            continue

        # Get timestamp
        meas_ms = (out_vals[0].get("measDt") if out_vals else None) or msg.get("crDt")
        ts = _to_dt(meas_ms)
        if ts is None:
            p3_1_log.debug(
                f"[build_sensor_values_from_pid_buffer] pid={pid}: could not parse ts"
            )
            continue

        st_no, st_nm = _extract_stock_from_prodlist_message(msg.get("prodList"))

        if ts not in label_meta:
            label_meta[ts] = {
                "good": msg.get("goodCnt"),
                "prSt": msg.get("prSt"),
                "job_order_reference_no": msg.get("joRef"),
                "prod_order_reference_no": msg.get("refNo"),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": msg.get("opNm"),
                "operationno": msg.get("opNo"),
                "operationtaskcode": msg.get("opTc")
            }

        # Add OUTPUT sensors
        for ov in out_vals:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })
        
        # Add INPUT sensors (NEW)
        for iv in in_vars:
            bucket[ts].append({
                "parameter": f"{iv['var_name']}",
                "counter_reading": _num_text(iv['var_value']),
                "equipment_name": iv['var_name'],
                "equipment_type": True,
            })

    if not bucket:
        p3_1_log.info(f"[build_sensor_values_from_pid_buffer] pid={pid}: 0 bundles from buffer")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "job_order_operation_id": pid,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(pid)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[build_sensor_values_from_pid_buffer] pid={pid}: built {len(sensor_values)} bundles from buffer"
    )
    return job_ids, dates, ids, sensor_values

# -------- Buffer'dan sensor_values üret (WS-level) -------- #
def build_sensor_values_from_ws_buffer(ws_id: int, messages):
    """
    UPDATED: Builds sensor_values from Kafka buffer with BOTH inputs and outputs.
    """
    p3_1_log.info(
        f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: received {len(messages or [])} buffered messages"
    )

    bucket = defaultdict(list)
    label_meta = {}

    for msg in messages or []:
        out_vals = msg.get("outVals") or []
        in_vars = _extract_invars_from_message(msg)
        
        if not out_vals and not in_vars:
            continue

        meas_ms = (out_vals[0].get("measDt") if out_vals else None) or msg.get("crDt")
        ts = _to_dt(meas_ms)
        if ts is None:
            p3_1_log.debug(
                f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: could not parse ts"
            )
            continue

        st_no, st_nm = _extract_stock_from_prodlist_message(msg.get("prodList"))

        if ts not in label_meta:
            label_meta[ts] = {
                "good": msg.get("goodCnt"),
                "prSt": msg.get("prSt"),
                "job_order_reference_no": msg.get("joRef"),
                "prod_order_reference_no": msg.get("refNo"),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": msg.get("opNm"),
                "operationno": msg.get("opNo"),
                "operationtaskcode": msg.get("opTc"),
            }

        # Add OUTPUTS
        for ov in out_vals:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": f"{str(pname)}",
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })
        
        # Add INPUTS (NEW)
        for iv in in_vars:
            bucket[ts].append({
                "parameter": f"{iv['var_name']}",
                "counter_reading": _num_text(iv['var_value']),
                "equipment_name": iv['var_name'],
                "equipment_type": True,
            })

    if not bucket:
        p3_1_log.info(f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: 0 bundles from buffer")
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "workstation_id": ws_id,
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode")
        }

        job_ids.append(ws_id)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[build_sensor_values_from_ws_buffer] ws_id={ws_id}: built {len(sensor_values)} bundles from buffer"
    )
    return job_ids, dates, ids, sensor_values

def _clean_stock(v):
    if v is None:
        return None
    s = str(v)
    if s in ("", "None", "nan", "NaN"):
        return None
    return s

def _sensor_values_has_stock(sensor_values, stock_no: str) -> bool:
    """
    sensor_values: list of bundles -> each bundle is [meta_dict, sensor1_dict, ...]
    We check meta["output_stock_no"].
    """
    stock_no = _clean_stock(stock_no)
    if not stock_no or not sensor_values:
        return False

    for bundle in sensor_values:
        if not bundle or not isinstance(bundle, list):
            p3_1_log.debug(f"[_sensor_values_has_stock] Skipping bundle with invalid format: {bundle}")
            continue
        meta = bundle[0] if len(bundle) > 0 else None
        if not isinstance(meta, dict):
            p3_1_log.debug(f"[_sensor_values_has_stock] Skipping bundle with invalid meta: {meta}")
            continue
        st = _clean_stock(meta.get("output_stock_no"))
        if st == stock_no:
            return True
    return False

def _count_points_for_stock(sensor_values, stock_no: str) -> int:
    """
    Count distinct time points for that stock. Uses meta["crDt"] if present, else counts bundles.
    """
    stock_no = _clean_stock(stock_no)
    if not stock_no or not sensor_values:
        return 0

    crdts = set()
    cnt = 0
    for bundle in sensor_values:
        if not bundle or not isinstance(bundle, list):
            continue
        meta = bundle[0] if len(bundle) > 0 else None
        if not isinstance(meta, dict):
            p3_1_log.debug(f"[_count_points_for_stock] Skipping bundle with invalid meta: {meta}")
            continue
        st = _clean_stock(meta.get("output_stock_no"))
        if st != stock_no:
            # p3_1_log.debug(f"[_count_points_for_stock] Skipping bundle with stock_no={st}")
            continue

        cr = meta.get("crDt")
        if cr is not None:
            crdts.add(str(cr))
        else:
            cnt += 1

    return len(crdts) if crdts else cnt


def fetch_for_optc_stock_via_dw(
    rows,
    input_list,
    output_list,
    _batch,
    op_tc=None,
    stock_no=None,
    ws_id=None,
    pid=None,
    prod_order_reference_no=None,
    max_rows: int = 20000,
):
    """
    DW snapshot fetch filtered by (op_tc + stock) PLUS optional ws_id/pid PLUS optional batch id.

    CRITICAL FIX:
      - If prod_order_reference_no is provided, we only return rows belonging to that batch.
      - We DO NOT use _combine_input_output_lists + zip(rows, combined_list) because it breaks row↔sensor alignment.
      - We infer INPUT/OUTPUT directly from row.equipment_type using _is_input_type.
    """
    # ---- normalize identifiers (avoid leading-zero mismatches) ----
    def _norm_ref(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() == "none":
            return None
        # If numeric-like, normalize by int to match "010330" vs 10330
        if s.isdigit():
            try:
                return str(int(s))
            except Exception:
                return (s.lstrip("0") or "0")
        return s

    op_tc = None if op_tc in (None, "", "None") else str(op_tc).strip()
    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no).strip()
    batch_norm = _normalize_batch_id(prod_order_reference_no)

    if not op_tc or not stock_no:
        p3_1_log.warning(
            f"[fetch_for_optc_stock_via_dw] Missing op_tc={op_tc} or stock_no={stock_no}, returning empty"
        )
        return [], [], [], []

    bucket = defaultdict(list)   # ts -> [sensor_dict, ...]
    label_meta = {}              # ts -> meta dict
    pid_by_ts = {}
    ws_by_ts = {}

    # rows already limited by fetchData(limit=...), but keep a defensive cap
    used = 0

    for r in rows or []:
        if used >= max_rows:
            break

        # ---- PRIMARY FILTERS ----
        row_optc = getattr(r, "operationtaskcode", None)
        if row_optc != op_tc:
            continue

        row_stock = getattr(r, "produced_stock_no", None)
        row_stock = None if row_stock in (None, "", "None", "nan", "NaN") else str(row_stock).strip()
        if row_stock != stock_no:
            continue

        # ---- OPTIONAL FILTERS ----
        if ws_id is not None:
            row_ws = getattr(r, "work_station_id", None) or getattr(r, "workstation_id", None)
            if row_ws != ws_id:
                continue

        if pid is not None:
            row_pid = getattr(r, "job_order_operation_id", None)
            if row_pid != pid:
                continue

        row_batch = getattr(r, "prod_order_reference_no", None)
        if batch_norm is not None:
            if _normalize_batch_id(row_batch) != batch_norm:
                continue

        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        # Track PID/WS per timestamp (best-effort)
        if ts not in pid_by_ts:
            pid_by_ts[ts] = getattr(r, "job_order_operation_id", None)
        if ts not in ws_by_ts:
            ws_by_ts[ts] = getattr(r, "work_station_id", None) or getattr(r, "workstation_id", None)

        # Meta per timestamp
        if ts not in label_meta:
            st_nm = getattr(r, "produced_stock_name", None)
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": row_stock,
                "output_stock_name": (None if st_nm in (None, "", "None") else str(st_nm)),
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": row_optc,
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None) or getattr(r, "workstation_id", None),
            }

        # Sensor extraction from row fields
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type)

        eq_no = getattr(r, "equipment_no", None)
        eq_nm = getattr(r, "equipment_name", None)

        if is_input:
            # INPUT: prefer gen_read_val, fallback to counter_reading if needed
            val = getattr(r, "gen_read_val", None)
            if val is None:
                val = getattr(r, "counter_reading", None)
            if val is None:
                continue

            param = eq_no or eq_nm
            if not param:
                continue

            bucket[ts].append(
                {
                    "parameter": str(param),
                    "counter_reading": _num_text(val),
                    "equipment_name": str(eq_nm or param),
                    "equipment_type": True,
                }
            )
        else:
            # OUTPUT: use counter_reading
            cval = getattr(r, "counter_reading", None)
            if cval is None:
                continue

            param = eq_no or getattr(r, "parameter", None) or eq_nm
            if not param:
                continue

            bucket[ts].append(
                {
                    "parameter": str(param),
                    "counter_reading": _num_text(cval),
                    "equipment_name": str(eq_nm or param),
                    "equipment_type": False,
                }
            )

        used += 1

    if not bucket:
        p3_1_log.info(
            f"[fetch_for_optc_stock_via_dw] 0 bundles for op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, pid={pid}, batch={prod_order_reference_no}"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []
    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {}) or {}
        meta = {
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
        }

        job_ids.append(pid_by_ts.get(ts) or 0)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[fetch_for_optc_stock_via_dw] Found {len(sensor_values)} bundles "
        f"for op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, "
        f"pid={pid}, batch={prod_order_reference_no}"
    )

    return job_ids, dates, ids, sensor_values


def build_sensor_values_from_buffer_for_optc_stock(
    messages,
    op_tc=None,
    stock_no=None,
    ws_id=None,
    pid=None,
    prod_order_reference_no=None,  # <-- ADD
):
    def _norm_ref(x):
        if x is None:
            return None
        s = str(x).strip()
        if not s or s.lower() == "none":
            return None
        if s.isdigit():
            try:
                return str(int(s))
            except Exception:
                return (s.lstrip("0") or "0")
        return s

    batch_norm = _norm_ref(prod_order_reference_no)
    """
    Build sensor_values from buffered messages filtered by op_tc + stock.
    Optionally filter by ws_id or pid.
    """
    op_tc = None if op_tc in (None, "", "None") else str(op_tc)
    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)

    if not op_tc or not stock_no:
        p3_1_log.warning(
            f"[build_sensor_values_from_buffer_for_optc_stock] Missing op_tc or stock_no"
        )
        return [], [], [], []

    bucket = defaultdict(list)
    label_meta = {}
    pid_by_ts = {}

    for msg in messages or []:
        msg_optc = msg.get("operationtaskcode") or msg.get("opTc")
        if msg_optc != op_tc:
            continue

        msg_stock, msg_stock_name = _extract_stock_from_prodlist_message(msg.get("prodList"))
        msg_stock = None if msg_stock in (None, "", "None", "nan", "NaN") else str(msg_stock)
        if msg_stock != stock_no:
            continue

        if ws_id is not None:
            msg_ws = msg.get("wsId")
            if msg_ws != ws_id:
                continue

        if pid is not None:
            msg_pid = msg.get("joOpId")
            if msg_pid != pid:
                continue

        if batch_norm is not None:
            msg_batch = msg.get("prod_order_reference_no") or msg.get("refNo")
            if _norm_ref(msg_batch) != batch_norm:
                continue

        # Extract inputs and outputs
        out_vals = msg.get("outVals") or []
        in_vars = _extract_invars_from_message(msg)

        if not out_vals and not in_vars:
            continue

        meas_ms = (out_vals[0].get("measDt") if out_vals else None) or msg.get("crDt")
        ts = _to_dt(meas_ms)
        if ts is None:
            continue

        # Track PID for this timestamp
        if ts not in pid_by_ts:
            pid_by_ts[ts] = msg.get("joOpId")

        if ts not in label_meta:
            label_meta[ts] = {
                "good": msg.get("goodCnt"),
                "prSt": msg.get("prSt"),
                "job_order_reference_no": msg.get("joRef"),
                "prod_order_reference_no": msg.get("refNo"),
                "output_stock_no": msg_stock,
                "output_stock_name": msg_stock_name,
                "operationname": msg.get("opNm"),
                "operationno": msg.get("opNo"),
                "operationtaskcode": msg_optc,
                "job_order_operation_id": msg.get("joOpId"),
                "work_station_id": msg.get("wsId"),
            }

        # Add OUTPUTS
        for ov in out_vals:
            pname = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")

            if pname and cval is not None:
                bucket[ts].append({
                    "parameter": str(pname),
                    "counter_reading": _num_text(cval),
                    "equipment_name": str(eq_name),
                    "equipment_type": False,
                })

        # Add INPUTS
        for iv in in_vars:
            bucket[ts].append({
                "parameter": iv['var_name'],
                "counter_reading": _num_text(iv['var_value']),
                "equipment_name": iv['var_name'],
                "equipment_type": True,
            })

    if not bucket:
        p3_1_log.info(
            f"[build_sensor_values_from_buffer_for_optc_stock] 0 bundles for "
            f"op_tc={op_tc}, stock={stock_no}"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {})
        meta = {
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
        }

        job_ids.append(pid_by_ts.get(ts) or 0)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[build_sensor_values_from_buffer_for_optc_stock] Built {len(sensor_values)} bundles "
        f"for op_tc={op_tc}, stock={stock_no}"
    )
    return job_ids, dates, ids, sensor_values


def fetch_latest_for_optc_stock_via_raw_table(
    op_tc,
    stock_no,
    *,
    max_rows=20000,
    ws_id=None,
    pid=None,
    prod_order_reference_no=None, 
):
    """
    Fetch data from RAW_TABLE filtered by operation task code + stock.
    Optionally filter by ws_id or pid.
    
    Returns data from ALL PIDs with matching op_tc + stock (unless pid is specified).
    UPDATED: Handles BOTH inputs and outputs from RAW_TABLE.
    """
    op_tc = None if op_tc in (None, "", "None") else str(op_tc)
    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no).strip()

    if not op_tc or not stock_no:
        p3_1_log.warning(
            f"[fetch_latest_for_optc_stock_via_raw_table] Missing op_tc={op_tc} or stock_no={stock_no}"
        )
        return [], [], [], []

    def _batch_variants(x):
        if x is None:
            return []
        s = str(x).strip()
        if not s or s.lower() == "none":
            return []
        out = [s]
        if s.isdigit():
            try:
                s_int = str(int(s))  # 010330 -> 10330
            except Exception:
                s_int = (s.lstrip("0") or "0")
            if s_int not in out:
                out.append(s_int)

            # Common padding length observed in this project: 6
            if len(s) < 6:
                z = s.zfill(6)
                if z not in out:
                    out.append(z)
            if len(s_int) < 6:
                z2 = s_int.zfill(6)
                if z2 not in out:
                    out.append(z2)
        return out

    # Build WHERE clause dynamically (base filters)
    base_where = "WHERE operationtaskcode = %s AND produced_stock_no = %s"
    base_params = [op_tc, stock_no]

    if ws_id is not None:
        base_where += " AND work_station_id = %s"
        base_params.append(ws_id)

    select_sql = (
        f"SELECT job_order_operation_id, work_station_id, measurement_date, "
        f"equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type, "
        f"good, workstation_state, job_order_reference_no, prod_order_reference_no, "
        f"produced_stock_no, produced_stock_name, operationname, operationno, operationtaskcode "
        f"FROM {RAW_TABLE} "
    )

    rows = []
    used_batch = None

    try:
        if prod_order_reference_no is None:
            where = base_where
            params = list(base_params)
            query = f"{select_sql} {where} LIMIT %s ALLOW FILTERING"
            params.append(max_rows)
            rows = list(session.execute(query, tuple(params)))
        else:
            for b in _batch_variants(prod_order_reference_no):
                where = base_where + " AND prod_order_reference_no = %s"
                params = list(base_params) + [b]
                query = f"{select_sql} {where} LIMIT %s ALLOW FILTERING"
                params.append(max_rows)
                tmp = list(session.execute(query, tuple(params)))
                if tmp:
                    rows = tmp
                    used_batch = b
                    break
    except Exception as e:
        p3_1_log.warning(
            f"[fetch_latest_for_optc_stock_via_raw_table] query failed for "
            f"op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, batch={prod_order_reference_no}: {e}"
        )
        return [], [], [], []

    if not rows:
        p3_1_log.info(
            f"[fetch_latest_for_optc_stock_via_raw_table] 0 rows for "
            f"op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, pid={pid}, batch={prod_order_reference_no}"
        )
        return [], [], [], []

    if prod_order_reference_no is not None:
        p3_1_log.info(
            f"[fetch_latest_for_optc_stock_via_raw_table] batch filter matched using variant={used_batch!r} "
            f"(requested={prod_order_reference_no!r}) rows={len(rows)}"
        )

    bucket = defaultdict(list)
    label_meta = {}
    pid_by_ts = {}
    ws_by_ts = {}

    for r in rows:
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue

        # Track which PID/WS this timestamp belongs to
        if ts not in pid_by_ts:
            pid_by_ts[ts] = getattr(r, "job_order_operation_id", None)
        if ts not in ws_by_ts:
            ws_by_ts[ts] = getattr(r, "work_station_id", None)

        # Label/meta info
        if ts not in label_meta:
            st_no = getattr(r, "produced_stock_no", None)
            st_nm = getattr(r, "produced_stock_name", None)

            st_no = None if st_no in (None, "", "None", "nan", "NaN") else str(st_no)
            st_nm = None if st_nm in (None, "", "None", "nan", "NaN") else str(st_nm)

            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": st_no,
                "output_stock_name": st_nm,
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": getattr(r, "operationtaskcode", None),
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None),
            }

        # ===== Check equipment_type for INPUT vs OUTPUT =====
        equipment_type = getattr(r, "equipment_type", None)
        is_input = _is_input_type(equipment_type)
        
        if is_input:
            # INPUT
            var_name = getattr(r, "equipment_name", None) or getattr(r, "parameter", None)
            var_value = getattr(r, "gen_read_val", None)
            
            if not var_name or var_value is None:
                continue
            
            bucket[ts].append({
                "parameter": str(var_name),
                "counter_reading": _num_text(var_value),
                "equipment_name": str(var_name),
                "equipment_type": True,
            })
        else:
            # OUTPUT
            pname = getattr(r, "parameter", None) or getattr(r, "equipment_no", None)
            cval = getattr(r, "counter_reading", None)
            eq_name = getattr(r, "equipment_name", None)
            
            if not pname or cval is None:
                continue
            
            bucket[ts].append({
                "parameter": str(pname),
                "counter_reading": _num_text(cval),
                "equipment_name": str(eq_name),
                "equipment_type": False,
            })

    if not bucket:
        p3_1_log.info(
            f"[fetch_latest_for_optc_stock_via_raw_table] 0 bundles after processing for "
            f"op_tc={op_tc}, stock={stock_no}"
        )
        return [], [], [], []

    job_ids, dates, ids, sensor_values = [], [], [], []

    for ts in sorted(bucket.keys()):
        sensors = bucket[ts]
        if not sensors:
            continue

        lm = label_meta.get(ts, {}) or {}
        meta = {
            "measurement_date": ts,
            "crDt": str(_to_epoch_ms_safe(ts)),
            "good": lm.get("good"),
            "prSt": lm.get("prSt"),
            "job_order_reference_no": lm.get("job_order_reference_no"),
            "prod_order_reference_no": lm.get("prod_order_reference_no"),
            "output_stock_no": lm.get("output_stock_no"),
            "output_stock_name": lm.get("output_stock_name"),
            "operationname": lm.get("operationname"),
            "operationno": lm.get("operationno"),
            "operationtaskcode": lm.get("operationtaskcode"),
            "job_order_operation_id": lm.get("job_order_operation_id"),
            "work_station_id": lm.get("work_station_id"),
        }

        job_ids.append(pid_by_ts.get(ts) or 0)
        dates.append(ts)
        ids.append(uuid.uuid4())
        sensor_values.append([_map_to_text(meta)] + [_map_to_text(s) for s in sensors])

    p3_1_log.info(
        f"[fetch_latest_for_optc_stock_via_raw_table] rows={len(rows)}, bundles={len(sensor_values)} "
        f"for op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, pid={pid}, batch={prod_order_reference_no}"
    )

    return job_ids, dates, ids, sensor_values

def _is_message_valid(message):
    """
    Validates required fields in message.
    Returns True if valid, False otherwise.
    """

    # 1) outVals → cust alanı boş mu?
    if message.get("outVals") and message["outVals"][0].get("cust") is None:
        p3_1_log.warning("[execute_phase_three] Skipping message: 'cust' field missing in outVals")
        return False

    # 2) plId → varsa None olamaz
    if "plId" in message and message["plId"] is None:
        p3_1_log.info("[execute_phase_three] Skipping message: plId is None")
        return False

    # 3) wsId → varsa None olamaz
    if "wsId" in message and message["wsId"] is None:
        p3_1_log.info("[execute_phase_three] Skipping message: wsId is None")
        return False

    # 4) joOpId → varsa None olamaz
    if "joOpId" in message and message["joOpId"] is None:
        p3_1_log.info("[execute_phase_three] Skipping message: joOpId is None")
        return False
    
    """if "refNo" in message and message["refNo"] is None:
        p3_1_log.info("[execute_phase_three] Skipping message: refNo is None")
        return False"""

    return True

def _predict_and_save_every_message(
    *,
    message: dict,
    pid,
    batch_id: str,
    p3_1_log,
    resample_seconds: int,
    scope_mode: str,
):
    """
    Fast-path: for EVERY Kafka message we attempt prediction and (if non-empty) save to Cassandra.

    - Keeps existing throttled tasks intact (correlation/FI/etc)
    - Uses RANDOM_FOREST (current focus)
    - Uses scope_mode:
        * 'batch' -> scope='batch', scope_id=batch_id
        * 'pid'   -> scope='pid',   scope_id=pid
    """
    try:
        algo = "RANDOM_FOREST"

        scope_mode = (scope_mode or "batch").strip().lower()
        if scope_mode == "pid":
            scope = "pid"
            scope_id = str(pid)
            group_by_stock = True
        else:
            scope = "batch"
            scope_id = str(batch_id)
            group_by_stock = True

        res = handle_realtime_prediction(
            message=message,
            p3_1_log=p3_1_log,
            algorithm=algo,
            scope=scope,
            scope_id=scope_id,
            group_by_stock=group_by_stock,
            resample_seconds=int(resample_seconds or 60),
            dry_run=False,
        )

        # Useful single-line status for debugging
        if p3_1_log:
            p3_1_log.info(
                f"[execute_phase_three][FAST_PRED] scope={scope} scope_id={scope_id} "
                f"ok={res.get('ok')} wrote={res.get('wrote')} reason={res.get('reason')}"
            )

        return res

    except Exception as e:
        if p3_1_log:
            p3_1_log.error(f"[execute_phase_three][FAST_PRED] failed: {e}", exc_info=True)
        return {"ok": False, "wrote": False, "reason": f"fast_pred_error:{e}"}


# -------- Phase 3 main loop -------- #
def execute_phase_three():
    """
    Phase 3 live loop:
    - Poll Kafka
    - Normalize message (plId filter optional, refNo/joRef mapping)
    - Persist raw message to Cassandra raw table
    - Buffer per pid (joOpId) and per wsId
    - Fire PID/WS tasks using _should_fire()
    - Fetch history via (DW snapshot -> RAW_TABLE fallback -> buffer fallback)
    - Run correlations + realtime predictions (+ FI) via _run_3_tasks_and_wait()

    Notes:
    - This function assumes the helper functions/vars already exist in this module,
      as in your current detailed codebase:
        kafka_consumer3, dw_tbl_raw_data, _is_message_valid, _extract_stock_from_prodlist_message,
        _to_dt, _should_fire, fetch_for_optc_stock_via_dw, fetch_latest_for_optc_stock_via_raw_table,
        build_sensor_values_from_buffer_for_optc_stock, fetch_latest_for_ws_via_dw,
        fetch_latest_for_ws_via_raw_table, build_sensor_values_from_ws_buffer,
        _clean_stock, _sensor_values_has_stock, _count_points_for_stock, _run_3_tasks_and_wait,
        buffer_for_process, buffer_for_ws,
        NONE_LIMIT, PID_COUNT_THRESHOLD, PID_TIME_THRESHOLD, WS_COUNT_THRESHOLD, WS_TIME_THRESHOLD.
    """
    p3_1_log.info("[execute_phase_three] Initializing Phase 3")

    global consumer3

    # ---------------------------------------------------------------------
    # 0) Take an initial DW snapshot (fast cache). RAW_TABLE fallback exists.
    # ---------------------------------------------------------------------
    try:
        rows, input_list, output_list, _batch = dw_tbl_raw_data.fetchData(limit=20_000)
        p3_1_log.info(f"[execute_phase_three] DW snapshot loaded rows={len(rows) if rows else 0}")
    except Exception as e:
        p3_1_log.error(f"[execute_phase_three] dw_tbl_raw_data.fetchData failed: {e}", exc_info=True)
        return

    # ---------------------------------------------------------------------
    # 1) Ensure consumer exists (IMPORTANT: do not create at import-time)
    # ---------------------------------------------------------------------
    if consumer3 is None:
        consumer3 = kafka_consumer3()

    none_counter = 0

    while True:
        # If consumer disappeared, recreate
        if consumer3 is None:
            p3_1_log.warning("[execute_phase_three] consumer3 is None -> recreating")
            consumer3 = kafka_consumer3()
            if consumer3 is None:
                p3_1_log.error("[execute_phase_three] kafka_consumer3() returned None; retrying in 2s")
                time.sleep(2.0)
                continue

        # -------------------------
        # Poll Kafka
        # -------------------------
        try:
            msg = consumer3.poll(1.0)
        except Exception as e:
            p3_1_log.error(f"[execute_phase_three] consumer.poll failed: {e}", exc_info=True)
            try:
                consumer3.close()
            except Exception:
                pass
            consumer3 = None
            time.sleep(2.0)
            continue

        # -------------------------
        # None handling
        # -------------------------
        if msg is None:
            none_counter += 1

            # Poll returning None is normal when topic is idle; log occasionally.
            if none_counter in (1, 10, 30, 60) or none_counter % 300 == 0:
                p3_1_log.warning(f"[execute_phase_three] Kafka Message is None (consecutive={none_counter})")
            else:
                p3_1_log.debug(f"[execute_phase_three] Kafka Message is None (consecutive={none_counter})")

            # Soft restart if too many consecutive None
            if none_counter >= NONE_LIMIT:
                p3_1_log.warning(
                    f"[execute_phase_three] Too many consecutive None messages ({none_counter}); restarting Kafka consumer"
                )
                try:
                    consumer3.close()
                except Exception as e:
                    p3_1_log.warning(f"[execute_phase_three] Error while closing consumer: {e}")
                consumer3 = None
                none_counter = 0
            continue

        # got a message
        none_counter = 0

        # -------------------------
        # Error handling
        # -------------------------
        if msg.error():
            p3_1_log.error(f"[execute_phase_three] Erroneous Message: {msg.error()}")

            # optional: hard restart after error
            try:
                consumer3.close()
            except Exception:
                pass
            consumer3 = None
            continue

        # -------------------------
        # Decode JSON
        # -------------------------
        try:
            raw_value = msg.value()
            if raw_value is None:
                p3_1_log.warning("[execute_phase_three] msg.value() is None -> skip")
                continue

            if isinstance(raw_value, dict):
                message = raw_value
            else:
                message = json.loads(raw_value.decode("utf-8"))
        except Exception as e:
            p3_1_log.error(f"[execute_phase_three] Failed to decode Kafka message: {e}", exc_info=True)
            continue

        # -------------------------
        # Validate schema early
        # -------------------------
        try:
            if not _is_message_valid(message):
                p3_1_log.info("[execute_phase_three] Skipping invalid message")
                continue
        except Exception as e:
            p3_1_log.warning(f"[execute_phase_three] _is_message_valid exception: {e} -> skipping", exc_info=True)
            continue

        # -------------------------
        # Optional test-plant filter (keep it if you need it)
        # -------------------------
        pl_raw = message.get("plId")
        pl_id = None
        try:
            pl_id = int(pl_raw) if pl_raw not in (None, "", "None") else None
        except Exception:
            pl_id = None

        # If you use the "test topic plant 76 only" rule, keep this on:
        # (Otherwise, comment out.)
        if pl_id is not None and pl_id != 149:
            p3_1_log.info("[execute_phase_three] Skipping non-test plant (plId != 149)")
            continue

        # -------------------------
        # Normalize reference numbers (batch identity)
        # -------------------------
        # If refNo is empty/0, promote joRef into refNo so main logic stays stable.
        if message.get("joRef") not in (None, "", "None", 0, "0") and message.get("refNo") in (None, "", "None", 0, "0"):
            message["refNo"] = message["joRef"]

        # Canonical batch key
        if message.get("refNo") not in (None, "", "None", 0, "0"):
            message["prod_order_reference_no"] = str(message["refNo"])
            message.setdefault("job_order_reference_no", str(message["refNo"]))

        # If still missing, normalize to something safe
        if not message.get("prod_order_reference_no"):
            message["job_order_reference_no"] = message.get("job_order_reference_no") or 0
            message["prod_order_reference_no"] = 0
            message["joRef"] = message.get("joRef") or 0
            message["refNo"] = message.get("refNo") or 0

        # -------------------------
        # Extract stock from prodList and attach to message
        # -------------------------
        try:
            st_no, st_nm = _extract_stock_from_prodlist_message(message.get("prodList"))
        except Exception:
            st_no, st_nm = None, None
        message["output_stock_no"] = st_no
        message["output_stock_name"] = st_nm

        # Normalize operation fields expected by downstream code
        message["operationname"] = message.get("opNm")
        message["operationno"] = message.get("opNo")
        message["operationtaskcode"] = message.get("opTc")

        # ---------------------------------------------------------------------
        # Save raw Kafka message into Cassandra raw table (DW)
        # ---------------------------------------------------------------------
        try:
            _ = dw_tbl_raw_data.saveData(message)
        except Exception as e:
            # For live Kafka, we do NOT want a Cassandra hiccup to kill the loop.
            p3_1_log.error(f"[execute_phase_three] dw_tbl_raw_data.saveData failed: {e}", exc_info=True)

        # ---------------------------------------------------------------------
        # Extract core routing identifiers
        # ---------------------------------------------------------------------
        wsSt = message.get("prSt")        # production state (e.g. "PRODUCTION")
        pid = message.get("joOpId")       # operation instance id (pid)
        ws_id = message.get("wsId")       # workstation id
        crDt_msg = _to_dt(message.get("crDt"))
        now_epoch = time.time()

        if pid in (None, "", "None"):
            p3_1_log.warning("[execute_phase_three] Missing joOpId (pid). Skipping message.")
            continue

        # Canonical batch id (used for scope_mode='batch' predictions and any buffering)
        batch_id = str(
            message.get("prod_order_reference_no")
            or message.get("refNo")
            or message.get("joRef")
            or "0"
        )

        # ---------------------------------------------------------------------
        # FAST PATH: predict & save for EVERY message (in addition to throttled tasks)
        # ---------------------------------------------------------------------
        _predict_and_save_every_message(
            message=message,
            pid=pid,
            batch_id=batch_id,
            p3_1_log=p3_1_log,
            resample_seconds=RESAMPLE_SECONDS,
            scope_mode=PREDICTION_SCOPE_MODE,   # respects config.json: 'batch' or 'pid'
        )

        cnt = buffer_for_process.get(pid, {}).get("count", 0)
        p3_1_log.info(
            f"[execute_phase_three] Incoming -- plId={message.get('plId')}, "
            f"batch={message.get('prod_order_reference_no')}, pid={pid}, wsId={ws_id}, wsSt={wsSt}, "
            f"in_pid_buffer={pid in buffer_for_process}, count={cnt}, stock={message.get('output_stock_name')}"
        )

        # ---------------------------------------------------------------------
        # Buffer management
        # ---------------------------------------------------------------------
        batch_id = str(message.get("prod_order_reference_no") or message.get("refNo") or message.get("joRef") or "0")
        if wsSt == "PRODUCTION":
            _LAST_PROD_MESSAGE_BY_BATCH[batch_id] = message
            # collect messages for finalize
            _BATCH_MESSAGE_BUFFER.setdefault(batch_id, []).append(message)
            # --- PID buffer ---
            if pid not in buffer_for_process:
                buffer_for_process[pid] = {
                    "count": 1,
                    "first_seen": now_epoch,
                    "last_time": now_epoch,
                    "first_crdt": crDt_msg,
                    "messages": [message],
                }
                p3_1_log.info(f"[execute_phase_three] Initialized PID buffer for pid={pid}")
            else:
                b = buffer_for_process[pid]
                b["count"] = b.get("count", 0) + 1
                b["last_time"] = now_epoch
                if b.get("first_crdt") is None and crDt_msg:
                    b["first_crdt"] = crDt_msg
                b.setdefault("messages", []).append(message)

            # --- WS buffer ---
            if ws_id is not None:
                if ws_id not in buffer_for_ws:
                    buffer_for_ws[ws_id] = {
                        "count": 1,
                        "first_seen": now_epoch,
                        "last_time": now_epoch,
                        "first_crdt": crDt_msg,
                        "messages": [message],
                    }
                    p3_1_log.info(f"[execute_phase_three] Initialized WS buffer for wsId={ws_id}")
                else:
                    w = buffer_for_ws[ws_id]
                    w["count"] = w.get("count", 0) + 1
                    w["last_time"] = now_epoch
                    if w.get("first_crdt") is None and crDt_msg:
                        w["first_crdt"] = crDt_msg
                    w.setdefault("messages", []).append(message)

            # -----------------------------------------------------------------
            # PID-level trigger + tasks
            # -----------------------------------------------------------------
            if pid in buffer_for_process:
                b = buffer_for_process[pid]
                fire_pid, pid_cnt, pid_time_lapsed = _should_fire(
                    b, PID_COUNT_THRESHOLD, PID_TIME_THRESHOLD, now_epoch
                )

                p3_1_log.info(
                    f"[execute_phase_three] PID buffer pid={pid} count={pid_cnt}, time_lapsed={pid_time_lapsed}"
                )

                if fire_pid:
                    batch_id = message.get("prod_order_reference_no") or message.get("refNo")
                    op_tc = message.get("operationtaskcode") or message.get("opTc")
                    stock_no = message.get("output_stock_no")
                    ws_id_current = message.get("wsId")

                    op_tc = None if op_tc in (None, "", "None") else str(op_tc)
                    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)

                    p3_1_log.info(
                        f"[execute_phase_three] PID={pid} FIRE -> fetching data for op_tc={op_tc}, stock={stock_no}, ws_id={ws_id_current}"
                    )

                    # (1) DW snapshot by op_tc + stock (NOT by pid)
                    job_ids, dates, ids, sensor_values_pid = fetch_for_optc_stock_via_dw(
                        rows,
                        input_list,
                        output_list,
                        _batch,
                        op_tc=op_tc,
                        stock_no=stock_no,
                        ws_id=ws_id,
                        pid=None,
                        prod_order_reference_no=batch_id,   # <-- ADD
                        max_rows=20000,
                    )

                    # (2) RAW_TABLE fallback
                    if not sensor_values_pid:
                        p3_1_log.info(
                            f"[execute_phase_three] pid={pid} — DW returned 0 bundles; trying RAW_TABLE fallback"
                        )
                        job_ids_pid, dates_pid, ids_pid, sensor_values_pid = fetch_latest_for_optc_stock_via_raw_table(
                            op_tc=op_tc,
                            stock_no=stock_no,
                            ws_id=ws_id_current,
                            pid=None,
                            prod_order_reference_no=batch_id,
                            max_rows=20_000,
                        )

                    # (3) Buffer fallback: scan all PID buffers
                    if not sensor_values_pid:
                        all_buffered_messages = []
                        for _pid, buf_entry in buffer_for_process.items():
                            all_buffered_messages.extend(buf_entry.get("messages", []))

                        p3_1_log.info(
                            f"[execute_phase_three] pid={pid} — RAW_TABLE returned 0; trying buffer fallback (total_buffered_messages={len(all_buffered_messages)})"
                        )
                        job_ids_pid, dates_pid, ids_pid, sensor_values_pid = build_sensor_values_from_buffer_for_optc_stock(
                            all_buffered_messages,
                            op_tc=op_tc,
                            stock_no=stock_no,
                            ws_id=ws_id_current,
                            pid=None,
                            prod_order_reference_no=batch_id,
                        )

                    if not sensor_values_pid:
                        p3_1_log.info(
                            f"[execute_phase_three] pid={pid} — no data after DW+RAW+buffer for op_tc={op_tc}, stock={stock_no}; skipping"
                        )
                    else:
                        p3_1_log.info(
                            f"[execute_phase_three] pid={pid} fetched points={len(dates_pid)} for op_tc={op_tc}, stock={stock_no}"
                        )
                        try:
                            _run_3_tasks_and_wait(
                                sensor_values=sensor_values_pid,
                                message=message,
                                dates=dates_pid,
                                input_list=input_list,
                                output_list=output_list,
                                p3_1_log=p3_1_log,
                                scope="pid",
                                scope_id=pid,
                                corr_group_by_output_stock=False,
                                pred_group_by_stock=True,
                                pred_algorithm="RANDOM_FOREST",
                                pred_epochs=125,
                                pred_min_train_points=250,
                                fi_algorithm="XGBOOST",
                            )
                        except Exception as e:
                            p3_1_log.error(
                                f"[execute_phase_three] PID parallel runner failed for pid={pid}: {e}",
                                exc_info=True
                            )

                    # Reset PID buffer after firing attempt
                    b["count"] = 0
                    b["first_seen"] = b.get("last_time", now_epoch)
                    b["first_crdt"] = None
                    b["messages"] = []
                else:
                    p3_1_log.info(
                        f"[execute_phase_three] PID={pid} not firing (cnt={pid_cnt}, elapsed={pid_time_lapsed})"
                    )

            # -----------------------------------------------------------------
            # WS-level trigger + tasks
            # -----------------------------------------------------------------
            if ws_id is not None and ws_id in buffer_for_ws:
                w = buffer_for_ws[ws_id]
                fire_ws, ws_cnt, ws_time_lapsed = _should_fire(
                    w, WS_COUNT_THRESHOLD, WS_TIME_THRESHOLD, now_epoch
                )

                p3_1_log.info(
                    f"[execute_phase_three] WS buffer wsId={ws_id} count={ws_cnt}, time_lapsed={ws_time_lapsed}"
                )

                if fire_ws:
                    p3_1_log.info(
                        f"[execute_phase_three] WS={ws_id} FIRE -> fetching ws-level data (cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                    )

                    # (1) DW snapshot
                    job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_dw(
                        ws_id, rows, input_list, output_list, _batch, max_rows=20_000
                    )

                    # (2) RAW_TABLE fallback
                    if not sensor_values_ws:
                        p3_1_log.info(
                            f"[execute_phase_three] wsId={ws_id} — DW returned 0 bundles; trying RAW_TABLE fallback"
                        )
                        job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_raw_table(
                            ws_id, max_rows=20_000
                        )

                    # (3) WS buffer fallback
                    if not sensor_values_ws:
                        buffered_n = len(w.get("messages", []))
                        p3_1_log.info(
                            f"[execute_phase_three] wsId={ws_id} — RAW_TABLE returned 0; trying WS buffer fallback (buffered_messages={buffered_n})"
                        )
                        job_ids_ws, dates_ws, ids_ws, sensor_values_ws = build_sensor_values_from_ws_buffer(
                            ws_id, w.get("messages", [])
                        )

                    if not sensor_values_ws:
                        p3_1_log.info(
                            f"[execute_phase_three] wsId={ws_id} — no data after DW+RAW+buffer; skipping"
                        )
                    else:
                        # Stock gate (avoid mixing stock contexts)
                        skip_ws = False
                        current_stock = _clean_stock(message.get("output_stock_no"))

                        if current_stock is None:
                            p3_1_log.warning(
                                f"[execute_phase_three] wsId={ws_id} group_by_output_stock=True but current message has no output_stock_no -> skip ws tasks"
                            )
                            skip_ws = True
                        else:
                            has_stock = _sensor_values_has_stock(sensor_values_ws, current_stock)
                            pts = _count_points_for_stock(sensor_values_ws, current_stock)

                            if (not has_stock) or (pts < 2):
                                p3_1_log.info(
                                    f"[execute_phase_three] wsId={ws_id} stock gate: current_stock={current_stock} "
                                    f"has_stock={has_stock} points={pts} -> skip ws tasks"
                                )
                                skip_ws = True

                        if not skip_ws:
                            try:
                                _run_3_tasks_and_wait(
                                    sensor_values=sensor_values_ws,
                                    message=message,
                                    dates=dates_ws,
                                    input_list=input_list,
                                    output_list=output_list,
                                    p3_1_log=p3_1_log,
                                    scope="ws",
                                    scope_id=ws_id,
                                    corr_group_by_output_stock=True,
                                    pred_group_by_stock=True,
                                    pred_algorithm="RANDOM_FOREST",
                                    pred_epochs=125,
                                    pred_min_train_points=250,
                                    fi_algorithm="XGBOOST",
                                )
                            except Exception as e:
                                p3_1_log.error(
                                    f"[execute_phase_three] WS parallel runner failed for wsId={ws_id}: {e}",
                                    exc_info=True
                                )

                            # Reset WS buffer on run
                            w["count"] = 0
                            w["first_seen"] = w.get("last_time", now_epoch)
                            w["first_crdt"] = None
                            w["messages"] = []
                        # else: keep WS buffer so next message can re-trigger correctly

                else:
                    p3_1_log.info(
                        f"[execute_phase_three] wsId={ws_id} not firing (cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                    )

        else:
            batch_id = str(message.get("prod_order_reference_no") or message.get("refNo") or message.get("joRef") or "0")
            last_prod = _LAST_PROD_MESSAGE_BY_BATCH.get(batch_id)
            batch_msgs = _BATCH_MESSAGE_BUFFER.get(batch_id, [])

            try:
                fin = finalize_batch_prediction(
                    batch_id=batch_id,
                    last_prod_message=last_prod,
                    batch_messages=batch_msgs,
                    p3_1_log=p3_1_log
                )
                p3_1_log.info(f"[execute_phase_three] finalize_batch_prediction status={fin}")
            except Exception as e:
                p3_1_log.error(f"[execute_phase_three] finalize_batch_prediction failed: {e}", exc_info=True)

            # cleanup
            _BATCH_MESSAGE_BUFFER.pop(batch_id, None)
            _LAST_PROD_MESSAGE_BY_BATCH.pop(batch_id, None)

            p3_1_log.info(
                f"[execute_phase_three] ws not in PRODUCTION -> clearing buffers pid={pid}, wsId={ws_id}, wsSt={wsSt}"
            )
            buffer_for_process.pop(pid, None)
            if ws_id is not None:
                buffer_for_ws.pop(ws_id, None)


import queue
import traceback

def _run_3_tasks_and_wait(
    *,
    sensor_values,
    message,
    dates,
    input_list,
    output_list,
    p3_1_log,
    scope: str,
    scope_id,
    # correlation
    corr_group_by_output_stock: bool,
    corr_algorithm: str = "SPEARMAN",
    # prediction (DISABLED HERE; fast path handles per-message prediction)
    pred_group_by_stock: bool = True,
    pred_algorithm: str = "RANDOM_FOREST",
    pred_lookback: int = 20,
    pred_epochs: int = 50,
    pred_min_train_points: int = 250,
    # feature importance
    fi_algorithm: str = "XGBOOST",
    fi_produce_col: str = "prSt",
    fi_good_col: str = "good",
    fi_drop_cols=("ts", "joOpId", "wsId"),
):
    """
    Runs (1) correlation, (2) feature importance in parallel threads.
    NOTE: Realtime prediction is intentionally DISABLED here because we now run
    prediction for EVERY Kafka message in execute_phase_three() (FAST_PRED path).

    This keeps correlation + FI periodic, while prediction is always-on.
    """
    err_q = queue.Queue()

    def _wrap(name, fn):
        try:
            fn()
        except Exception as e:
            tb = traceback.format_exc()
            err_q.put((name, str(e), tb))

    # ---- task 1: correlation ----
    def _task_correlation():
        compute_correlation(
            sensor_values,
            message,
            p3_1_log=p3_1_log,
            algorithm=corr_algorithm,
            scope=scope,
            scope_id=scope_id,
            group_by_output_stock=corr_group_by_output_stock,
        )

    # ---- task 2: feature importance ----
    def _task_feature_importance():
        compute_and_save_feature_importance(
            sensor_values,
            message,
            input_feature_names=input_list,
            output_feature_names=output_list,
            produce_col=fi_produce_col,
            good_col=fi_good_col,
            drop_cols=fi_drop_cols,
            algorithm=fi_algorithm,
            p3_1_log=p3_1_log,
            scope=scope,
            scope_id=scope_id,
        )

    t1 = threading.Thread(target=lambda: _wrap("correlation", _task_correlation), daemon=True)
    t2 = threading.Thread(target=lambda: _wrap("feature_importance", _task_feature_importance), daemon=True)

    start = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.time() - start

    had_err = False
    while not err_q.empty():
        had_err = True
        name, msg, tb = err_q.get()
        p3_1_log.error(f"[parallel_task:{name}] failed: {msg}\n{tb}")

    p3_1_log.info(
        f"[execute_phase_three] Parallel tasks done for scope={scope} scope_id={scope_id} "
        f"(elapsed={elapsed:.3f}s, errors={had_err})"
    )
