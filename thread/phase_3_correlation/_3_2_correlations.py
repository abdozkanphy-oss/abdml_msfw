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

# Helper functions for correlation matrix
from thread.phase_3_correlation._3_1_helper_functions import map_to_text, compute_correlation, _is_input_type, aggregate_correlation_data

# Real-time LSTM prediction imports
from thread.phase_3_correlation._3_3_predictions import handle_realtime_prediction, history_from_fetch
from thread.phase_3_correlation._3_5_feature_importance import compute_and_save_feature_importance


consumer3 = kafka_consumer3()
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

def _should_fire(buffer_entry, min_points, max_interval_sec, now_epoch=None):
    """
    Decide whether a buffer has enough points or time to trigger processing.

    Returns: (fire_bool, count, time_lapsed)
    """
    if now_epoch is None:
        now_epoch = time.time()

    cnt = buffer_entry.get("count", 0)
    first_seen = buffer_entry.get("first_seen", now_epoch)
    time_lapsed = now_epoch - first_seen

    fire = (cnt >= 2) or (time_lapsed >= max_interval_sec)
    return True, cnt, time_lapsed #fire, cnt, time_lapsed

p3_1_log = setup_logger(
    "p3_1_logger", "logs/p3_1.log"
)

RAW_TABLE = cfg["cassandra_props"]["raw_data_table"]  # "dw_tbl_raw_data"


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
            #p3_1_log.debug(f"[_count_points_for_stock] Skipping bundle with stock_no={st}")
            continue

        cr = meta.get("crDt")
        if cr is not None:
            crdts.add(str(cr))
        else:
            cnt += 1

    return len(crdts) if crdts else cnt


def fetch_for_optc_stock_via_dw(rows, input_list, output_list, _batch, 
                                 op_tc=None, stock_no=None, 
                                 ws_id=None, pid=None, max_rows: int = 20000):
    """
    Fetch data by operation task code + stock (NOT by single PID).
    Optionally filter by ws_id or pid for additional constraints.
    
    This enables training across ALL PIDs with same op_tc + stock.
    """
    bucket = defaultdict(list)
    label_meta = {}
    pid_by_ts = {}
    ws_by_ts = {}
    
    # Clean inputs
    op_tc = None if op_tc in (None, "", "None") else str(op_tc)
    stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)
    
    if not op_tc or not stock_no:
        p3_1_log.warning(
            f"[fetch_for_optc_stock_via_dw] Missing op_tc={op_tc} or stock_no={stock_no}, returning empty"
        )
        return [], [], [], []
    
    combined_list = _combine_input_output_lists(input_list, output_list)
    
    for r, wrapped_ov in zip(rows, combined_list):
        # ✅ PRIMARY FILTER: operation task code
        row_optc = getattr(r, "operationtaskcode", None)
        if row_optc != op_tc:
            continue
        
        # ✅ PRIMARY FILTER: stock
        row_stock = getattr(r, "produced_stock_no", None)
        row_stock = None if row_stock in (None, "", "None", "nan", "NaN") else str(row_stock)
        if row_stock != stock_no:
            continue
        
        # ✅ OPTIONAL FILTER: workstation (eğer belirtilmişse)
        if ws_id is not None:
            row_ws = getattr(r, "work_station_id", None) or getattr(r, "workstation_id", None)
            if row_ws != ws_id:
                continue
        
        # ✅ OPTIONAL FILTER: pid (eğer sadece o PID'i istiyorsak - genelde istemeyiz)
        if pid is not None:
            row_pid = getattr(r, "job_order_operation_id", None)
            if row_pid != pid:
                continue
        
        ts = getattr(r, "measurement_date", None)
        if ts is None:
            continue
        
        # Track which PID/WS this timestamp belongs to
        if ts not in pid_by_ts:
            pid_by_ts[ts] = getattr(r, "job_order_operation_id", None)
        if ts not in ws_by_ts:
            ws_by_ts[ts] = getattr(r, "work_station_id", None)
        
        # Label/meta
        if ts not in label_meta:
            label_meta[ts] = {
                "good": getattr(r, "good", None),
                "prSt": getattr(r, "workstation_state", None),
                "job_order_reference_no": getattr(r, "job_order_reference_no", None),
                "prod_order_reference_no": getattr(r, "prod_order_reference_no", None),
                "output_stock_no": row_stock,
                "output_stock_name": getattr(r, "produced_stock_name", None),
                "operationname": getattr(r, "operationname", None),
                "operationno": getattr(r, "operationno", None),
                "operationtaskcode": row_optc,
                "job_order_operation_id": getattr(r, "job_order_operation_id", None),
                "work_station_id": getattr(r, "work_station_id", None) or getattr(r, "workstation_id", None),
            }
        
        # Sensor data
        ov = wrapped_ov[0] if wrapped_ov else {}
        if ov.get("equipment_type", True):  # INPUT
            var_name = ov.get("varNo")
            var_value = ov.get("genReadVal")
            
            if var_name and var_value is not None:
                bucket[ts].append({
                    "parameter": str(var_name),
                    "counter_reading": _num_text(var_value),
                    "equipment_name": ov.get("varNm", var_name),
                    "equipment_type": True,
                })
        else:  # OUTPUT
            param = ov.get("eqNo")
            cval = ov.get("cntRead")
            eq_name = ov.get("eqNm")
            
            if param and cval is not None:
                bucket[ts].append({
                    "parameter": str(param),
                    "counter_reading": _num_text(cval),
                    "equipment_name": str(eq_name),
                    "equipment_type": False,
                })
    
    if not bucket:
        p3_1_log.info(
            f"[fetch_for_optc_stock_via_dw] 0 rows for op_tc={op_tc}, stock={stock_no}"
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
        f"[fetch_for_optc_stock_via_dw] Found {len(sensor_values)} bundles for "
        f"op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, pid={pid}"
    )
    return job_ids, dates, ids, sensor_values

def build_sensor_values_from_buffer_for_optc_stock(messages, op_tc=None, stock_no=None, 
                                                     ws_id=None, pid=None):
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
        # ✅ Filter by operation task code
        msg_optc = msg.get("operationtaskcode") or msg.get("opTc")
        if msg_optc != op_tc:
            continue
        
        # ✅ Filter by stock
        msg_stock, msg_stock_name = _extract_stock_from_prodlist_message(msg.get("prodList"))
        msg_stock = None if msg_stock in (None, "", "None", "nan", "NaN") else str(msg_stock)
        if msg_stock != stock_no:
            continue
        
        # ✅ Optional filter by workstation
        if ws_id is not None:
            msg_ws = msg.get("wsId")
            if msg_ws != ws_id:
                continue
        
        # ✅ Optional filter by pid
        if pid is not None:
            msg_pid = msg.get("joOpId")
            if msg_pid != pid:
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

    # Build WHERE clause dynamically
    where = "WHERE operationtaskcode = %s AND produced_stock_no = %s"
    params = [op_tc, stock_no]

    if ws_id is not None:
        where += " AND work_station_id = %s"
        params.append(ws_id)
    
    #if pid is not None:
        #where += " AND job_order_operation_id = %s"
        #params.append(pid)

    query = (
        f"SELECT job_order_operation_id, work_station_id, measurement_date, "
        f"equipment_no, equipment_name, counter_reading, gen_read_val, equipment_type, "
        f"good, workstation_state, job_order_reference_no, prod_order_reference_no, "
        f"produced_stock_no, produced_stock_name, operationname, operationno, operationtaskcode "
        f"FROM {RAW_TABLE} "
        f"{where} "
        f"LIMIT %s ALLOW FILTERING"
    )

    params.append(max_rows)

    try:
        rows = list(session.execute(query, tuple(params)))
    except Exception as e:
        p3_1_log.warning(
            f"[fetch_latest_for_optc_stock_via_raw_table] query failed for "
            f"op_tc={op_tc}, stock={stock_no}: {e}"
        )
        return [], [], [], []

    if not rows:
        p3_1_log.info(
            f"[fetch_latest_for_optc_stock_via_raw_table] 0 rows for "
            f"op_tc={op_tc}, stock={stock_no}, ws_id={ws_id}, pid={pid}"
        )
        return [], [], [], []

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
        f"[fetch_latest_for_optc_stock_via_raw_table] rows={len(rows)}, "
        f"bundles={len(sensor_values)} for op_tc={op_tc}, stock={stock_no}, "
        f"ws_id={ws_id}, pid={pid}"
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

# -------- Phase 3 main loop -------- #
def execute_phase_three():
    p3_1_log.info("[execute_phase_three] Initializing Phase 3")

    global consumer3

    # İlk başta DW snapshot'ı al
    try:
        rows, input_list, output_list, _batch = dw_tbl_raw_data.fetchData(limit=20_000)
    except Exception as e:
        p3_1_log.error(f"[dw_tbl_raw_data.fetchData] fetchData failed: {e}")
        return [], [], [], []

    # Ardışık None sayacı
    none_counter = 0

    try:
        while True:        
            # 1) Kafka'dan mesaj çek
            msg = consumer3.poll(1.0)
            # ---- NONE HANDLING (soft restart) ----
            if msg is None:
                none_counter += 1
                p3_1_log.warning(
                    f"[execute_phase_three] Kafka Message is None "
                    f"(consecutive={none_counter})"
                )

                if none_counter >= NONE_LIMIT:
                    p3_1_log.warning(
                        "[execute_phase_three] Too many consecutive None messages "
                        f"({none_counter}), restarting Kafka consumer"
                    )
                    # Eski consumer'i kapat
                    try:
                        consumer3.close()
                    except Exception as e:
                        p3_1_log.warning(
                            f"[execute_phase_three] Error while closing consumer: {e}"
                        )

                    # Aynı group.id ve config ile yeni consumer
                    try:
                        consumer3 = kafka_consumer3()
                        p3_1_log.info(
                            "[execute_phase_three] Kafka consumer recreated successfully "
                            "after None storm"
                        )
                    except Exception as e:
                        p3_1_log.error(
                            f"[execute_phase_three] Failed to recreate consumer: {e}",
                            exc_info=True,
                        )

                    # Sayaç sıfırlansın, buffer'lara dokunmuyoruz
                    none_counter = 0

                # Bu iterasyonu atla
                continue

            # Buraya geldiysek gerçek bir mesaj var, sayaç sıfırla
            none_counter = 0

            # ---- ERROR HANDLING (opsiyonel: burada da restart edebilirsin) ----
            if msg.error():
                p3_1_log.error(
                    f"[execute_phase_three] Errornous Message: {msg.error()}"
                )
                # İstersen burada da aynı soft-restart mantığını kullan:
                try:
                    consumer3.close()
                except Exception as e:
                    p3_1_log.warning(
                        f"[execute_phase_three] Error while closing consumer after error: {e}"
                    )
                try:
                    consumer3 = kafka_consumer3()
                    p3_1_log.info(
                        "[execute_phase_three] Kafka consumer recreated after error"
                    )
                except Exception as e:
                    p3_1_log.error(
                        f"[execute_phase_three] Failed to recreate consumer after error: {e}",
                        exc_info=True,
                    )
                continue
            
            """try:
                rows, input_list, output_list, _batch = dw_tbl_raw_data.fetchData(limit=20_000)
            except Exception as e:
                p3_1_log.error(f"[dw_tbl_raw_data.fetchData] fetchData failed: {e}")
                return [], [], [], []"""
            
            try:
                raw_value = msg.value().decode("utf-8")
                message = json.loads(raw_value)

                if not _is_message_valid(message):
                    p3_1_log.info("[execute_phase_three] Skipping invalid message")
                    continue

                # Map joRef from Kafka message into raw table column name
                if "joRef" in message:
                    message["job_order_reference_no"] = message["joRef"]

                if "refNo" in message:
                    message["prod_order_reference_no"] = message["refNo"]

                # NEW: make sure missing refs become 0, without KeyError
                if not message.get("prod_order_reference_no"):
                    message["job_order_reference_no"] = 0
                    message["prod_order_reference_no"] = 0
                    message["joRef"] = 0
                    message["refNo"] = 0

                # NEW: extract stock from prodList and put on message
                st_no, st_nm = _extract_stock_from_prodlist_message(
                    message.get("prodList")
                )
                message["output_stock_no"] = st_no
                message["output_stock_name"] = st_nm

                message["operationname"] = message.get("opNm")
                message["operationno"] = message.get("opNo")
                message["operationtaskcode"] = message.get("opTc")

                # DW raw table'a tek satır kaydet
                _ = dw_tbl_raw_data.saveData(message)  # her şey buna göre set edilmiş

                
                # Belirli müşteri skip
                """if message.get("outVals") and message["outVals"][0].get("cust") == "teknia_group":
                    p3_1_log.info("[execute_phase_three] Skipping teknia_group customer")
                    continue

                if message.get("plId") and message["plId"] == 20:
                    p3_1_log.info("[execute_phase_three] Skipping Savola customer")
                    continue

                if message.get("plId") and message["plId"] == 162:
                    p3_1_log.info("[execute_phase_three] Skipping Meriç customer")
                    continue
                
                if message.get("prodList") and message["prodList"][0].get("stNo") == "Antares PV":
                    p3_1_log.info("[execute_phase_three] Skipping Antares PV stock")
                    continue

                if message.get("prodList") and message["prodList"][0].get("stNo") == "Loperamide 2 mg granulate":
                    p3_1_log.info("[execute_phase_three] Skipping Loperamide 2 mg granulate stock")
                    continue"""
                
                plant = message.get("plId")
                wsSt = message["prSt"]
                pid = message["joOpId"]
                ws_id = message.get("wsId")
                crDt_msg = _to_dt(message.get("crDt"))
                now_epoch = time.time()

                cnt = buffer_for_process.get(pid, {}).get("count", 0)
                p3_1_log.info(
                    f"[execute_phase_three] Initializing for -- plId={plant}, prod={message.get('refNo')}, pid={pid}, wsId={ws_id}, "
                    f"wsSt={wsSt}, In Buffer={pid in buffer_for_process}, count={cnt}"
                )

                # --- PID buffer yönetimi ---
                if wsSt == "PRODUCTION":
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
                        # mesajı buffer'a ekle
                        b.setdefault("messages", []).append(message)

                # --- WS buffer yönetimi ---
                if wsSt == "PRODUCTION" and ws_id is not None:
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

                # --- PID / WS için threshold check ---
                if wsSt == "PRODUCTION":
                    # ---------- PID-LEVEL CHECK ----------
                    if pid in buffer_for_process:
                        b = buffer_for_process[pid]
                        fire_pid, pid_cnt, pid_time_lapsed = _should_fire(
                            b, PID_COUNT_THRESHOLD, PID_TIME_THRESHOLD, now_epoch
                        )

                        p3_1_log.info(
                            f"[execute_phase_three] Pid={pid}, wsId={ws_id}, wsSt={wsSt} "
                            f"Updated PID buffer count={pid_cnt}, time_lapsed={pid_time_lapsed}"
                        )

                        if fire_pid:
                            # Extract current message's op_tc and stock
                            op_tc = message.get("operationtaskcode") or message.get("opTc")
                            stock_no = message.get("output_stock_no")
                            ws_id_current = message.get("wsId")
                            
                            # Clean them
                            op_tc = None if op_tc in (None, "", "None") else str(op_tc)
                            stock_no = None if stock_no in (None, "", "None", "nan", "NaN") else str(stock_no)
                            
                            p3_1_log.info(
                                f"[execute_phase_three] PID={pid} FETCHING DATA for op_tc={op_tc}, "
                                f"stock={stock_no}, ws_id={ws_id_current}"
                            )
                            
                            # === (1) DW SNAPSHOT - by op_tc + stock (NOT by PID!) ===
                            job_ids_pid, dates_pid, ids_pid, sensor_values_pid = fetch_for_optc_stock_via_dw(
                                rows, input_list, output_list, _batch,
                                op_tc=op_tc,
                                stock_no=stock_no,
                                ws_id=ws_id_current,  # opsiyonel: sadece bu workstation
                                pid=None,              # ✅ PID filtresi YOK - tüm PID'ler gelsin
                                max_rows=20_000
                            )
                            
                            # === (2) RAW_TABLE FALLBACK ===
                            if not sensor_values_pid:
                                p3_1_log.info(
                                    f"[execute_phase_three] pid={pid} — DW returned 0 bundles; "
                                    f"trying RAW_TABLE fallback"
                                )
                                job_ids_pid, dates_pid, ids_pid, sensor_values_pid = fetch_latest_for_optc_stock_via_raw_table(
                                    op_tc=op_tc,
                                    stock_no=stock_no,
                                    ws_id=ws_id_current,
                                    pid=None,  # ✅ PID filtresi YOK
                                    max_rows=20_000
                                )
                            
                            # === (3) BUFFER FALLBACK ===
                            if not sensor_values_pid:
                                # Burada buffer'dan çekerken TÜM buffer'ı tara (sadece PID buffer değil)
                                all_buffered_messages = []
                                for buffered_pid, buf_entry in buffer_for_process.items():
                                    all_buffered_messages.extend(buf_entry.get("messages", []))
                                
                                buffered_n = len(all_buffered_messages)
                                p3_1_log.info(
                                    f"[execute_phase_three] pid={pid} — RAW_TABLE returned 0 bundles; "
                                    f"trying buffer fallback (total_buffered_messages={buffered_n})"
                                )
                                
                                job_ids_pid, dates_pid, ids_pid, sensor_values_pid = build_sensor_values_from_buffer_for_optc_stock(
                                    all_buffered_messages,
                                    op_tc=op_tc,
                                    stock_no=stock_no,
                                    ws_id=ws_id_current,
                                    pid=None  # ✅ PID filtresi YOK
                                )
                            
                            if not sensor_values_pid:
                                p3_1_log.info(
                                    f"[execute_phase_three] pid={pid} — still no data after "
                                    f"DW+RAW+buffer for op_tc={op_tc}, stock={stock_no}; skipping"
                                )
                                continue
                            
                            # ✅ Artık sensor_values_pid, AYNI op_tc + stock kombinasyonuna sahip
                            #    TÜM PID'lerden gelen datayı içerir
                            
                            p3_1_log.info(
                                f"[execute_phase_three] Fetched {len(dates_pid)} time points "
                                f"for op_tc={op_tc}, stock={stock_no} (from multiple PIDs)"
                            )
                            # --- PID: run correlation + prediction + feature-importance in parallel ---
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
                                    corr_group_by_output_stock=False,   # keep your PID behavior
                                    pred_group_by_stock=True,          # keep your PID behavior
                                    pred_algorithm="RANDOM_FOREST",
                                    pred_epochs=125,
                                    pred_min_train_points=250,
                                    fi_algorithm="XGBOOST",
                                )
                            except Exception as e:
                                p3_1_log.error(f"[execute_phase_three] PID parallel runner failed for pid={pid}: {e}", exc_info=True)


                            # ---- buffer reset policy ----
                            # If we skipped due to stock gate, DON'T clear the PID buffer,
                            # so the next message can trigger again quickly.
                            b["count"] = 0
                            b["first_seen"] = b["last_time"]
                            b["first_crdt"] = None
                            b["messages"] = []

                        else:
                            p3_1_log.info(
                                f"[execute_phase_three] Pid={pid}, wsId={ws_id}, wsSt={wsSt} "
                                f"SKIPPING PID-level Correlation (cnt={pid_cnt}, elapsed={pid_time_lapsed})"
                            )

                    # ---------- WS-LEVEL CHECK (AGGREGATED) ----------
                    if ws_id is not None and ws_id in buffer_for_ws:
                        w = buffer_for_ws[ws_id]
                        fire_ws, ws_cnt, ws_time_lapsed = _should_fire(
                            w, WS_COUNT_THRESHOLD, WS_TIME_THRESHOLD, now_epoch
                        )

                        p3_1_log.info(
                            f"[execute_phase_three] wsId={ws_id}, wsSt={wsSt} "
                            f"Updated WS buffer count={ws_cnt}, time_lapsed={ws_time_lapsed}"
                        )

                        if fire_ws:
                            p3_1_log.info(
                                f"[execute_phase_three] Fetching WS-level raw data for wsId={ws_id} "
                                f"(cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                            )

                            # (1) DW SNAPSHOT
                            job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_dw(
                                ws_id, rows, input_list, output_list, _batch, max_rows=20_000
                            )

                            # (2) RAW_TABLE FALLBACK
                            if not sensor_values_ws:
                                p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — DW returned 0 bundles; trying RAW_TABLE fallback")
                                job_ids_ws, dates_ws, ids_ws, sensor_values_ws = fetch_latest_for_ws_via_raw_table(
                                    ws_id, max_rows=20_000
                                )

                            # (3) WS BUFFER FALLBACK
                            if not sensor_values_ws:
                                buffered_n = len(w.get("messages", []))
                                p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — RAW_TABLE returned 0 bundles; trying WS buffer fallback (buffered_messages={buffered_n})")
                                job_ids_ws, dates_ws, ids_ws, sensor_values_ws = build_sensor_values_from_ws_buffer(
                                    ws_id, w.get("messages", [])
                                )

                            if not sensor_values_ws:
                                p3_1_log.info(f"[execute_phase_three] wsId={ws_id} — still no data after DW+RAW+buffer; skipping")
                                continue
                            else:
                                # ---- STOCK GATE (WS + group_by_output_stock) ----
                                skip_ws = False
                                current_stock = _clean_stock(message.get("output_stock_no"))

                                if current_stock is None:
                                    p3_1_log.warning(
                                        f"[execute_phase_three] wsId={ws_id} group_by_output_stock=True but current message has no output_stock_no. "
                                        "Skipping WS correlation/LSTM/feature-importance to avoid mixing stocks."
                                    )
                                    skip_ws = True
                                else:
                                    has_stock = _sensor_values_has_stock(sensor_values_ws, current_stock)
                                    pts = _count_points_for_stock(sensor_values_ws, current_stock)

                                    if (not has_stock) or (pts < 2):
                                        p3_1_log.info(
                                            f"[execute_phase_three] wsId={ws_id} stock gate: current_stock={current_stock} "
                                            f"has_stock={has_stock} points={pts} (<2 or missing). "
                                            "Skipping WS correlation/LSTM/feature-importance; waiting for next message."
                                        )
                                        skip_ws = True

                                if not skip_ws:
                                    # --- WS: run correlation + prediction + feature-importance in parallel ---
                                    try:
                                        _run_3_tasks_and_wait(
                                            sensor_values=sensor_values_ws,   # (NOTE) use what you already feed to correlation/fi
                                            message=message,
                                            dates=dates_ws,
                                            input_list=input_list,
                                            output_list=output_list,
                                            p3_1_log=p3_1_log,
                                            scope="ws",
                                            scope_id=ws_id,
                                            corr_group_by_output_stock=True,  # keep your WS behavior
                                            pred_group_by_stock=True,         # keep your WS behavior
                                            pred_algorithm="RANDOM_FOREST",
                                            pred_epochs=125,
                                            pred_min_train_points=250,
                                            fi_algorithm="XGBOOST",
                                        )
                                    except Exception as e:
                                        p3_1_log.error(f"[execute_phase_three] WS parallel runner failed for wsId={ws_id}: {e}", exc_info=True)

                                # ---- WS buffer reset policy ----
                                if not skip_ws:
                                    w["count"] = 0
                                    w["first_seen"] = w["last_time"]
                                    w["first_crdt"] = None
                                    w["messages"] = []
                                else:
                                    # keep buffer so next message can re-trigger and fetch includes the stock
                                    pass


                        else:
                            p3_1_log.info(
                                f"[execute_phase_three] wsId={ws_id}, wsSt={wsSt} "
                                f"SKIPPING WS-level Correlation (cnt={ws_cnt}, elapsed={ws_time_lapsed})"
                            )

                # PRODUCTION değilse PID / WS bufferları temizle
                else:
                    p3_1_log.info(
                        f"[execute_phase_three] pid={pid}, wsId={ws_id}, ws not in PRODUCTION, "
                        f"clearing buffers, wsSt={wsSt}"
                    )
                    buffer_for_process.pop(pid, None)
                    if ws_id is not None:
                        buffer_for_ws.pop(ws_id, None)


            except Exception as e2:
                p3_1_log.error(f"[execute_phase_three] Error Level 2: {e2}", exc_info=True)

    except Exception as e1:
        p3_1_log.error(f"[execute_phase_three] Error Level 1: {e1}", exc_info=True)


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
    # prediction
    pred_group_by_stock: bool,
    pred_algorithm: str = "LSTM",
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
    Runs (1) correlation, (2) realtime prediction, (3) feature importance in parallel threads.
    Waits until all complete. Exceptions are logged; main loop continues safely.
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

    # ---- task 2: realtime prediction ----
    def _task_prediction():
        if scope == "pid":
            op_tc = message.get("operationtaskcode") or message.get("opTc")
            stock_no = message.get("output_stock_no") or (
                (message.get("prodList") or [{}])[0].get("stNo")
                if isinstance(message.get("prodList"), list) else None
            )

            # DATASET: RAW_TABLE’da opTc + stock (+ ws) eşleşen tüm rowlar
            _, dates_rt, _, sv_rt = fetch_latest_for_optc_stock_via_raw_table(
                op_tc=str(op_tc) if op_tc not in (None, "", "None") else None,
                stock_no=str(stock_no) if stock_no not in (None, "", "None", "nan", "NaN") else None,
                max_rows=20000,
                ws_id=message.get("wsId")
                # pid=message.get("joOpId"),  # istersen aç
            )

            if not sv_rt:
                p3_1_log.warning(
                    f"[prediction] RAW_TABLE returned 0 rows for opTc={op_tc}, stock={stock_no}. "
                    f"Falling back to in-memory sensor_values (mix risk)."
                )
                hist_in, hist_out = history_from_fetch(dates, sensor_values)
            else:
                p3_1_log.info(
                    f"[prediction] Fetched {len(dates_rt)} time points from RAW_TABLE for "
                    f"opTc={op_tc}, stock={stock_no}."
                )
                hist_in, hist_out = history_from_fetch(dates_rt, sv_rt)

            # seed_history olarak SADECE OUTPUT geçmişini ver
            seed = hist_out

            res = handle_realtime_prediction(
                message=message,
                lookback=(pred_lookback // 2),
                epochs=pred_epochs,
                min_train_points=pred_min_train_points,
                p3_1_log=p3_1_log,
                algorithm=pred_algorithm,
                seed_history=seed,
                scope="pid",
                scope_id=scope_id,   # joOpId (pid) kalsın
                group_by_stock=True  # key = OPTC_<opTc>_WS_<wsId>_ST_<stock>_OUTPUT / INPUT
            )

        else:
            # WS tarafı: aynı mantık, yine sadece OUTPUT history seed
            hist_in, hist_out = history_from_fetch(dates, sensor_values)
            seed = hist_out

            res = handle_realtime_prediction(
                message=message,
                lookback=pred_lookback,
                epochs=pred_epochs,
                min_train_points=pred_min_train_points,
                p3_1_log=p3_1_log,
                algorithm=pred_algorithm,
                seed_history=seed,
                scope=scope,
                scope_id=scope_id,
                group_by_stock=pred_group_by_stock,
            )

        p3_1_log.info(f"[execute_phase_three] {scope}-level realtime_prediction status={res}")


    # ---- task 3: feature importance ----
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

    # Start 3 threads
    t1 = threading.Thread(target=lambda: _wrap("correlation", _task_correlation), daemon=True)
    t2 = threading.Thread(target=lambda: _wrap("realtime_prediction", _task_prediction), daemon=True)
    t3 = threading.Thread(target=lambda: _wrap("feature_importance", _task_feature_importance), daemon=True)

    start = time.time()
    t1.start(); t2.start(); t3.start()

    # Wait all
    t1.join()
    t2.join()
    t3.join()
    elapsed = time.time() - start

    # Log any errors (don’t crash the main loop)
    had_err = False
    while not err_q.empty():
        had_err = True
        name, msg, tb = err_q.get()
        p3_1_log.error(f"[parallel_task:{name}] failed: {msg}\n{tb}")

    p3_1_log.info(
        f"[execute_phase_three] Parallel tasks done for scope={scope} scope_id={scope_id} "
        f"(elapsed={elapsed:.3f}s, errors={had_err})"
    )
