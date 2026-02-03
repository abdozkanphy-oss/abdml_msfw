import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import spearmanr

from cassandra_utils.models.scada_correlation_matrix import ScadaCorrelationMatrix
from cassandra_utils.models.scada_correlation_matrix_summary import ScadaCorrelationMatrixSummary


# --------------------------- utilities ---------------------------

def map_to_text(obj):
    if obj is None or not isinstance(obj, dict):
        return {}
    return {k: str(v) if v is not None else '' for k, v in obj.items()}


def _to_epoch_ms(v):
    if v is None:
        return None
    if isinstance(v, datetime):
        dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    # numeric?
    try:
        iv = int(v)
        # if it's clearly in seconds, convert to ms
        return iv * 1000 if iv < 10**11 else iv
    except Exception:
        pass

    # ISO string?
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def _coerce_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _unique_names(names):
    """
    Make sensor names unique (Sensor, Sensor_2, Sensor_3, ...)
    if needed. Keep for future use if you want to deduplicate.
    """
    seen = {}
    out = []
    for n in names:
        base = (n or "Sensor").strip() or "Sensor"
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def _sensor_name(d: dict):
    """
    Build sensor name, including equipment_name if present:
      "<parameter>-<equipment_name>" or just "<parameter>".
    """
    nm_raw = d.get('parameter') or d.get('param') or d.get('eqNo')
    if nm_raw is None:
        return None

    nm_raw = str(nm_raw).strip()
    if not nm_raw:
        return None

    eq_name = d.get("equipment_name")
    if eq_name:
        return f"{nm_raw}"
    return nm_raw


def _sensor_value(d: dict):
    """
    Extract numeric reading from sensor dict.
    Supports keys: 'counter_reading', 'cntRead', 'value'.
    """
    v = d.get('counter_reading', d.get('cntRead', d.get('value', None)))
    if v is None:
        return None
    try:
        return float(str(v).replace(',', '.'))
    except Exception:
        return None

def _to_equipment_label(var_name: str) -> str:
    """
    'parameter-equipment_name' -> 'equipment_name'
    'parameter' (no '-')      -> 'parameter' (fallback)
    """
    if not isinstance(var_name, str):
        return str(var_name)
    if "-" in var_name:
        # ilk '-' sonrası ekipman adı
        return var_name.split("-", 1)[1]
    return var_name

def extract_cntReads_to_df(sensor_values):
    """
    Build a tidy DataFrame:
      columns = ['crDt'] + sorted(unique sensor names across all timestamps)
      rows    = one per timestamp; values = float or NaN

    Sensor name = "<parameter>-<equipment_name>" if equipment_name exists,
                  otherwise "<parameter>".
    """
    if not sensor_values or not isinstance(sensor_values, list):
        return pd.DataFrame()

    # 1) collect all distinct sensor names
    name_set = set()
    for grp in sensor_values:
        if not isinstance(grp, (list, tuple)) or len(grp) < 2:
            continue
        for sensor in grp[1:]:
            d = dict(sensor) if sensor is not None else {}
            nm = _sensor_name(d)
            if nm:
                name_set.add(nm)

    sensor_names = sorted(name_set)
    if not sensor_names:
        return pd.DataFrame()

    columns = ['crDt'] + sensor_names
    rows = []

    # 2) build rows
    for grp in sensor_values:
        if not isinstance(grp, (list, tuple)) or len(grp) < 2:
            continue

        meta = dict(grp[0]) if grp[0] is not None else {}
        crdt_ms = (
            _to_epoch_ms(meta.get('crDt'))
            or _to_epoch_ms(meta.get('measurement_date'))
            or _to_epoch_ms(meta.get('mdate'))
            or 0
        )

        row = {'crDt': crdt_ms, **{nm: np.nan for nm in sensor_names}}

        # last value wins; (you can switch to averaging if you want)
        for sensor in grp[1:]:
            d = dict(sensor) if sensor is not None else {}
            nm = _sensor_name(d)
            if nm and (nm in row):
                val = _sensor_value(d)
                if val is not None:
                    row[nm] = val

        rows.append(row)

    df = pd.DataFrame(rows)

    # enforce column order
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[columns]

def extract_cntReads_to_df_with_message(sensor_values, message) -> pd.DataFrame:
    """
    1) History (sensor_values) -> df_hist
    2) Mesaj (outVals)        -> df_msg
    3) Kolonları birleştirip concat et
    """
    df_hist = extract_cntReads_to_df(sensor_values)

    row_msg = _row_from_message_out_for_corr(message)
    df_msg = pd.DataFrame([row_msg]) if row_msg else pd.DataFrame()

    if df_hist.empty and df_msg.empty:
        return pd.DataFrame()

    if df_hist.empty:
        return df_msg

    if df_msg.empty:
        return df_hist

    # kolonları hizala
    all_cols = sorted(set(df_hist.columns) | set(df_msg.columns))
    df_hist = df_hist.reindex(columns=all_cols)
    df_msg  = df_msg.reindex(columns=all_cols)

    df = pd.concat([df_hist, df_msg], ignore_index=True)

    if "crDt" in df.columns:
        df = df.sort_values("crDt").reset_index(drop=True)

    return df

# ------------------- correlation matrix helpers -------------------

def _sanitize_corr_df(df_corr: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure: numeric dtype, diagonal=1.0, no NaN/Inf (converted to 0.0).
    """
    # Coerce to float matrix
    df_corr = df_corr.astype(float, copy=False)

    # Replace inf/-inf -> NaN, then fill with 0
    df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_corr.fillna(0.0, inplace=True)

    # Force symmetry just in case (average with its transpose)
    df_corr = (df_corr + df_corr.T) / 2.0

    # Force diagonal to 1.0
    np.fill_diagonal(df_corr.values, 1.0)

    # Final pass to guarantee no NaNs remain (paranoia)
    df_corr.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_corr.fillna(0.0, inplace=True)

    return df_corr


def _matrix_to_frozen(df_numeric: pd.DataFrame):
    """
    Convert DataFrame to the frozen list-of-maps format expected by
    ScadaCorrelationMatrix.correlation_data.

    Result shape:
      [
        {"var1": {"var1": 1.0, "var2": 0.3, ...}},
        {"var2": {"var1": 0.3, "var2": 1.0, ...}},
        ...
      ]
    """
    frozen = []
    for row_var in df_numeric.index:
        row = {}
        for col_var, val in df_numeric.loc[row_var].items():
            # sanitize per-value
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                row[col_var] = 0.0
            else:
                row[col_var] = float(val)
        frozen.append({row_var: row})
    return frozen


def convert_corr_matrix_to_frozen_structure(corr_df: pd.DataFrame):
    # sanitize before freezing
    corr_df = _sanitize_corr_df(corr_df)
    return _matrix_to_frozen(corr_df)

def _row_from_message_out_for_corr(message: dict) -> dict:
    """
    outVals'tan correlation için tek satır üretir.
    Kolon adları extract_cntReads_to_df ile uyumlu olsun diye
    _sensor_name / _sensor_value kullanıyoruz.
    """
    row = {}

    out_list = message.get("outVals") or []
    if not isinstance(out_list, list):
        return row

    for ov in out_list:
        if not isinstance(ov, dict):
            continue

        fake_sensor = {
            "parameter": ov.get("eqNo") or ov.get("param") or ov.get("eqNm"),
            "equipment_name": ov.get("eqNm"),
            "counter_reading": ov.get("cntRead"),
        }
        nm = _sensor_name(fake_sensor)
        if not nm:
            continue

        val = _sensor_value(fake_sensor)
        if val is not None:
            row[nm] = val

    # zaman
    ts = (
        message.get("crDt")
        or (out_list[0].get("measDt") if out_list else None)
    )
    row["crDt"] = _to_epoch_ms(ts)

    return row

# --------------------------- main API ---------------------------

def compute_correlation(
    sensor_values,
    message,
    p3_1_log=None,
    algorithm: str = "SPEARMAN",
    scope: str = "pid",
    scope_id=None
):
    """
    Compute correlation matrix for the given sensor_values and save it to Cassandra.

    - For scope="pid": saves to ScadaCorrelationMatrix (per process/joOpId)
      using a frozen list-of-maps structure.
    - For scope="ws":  saves to ScadaCorrelationMatrixSummary (workstation-level,
      aggregated) using a dict structure.

    Parameters
    ----------
    sensor_values : list
        Output of fetch_latest_for_pid_via_dw / fetch_latest_for_ws_via_dw.
    message : dict
        Original Kafka message containing metadata (plId, wcNm, wsNo, prodList, outVals, etc.).
    p3_1_log : logger
        Logger for debug/info/warning.
    algorithm : str
        Currently only "SPEARMAN" is implemented.
    scope : {"pid", "ws"}
        When "pid", result goes to ScadaCorrelationMatrix.
        When "ws",  result goes to ScadaCorrelationMatrixSummary.
    scope_id : any
        pid for scope="pid", wsId for scope="ws" (used only in logs for now).
    """

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Starting Correlation Computation "
            f"(scope={scope}, scope_id={scope_id}, algorithm={algorithm})"
        )

    # --- build DataFrame ---
    #df = extract_cntReads_to_df(sensor_values)
    df = extract_cntReads_to_df_with_message(sensor_values, message)
    if df is None or df.empty or df.shape[1] <= 2:  # crDt + at least 2 sensors needed
        if p3_1_log:
            p3_1_log.warning(
                f"[compute_correlation] not enough distinct sensors; skipping "
                f"(scope={scope}, scope_id={scope_id})"
            )
        return

    if p3_1_log:
        p3_1_log.info(f"[compute_correlation] Final sensor columns: {df.columns.tolist()}")

    # Ensure timestamp isn't used for correlation: place 'crDt' as first column so the
    # subsequent df.iloc[:, 1:] call effectively drops it from the variables.
    if 'crDt' in df.columns:
        cols = list(df.columns)
        if cols[0] != 'crDt':
            cols.remove('crDt')
            cols.insert(0, 'crDt')
            df = df.reindex(columns=cols)
        if p3_1_log:
            p3_1_log.info("[compute_correlation] Excluding 'crDt' (timestamp) from correlation computation")

    # drop crDt, ensure numeric
    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    cols = X.columns
    n = len(cols)
    corr = pd.DataFrame(0.0, index=cols, columns=cols)

    if algorithm.upper() != "SPEARMAN":
        # you can extend here for other algorithms later
        if p3_1_log:
            p3_1_log.warning(
                f"[compute_correlation] Unsupported algorithm={algorithm}, "
                f"falling back to SPEARMAN"
            )

    if p3_1_log:
        p3_1_log.info("[compute_correlation] Computing Spearman correlations")

    # --- compute only upper triangle, mirror to keep symmetry ---
    for i in range(n):
        xi = X.iloc[:, i].to_numpy(dtype=float)
        for j in range(i, n):
            yj = X.iloc[:, j].to_numpy(dtype=float)

            mask = ~np.isnan(xi) & ~np.isnan(yj)
            nij = int(mask.sum())

            if i == j:
                # self-correlation = 1 by definition (even if constant)
                r = 1.0
            elif nij >= 3 and np.nanstd(xi[mask]) > 0 and np.nanstd(yj[mask]) > 0:
                r, _ = spearmanr(xi[mask], yj[mask], nan_policy='omit')
                # clamp/clean
                if r is None or np.isnan(r) or np.isinf(r):
                    r = 0.0
                else:
                    r = float(np.clip(r, -1.0, 1.0))
            else:
                r = 0.0

            corr.iat[i, j] = r
            corr.iat[j, i] = r  # mirror

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Raw correlation matrix for scope={scope}, "
            f"scope_id={scope_id}:\n{corr}"
        )

    # --- sanitize matrix (diag=1.0, no NaN/Inf) before persisting ---
    """corr = _sanitize_corr_df(corr)

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Sanitized correlation matrix for scope={scope}, "
            f"scope_id={scope_id}:\n{corr}"
        )

    # --- build representations for saving ---
    # 1) frozen list-of-maps for ScadaCorrelationMatrix (PID-based)
    frozen_corr = convert_corr_matrix_to_frozen_structure(corr)

    # 2) dict-of-dicts for ScadaCorrelationMatrixSummary (WS-based)
    corr_dict = {
        row_var: {
            col_var: float(val) if not (val is None or np.isnan(val) or np.isinf(val)) else 0.0
            for col_var, val in corr.loc[row_var].items()
        }
        for row_var in corr.index
    }"""

        # --- sanitize matrix (diag=1.0, no NaN/Inf) before persisting ---
    corr = _sanitize_corr_df(corr)

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Sanitized correlation matrix for scope={scope}, "
            f"scope_id={scope_id}:\n{corr}"
        )

    # ------------------ LABEL RENAMING (VISUAL ONLY) ------------------
    # Hesaplama 'corr' üzerinde yapıldı. Şimdi sadece görsel amaçla
    # kolon/indeks isimlerini equipment_name'e çeviriyoruz.
    viz_names = [ _to_equipment_label(v) for v in corr.index ]

    # Eğer duplicate oluşursa, bu ciddi bir durum: sonradan key'ler çakışır.
    # İstersen log at:
    if len(set(viz_names)) < len(viz_names) and p3_1_log:
        p3_1_log.warning(
            "[compute_correlation] Duplicate equipment labels detected after renaming; "
            "some variables may be merged in persisted view."
        )

    corr_viz = corr.copy()
    corr_viz.index = viz_names
    corr_viz.columns = viz_names  # simetrik kalması için

    # --- build representations for saving (equipment_name-only labels) ---

    # 1) frozen list-of-maps for ScadaCorrelationMatrix (PID-based)
    frozen_corr = convert_corr_matrix_to_frozen_structure(corr_viz)

    # 2) dict-of-dicts for ScadaCorrelationMatrixSummary (WS-based)
    corr_dict = {
        row_var: {
            col_var: float(val) if not (val is None or np.isnan(val) or np.isinf(val)) else 0.0
            for col_var, val in corr_viz.loc[row_var].items()
        }
        for row_var in corr_viz.index
    }


    # --- persist according to scope ---
    try:
        if scope == "pid":
            # PID-based: use ScadaCorrelationMatrix
            if p3_1_log:
                p3_1_log.info(
                    f"[compute_correlation] Saving PID-level correlation "
                    f"(pid={scope_id}) via ScadaCorrelationMatrix"
                )
            ScadaCorrelationMatrix.saveData(message, frozen_corr, p3_1_log=p3_1_log)

        elif scope == "ws":
            # Workstation-based: use ScadaCorrelationMatrixSummary
            if p3_1_log:
                p3_1_log.info(
                    f"[compute_correlation] Saving WS-level correlation "
                    f"(wsId={scope_id}) via ScadaCorrelationMatrixSummary"
                )
            ScadaCorrelationMatrixSummary.saveData(message, corr_dict, p3_1_log=p3_1_log)

        else:
            if p3_1_log:
                p3_1_log.warning(
                    f"[compute_correlation] Unknown scope={scope}; nothing persisted."
                )

    except Exception as e:
        if p3_1_log:
            p3_1_log.error(
                f"[compute_correlation] Error saving correlation matrix "
                f"(scope={scope}, scope_id={scope_id}): {e}",
                exc_info=True
            )
        # you can choose to re-raise if you want the caller to handle it
        raise

    if p3_1_log:
        p3_1_log.info(
            f"[compute_correlation] Saved correlation matrix to Cassandra "
            f"(scope={scope}, scope_id={scope_id})"
        )


#### Helper Functions for Correlation Matrix Summary
import numpy as np
from collections import defaultdict

def _frozen_list_to_dict(frozen_list):
    """
    Convert stored frozen list format:
      [{A:{B:r,...}}, {C:{...}}, ...]  ->  {A:{B:r,...}, C:{...}}
    """
    out = {}
    for item in (frozen_list or []):
        if isinstance(item, dict):
            for k, v in item.items():
                out[k] = v or {}
    return out

def aggregate_correlation_data(corr_list, p3_1_log=None):
    """
    corr_list: List of frozen correlation matrices
               each like [{A:{B:ρ,...}}, {B:{...}}, ...]
    Returns: nested dict {A:{B: ρ*}} aggregated via equal-weight Fisher z.
    """
    p3_1_log.info("[aggregate_correlation_data] Starting Correlation Aggregation")
    acc = defaultdict(lambda: defaultdict(lambda: {'sum_z': 0.0, 'k': 0}))

    for corr_frozen in corr_list:
        C = _frozen_list_to_dict(corr_frozen)
        for s1, row in (C or {}).items():
            if not isinstance(row, dict):
                continue
            for s2, r in row.items():
                if r is None or np.isnan(r):
                    continue
                # clamp to avoid atanh(±1)
                r_clip = float(np.clip(r, -0.999999, 0.999999))
                z = np.arctanh(r_clip)
                acc[s1][s2]['sum_z'] += z
                acc[s1][s2]['k']     += 1

    out = {}
    for s1, row in acc.items():
        out[s1] = {}
        for s2, v in row.items():
            if v['k'] > 0:
                out[s1][s2] = float(np.tanh(v['sum_z'] / v['k']))
    p3_1_log.info(f"[aggregate_correlation_data] Completed Correlation Aggregation: {out}")
    return out