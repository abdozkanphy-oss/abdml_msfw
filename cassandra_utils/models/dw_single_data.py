import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table, drop_table
from cassandra.cqlengine.models import Model
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

"""
UPDATED: Now supports both inVars (inputs) and outVals (outputs)
- equipment_type: True = INPUT, False = OUTPUT
- gen_read_val: input values (for inVars)
- counter_reading: output values (for outVals)
"""

cassandra_host = "209.250.235.243"
username = "cassandra"
password = "cassAdmin%&.."
keyspace = "das_new_pm"

connection.setup(
    hosts=[cassandra_host],
    default_keyspace=keyspace,
    auth_provider=PlainTextAuthProvider(username=username, password=password),
    protocol_version=4,
    lazy_connect=True, 
    retry_connect=True, 
    connect_timeout=20,
)

auth_provider = PlainTextAuthProvider(username=username, password=password)
cluster = Cluster([cassandra_host], auth_provider=auth_provider)
session = cluster.connect()


# --- small helpers for type conversion ---
def to_dt(ms_or_dt):
    if ms_or_dt is None:
        return None
    if isinstance(ms_or_dt, datetime):
        return ms_or_dt
    try:
        v = int(ms_or_dt)
        return datetime.utcfromtimestamp(v / 1000.0) if v > 10**11 else datetime.utcfromtimestamp(v)
    except Exception:
        return None

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def to_bool(x):
    if isinstance(x, bool):
        return x
    if x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y")
    return None


class dw_tbl_raw_data(Model):
    __keyspace__   = keyspace
    __table_name__ = "dw_tbl_raw_data"

    # PRIMARY KEY
    unique_code = columns.Text(partition_key=True)

    # === EXISTING COLUMNS ===
    active                                  = columns.Boolean()
    anomaly_detection_active                = columns.Boolean(default=False)
    good                                    = columns.Boolean()
    #bias                                    = columns.Double()
    #coefficient                             = columns.Double()
    counter_reading                         = columns.Double()  # For OUTPUT values
    #create_auto_maintenance_order_notification = columns.Boolean()
    create_date                             = columns.DateTime()
    customer                                = columns.Text()
    #employee_id                             = columns.Integer()
    #equipment_code_group_header_id          = columns.Integer()
    equipment_id                            = columns.Integer()
    #equipment_measuring_point_id            = columns.Integer()
    equipment_name                          = columns.Text()
    equipment_no                            = columns.Text()
    job_order_operation_id                  = columns.Integer()
    machine_state                           = columns.Text()
    #max_error_count_in_hour                 = columns.Integer()
    #max_error_count_in_min                  = columns.Integer()
    #max_error_count_in_shift                = columns.Integer()
    #mean_square_error                       = columns.Double()
    measurement_date                        = columns.DateTime()
    #measurement_document_id                 = columns.Integer()
    #month_year                              = columns.Text()
    operationname                           = columns.Text()
    operationno                             = columns.Text()
    operationtaskcode                       = columns.Text()
    #organization_id                         = columns.Integer()
    #organization_name                       = columns.Text()
    #organization_no                         = columns.Text()
    #parameter                               = columns.Text()
    plant_id                                = columns.Integer()
    plant_name                              = columns.Text()
    produced_stock_id                       = columns.Integer()
    produced_stock_name                     = columns.Text()
    produced_stock_no                       = columns.Text()
    #regression_value                        = columns.Double()
    shift_finish_time                       = columns.DateTime()
    shift_start_time                        = columns.DateTime()
    stock_id                                = columns.Integer()
    trend_calculation_active                = columns.Boolean()
    update_date                             = columns.DateTime()
    valuation_code                          = columns.Text()
    work_center_id                          = columns.Integer()
    work_center_name                        = columns.Text()
    work_center_no                          = columns.Text()
    work_station_id                         = columns.Integer()
    work_station_name                       = columns.Text()
    work_station_no                         = columns.Text()
    workstation_state                       = columns.Text()
    year                                    = columns.Text()
    job_order_reference_no                  = columns.Text()
    prod_order_reference_no                 = columns.Text()
    
    # === NEW COLUMNS for INPUT support ===
    gen_read_val                            = columns.Double()    # For INPUT values
    equipment_type                          = columns.Boolean()   # True = INPUT, False = OUTPUT

    @classmethod
    def saveData(cls, message: dict):
        """
        UPDATED: Save BOTH inVars (inputs) and outVals (outputs) to dw_tbl_raw_data.
        
        - outVals → equipment_type=False, counter_reading=value
        - inVars  → equipment_type=True, gen_read_val=value
        """
        top = message or {}
        
        # === PART 1: Save OUTPUT variables (outVals) ===
        outvals = top.get("outVals") or []
        if not isinstance(outvals, list):
            outvals = []

        prod_raw = top.get("prodList") or []
        if isinstance(prod_raw, dict):
            prod = prod_raw
        elif isinstance(prod_raw, list) and prod_raw and isinstance(prod_raw[0], dict):
            prod = prod_raw[0]
        else:
            prod = {}

        for ov in outvals:
            row = {
                "unique_code": str(uuid.uuid4()),
                
                # NEW: Mark as OUTPUT
                "equipment_type": False,  # False = OUTPUT
                
                # OUTPUT value goes to counter_reading
                "counter_reading": to_float(ov.get("cntRead")),
                
                # gen_read_val stays None for outputs (safety)
                # "gen_read_val": None,  # Implicit
                
                # All other fields same as before
                "active": True, #to_bool(ov.get("act")),
                "anomaly_detection_active": to_bool(ov.get("anomDetAct")),
                #"bias": to_float(ov.get("bias")),
                #"coefficient": to_float(ov.get("coef")),
                #"create_auto_maintenance_order_notification": to_bool(ov.get("crAutoMaintOrdNot")),
                "create_date": to_dt(top.get("crDt")),
                "customer": (ov.get("cust") or top.get("cust")),
                #"employee_id": to_int(top.get("empId")),
                #"equipment_code_group_header_id": to_int(ov.get("eqCodeGrpHdrId")),
                "equipment_id": to_int(ov.get("eqId")),
                #"equipment_measuring_point_id": to_int(ov.get("eqMeasPtId")),
                "equipment_name": ov.get("eqNm"),
                "equipment_no": ov.get("eqNo"),
                "job_order_operation_id": to_int(top.get("joOpId")),
                "job_order_reference_no": str(top.get("joRef", "")),
                "prod_order_reference_no": str(top.get("refNo", "")),
                "good": to_bool(top.get("goodCnt")),
                "machine_state": top.get("mcSt"),
                #"max_error_count_in_hour": to_int(ov.get("maxErrCntHr")),
                #"max_error_count_in_min": to_int(ov.get("maxErrCntMin")),
                #"max_error_count_in_shift": to_int(ov.get("maxErrCntShft")),
                #"mean_square_error": to_float(ov.get("msErr")),
                "measurement_date": to_dt(ov.get("measDt")),
                #"measurement_document_id": to_int(ov.get("measDocId")),
                #"month_year": ov.get("monYr"),
                "operationname": top.get("opNm"),
                "operationno": top.get("opNo"),
                "operationtaskcode": top.get("opTc"),
                #"organization_id": to_int(top.get("orgId")),
                #"organization_name": top.get("orgNm"),
                #"organization_no": top.get("orgNo"),
                #"parameter": ov.get("param"),
                "plant_id": to_int(top.get("plId")),
                "plant_name": top.get("plNm"),
                "produced_stock_id": to_int(prod.get("stId")),
                "produced_stock_name": prod.get("stNm"),
                "produced_stock_no": prod.get("stNo"),
                #"regression_value": to_float(ov.get("regVal")),
                "shift_finish_time": to_dt(top.get("shFt")),
                "shift_start_time": to_dt(top.get("shSt")),
                "stock_id": to_int(prod.get("joStId")),
                "trend_calculation_active": to_bool(ov.get("trendCalcAct")),
                "update_date": to_dt(ov.get("updDt")),
                "valuation_code": ov.get("valCd"),
                "work_center_id": to_int(top.get("wcId")),
                "work_center_name": top.get("wcNm"),
                "work_center_no": top.get("wcNo"),
                "work_station_id": to_int(top.get("wsId")),
                "work_station_name": top.get("wsNm"),
                "work_station_no": top.get("wsNo"),
                "workstation_state": top.get("prSt"),
                "year": ov.get("year") or top.get("year"),
            }
            clean = {k: v for k, v in row.items() if v is not None}
            
            # Skip rows without timestamp
            if not clean.get("measurement_date"):
                continue
            
            cls.create(**clean)

        # === PART 2: Save INPUT variables (inVars) ===
        # Support both field names
        invars = top.get("inVars") or top.get("inputVariableList") or []
        if not isinstance(invars, list):
            invars = []

        """for iv in invars:
            # Determine variable name (priority: varNm > varNo > param > varId)
            var_name = (
                iv.get("varNm") 
                or iv.get("varNo") 
                or iv.get("param")
                or iv.get("varId")
            )
            
            # Determine value (priority: genReadVal > actVal > value > cntRead)
            var_value = None
            if "genReadVal" in iv:
                var_value = iv.get("genReadVal")
            elif "actVal" in iv:
                var_value = iv.get("actVal")
            elif "value" in iv:
                var_value = iv.get("value")
            elif "cntRead" in iv:
                var_value = iv.get("cntRead")
            
            if var_name is None or var_value is None:
                continue  # Skip if we can't determine name or value
            
            row = {
                "unique_code": str(uuid.uuid4()),
                
                "active": to_bool(ov.get("act")),
                # NEW: Mark as INPUT
                "equipment_type": True,  # True = INPUT
                
                # INPUT value goes to gen_read_val
                "gen_read_val": to_float(var_value),
                
                # counter_reading stays None for inputs (safety)
                # "counter_reading": None,  # Implicit
                
                # Use variable name as equipment info
                "equipment_name": var_name,
                "equipment_no": str(var_name),  # Use name as number too
                #"parameter": var_name,
                
                # Copy all top-level metadata
                "create_date": to_dt(top.get("crDt")),
                "measurement_date": to_dt(iv.get("measDt") or top.get("crDt")),  # Use inVar's date or fallback
                "customer": (ov.get("cust") or top.get("cust")),
                #"employee_id": to_int(top.get("empId")),
                "job_order_operation_id": to_int(top.get("joOpId")),
                "job_order_reference_no": str(top.get("joRef", "")),
                "prod_order_reference_no": str(top.get("refNo", "")),
                "good": to_bool(top.get("goodCnt")),
                "machine_state": top.get("mcSt"),
                "operationname": top.get("opNm"),
                "operationno": top.get("opNo"),
                "operationtaskcode": top.get("opTc"),
                #"organization_id": to_int(top.get("orgId")),
                #"organization_name": top.get("orgNm"),
                #"organization_no": top.get("orgNo"),
                "plant_id": to_int(top.get("plId")),
                "plant_name": top.get("plNm"),
                "produced_stock_id": to_int(prod.get("stId")),
                "produced_stock_name": prod.get("stNm"),
                "produced_stock_no": prod.get("stNo"),
                "shift_finish_time": to_dt(top.get("shFt")),
                "shift_start_time": to_dt(top.get("shSt")),
                "stock_id": to_int(prod.get("joStId")),
                "work_center_id": to_int(top.get("wcId")),
                "work_center_name": top.get("wcNm"),
                "work_center_no": top.get("wcNo"),
                "work_station_id": to_int(top.get("wsId")),
                "work_station_name": top.get("wsNm"),
                "work_station_no": top.get("wsNo"),
                "workstation_state": top.get("prSt"),
                "year": iv.get("year") or top.get("year"),
                
                # Input-specific fields (if available)
                "active": to_bool(iv.get("act")),
                "equipment_id": to_int(iv.get("varId")),  # Use varId as equipment_id
            }
            
            clean = {k: v for k, v in row.items() if v is not None}
            
            # Skip rows without timestamp
            if not clean.get("measurement_date"):
                continue
            
            cls.create(**clean)"""

        # === Handle rare top-level single reading ===
        if not outvals and not invars and (top.get("measDt") or top.get("cntRead")):
            row = {
                "unique_code": str(uuid.uuid4()),
                "equipment_type": False,  # Assume OUTPUT for top-level
                "counter_reading": to_float(top.get("cntRead")),
                "active": True, #to_bool(top.get("act")),
                "anomaly_detection_active": to_bool(top.get("anomDetAct")),
                "good": to_bool(top.get("goodCnt")),
                #"bias": to_float(top.get("bias")),
                #"coefficient": to_float(top.get("coef")),
                #"create_auto_maintenance_order_notification": to_bool(top.get("crAutoMaintOrdNot")),
                "create_date": to_dt(top.get("crDt")),
                "customer": top.get("cust"),
                #"employee_id": to_int(top.get("empId")),
                #"equipment_code_group_header_id": to_int(top.get("eqCodeGrpHdrId")),
                "equipment_id": to_int(top.get("eqId")),
                #"equipment_measuring_point_id": to_int(top.get("eqMeasPtId")),
                "equipment_name": top.get("eqNm"),
                "equipment_no": top.get("eqNo"),
                "job_order_operation_id": to_int(top.get("joOpId")),
                "job_order_reference_no": str(top.get("joRef", "")),
                "prod_order_reference_no": str(top.get("refNo", "")),
                "machine_state": top.get("mcSt"),
                #"mean_square_error": to_float(top.get("msErr")),
                "measurement_date": to_dt(top.get("measDt")),
                #"measurement_document_id": to_int(top.get("measDocId")),
                #"month_year": top.get("monYr"),
                "organization_id": to_int(top.get("orgId")),
                "organization_name": top.get("orgNm"),
                "organization_no": top.get("orgNo"),
                #"parameter": top.get("param"),
                "plant_id": to_int(top.get("plId")),
                "plant_name": top.get("plNm"),
                #"regression_value": to_float(top.get("regVal")),
                "shift_finish_time": to_dt(top.get("shFt")),
                "shift_start_time": to_dt(top.get("shSt")),
                "trend_calculation_active": to_bool(top.get("trendCalcAct")),
                "update_date": to_dt(top.get("updDt")),
                "valuation_code": top.get("valCd"),
                "work_center_id": to_int(top.get("wcId")),
                "work_center_name": top.get("wcNm"),
                "work_center_no": top.get("wcNo"),
                "work_station_id": to_int(top.get("wsId")),
                "work_station_name": top.get("wsNm"),
                "work_station_no": top.get("wsNo"),
                "workstation_state": top.get("workstation_state"),
                "year": top.get("year"),
            }
            clean = {k: v for k, v in row.items() if v is not None}
            if clean.get("measurement_date"):
                cls.create(**clean)

        return message

    @classmethod
    def fetchData(cls, limit=60):
        """
        UPDATED: Returns both inputs and outputs.
        
        Returns:
            (rows, inputList, outputList, batchList)
            - inputList: list of input variable dicts (from inVars)
            - outputList: list of single-item lists (from outVals)
        """
        rows = list(cls.objects.allow_filtering().limit(limit))
        input_list, batch_list, output_list = [], [], []

        for r in rows:
            # Check equipment_type to determine if INPUT or OUTPUT
            is_input = r.equipment_type if r.equipment_type is not None else False
            
            if is_input:
                # INPUT variable
                iv = {
                    "varNm": r.equipment_name,
                    "varNo": r.equipment_no,
                    "genReadVal": r.gen_read_val,
                    "measDt": int(r.measurement_date.timestamp() * 1000) if r.measurement_date else None,
                    "varId": r.equipment_id,
                    "act": r.active,
                    "joOpId": r.job_order_operation_id,
                    "plId": r.plant_id,
                    "wsId": r.work_station_id,
                    "wcId": r.work_center_id,
                    "good": r.good,
                    "eqId": r.equipment_id,
                    "cust": r.customer,
                    "joRef": r.job_order_reference_no,
                    "refNo": r.prod_order_reference_no,
                    "anomDetAct": r.anomaly_detection_active,
                    "trendCalcAct": r.trend_calculation_active
                }
                input_list.append(iv)
            else:
                # OUTPUT variable
                ov = {
                    "cntRead": r.counter_reading,
                    "measDt": int(r.measurement_date.timestamp() * 1000) if r.measurement_date else None,
                    #"valCd": r.valuation_code,
                    #"param": r.parameter,
                    "eqId": r.equipment_id,
                    #"eqMeasPtId": r.equipment_measuring_point_id,
                    #"eqCodeGrpHdrId": r.equipment_code_group_header_id,
                    "eqNm": r.equipment_name,
                    "eqNo": r.equipment_no,
                    "cust": r.customer,
                    "joRef": r.job_order_reference_no,
                    "refNo": r.prod_order_reference_no,
                    #"monYr": r.month_year,
                    #"measDocId": r.measurement_document_id,
                    "anomDetAct": r.anomaly_detection_active,
                    "trendCalcAct": r.trend_calculation_active,
                    #"msErr": r.mean_square_error,
                    #"coef": r.coefficient,
                    #"regVal": r.regression_value,
                    "plId": r.plant_id,
                    "wsId": r.work_station_id,
                    "wcId": r.work_center_id,
                    "good": r.good
                }
                output_list.append([ov])  # single-item list for compatibility

        return rows, input_list, output_list, batch_list


if __name__ == "__main__":
    sync_table(dw_tbl_raw_data)


"""import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table, drop_table
from cassandra.cqlengine.models import Model
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster

cassandra_host = "209.250.235.243"
username = "cassandra"
password = "cassAdmin%&.."
keyspace = "das_new_pm"

connection.setup(
    hosts=[cassandra_host],
    default_keyspace=keyspace,
    auth_provider=PlainTextAuthProvider(username=username, password=password),
    protocol_version=4,
    lazy_connect=True, 
    retry_connect=True, 
    connect_timeout=20,
)
# The connection setup can also be done using Cluster and Session directly
auth_provider = PlainTextAuthProvider(username=username, password=password)
cluster = Cluster([cassandra_host], auth_provider=auth_provider)
session = cluster.connect()


# --- small helpers for type conversion ---
def to_dt(ms_or_dt):
    if ms_or_dt is None:
        return None
    if isinstance(ms_or_dt, datetime):
        return ms_or_dt
    try:
        v = int(ms_or_dt)
        # treat as milliseconds if big enough
        return datetime.utcfromtimestamp(v / 1000.0) if v > 10**11 else datetime.utcfromtimestamp(v)
    except Exception:
        return None

def to_int(x):
    try:
        return int(x)
    except Exception:
        return None

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def to_bool(x):
    if isinstance(x, bool):
        return x
    if x in (0, 1):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "t", "yes", "y")
    return None


# ============================
#  MODEL maps to dw_tbl_raw_data
# ============================
class dw_tbl_raw_data(Model):
    __keyspace__   = keyspace
    __table_name__ = "dw_tbl_raw_data"

    # PRIMARY KEY
    unique_code = columns.Text(partition_key=True)

    # Columns (exactly as in your raw table)
    active                                  = columns.Boolean()
    anomaly_detection_active                = columns.Boolean(default=False)
    good                                    = columns.Boolean()
    bias                                    = columns.Double()
    coefficient                             = columns.Double()
    counter_reading                         = columns.Double()
    create_auto_maintenance_order_notification = columns.Boolean()
    create_date                             = columns.DateTime()
    customer                                = columns.Text()
    employee_id                             = columns.Integer()
    equipment_code_group_header_id          = columns.Integer()
    equipment_id                            = columns.Integer()
    equipment_measuring_point_id            = columns.Integer()
    equipment_name                          = columns.Text()
    equipment_no                            = columns.Text()
    job_order_operation_id                  = columns.Integer()
    machine_state                           = columns.Text()
    max_error_count_in_hour                 = columns.Integer()
    max_error_count_in_min                  = columns.Integer()
    max_error_count_in_shift                = columns.Integer()
    mean_square_error                       = columns.Double()
    measurement_date                        = columns.DateTime()
    measurement_document_id                 = columns.Integer()
    month_year                              = columns.Text()
    operationname                           = columns.Text()
    operationno                             = columns.Text()
    operationtaskcode                       = columns.Text()
    organization_id                         = columns.Integer()
    organization_name                       = columns.Text()
    organization_no                         = columns.Text()
    parameter                               = columns.Text()
    plant_id                                = columns.Integer()
    plant_name                              = columns.Text()
    produced_stock_id                       = columns.Integer()
    produced_stock_name                     = columns.Text()
    produced_stock_no                       = columns.Text()
    regression_value                        = columns.Double()
    shift_finish_time                       = columns.DateTime()
    shift_start_time                        = columns.DateTime()
    stock_id                                = columns.Integer()
    trend_calculation_active                = columns.Boolean()
    update_date                             = columns.DateTime()
    valuation_code                          = columns.Text()
    work_center_id                          = columns.Integer()
    work_center_name                        = columns.Text()
    work_center_no                          = columns.Text()
    work_station_id                         = columns.Integer()
    work_station_name                       = columns.Text()
    work_station_no                         = columns.Text()
    workstation_state                       = columns.Text()
    year                                    = columns.Text()
    job_order_reference_no                 = columns.Text()
    prod_order_reference_no                 = columns.Text()

    # ---------- write: one row per outVals ----------
    @classmethod
    def saveData(cls, message: dict):
        #Save ONE ROW PER outVals item to dw_tbl_raw_data. This function returns the original message.
        top = message or {}
        #print(f"\n\nReceived message for dw_tbl_raw_data: {top}")
        
        outvals = top.get("outVals") or []
        if not isinstance(outvals, list):
            outvals = []
        #print(f"\n\nOutvals: {outvals}")

        prod_raw = top.get("prodList") or []
        if isinstance(prod_raw, dict):
            prod = prod_raw
        elif isinstance(prod_raw, list) and prod_raw and isinstance(prod_raw[0], dict):
            prod = prod_raw[0]           # or choose by some rule/index
        else:
            prod = {}
        #print(f"\n\nProdList: {prodList}")

        for ov in outvals:
            row = {
                "unique_code": str(uuid.uuid4()),
                "active":                          to_bool(ov.get("act")),
                "anomaly_detection_active":        to_bool(ov.get("anomDetAct")),
                "bias":                            to_float(ov.get("bias")),
                "coefficient":                     to_float(ov.get("coef")),
                "counter_reading":                 to_float(ov.get("cntRead")),
                "create_auto_maintenance_order_notification": to_bool(ov.get("crAutoMaintOrdNot")),
                "create_date":                     to_dt(top.get("crDt")),
                "customer":                        (ov.get("cust") or top.get("cust")),
                "employee_id":                     to_int(top.get("empId")),
                "equipment_code_group_header_id":  to_int(ov.get("eqCodeGrpHdrId")),
                "equipment_id":                    to_int(ov.get("eqId")),
                "equipment_measuring_point_id":    to_int(ov.get("eqMeasPtId")),
                "equipment_name":                  ov.get("eqNm"),
                "equipment_no":                    ov.get("eqNo"),
                "job_order_operation_id":          to_int(top.get("joOpId")),
                "job_order_reference_no":          str(top.get("joRef", "")),
                "prod_order_reference_no":         str(top.get("refNo", "")),
                "good":                            to_bool(top.get("goodCnt")),
                "machine_state":                   top.get("mcSt"),
                "max_error_count_in_hour":         to_int(ov.get("maxErrCntHr")),
                "max_error_count_in_min":          to_int(ov.get("maxErrCntMin")),
                "max_error_count_in_shift":        to_int(ov.get("maxErrCntShft")),
                "mean_square_error":               to_float(ov.get("msErr")),
                "measurement_date":                to_dt(ov.get("measDt")),
                "measurement_document_id":         to_int(ov.get("measDocId")),
                "month_year":                      ov.get("monYr"),
                "operationname":                   top.get("opNm"),
                "operationno":                     top.get("opNo"),
                "operationtaskcode":               top.get("opTc"),
                "organization_id":                 to_int(top.get("orgId")),
                "organization_name":               top.get("orgNm"),
                "organization_no":                 top.get("orgNo"),
                "parameter":                       ov.get("param"),
                "plant_id":                        to_int(top.get("plId")),
                "plant_name":                      top.get("plNm"),
                "produced_stock_id":               to_int(prod.get("stId")),
                "produced_stock_name":             prod.get("stNm"),
                "produced_stock_no":               prod.get("stNo"),
                "regression_value":                to_float(ov.get("regVal")),
                "shift_finish_time":               to_dt(top.get("shFt")),
                "shift_start_time":                to_dt(top.get("shSt")),
                "stock_id":                        to_int(prod.get("joStId")),
                "trend_calculation_active":        to_bool(ov.get("trendCalcAct")),
                "update_date":                     to_dt(ov.get("updDt")),
                "valuation_code":                  ov.get("valCd"),
                "work_center_id":                  to_int(top.get("wcId")),
                "work_center_name":                top.get("wcNm"),
                "work_center_no":                  top.get("wcNo"),
                "work_station_id":                 to_int(top.get("wsId")),
                "work_station_name":               top.get("wsNm"),
                "work_station_no":                 top.get("wsNo"),
                "workstation_state":               top.get("prSt"),
                "year":                            ov.get("year") or top.get("year"),
            }
            clean = {k: v for k, v in row.items() if v is not None}
            # skip rows that don't have a timestamp
            if not clean.get("measurement_date"):
                continue
            cls.create(**clean)

        # handle rare messages with a single top-level reading but no outVals
        if not outvals and (top.get("measDt") or top.get("cntRead")):
            ov = {
                "measDt": top.get("measDt"),
                "cntRead": top.get("cntRead"),
                "param": top.get("param"),
                "valCd": top.get("valCd"),
                "eqId": top.get("eqId"),
                "eqMeasPtId": top.get("eqMeasPtId"),
                "eqCodeGrpHdrId": top.get("eqCodeGrpHdrId"),
                "eqNm": top.get("eqNm"),
                "eqNo": top.get("eqNo"),
                "cust": top.get("cust"),
            }
            row = {
                **{k: v for k, v in clean.items()},  # not defined yet; build fresh instead
            }
            row = {
                "unique_code": str(uuid.uuid4()),
                "active": to_bool(top.get("act")),
                "anomaly_detection_active": to_bool(top.get("anomDetAct")),
                "good": to_bool(top.get("goodCnt")),
                "bias": to_float(top.get("bias")),
                "coefficient": to_float(top.get("coef")),
                "counter_reading": to_float(top.get("cntRead")),
                "create_auto_maintenance_order_notification": to_bool(top.get("crAutoMaintOrdNot")),
                "create_date": to_dt(top.get("crDt")),
                "customer": top.get("cust"),
                "employee_id": to_int(top.get("empId")),
                "equipment_code_group_header_id": to_int(top.get("eqCodeGrpHdrId")),
                "equipment_id": to_int(top.get("eqId")),
                "equipment_measuring_point_id": to_int(top.get("eqMeasPtId")),
                "equipment_name": top.get("eqNm"),
                "equipment_no": top.get("eqNo"),
                "job_order_operation_id": to_int(top.get("joOpId")),
                "job_order_reference_no":          str(top.get("joRef", "")),
                "prod_order_reference_no":         str(top.get("refNo", "")),
                "machine_state": top.get("mcSt"),
                "mean_square_error": to_float(top.get("msErr")),
                "measurement_date": to_dt(top.get("measDt")),
                "measurement_document_id": to_int(top.get("measDocId")),
                "month_year": top.get("monYr"),
                "organization_id": to_int(top.get("orgId")),
                "organization_name": top.get("orgNm"),
                "organization_no": top.get("orgNo"),
                "parameter": top.get("param"),
                "plant_id": to_int(top.get("plId")),
                "plant_name": top.get("plNm"),
                "regression_value": to_float(top.get("regVal")),
                "shift_finish_time": to_dt(top.get("shFt")),
                "shift_start_time": to_dt(top.get("shSt")),
                "trend_calculation_active": to_bool(top.get("trendCalcAct")),
                "update_date": to_dt(top.get("updDt")),
                "valuation_code": top.get("valCd"),
                "work_center_id": to_int(top.get("wcId")),
                "work_center_name": top.get("wcNm"),
                "work_center_no": top.get("wcNo"),
                "work_station_id": to_int(top.get("wsId")),
                "work_station_name": top.get("wsNm"),
                "work_station_no": top.get("wsNo"),
                "workstation_state": top.get("workstation_state"),
                "year": top.get("year"),
            }
            clean2 = {k: v for k, v in row.items() if v is not None}
            if clean2.get("measurement_date"):
                cls.create(**clean2)

        # Return the original message so the rest of your pipeline keeps working
        return message

    # ---------- read: keep return shape compatible ----------
    @classmethod
    def fetchData(cls, limit=60):
        
        #Returns (rows, inputList, outputList, batchList)
        #- outputList is a list of single-item lists that look like old outVals dicts
        #  so your Phase1 training code can keep using them.
        
        rows = list(cls.objects.allow_filtering().limit(limit))
        input_list, batch_list, output_list = [], [], []

        for r in rows:
            ov = {
                "cntRead": r.counter_reading,
                "measDt": int(r.measurement_date.timestamp() * 1000) if r.measurement_date else None,
                "valCd": r.valuation_code,
                "param": r.parameter,
                "eqId": r.equipment_id,
                "eqMeasPtId": r.equipment_measuring_point_id,
                "eqCodeGrpHdrId": r.equipment_code_group_header_id,
                "eqNm": r.equipment_name,
                "eqNo": r.equipment_no,
                "cust": r.customer,
                "joRef": r.job_order_reference_no,
                "refNo": r.prod_order_reference_no,
                "monYr": r.month_year,
                "measDocId": r.measurement_document_id,
                "anomDetAct": r.anomaly_detection_active,
                "trendCalcAct": r.trend_calculation_active,
                "msErr": r.mean_square_error,
                "coef": r.coefficient,
                "regVal": r.regression_value,
                "plId": r.plant_id,
                "wsId": r.work_station_id,
                "wcId": r.work_center_id,
                "good": r.good
            }
            output_list.append([ov])  # single-item list for compatibility

        return rows, input_list, output_list, batch_list


if __name__ == "__main__":
    # Only run this if you truly want cqlengine to manage the schema.
    # For an existing production table, it's safer NOT to run this file directly.
    sync_table(dw_tbl_raw_data)
    



import uuid
from cassandra.cqlengine import columns
from cassandra.cqlengine import connection
from datetime import datetime
from cassandra.cqlengine.management import sync_table, drop_table
from cassandra.cqlengine.models import Model
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster


cassandra_host = "209.250.235.243"
username = "cassandra"
password = "cassAdmin%&.."
keyspace = "das_new_pm"

connection.setup(
    hosts=[cassandra_host],
    default_keyspace=keyspace,
    auth_provider=PlainTextAuthProvider(username=username, password=password),
    protocol_version=4
)
auth_provider = PlainTextAuthProvider(username=username, password=password)
cluster = Cluster([cassandra_host], auth_provider=auth_provider)
session = cluster.connect()


def timestamp_to_datetime(ms):
    return datetime.fromtimestamp(ms / 1000) if ms is not None else None


def map_to_text(obj):
    if obj is None or not isinstance(obj, dict):
        return {}
    return {k: str(v) if v is not None else '' for k, v in obj.items()}




class dw_single_data_data(Model):
    __keyspace__ = keyspace
    partition_key = columns.Text(primary_key=True, default="latest")
    createDate = columns.DateTime(primary_key=True, clustering_order="DESC")

    ##new fields
    quantitychanged = columns.Integer
    good = columns.Boolean()
    # unique_code = columns.UUID(primary_key=True, default=uuid.uuid4)
    unique_code = columns.UUID(default=uuid.uuid4)
    plantId = columns.Integer(required=False)
    plantName = columns.Text(required=False)
    jobOrderReferenceNo = columns.Text(required=False)
    jobOrderOperationId = columns.Integer(required=False)
    operationNo = columns.Text(required=False)
    operationName = columns.Text(required=False)
    operationTaskCode = columns.Text(required=False)
    orderNo = columns.Text(required=False)
    employeeId = columns.Integer(required=False)
    employeeNo = columns.Text(required=False)
    employeeFirstName = columns.Text(required=False)
    employeeLastName = columns.Text(required=False)
    workStationId = columns.Integer(required=False)
    workStationNo = columns.Text(required=False)
    workStationName = columns.Text(required=False)
    workCenterId = columns.Integer(required=False)
    workCenterNo = columns.Text(required=False)
    workCenterName = columns.Text(required=False)
    machineState = columns.Text(required=False)
    productionState = columns.Text(required=False)
    shiftId = columns.Integer(required=False)
    shiftName = columns.Text(required=False)
    shiftStartTime = columns.DateTime(required=False)
    shiftFinishTime = columns.DateTime(required=False)
    organizationId = columns.Text(required=False)
    organizationNo = columns.Text(required=False)
    organizationName = columns.Text(required=False)
    inputVariableList = columns.List(columns.Map(
        columns.Text, columns.Text), required=False)
    qualityCheckList = columns.List(columns.Map(
        columns.Text, columns.Text), required=False)
    outputValueList = columns.List(columns.Map(
        columns.Text, columns.Text), required=False)
    componentBatchList = columns.List(columns.Map(
        columns.Text, columns.Text), required=False)
    produceList = columns.List(columns.Map(
        columns.Text, columns.Text), required=False)

    @classmethod
    def saveData(cls, message):
        in_vars = message.get('inVars', []) or []
        comp_bats = message.get('compBats', []) or []
        out_vals = message.get('outVals', []) or []
        prod_list = message.get('prodList', []) or []
        qc_list = message.get('qCList', []) or []


        static_fields = {
            "partition_key": "latest",
            "unique_code": uuid.uuid4(),
        }
        dynamic_fields = {
            "quantitychanged" : message.get("chngCycQty"),
            "good" : message.get("goodCnt"),
            "createDate": timestamp_to_datetime(message.get('crDt')),  # Date
            "plantId": message.get("plId"),
            "plantName": message.get("plNm"),
            "jobOrderReferenceNo": message.get("joRef"),
            "jobOrderOperationId": message.get("joOpId"),
            "operationNo": message.get("opNo"),
            "operationName": message.get("opNm"),
            "operationTaskCode": message.get("opTc"),
            "orderNo": message.get("ordNo"),
            "employeeId": message.get("empId"),
            "employeeNo": message.get("empNo"),
            "employeeFirstName": message.get("empFn"),
            "employeeLastName": message.get("empLn"),
            "workStationId": message.get("wsId"),
            "workStationNo": message.get("wsNo"),
            "workStationName": message.get("wsNm"),
            "workCenterId": message.get("wcId"),
            "workCenterNo": message.get("wcNo"),
            "workCenterName": message.get("wcNm"),
            "machineState": message.get("mcSt"),
            "productionState": message.get("prSt"),
            "shiftId": message.get("shId"),
            "shiftName": message.get("shNm"),
            "shiftStartTime": timestamp_to_datetime(message.get("shSt")),
            "shiftFinishTime": timestamp_to_datetime(message.get("shFt")),
            "organizationId": message.get("orgId"),
            "organizationNo": message.get("orgNo"),
            "organizationName": message.get("orgNm"),
            "inputVariableList": [map_to_text(var) for var in in_vars],
            "componentBatchList": [map_to_text(batch) for batch in comp_bats],
            "outputValueList": [map_to_text(out) for out in out_vals],
            "produceList": [map_to_text(prod) for prod in prod_list],
            "qualityCheckList": [map_to_text(qc) for qc in qc_list]
        }

        filtered_fields = {k: v for k,
                        v in dynamic_fields.items() if v is not None }

        # Merge static and filtered dynamic fields
        final_fields = {**static_fields, **filtered_fields}
        return cls.create(**final_fields)

    @classmethod
    def fetchData(cls, limit = 60):
        rows = cls.objects(partition_key = "latest").allow_filtering().order_by('-createDate').limit(limit)
        returnList = []
        outputList = []
        inputList = []
        batchList = []
        for row in rows:
            returnList.append(row)
            outputList.append(row['outputValueList']) 
            inputList.append(row['inputVariableList'])
            batchList.append(row['componentBatchList'])
        return returnList, inputList, outputList, batchList
        

if __name__ == "__main__":

    def delete():
        drop_table(dw_single_data_data)
    # delete()
    sync_table(dw_single_data_data)
"""