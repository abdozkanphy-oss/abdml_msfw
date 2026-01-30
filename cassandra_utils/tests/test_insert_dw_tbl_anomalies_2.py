import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datetime import datetime
import uuid
from cassandra_utils.models.dw_tbl_anomalies_2 import DwTblAnomalies2
from cassandra.cqlengine.management import sync_table

# Sync the table
sync_table(DwTblAnomalies2)

# Current date for partition_date and year_month
current_date = datetime.now()
partition_date = current_date.strftime("%Y-%m-%d")
year_month = current_date.strftime("%Y-%m")

# Dummy message to save
dummy_message = {
    "year_month": year_month,
    "partition_date": partition_date,
    "unique_code": str(uuid.uuid4()),
    "measurement_date": int(datetime.now().timestamp() * 1000),
    "active": True, #out['act']
    "anomaly": 1, "1 if detected else 0"
    "anomaly_detection_active": True, # NULL
    "anomaly_importance": 0.88, # 0?
    "bias": 0.12, # out['bias']
    "coefficient": 1.05, # out['coef']
    "counter_reading": 1450.25, # out['cntRead']
    "create_auto_maintenance_order_notification": False, # NULL
    "create_date": int(datetime.now().timestamp() * 1000), 
    "customer": "ABC Corp", # NF
    "employee_id": 101, # out['eqId]
    "equipment_code_group_header_id": 201, # out['eqCodeGrpHdrId]
    "equipment_id": 301, # out['eqId']
    "equipment_measuring_point_id": 401, # out['eqMeasPtId']
    "equipment_name": "Pump 3", # out['eqNm']
    "equipment_no": "EQ-0003", # out['eqNo']
    "job_order_operation_id": 501, # main[jobOrderOperationId]
    "machine_state": "Running", # main machineState
    "max_error_count_in_hour": 5, # NULL
    "max_error_count_in_min": 1, # NULL
    "max_error_count_in_shift": 12, # NULL
    "mean_square_error": 0.03, # NULL
    "measurement_document_id": 601, # NULL
    "message_key": "TEST", # Key
    "month_year": "Apr-2025", # out['monYr']
    "organization_id": 701, # main
    "organization_name": "TEST", # main
    "organization_no": "TEST", # main
    "parameter": "TEST", # NULL
    "plant_id": 801, # main
    "plant_name": "TEST", # main
    "predicted_lower": 99.1, # predict
    "predicted_upper": 101.5, # preict
    "predicted_value": 100.2, # predict
    "produced_stock_id": 901, #
    "produced_stock_name": "TEST",
    "produced_stock_no": "TEST",
    "regression_value": 0.92, # NULL
    "shift_finish_text": "TEST", # same as below
    "shift_finish_time": int(datetime.now().timestamp() * 1000), # main
    "shift_start_text": "TEST",  # same as below in str
    "shift_start_time": int(datetime.now().timestamp() * 1000), # main
    "stock_id": 1001,
    "trend": 1.23,
    "trend_calculation_active": True,
    "trend_lower": 0.95,
    "trend_upper": 1.5,
    "update_date": int(datetime.now().timestamp() * 1000),
    "valuation_code": "TEST",
    "windowing_timestamp": int(datetime.now().timestamp() * 1000),
    "work_center_id": 1101,
    "work_center_name": "TEST",
    "work_center_no": "TEST",
    "work_station_id": 1201,
    "work_station_name": "TEST",
    "work_station_no": "TEST",
    "workstation_state": "TEST",
    "year": "2025"
}

# Insert using the save_data method
try:
    DwTblAnomalies2.save_data(dummy_message)
    print("✅ Dummy data inserted using save_data() in dw_tbl_anomalies_2.")
except Exception as e:
    print("❌ Error inserting dummy data:", e) 