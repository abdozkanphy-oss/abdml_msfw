import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datetime import datetime
import uuid
from cassandra_utils.models.dw_tbl_anomalies import DwTblAnomalies
from cassandra.cqlengine.management import sync_table

# Sync the table
sync_table(DwTblAnomalies)

# Dummy message to save
dummy_message = {
    "partition_date": "2025-04-18",
    "unique_code": str(uuid.uuid4()),
    "active": True,
    "anomaly": 1,
    "anomaly_detection_active": True,
    "anomaly_importance": 0.88,
    "bias": 0.12,
    "coefficient": 1.05,
    "counter_reading": 1450.25,
    "create_auto_maintenance_order_notification": False,
    "create_date": int(datetime.now().timestamp() * 1000),
    "customer": "TEST",
    "employee_id": 101,
    "equipment_code_group_header_id": 201,
    "equipment_id": 301,
    "equipment_measuring_point_id": 401,
    "equipment_name": "TEST",
    "equipment_no": "TEST",
    "job_order_operation_id": 501,
    "machine_state": "Running",
    "max_error_count_in_hour": 5,
    "max_error_count_in_min": 1,
    "max_error_count_in_shift": 12,
    "mean_square_error": 0.03,
    "measurement_date": int(datetime.now().timestamp() * 1000),
    "measurement_document_id": 601,
    "message_key": "TEST",
    "month_year": "TEST",
    "organization_id": 701,
    "organization_name": "TEST",
    "organization_no": "TEST",
    "parameter": "TEST",
    "plant_id": 801,
    "plant_name": "TEST",
    "predicted_lower": 99.1,
    "predicted_upper": 101.5,
    "predicted_value": 100.2,
    "produced_stock_id": 901,
    "produced_stock_name": "TEST",
    "produced_stock_no": "TEST",
    "regression_value": 0.92,
    "shift_finish_text": "TEST",
    "shift_finish_time": int(datetime.now().timestamp() * 1000),
    "shift_start_text": "TEST",
    "shift_start_time": int(datetime.now().timestamp() * 1000),
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
    DwTblAnomalies.save_data(dummy_message)
    print("✅ Dummy data inserted using save_data() in dw_tbl_anomalies.")
except Exception as e:
    print("❌ Error inserting dummy data:", e)
