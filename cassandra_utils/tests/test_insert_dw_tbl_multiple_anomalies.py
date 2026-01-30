import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import uuid
from datetime import datetime
from cassandra_utils.models.dw_tbl_multiple_anomalies import DwTblMultipleAnomalies


def test_insert_dw_tbl_multiple_anomalies():
    dummy_message = {
        "partition_date": "2025-04-20",
        "unqCd": str(uuid.uuid4()),
        "act": True,
        "anomDetAct": True,
        "crAutoMaintOrdNot": False,
        "anomaly_importance": 0.8,
        "anomaly_score": 0.95,
        "bias": 0.1,
        "coef": 1.2,
        "cntRead": 123.45,
        "detected_anomaly": 0.5,
        "eqCodeGrpHdrId": 101,
        "eqId": 202,
        "eqMeasPtId": 303,
        "eqNm": "Test Equipment",
        "eqNo": "EQ-001",
        "max_error_count_in_hour": 2,
        "max_error_count_in_min": 1,
        "max_error_count_in_shift": 3,
        "mean_square_error": 0.02,
        "measurement_document_id": 404,
        "message_key": "MSG-001",
        "monYr": "4-2025",
        "parameter": "temp",
        "plantNo": "PL-001",
        "produced_stock_id": 505,
        "produced_stock_name": "StockName",
        "produced_stock_no": "ST-001",
        "regression_value": 0.7,
        "sensor_values": {"sensor1": {"val": "10"}},
        "stock_id": 606,
        "trend_calculation_active": True,
        "valuation_code": "VAL-001",
        "windowing_timestamp": 1680000000000,
        "workstation_state": "Running",
        "year": "2025"
    }
    dummy_main_message = {
        "employeeId": 1,
        "jobOrderOperationId": 2,
        "organizationId": 3,
        "plantId": 4,
        "workCenterId": 5,
        "workStationId": 6,
        "machineState": "Running",
        "organizationName": "OrgName",
        "organizationNo": "ORG-001",
        "plantName": "PlantName",
        "plantNo": "PL-001",
        "workCenterName": "CenterName",
        "workCenterNo": "WC-001",
        "workStationName": "StationName",
        "workStationNo": "WS-001",
        "shiftFinishTime": datetime.now(),
        "shiftStartTime": datetime.now(),
    }
    y = 123.45
    key = "TESTKEY"
    try:
        result = DwTblMultipleAnomalies.saveData(
            message=dummy_message, main_message=dummy_main_message, y=y, key=key)
        print("✅ Dummy data inserted successfully:")
        print(result)
    except Exception as e:
        print("❌ Failed to insert dummy data:")
        print(e)


if __name__ == "__main__":
    test_insert_dw_tbl_multiple_anomalies()
