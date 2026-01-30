import uuid
from datetime import datetime
from cassandra.cqlengine.management import sync_table
from cassandra.cqlengine import connection
from cassandra.auth import PlainTextAuthProvider

from cassandra_utils.models.dw_single_data import dw_single_data_data

# ---- Cassandra Setup ----


CASSANDRA_HOST = "108.61.178.52"
USERNAME = "cassandra"
PASSWORD = "cassAdmin%&.."
KEYSPACE = "das_new_pm"

# Initialize connection
connection.setup(
    hosts=[CASSANDRA_HOST],
    default_keyspace=KEYSPACE,
    auth_provider=PlainTextAuthProvider(username=USERNAME, password=PASSWORD),
    protocol_version=4
)

# Sync table if not already created
sync_table(dw_single_data_data)

# ---- Dummy Data to Insert ----
dummy_message = {
    "crDt": int(datetime.now().timestamp() * 1000),
    "plId": 101,
    "plNm": "Plant A",
    "joRef": "JO-001",
    "joOpId": 201,
    "opNo": "OP-100",
    "opNm": "Welding",
    "opTc": "WLD-123",
    "ordNo": "ORD-9999",
    "empId": 301,
    "empNo": "EMP-009",
    "empFn": "John",
    "empLn": "Doe",
    "wsId": 401,
    "wsNo": "WS-001",
    "wsNm": "Station 1",
    "wcId": 501,
    "wcNo": "WC-001",
    "wcNm": "Center 1",
    "mcSt": "Running",
    "prSt": "InProgress",
    "shId": 601,
    "shNm": "Morning",
    "shSt": int(datetime.now().timestamp() * 1000),
    "shFt": int(datetime.now().timestamp() * 1000),
    "orgId": "ORG-01",
    "orgNo": "ON-01",
    "orgNm": "OrgName Ltd",
    "inVars": [{"temp": "200", "speed": "100"}],
    "compBats": [{"batchNo": "B123", "material": "Steel"}],
    "outVals": [{"length": "10m", "quality": "A+"}],
    "prodList": [{"item": "Widget A", "qty": "10"}],
    "qCList": [{"check": "Visual", "result": "Pass"}]
}

# ---- Insert ----
try:
    result = dw_single_data_data.saveData(dummy_message)
    print("✅ Dummy data inserted successfully:")
    print(result)
except Exception as e:
    print("❌ Failed to insert dummy data:")
    print(e)
