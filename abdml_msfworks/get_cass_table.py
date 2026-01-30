import os
import json
import csv
from cassandra.cluster import Cluster, ExecutionProfile, EXEC_PROFILE_DEFAULT
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import DCAwareRoundRobinPolicy, RetryPolicy, TokenAwarePolicy
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement
from utils.config_reader import ConfigReader


# ---------------- Cassandra connection ----------------
cfg = ConfigReader()
cassandra_config = cfg["cassandra"]

CASSANDRA_HOST = cassandra_config["host"]
USERNAME = cassandra_config["username"]
PASSWORD = cassandra_config["password"]
KEYSPACE = cassandra_config["keyspace"]

auth_provider = PlainTextAuthProvider(USERNAME, PASSWORD)

profile = ExecutionProfile(
    load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy(local_dc=None)),
    request_timeout=60.0,
    consistency_level=ConsistencyLevel.LOCAL_ONE,
    retry_policy=RetryPolicy(),
)

cluster = Cluster(
    [CASSANDRA_HOST],
    auth_provider=auth_provider,
    execution_profiles={EXEC_PROFILE_DEFAULT: profile},
)
session = cluster.connect(KEYSPACE)


# ---------------- Query ----------------
# ⚠️ This works ONLY if work_station_id is in PRIMARY KEY
query = """
SELECT *
FROM dw_tbl_raw_data
WHERE work_station_id = 441165 AND produced_stock_name = 'Loperamide 2 mg granulate'
ALLOW FILTERING
"""

statement = SimpleStatement(query, fetch_size=5000)
rows = session.execute(statement)


# ---------------- CSV export ----------------
output_dir = "tables"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "dw_tbl_raw_data_ws_441165_loperamide_2_mg_granulate.csv")

csv_file = open(output_file, mode="w", newline="", encoding="utf-8")
writer = None

for row in rows:
    row_dict = row._asdict()

    # Convert Cassandra complex types to string
    for k, v in row_dict.items():
        if isinstance(v, (dict, list, tuple, set)):
            row_dict[k] = json.dumps(v, ensure_ascii=False, default=str)

    if writer is None:
        writer = csv.DictWriter(csv_file, fieldnames=row_dict.keys())
        writer.writeheader()

    writer.writerow(row_dict)

csv_file.close()

print(f"✅ Exported to CSV: {output_file}")

session.shutdown()
cluster.shutdown()
