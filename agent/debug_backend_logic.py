import os
import pytz
from datetime import datetime
from influxdb_client import InfluxDBClient

url = 'http://175.45.201.123:8086'
token = 'vbQW6tkkqW5G867zF2I0qXq9tF-5T_UEH-76F0E0a5wPtX83X2LgqW9-11p6Z0k-rJOq5C3E57H1X4_a8oD8lA=='
org = 'spint'
bucket = 'spint'

# Increased timeout to 60s
client = InfluxDBClient(url=url, token=token, org=org, debug=False, timeout=60000)
query_api = client.query_api()

kst = pytz.timezone('Asia/Seoul')
now = datetime.now(kst)
start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
start_str = start_of_day.isoformat()
print(f"Querying from Start of Day: {start_str}")

# Check Count first (Data Density)
flux_query_count = f'''
from(bucket: "{bucket}")
  |> range(start: {start_str})
  |> filter(fn: (r) => r["_measurement"] == "cnc_analyze")
  |> filter(fn: (r) => r["_field"] == "Run")
  |> filter(fn: (r) => r["tag_id"] == "OP10-1-260123-00548")
  |> count()
'''

print("\n--- Checking Data Density (Count) ---")
try:
    result = query_api.query(org=org, query=flux_query_count)
    if not result:
        print("No data found (Count is 0)")
    for table in result:
        for record in table:
            print(f"Total Records: {record.get_value()}")
except Exception as e:
    print(f"Error in Count query: {e}")

# Backend Logic: Sum
flux_query_sum = f'''
from(bucket: "{bucket}")
  |> range(start: {start_str})
  |> filter(fn: (r) => r["_measurement"] == "cnc_analyze")
  |> filter(fn: (r) => r["_field"] == "Run")
  |> filter(fn: (r) => r["tag_id"] == "OP10-1-260123-00548")
  |> aggregateWindow(every: 1d, fn: sum, createEmpty: false)
'''

print("\n--- Testing Backend Logic (SUM) ---")
try:
    result = query_api.query(org=org, query=flux_query_sum)
    for table in result:
        for record in table:
            val = record.get_value()
            print(f"Sum Result (Raw): {val}")
            print(f"As Seconds -> Hours: {val / 3600:.4f}")
            print(f"As Milliseconds -> Hours: {val / 3600000:.4f}")
except Exception as e:
    print(f"Error in Sum query: {e}")

# Check Values again
flux_query_values = f'''
from(bucket: "{bucket}")
  |> range(start: {start_str})
  |> filter(fn: (r) => r["_measurement"] == "cnc_analyze")
  |> filter(fn: (r) => r["_field"] == "Run")
  |> filter(fn: (r) => r["tag_id"] == "OP10-1-260123-00548")
  |> limit(n: 20)
'''
print("\n--- Sampling Values ---")
try:
    result = query_api.query(org=org, query=flux_query_values)
    for table in result:
        for record in table:
            print(f"Time: {record.get_time()}, Value: {record.get_value()}")
except Exception as e:
    print(f"Error in sampling: {e}")
