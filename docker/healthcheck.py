import json
import urllib.request

with urllib.request.urlopen("http://127.0.0.1:8000/ping", timeout=10) as response:
    status = json.load(response)
if status.get("status") != "alive":
    raise SystemExit(1)
