"""Read aggregate device telemetry and the deployed server's activity counters."""
import json
import time
import urllib.request
from pathlib import Path

import pynvml as nv

names = {"vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:prompt_tokens_total",
         "vllm:generation_tokens_total"}
nv.nvmlInit()
handle = nv.nvmlDeviceGetHandleByIndex(0)
temp = next((p.with_name("temp") for p in Path("/sys/class/thermal").glob("thermal_zone*/type")
             if p.read_text().strip() == "gpu-thermal"), None)
while True:
    sample = {"unix_ns": time.time_ns()}
    try:
        sample["gpu_busy_pct"] = nv.nvmlDeviceGetUtilizationRates(handle).gpu
        sample["gpc_hz"] = int(Path("/sys/class/devfreq/gpu-gpc-0/cur_freq").read_text())
        sample["temperature_mc"] = int(temp.read_text()) if temp else None
        with urllib.request.urlopen("http://127.0.0.1:58000/metrics", timeout=0.3) as response:
            for line in response.read().decode().splitlines():
                name = line.partition("{")[0]
                if name in names:
                    sample[name] = float(line.split()[-1])
    except Exception as error:
        sample["read_error"] = str(error)
    print(json.dumps(sample), flush=True)
    time.sleep(.08)
