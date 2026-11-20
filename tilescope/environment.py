"""Read-only resource and dependency checks for the selected Thor environment."""
import csv
import json
import os
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request

from .common import ROOT, sha256

DEFAULT_PYTHON = "/opt/tilescope/venv/bin/python"
DEFAULT_MODEL = "/opt/models/Qwen3.8-27B-NVFP4"
DEFAULT_ENDPOINT = "http://127.0.0.1:58000"
CUDA = "/usr/local/cuda-13.3"

# Executed by the caller's chosen deployment Python, not by the offline CLI.
PROBE = """
import json, importlib, importlib.metadata as m, pathlib, torch, sys
out = {'python':sys.version.split()[0], 'versions':{}, 'cuda_build':torch.version.cuda}
for p in ('torch','cuda-tile','triton','vllm','flashinfer-python','safetensors','nvidia-ml-py'):
    try: out['versions'][p]=m.version(p)
    except m.PackageNotFoundError: out['versions'][p]=None
import cuda.tile, safetensors, pynvml
from vllm import _custom_ops
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm_out
props=torch.cuda.get_device_properties(0)
out.update(name=props.name,cc=[props.major,props.minor],sm_count=props.multi_processor_count,
           global_memory_bytes=props.total_memory,device_uuid=str(props.uuid),
           cuda_free_total_bytes=list(torch.cuda.mem_get_info()))
out['libraries']=sorted({l.split()[-1] for l in pathlib.Path('/proc/self/maps').read_text().splitlines()
                         if '/libcudart.so' in l or '/libcuda.so' in l})
out['vllm_dir']=str(pathlib.Path(_custom_ops.__file__).parent)
print(json.dumps(out))
"""


def file_text(path):
    try:
        return Path(path).read_text().replace("\x00", " ").strip()
    except OSError:
        return None


def cmd(argv, timeout=12):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "exit": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "exit": None, "stdout": "", "stderr": str(error)}


def service_activity(endpoint):
    """Observe aggregate counters; no user text, requests or credentials collected."""
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/metrics", timeout=1) as r:
            lines = r.read().decode().splitlines()
        names = {"vllm:num_requests_running", "vllm:num_requests_waiting",
                 "vllm:prompt_tokens_total", "vllm:generation_tokens_total"}
        return {line.split("{", 1)[0]: float(line.split()[-1]) for line in lines
                if line.split("{", 1)[0] in names}
    except (OSError, ValueError) as e:
        return {"unavailable": str(e)}


def buffer_floor(selected):
    """At least the largest single materialized input must fit."""
    sizes = []
    for task in selected:
        for dtype, shape, _ in task["inputs"].values():
            width = {"torch.bfloat16":2,"torch.float32":4,"torch.float8_e4m3fn":1,"torch.uint8":1}.get(dtype)
            if width is not None:
                sizes.append(math.prod(shape)*width)
    return max(sizes, default=0)


def inspect(python, checkpoint, endpoint, minimum_bytes=0, toolkit=CUDA, selected=()):
    checkpoint = Path(checkpoint)
    prop = {"model": file_text("/proc/device-tree/model"), "compatible": file_text("/proc/device-tree/compatible"),
            "sku": file_text("/sys/class/dmi/id/product_sku"), "arch": platform.machine(),
            "kernel": platform.release(), "l4t": file_text("/etc/nv_tegra_release")}
    blocks, warnings = [], []
    power = cmd(["nvpmodel", "-q"])
    tools = {name: cmd([str(Path(toolkit) / "bin" / name), "--version"]) for name in ("nvcc", "tileiras", "ptxas")}
    if not prop["model"] or "thor" not in (prop["model"] + str(prop["compatible"])).lower():
        blocks.append("device: expected a Thor device-tree identity")
    if power["exit"] != 0:
        warnings.append("power mode unavailable")
    if tools["tileiras"]["exit"] != 0:
        blocks.append("tileiras compiler unavailable in selected toolkit")

    meminfo = file_text("/proc/meminfo") or ""
    mem = {line.split(":")[0]: int(line.split()[1])*1024 for line in meminfo.splitlines()
           if line.startswith(("MemAvailable:", "MemTotal:"))}
    floor = max(minimum_bytes, buffer_floor(selected))
    resources = {"linux_meminfo_bytes": mem, "cuda_free_total_bytes": None,
                 "disk_free_bytes": shutil.disk_usage(".").free, "minimum_free_bytes": minimum_bytes,
                 "largest_required_buffer_bytes": buffer_floor(selected),
                 "cpu_load_1_5_15": list(os.getloadavg())}
    if mem.get("MemAvailable", floor) < floor:
        blocks.append("system MemAvailable below workload or requested minimum")
    runtime = {}
    probe = cmd([python, "-c", PROBE], timeout=45)
    if probe["exit"] == 0:
        runtime = json.loads(probe["stdout"].splitlines()[-1])
        resources["cuda_free_total_bytes"] = runtime.pop("cuda_free_total_bytes")
        if resources["cuda_free_total_bytes"][0] < floor:
            blocks.append("CUDA allocatable free bytes below workload or requested minimum")
        if runtime["cc"] != [11, 0]:
            blocks.append("CUDA device 0 is not CC 11.0")
        absent = [k for k,v in runtime["versions"].items() if v is None]
        if absent:
            blocks.append("missing deployment packages: " + ", ".join(absent))
    else:
        blocks.append("selected Python/CUDA deployment import or device query failed: " + probe["stderr"][-420:])

    expected = {name: value for value,name in (line.split(maxsplit=1) for line in
                (ROOT/"round2/evidence/model/source-hashes.txt").read_text().splitlines())}
    model = {"path": str(checkpoint), "files": {}}
    for name in ("config.json","model.safetensors.index.json","model.safetensors","model_mtp.safetensors"):
        p = checkpoint/name
        if not p.is_file():
            blocks.append("missing checkpoint file: "+name)
            continue
        item = {"bytes": p.stat().st_size, "expected_sha256": expected[name]}
        if name.endswith(".json"):
            item["sha256"] = sha256(p)
            if item["sha256"] != expected[name]:
                blocks.append(f"checkpoint identity mismatch: {name}")
        model["files"][name] = item

    observed = []
    for folder in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        if file_text(folder/"type") == "gpu-thermal":
            observed.append({"path":str(folder/"temp"), "unit":"m°C", "value":file_text(folder/"temp")})
    for name in ("gpu-gpc-0","gpu-nvd-0"):
        path = f"/sys/class/devfreq/{name}/cur_freq"
        observed.append({"path":path,"unit":"Hz","value":file_text(path)})
    gpu = cmd(["nvidia-smi","--query-gpu=name,compute_cap,driver_version,utilization.gpu,temperature.gpu,clocks.gr",
               "--format=csv,noheader,nounits"])
    driver = None
    if gpu["exit"]==0:
        fields=next(csv.reader([gpu["stdout"].splitlines()[0]],skipinitialspace=True))
        driver=fields[2]
    activity = [{"time_ns":time.time_ns(), "metrics":service_activity(endpoint)}]
    if "unavailable" in activity[0]["metrics"]:
        warnings.append("service activity unavailable")
    elif activity[0]["metrics"].get("vllm:num_requests_running",0)>0:
        warnings.append("model requests active during preflight; inspect run telemetry")
    if any(v["value"] is None for v in observed) or not any(v["unit"]=="m°C" for v in observed):
        warnings.append("some GPU sensor readings unavailable")
    processes=cmd(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"])
    binaries = [Path(toolkit)/"bin"/p for p in ("tileiras","ptxas","nvcc")]
    if runtime.get("vllm_dir"):
        binaries.extend(Path(runtime["vllm_dir"]).glob("_C*.so"))
    return {"schema":"tilescope.environment.v2","time_ns":time.time_ns(),"device":prop,"runtime":runtime,
            "binary_sha256":{p.name:sha256(p) for p in binaries if p.is_file()},
            "clock_policy":{key:file_text("/sys/class/devfreq/gpu-gpc-0/"+key)
                            for key in ("governor","min_freq","max_freq")},
            "driver":driver,"resources":resources,"power_mode":power,"compiler":tools,"checkpoint":model,
            "sensors":observed,"gpu_snapshot":gpu,"gpu_processes":processes,"endpoint":endpoint,
            "activity":activity,"python_probe_error":probe["stderr"] if probe["exit"] else None,
            "measurement_scope":"Linux available system RAM and CUDA free device memory are distinct views of unified memory; GPU utilization is aggregate; power mode is a policy.",
            "status":"blocked" if blocks else ("warning" if warnings else "ready"),"blocked":blocks,"warnings":warnings}
