"""Diagnostic requests for an explicitly owned loopback vLLM deployment."""
import hashlib
import http.client
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from parity.trace_client import control, write


def guard(root):
    from observation.collect import memory_available
    sensor = next(p.with_name("temp") for p in Path("/sys/class/thermal").glob("thermal_zone*/type")
                  if p.read_text().strip() == "gpu-thermal")
    temp, available = int(sensor.read_text()), memory_available()
    with (root/"guards.jsonl").open("a") as f:
        f.write(json.dumps({"mono_ns":time.monotonic_ns(), "temp_path":str(sensor),
                            "gpu_millidegC":temp, "mem_available_bytes":available})+"\n")
    if temp >= 85000 or available < 12*1024**3:
        raise RuntimeError("diagnostic reached the device protection limit")


def call(root, endpoint, model, label, prompt, budget, cancel=False):
    rid = "depth-" + uuid.uuid4().hex + "-" + label
    body = {"model":model, "request_id":rid, "prompt":prompt, "temperature":0, "seed":381127,
            "max_tokens":budget, "min_tokens":budget, "ignore_eos":True, "return_token_ids":True,
            "stream":cancel}
    if cancel:
        body["stream_options"] = {"include_usage":True}
    raw = json.dumps(body, separators=(",",":")).encode()
    dest = root/"http"
    (dest/(label+".request")).write_bytes(raw)
    conn = http.client.HTTPConnection(endpoint.hostname, endpoint.port, timeout=150)
    record = {"status":"running", "label":label, "run_id":rid, "request":body,
              "role":"diagnostic", "done":False, "chunks":[],
              "prompt_sha256":hashlib.sha256(json.dumps(prompt,separators=(",",":")).encode()).hexdigest()}
    record.update(start_ns=time.monotonic_ns(), wall_start_ns=time.time_ns())
    response = None
    try:
        conn.request("POST", "/v1/completions", body=raw, headers={"Content-Type":"application/json"})
        response = conn.getresponse()
        record.update(http_status=response.status)
        with (dest/(label+".response")).open("xb") as f:
            if response.status != 200:
                error = response.read()
                f.write(error)
                raise RuntimeError(f"HTTP {response.status}: {error[:500]!r}")
            if not cancel:
                raw = response.read()
                f.write(raw)
                record.update(response=json.loads(raw), status="ok")
            else:
                for line in response:
                    when = time.monotonic_ns()
                    f.write(line)
                    if line.startswith(b"data:"):
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            break
                        message = json.loads(data)
                        record["chunks"].append({"mono_ns":when,"payload":message})
                        if any(c.get("token_ids") for c in message["choices"]):
                            record.update(status="disconnected", client_close_start_ns=time.monotonic_ns())
                            break
                if record["status"] != "disconnected":
                    raise ValueError("stream ended before controlled cancel")
    except Exception as exc:
        record.update(status="failed", error=repr(exc))
        raise
    finally:
        if response is not None:
            response.close()
        conn.close()
        record.update(end_ns=time.monotonic_ns(),wall_end_ns=time.time_ns())
        write(dest/(label+".json"), record)
    return rid


def collect(root, profile, cancel, *, legacy=False):
    plan = json.loads((root/"depth-plan.json").read_text())
    endpoint = urlsplit(plan["endpoint"])
    base = plan["endpoint"].rstrip("/")
    model = plan["served_model"]
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or endpoint.path not in ("","/"):
        raise ValueError("owned loopback HTTP endpoint required for depth observation")
    advertised = json.load(urlopen(base+"/v1/models", timeout=5))["data"]
    matched = [m for m in advertised if m["id"] == model]
    if len(matched) != 1 or Path(matched[0]["root"]).resolve() != Path(plan["checkpoint"]).resolve():
        raise ValueError("diagnostic ownership/model binding failed")
    (root/"http").mkdir()
    write(root/"discovery.json", advertised)
    write(root/"diagnostics.json", {"profile":profile, "cancel":cancel, "role":"diagnostic",
                                   "count_mode":"forced finite token count outside formal samples"})
    def invoke(label, out, cancel=False):
        guard(root)
        text = "TileScope public diagnostic " + label + ". Name three colors and explain each in one sentence."
        body = json.dumps({"model":model, "prompt":text}).encode()
        request = Request(base+"/tokenize",data=body,headers={"Content-Type":"application/json"})
        with urlopen(request, timeout=10) as r:
            tokenized = json.load(r)
        write(root/"http"/(label+"-tokenize.json"), {"text":text,"response":tokenized})
        return call(root, endpoint, model, label, tokenized["tokens"], out, cancel)
    def profiler(route):
        req = Request(base+"/"+route, data=b"")
        with urlopen(req, timeout=90) as res:
            write(root/(route+".json"), {"status":res.status, "body":res.read().decode(), "wall_ns":time.time_ns()})
    try:
        control(root, "tensors" if legacy else "dense")
        invoke("layer-replay" if legacy else "dense", 4)
        if not legacy and not (root/"dense.json").is_file():
            raise ValueError("no real first-prefill tensor observation from the owned worker")
        if profile:
            control(root, "profile")
            profiler("start_profile")
            try:
                invoke("profile", 8)
            finally:
                profiler("stop_profile")
        if cancel:
            control(root, "events")
            rid = invoke("cancel", 256, cancel=True)
            limit = time.monotonic()+15
            while time.monotonic() < limit:
                if any('"external_finish"' in line and '"FINISHED_ABORTED"' in line and rid in line
                       for path in root.glob("engine-*.jsonl") for line in path.read_text().splitlines()):
                    break
                time.sleep(.05)
            else:
                raise RuntimeError("request-specific scheduler abort not observed")
            invoke("followup", 2)
    finally:
        control(root,"off")
