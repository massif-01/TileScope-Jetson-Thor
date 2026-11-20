"""Capture controlled online request timing and separate private runner shape evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request
import uuid


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def fetch(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not readable")


def serving_processes(checkpoint):
    result = []
    keys = {"--tensor-parallel-size", "--max-num-seqs", "--max-num-batched-tokens",
            "--max-model-len", "--gpu-memory-utilization", "--attention-backend",
            "--speculative-config", "--enable-log-requests", "--enforce-eager",
            "--enable-prefix-caching", "--enable-chunked-prefill", "--port"}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0")
            if "serve" not in args or not any(arg.rstrip("/") == str(checkpoint) for arg in args):
                continue
            config = {arg: args[i+1] if i+1 < len(args) and not args[i+1].startswith("--") else True
                      for i,arg in enumerate(args) if arg in keys}
            result.append(dict(pid=int(proc.name), exe=os.readlink(proc / "exe"), config=config))
        except (OSError, UnicodeDecodeError):
            continue
    return result


def sample(endpoint):
    item = dict(unix_ns=time.time_ns(), available_bytes=memory_available())
    names = {"vllm:num_requests_running","vllm:num_requests_waiting",
             "vllm:prompt_tokens_total","vllm:generation_tokens_total"}
    try:
        for line in fetch(endpoint + "/metrics").decode().splitlines():
            name = line.partition("{")[0]
            if name in names:
                item[name] = float(line.split()[-1])
    except (OSError, ValueError) as exc:
        item["metrics_error"] = str(exc)
    return item


def online_request(spec, endpoint, model):
    body = {"model":model, "request_id":spec["id"], "prompt":spec["ids"],
            "temperature":0, "seed":spec["seed"], "max_tokens":spec["outputs"],
            "min_tokens":spec["outputs"], "ignore_eos":True, "stream":True,
            "stream_options":{"include_usage":True}, "return_token_ids":True}
    req = urllib.request.Request(endpoint + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type":"application/json"})
    events = []
    begin = time.perf_counter_ns()
    wall = time.time_ns()
    with urllib.request.urlopen(req, timeout=120) as response:
        for raw in response:
            if not raw.startswith(b"data: ") or raw.strip()==b"data: [DONE]":
                continue
            delta = json.loads(raw[6:])
            elapsed = (time.perf_counter_ns() - begin) / 1e6
            choices = delta.get("choices",[])
            events.append(dict(elapsed_ms=elapsed, response_id=delta.get("id"),
                               token_ids=[t for c in choices for t in (c.get("token_ids") or [])],
                               nonempty_text=any(c.get("text") for c in choices),
                               usage=delta.get("usage")))
    total = (time.perf_counter_ns()-begin)/1e6
    nonempty = [e["elapsed_ms"] for e in events if e["nonempty_text"]]
    usage = next(e["usage"] for e in reversed(events) if e["usage"] is not None)
    if usage["prompt_tokens"] != len(spec["ids"]) or usage["completion_tokens"]!=spec["outputs"]:
        raise ValueError(f"Unexpected HTTP request usage: {usage}")
    return dict(id=spec["id"], start_unix_ns=wall, end_unix_ns=time.time_ns(), request=body,
                total_ms=total, first_nonempty_chunk_ms=nonempty[0] if nonempty else None,
                nonempty_chunk_gaps_ms=[b-a for a,b in zip(nonempty, nonempty[1:])],
                usage=usage, events=events)


def start_private(args):
    # Start only the observer's process group. Bound its memory footprint on shared LPDDR.
    cache = args.output / "temporary-cache"
    historical = Path.home()/".cache/flashinfer/0.6.11.post2/110a"
    # The JIT build embeds absolute source paths in its Ninja file. Copying a
    # built cache to a new path triggers a full rebuild. Use the package's AOT
    # lookup to load the unmodified existing deployment binary in this process.
    module_dir = Path(subprocess.check_output([str(args.python), "-c",
        "import pathlib,flashinfer_jit_cache as j; print(pathlib.Path(j.__file__).parent)"], text=True).strip())
    local_module = cache / "python/flashinfer_jit_cache"
    shutil.copytree(module_dir, local_module, ignore=shutil.ignore_patterns("jit_cache","__pycache__"))
    aot = local_module / "jit_cache"
    aot.mkdir()
    for child in (module_dir / "jit_cache").iterdir():
        (aot / child.name).symlink_to(child)
    reused = []
    for binary in (historical / "cached_ops").glob("*/*.so"):
        if not (aot / binary.parent.name).exists():
            target = aot / binary.parent.name
            target.mkdir()
            shutil.copy2(binary, target / binary.name)
            reused.append(dict(path=str(binary),sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
    save(args.output / "reused-libraries.json",reused)
    inductor = Path.home() / ".cache/vllm/torch_compile_cache/torch_aot_compile"
    seeds = list(inductor.glob("*/inductor_cache/triton/0"))
    if len(seeds) == 1:
        shutil.copytree(seeds[0], cache / "triton")
    save(args.output/"cache-seed.json", {"triton_seed":str(seeds[0]) if len(seeds)==1 else None})
    env = {**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING":"0", "OMP_NUM_THREADS":"2",
           "VLLM_CACHE_ROOT":str(cache/"vllm"), "TRITON_CACHE_DIR":str(cache/"triton"),
           "CUDA_CACHE_PATH":str(cache/"cuda"), "FLASHINFER_WORKSPACE_BASE":str(cache),
           "CUDA_HOME":"/usr/local/cuda-13.3", "PATH":"/usr/local/cuda-13.3/bin:"+os.environ["PATH"],
           "TOKENIZERS_PARALLELISM":"false", "PYTHONPATH":str(cache / "python")+os.pathsep+os.environ.get("PYTHONPATH",""),
           "MAX_JOBS":"1","FLASHINFER_NVCC_THREADS":"1","VLLM_NO_USAGE_STATS":"1"}
    cmd = [str(args.python), "-u", str(Path(__file__).with_name("runner.py")),
           "--checkpoint", str(args.checkpoint), "--output", str(args.output)]
    save(args.output/"isolated-command.json",cmd)
    deadline = time.monotonic()+args.timeout
    proc = None
    try:
        with (args.output/"isolated.stdout").open("w") as out, (args.output/"isolated.stderr").open("w") as err, \
                (args.output/"isolated-resources.jsonl").open("w",buffering=1) as stats:
            proc = subprocess.Popen(cmd,env=env,stdout=out,stderr=err,start_new_session=True)
            while proc.poll() is None:
                row = sample(args.endpoint)
                stats.write(json.dumps(row)+"\n")
                if row["available_bytes"] < 12 * 1024**3:
                    raise RuntimeError("Observer stopped: system MemAvailable below 12 GiB")
                if time.monotonic() > deadline:
                    raise TimeoutError("Private observation timed out")
                time.sleep(1)
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode,cmd)
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        if proc is not None:
            save(args.output/"isolated-exit.json",dict(pid=proc.pid,returncode=proc.returncode))
        shutil.rmtree(cache, ignore_errors=True)


def summarize(args):
    rows = [json.loads(line) for line in (args.output/"runner.jsonl").read_text().splitlines()]
    specs = {r["id"]:r for r in json.loads((args.output/"requests.json").read_text())}
    steps = {r["step"]:r for r in rows if r["event"]=="target_forward"}
    findings = []
    for r in rows:
        if r["event"]!="operator":
            continue
        step = steps[r["step"]]
        b, = step["batch"]
        spec = specs[r["request"]]
        expected = hashlib.sha256(json.dumps(spec["ids"],separators=(",",":")).encode()).hexdigest()
        if b["prompt_token_hash"]!=expected:
            raise ValueError("Forward not bound to controlled prompt")
        info = r["inputs"][0]
        key = (r["label"], b["phase"],info["shape"][0])
        if any(tuple(item["key"])==key for item in findings):
            continue
        findings.append(dict(key=key,step=r["step"],request_id=r["request"],
                             scheduler=b,actual_input=info,actual_output=r["output"],
                             actual_weight=r["weight"],module=r["module"],module_class=r["cls"],
                             input_args=r["inputs"],model_input_ids=step["input_ids"]))
    phases = {r["key"][1] for r in findings if r["key"][0]=="bf16_ba"}
    if "prefill" not in phases or not phases.intersection({"decode_target_verify","decode_target"}):
        raise ValueError("Missing observed prefill or decode-target BF16 projection")
    if {r["request"] for r in rows if r["event"]=="completed"}!=set(specs):
        raise ValueError("Missing completed controlled request")
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:58000")
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("CHECKPOINT", "/opt/models/Qwen3.8-27B-NVFP4"))
    parser.add_argument("--python", type=Path, default=os.environ.get("PROBE_PYTHON", "/opt/tilescope/venv/bin/python"))
    parser.add_argument("--timeout", type=int, default=600)
    args=parser.parse_args()
    args.output=args.output.resolve()
    args.checkpoint=args.checkpoint.resolve()
    args.output.mkdir(parents=True)
    deployment=serving_processes(args.checkpoint)
    try:
        if not deployment:
            raise RuntimeError("Could not bind online service to specified checkpoint")
        models=json.loads(fetch(args.endpoint+"/v1/models"))["data"]
        model=next(m for m in models if Path(m.get("root","")).resolve()==args.checkpoint)
        routes=json.loads(fetch(args.endpoint+"/openapi.json"))["paths"]
        identity={f:hashlib.sha256((args.checkpoint/f).read_bytes()).hexdigest()
                  for f in ("config.json","tokenizer.json","tokenizer_config.json")}
        before=sample(args.endpoint)
        save(args.output/"deployment.json",dict(processes=deployment,model=model["id"],checkpoint=str(args.checkpoint),
             identity=identity,metrics=before,profiling_routes=[r for r in routes if "profile" in r]))
        if before.get("vllm:num_requests_running")!=0 or before.get("vllm:num_requests_waiting")!=0:
            raise RuntimeError("Existing service busy; retry the controlled observation later")
        rid=uuid.uuid4().hex[:12]
        specs=[dict(id=f"tilescope-{rid}-{length}",ids=list(range(start,start+length)),seed=381127,outputs=12)
               for length,start in ((64,12331),(256,15571))]
        save(args.output/"requests.json",specs)
        timing=[]
        for spec in specs:
            pre=sample(args.endpoint)
            result=online_request(spec,args.endpoint,model["id"])
            timing.append({**result,"resources_before":pre,"resources_after":sample(args.endpoint)})
            save(args.output/"online-requests.json",timing)
        if memory_available()<38*1024**3:
            raise RuntimeError("Private model observer needs 38 GiB available unified memory")
        start_private(args)
        found=summarize(args)
        save(args.output/"summary.json",dict(schema="tilescope.observation.v1", checkpoint=str(args.checkpoint),
            online_request_count=len(timing),online_timing="online-requests.json",
            shape_source="isolated in-process eager vLLM target forward using real checkpoint; overlay instrumentation",
            graph_padding="eager observer; deployed CUDA graph replay geometry not captured",
            observation_records=found))
    finally:
        save(args.output/"postflight.json",dict(metrics=sample(args.endpoint),serving_processes=serving_processes(args.checkpoint)))


if __name__ == "__main__":
    main()
