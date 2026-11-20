"""Run one private stock engine until its stop file appears or a resource limit trips."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen

from parity.trace_lifecycle import stop_group
from tilescope.common import device_slot, write_json
from .targets import resolve
from tools.multimodel.guard import guard, sample


def recipe(target, root, depth):
    d = target["deployment"]
    if d["mode"] != "owned_instance":
        raise ValueError("private launcher requires owned_instance")
    endpoint = urlsplit(d["endpoint"])
    limits = d["launch_parameters"]
    opts = [d["runtime_executable"]]
    extra = {}
    if target["engine"]["adapter"]=="llama_cpp.native.v1":
        opts += ["--model",d["checkpoint"],"--alias",d["served_model"],"--host",endpoint.hostname,
                 "--port",str(endpoint.port),"--ctx-size",str(limits["max_context"]*limits["max_sequences"]),
                 "--parallel",str(limits["max_sequences"]),"--n-gpu-layers","all","--metrics",
                 "--no-webui","--no-warmup"]
    else:
        opts += ["-m","vllm.entrypoints.cli.main","serve",d["checkpoint"],"--host",endpoint.hostname,
                 "--port",str(endpoint.port),"--served-model-name",d["served_model"],"--dtype","bfloat16",
                 "--tensor-parallel-size",str(limits["tensor_parallel"]),"--enforce-eager","--no-enable-prefix-caching",
                 "--max-model-len",str(limits["max_context"]),"--max-num-seqs",str(limits["max_sequences"]),
                 "--max-num-batched-tokens",str(limits["max_context"]),"--kv-cache-memory-bytes",str(limits["memory_budget_bytes"]),
                 "--gpu-memory-utilization","0.15"]
        if depth:
            opts += ["--profiler-config",json.dumps({"profiler":"torch","torch_profiler_dir":str(depth/"profiles"),
                     "torch_profiler_with_stack":False,"torch_profiler_record_shapes":False,"ignore_frontend":True})]
            # The audited worker hook is on the upstream V1 runner; V2 has different observation boundaries.
            extra.update(VLLM_USE_V2_MODEL_RUNNER="0",TILESCOPE_TRACE_DIR=str(depth),PYTHONPATH=os.pathsep.join([
                str(Path(__file__).parent/"depth/hook"),str(Path(__file__).resolve().parents[2]),os.environ.get("PYTHONPATH","")]))
        extra.update(VLLM_CACHE_ROOT=str(root/"cache/vllm"),TORCHINDUCTOR_CACHE_DIR=str(root/"cache/inductor"),
                     TRITON_CACHE_DIR=str(root/"cache/triton"),CUDA_CACHE_PATH=str(root/"cache/cuda"),VLLM_NO_USAGE_STATS="1",
                     CUDA_HOME="/usr/local/cuda-13.3",PATH="/usr/local/cuda-13.3/bin:"+os.environ["PATH"])
    return opts,extra


def serve(target_path, out, depth):
    t = resolve(target_path)
    d=t["deployment"]
    root=out.resolve()
    root.mkdir(parents=True,exist_ok=False)
    if depth:
        depth=depth.resolve()
        subprocess.run([sys.executable,"-m","tilescope.probe.depth","prepare","--target",str(target_path),
                        "--out",str(depth)],check=True)
    cmd,extra=recipe(t,root,depth)
    write_json(root/"launch.json",{"argv":cmd,"env_changes":extra,"profile":t,"launcher_pid":os.getpid()})
    lock=Path.home()/".cache/tilescope/owned-engine.lock"
    lock.parent.mkdir(parents=True,exist_ok=True)
    stop=[]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda s,_:stop.append(s))
    with lock.open("a") as fd,(root/"stdout").open("x") as stdout,(root/"stderr").open("x") as stderr, \
            (root/"resources.jsonl").open("x",buffering=1) as trend:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        def protected():
            s=sample()
            print(json.dumps(s),file=trend)
            guard(s)
            return not stop and not (root/"STOP").exists()
        p=None
        try:
            with device_slot():
                if not protected():
                    raise RuntimeError("startup cancelled")
                with socket.socket() as sock:
                    u=urlsplit(d["endpoint"]);sock.bind((u.hostname,u.port))
                p=subprocess.Popen(cmd,env={**os.environ,**extra},stdout=stdout,stderr=stderr,start_new_session=True)
                start_ticks=int(Path(f"/proc/{p.pid}/stat").read_text().rsplit(")",1)[1].split()[19])
                deadline=time.monotonic()+480
                while time.monotonic()<deadline and protected():
                    if p.poll() is not None:
                        raise RuntimeError(f"private service exited {p.returncode}")
                    try:
                        with urlopen(d["endpoint"]+"/health",timeout=1) as res:
                            if res.status==200:
                                break
                    except OSError:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError("private startup timed out or was cancelled")
                write_json(root/"ready.json",{"pid":p.pid,"start_ticks":start_ticks,"wall_ns":time.time_ns()})
            while protected():
                if p.poll() is not None:
                    raise RuntimeError(f"owned server exited unexpectedly: {p.returncode}")
                time.sleep(1)
        finally:
            if p:
                stop_group(p)
            write_json(root/"shutdown.json",{"pid":None if p is None else p.pid,"exit":None if p is None else p.returncode,
                                             "wall_ns":time.time_ns(),"signal":stop})


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--depth",type=Path)
    args=ap.parse_args()
    serve(args.target,args.out,args.depth)
