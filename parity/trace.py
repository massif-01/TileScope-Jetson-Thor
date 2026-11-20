"""Run a bounded production-configuration trace on an owned, loopback-only service."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

from observation.collect import memory_available, serving_processes
from parity.trace_client import write
from parity.trace_resources import processes, snapshot
from parity.trace_graph import extract
from parity.trace_nvmap import heaps
from parity.trace_lifecycle import locked_slot, raise_if_requested, stop_group

def private_env(args):
    cache=args.output/"private-cache"
    cache.mkdir()
    module=Path(subprocess.check_output([str(args.python),"-c",
        "import pathlib,flashinfer_jit_cache as j;print(pathlib.Path(j.__file__).parent)"],text=True).strip())
    shell=cache/"python/flashinfer_jit_cache"
    shutil.copytree(module,shell,ignore=shutil.ignore_patterns("jit_cache","__pycache__"))
    aot=shell/"jit_cache"
    aot.mkdir()
    for x in (module/"jit_cache").iterdir():
        (aot/x.name).symlink_to(x)
    flash=Path.home()/".cache/flashinfer/0.6.11.post2/110a/cached_ops"
    libraries=[]
    for lib in sorted(flash.glob("*/*.so")):
        dest=aot/lib.parent.name
        if dest.is_symlink():
            continue
        dest.mkdir(exist_ok=True)
        shutil.copy2(lib,dest/lib.name)
        libraries.append({"path":str(lib),"sha256":hashlib.sha256(lib.read_bytes()).hexdigest()})
    source=Path.home()/".cache/vllm/torch_compile_cache"
    shutil.copytree(source,cache/"vllm/torch_compile_cache")
    tritons=list(source.glob("torch_aot_compile/*/inductor_cache/triton/0"))
    if tritons:
        for source_triton in tritons:
            shutil.copytree(source_triton,cache/"triton",dirs_exist_ok=True)
    write(args.output/"cache-seed.json",{"deployment_compile_directories":[p.name for p in source.iterdir()],
              "triton_seeds":[str(p) for p in tritons],"flashinfer_libraries":libraries,
              "page_cache":"uncontrolled; existing live service on same checkpoint"})
    hooks=Path(__file__).with_name("trace_hook").resolve()
    return {**os.environ,"TILESCOPE_TRACE_DIR":str(args.output),"PYTHONPATH":os.pathsep.join([
             str(hooks),str(cache/"python"),os.environ.get("PYTHONPATH","")]),
            "VLLM_CACHE_ROOT":str(cache/"vllm"),"TORCHINDUCTOR_CACHE_DIR":str(cache/"inductor"),
            "TRITON_CACHE_DIR":str(cache/"triton"),"CUDA_CACHE_PATH":str(cache/"cuda"),
            "FLASHINFER_WORKSPACE_BASE":str(cache),"VLLM_NO_USAGE_STATS":"1",
            "CUDA_HOME":"/usr/local/cuda-13.3","PATH":"/usr/local/cuda-13.3/bin:"+os.environ["PATH"],
            "MAX_JOBS":"1","FLASHINFER_NVCC_THREADS":"1","TORCHINDUCTOR_COMPILE_THREADS":"1"}

def resource_snapshot(path, query, owners):
    try:
        value=query(owners)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        value={"error":repr(exc),"unix_ns":time.time_ns()}
    write(path,value)

def gpu_free(path):
    try:
        result=subprocess.run([sys.executable,"-m","parity.trace_gpu_memory"],capture_output=True,text=True,timeout=30)
        data={"query_returncode":result.returncode,"stderr":result.stderr}
        if result.stdout:
            data["result"]=json.loads(result.stdout)
    except (OSError,ValueError,subprocess.SubprocessError) as exc:
        data={"error":repr(exc),"unix_ns":time.time_ns()}
    write(path,data)

def launch(args, *, client_command=None, guard_tick=None, sequence=None):
    root=args.output
    root.mkdir(parents=True)
    (root/"trace-control").write_text("off")
    services=serving_processes(args.checkpoint)
    if len(services)!=1:
        raise RuntimeError("expected one existing service bound to checkpoint")
    service=services[0]
    known=service["config"]
    for key,value in (("--max-num-seqs","4"),("--max-num-batched-tokens","2048"),("--attention-backend","FLASHINFER")):
        if known.get(key)!=value:
            raise ValueError(f"production config drift for {key}: {known}")
    identity={f:hashlib.sha256((args.checkpoint/f).read_bytes()).hexdigest() for f in
              ("config.json","tokenizer.json","tokenizer_config.json")}
    # Reuse current parser/scheduler/model flags. Record only the permitted CLI fields.
    argv=Path(f"/proc/{service['pid']}/cmdline").read_bytes().decode().rstrip("\x00").split("\x00")
    flags=argv[argv.index("serve")+2:]
    allowed={"--host","--port","--gpu-memory-utilization","--max-model-len","--served-model-name",
             "--enable-prefix-caching","--enable-chunked-prefill","--enable-auto-tool-choice",
             "--tool-call-parser","--reasoning-parser","--tensor-parallel-size","--default-chat-template-kwargs",
             "--attention-backend","--max-num-seqs","--max-num-batched-tokens","--speculative-config"}
    if any(x.startswith("--") and x not in allowed for x in flags):
        raise ValueError("deployment has previously unreviewed CLI flags; inspect before cloning")
    changed={"--host":"127.0.0.1","--port":str(args.port),"--max-model-len":str(args.max_len)}
    private=flags.copy()
    for key,value in changed.items():
        private[private.index(key)+1]=value
    private.extend(["--kv-cache-memory-bytes",str(4*1024**3),"--skip-mm-profiling",
                    "--limit-mm-per-prompt",json.dumps({"image":0,"video":0}),
                    "--profiler-config",json.dumps({"profiler":"torch","torch_profiler_dir":str(root/"profiles"),
                    "torch_profiler_with_stack":False,"torch_profiler_record_shapes":False,"ignore_frontend":True})])
    guard=args.min_free_gib*1024**3
    cmd=[str(args.python),"-m","vllm.entrypoints.cli.main","serve",str(args.checkpoint),*private]
    write(root/"plan.json",{"schema":1,"deployed":services,"deployment_flags":flags,"private_cmd":cmd,
          "model_files":identity,"checkpoint":str(args.checkpoint),"private_only":"loopback HTTP",
          "max_runtime_s":args.timeout,"min_free_gib":args.min_free_gib,"cache":"private seeded cache",
          "sequence":sequence if sequence is not None else (["warmup","unobserved","event-logged","profiled","two-row replay","chunked and reuse",
                      "stream/nonstream","bounded cancel/slow/queue with health requests"] if args.panel=="main" else
                      ["cache-prime-3840","cache-repeat-3840","checkpoint-numerics","over-limit","capacity-next"]),
          "panel":args.panel,"queue_clients":6 if args.panel=="main" else 0,"sampling_interval_s":1,"started":datetime.datetime.now(datetime.timezone.utc).isoformat()})
    lock=Path.home()/".cache/tilescope/gpu-0.lock"
    lock.parent.mkdir(exist_ok=True,parents=True)
    with locked_slot(lock) as pending:
        if memory_available()<guard+30*1024**3:
            raise RuntimeError("insufficient system memory before own process")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1",args.port))
        raise_if_requested(pending)
        if guard_tick is not None:
            guard_tick(root)
        env=private_env(args)
        resource_snapshot(root/"nvmap-before.json",snapshot,set())
        gpu_free(root/"cuda-free-before.json")
        resource_snapshot(root/"all-heaps-before.json",heaps,set())
        deadline=time.monotonic()+args.timeout
        child=client=None
        owners=set()
        states={}
        try:
            with (root/"server.stdout").open("w") as stdout,(root/"server.stderr").open("w") as stderr, \
                    (root/"resources.jsonl").open("x",buffering=1) as samples, \
                    (root/"client.stdout").open("w") as cout,(root/"client.stderr").open("w") as cerr:
                raise_if_requested(pending)
                child=subprocess.Popen(cmd,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
                while True:
                    raise_if_requested(pending)
                    now=time.time_ns()
                    procs=processes(child.pid, owners)
                    states.update(procs)
                    owners.update(procs)
                    if guard_tick is not None:
                        guard_tick(root)
                    avail=memory_available()
                    samples.write(json.dumps({"unix_ns":now,"available_bytes":avail,
                                              "processes":list(procs.values())})+"\n")
                    if avail<guard or time.monotonic()>deadline:
                        raise RuntimeError(f"private guard reached; available={avail} deadline={time.monotonic()>deadline}")
                    if child.poll() is not None:
                        raise RuntimeError(f"private service exited before client completion: {child.returncode}")
                    if client is None:
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health",timeout=.35) as response:
                                ready=response.status==200
                        except OSError:
                            ready=False
                        if ready:
                            write(root/"ready.json",{"unix_ns":time.time_ns(),"pid":child.pid})
                            resource_snapshot(root/"nvmap-ready.json",snapshot,owners)
                            resource_snapshot(root/"all-heaps-ready.json",heaps,owners)
                            gpu_free(root/"cuda-free-ready.json")
                            extract(root)
                            command = client_command(root) if client_command is not None else [
                                sys.executable,"-m","parity.trace_client","--output",str(root),
                                "--port",str(args.port),"--panel",args.panel]
                            client=subprocess.Popen(command,stdout=cout,stderr=cerr,start_new_session=True)
                    elif client.poll() is not None:
                        if client.returncode:
                            raise RuntimeError(f"client failed: {client.returncode}")
                        break
                    time.sleep(1)
        finally:
            if child:
                resource_snapshot(root/"nvmap-before-exit.json",snapshot,owners)
            try:
                stop_group(client)
            finally:
                stop_group(child)
            # Account for children which make their own process group but were authenticated from the task group.
            for delay in (0,3):
                time.sleep(delay)
                resource_snapshot(root/f"nvmap-after-{delay}s.json",snapshot,owners)
            gpu_free(root/"cuda-free-after.json")
            resource_snapshot(root/"all-heaps-after.json",heaps,owners)
            living=processes(child.pid if child else -1,owners)
            write(root/"shutdown.json",{"service":None if not child else child.returncode,
                  "client":None if not client else client.returncode,"owner_identities":list(states.values()),
                  "remaining_same_identity":[v for k,v in living.items() if k in states and
                    states[k]["start_ticks"]==v["start_ticks"]],"original_service":serving_processes(args.checkpoint),
                  "available_bytes":memory_available(),"unix_ns":time.time_ns()})
            shutil.rmtree(root/"private-cache",ignore_errors=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--checkpoint",type=Path,default=Path("/opt/models/Qwen3.8-27B-NVFP4"))
    p.add_argument("--python",type=Path,default=Path("/opt/tilescope/venv/bin/python"))
    p.add_argument("--port",type=int,default=58191)
    p.add_argument("--max-len",type=int,default=4096)
    p.add_argument("--panel",choices=("main","cache"),default="main")
    p.add_argument("--timeout",type=int,default=900)
    p.add_argument("--min-free-gib",type=int,default=12)
    a=p.parse_args(); a.output=a.output.resolve(); launch(a)

if __name__=="__main__":
    main()
