"""Serial dispatch with immutable per-attempt data and a resumable batch index."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import signal
import time
from uuid import uuid4

from .common import (ROOT, digest, read_json, seal_evidence, sha256, task_contract, verify_evidence,
                     write_json)
from .environment import inspect
from .tasks import review_run


def unique_name():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


def identities(task, environment):
    files = ["tilescope/gpu.py","tilescope/kernels.py","tilescope/tasks.py"]
    if task["adapter"].startswith("v1_"):
        files += ["round2/run.sh","round2/kernels.py","round2/probe.py","round2/monitor.py","round2/WORKLOADS.jsonl"]
    runtime = environment["runtime"]
    implementation = {"adapter":task["adapter"],"descriptions":task["implementations"],
                      "source_sha256":{f:sha256(ROOT/f) for f in files if (ROOT/f).is_file()},
                      "libraries":runtime.get("versions"),"cuda_build":runtime.get("cuda_build"),
                      "compiler":{k:v["stdout"] for k,v in environment["compiler"].items()},
                      "binary_sha256":environment.get("binary_sha256",{})}
    conditions = {"schema":2,"protocol":task["protocol"],"model":task["model"],
                  "device":environment["device"],"device_uuid":runtime.get("device_uuid"),
                  "driver":environment["driver"],"cc":runtime.get("cc"),"sms":runtime.get("sm_count"),
                  "power_policy":environment["power_mode"]["stdout"],
                  "performance_clock_policy":environment.get("clock_policy"),
                  "monitor_endpoint":environment["endpoint"]}
    return dict(task=task, implementation=implementation, conditions=conditions,
                task_hash=digest(task_contract(task)), implementation_hash=digest(implementation),
                condition_hash=digest(conditions))


def new_attempt(root, task, identity):
    path = root/"attempts"/(task["task_id"]+"-"+unique_name())
    path.mkdir(parents=True)
    write_json(path/"task.json", task)
    write_json(path/"identity.json", identity)
    write_json(path/"status.json", {"status":"not_run","reason":"queued"})
    return {"task_id":task["task_id"],"attempt":str(path.relative_to(root)),
            **{k:identity[k] for k in ("task_hash","implementation_hash","condition_hash")},
            "status":"not_run","reason":"queued"}


def set_status(root, batch, entry, status, reason):
    entry.update(status=status, reason=reason)
    write_json(root/entry["attempt"]/"status.json", {"status":status,"reason":reason})
    write_json(root/"batch.json", batch)


def execute_worker(argv, env, path, lock_fd):
    """Forward interruption to the complete worker tree before giving up the device slot."""
    with (path/"launcher.stdout").open("wb") as stdout,(path/"launcher.stderr").open("wb") as stderr:
        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True, pass_fds=(lock_fd,))
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        old_term = signal.signal(signal.SIGTERM, interrupt)
        try:
            return process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            (path/"interrupted.txt").write_text("Worker group interrupted; resume creates a new attempt.\n")
            raise
        finally:
            signal.signal(signal.SIGTERM, old_term)


def run_batch(args, catalog, selected, lock_fd):
    root = Path(args.out).resolve()
    if args.resume:
        if not (root/"batch.json").is_file():
            raise ValueError("resume requires an existing batch.json")
        batch = read_json(root/"batch.json")
        if (batch["schema"] != "tilescope.batch.v2" or batch["mode"] != args.mode
                or [e["task_id"] for e in batch["tasks"]] != [task["task_id"] for task in selected]):
            raise ValueError("resume selection or comparison mode differs from saved batch")
    else:
        root.mkdir(parents=True,exist_ok=False)
        batch = {"schema":"tilescope.batch.v2","mode":args.mode,"run_id":root.name,
                 "created_ns":time.time_ns(),"tasks":[]}
    env = inspect(args.python,args.checkpoint,args.endpoint,args.min_free_mib*1024**2,args.toolkit,selected)
    frozen = [identities(t,env) for t in selected]
    if args.resume:
        for entry, identity in zip(batch["tasks"],frozen):
            for key in ("task_hash","implementation_hash","condition_hash"):
                if entry[key] != identity[key]:
                    raise ValueError(f"resume refused: {entry['task_id']} {key} changed; start a new output directory")
            old_path = (root / entry["attempt"]).resolve()
            if old_path.parent != (root/"attempts").resolve():
                raise ValueError("resume refused: attempt path is outside the batch")
        write_json(root/"history"/(unique_name()+".json"),batch)
    else:
        batch["environment"] = env
        if args.observation:
            import shutil
            source=Path(args.observation)
            source_data=read_json(source)
            target=root/"observations"/"summary.json"
            target.parent.mkdir()
            shutil.copyfile(source,target)
            for filename in ("online-requests.json","runner.jsonl","requests.json","deployment.json",
                             "isolated-options.json","isolated-exit.json","postflight.json"):
                companion=source.parent/filename
                if companion.is_file():
                    shutil.copyfile(companion,target.parent/filename)
            batch["observation"]={"summary":source_data.get("summary","受控请求实机观测"),
                                  "evidence":str(target.relative_to(root)),
                                  "requests":source_data.get("requests",[])}
        request = {"mode": args.mode, "task_ids": [t["task_id"] for t in selected]}
        write_json(root/"selection.json", request)
        batch["selection_sha256"] = digest(request)
        write_json(root/"catalog.json",catalog)
        batch["tasks"] = [new_attempt(root,t,i) for t,i in zip(selected,frozen)]
        write_json(root/"batch.json",batch)
    write_json(root/"preflight"/(unique_name()+".json"),env)
    for ix,(task,identity) in enumerate(zip(selected,frozen)):
        entry=batch["tasks"][ix]
        if args.resume and entry["status"]=="passed":
            path=root/entry["attempt"]
            # Detect divergence between the indexed experiment and its on-disk evidence.
            if read_json(path/"identity.json")!=identity or read_json(path/"task.json")!=task:
                raise ValueError("completed attempt snapshot differs from selected task")
            verify_evidence(path)
            review_run(task,path/"native")
            continue
        if args.resume:
            entry=batch["tasks"][ix]=new_attempt(root,task,identity)
            write_json(root/"batch.json",batch)
        path=root/entry["attempt"]
        if env["blocked"]:
            set_status(root,batch,entry,"environment_blocked"," | ".join(env["blocked"]))
            continue
        argv=[args.python,"-m","tilescope.gpu","--task-json",str(path/"task.json"),"--out",str(path)]
        child_env=dict(os.environ, CHECKPOINT=str(args.checkpoint), PROBE_PYTHON=args.python,
                       TILESCOPE_ENDPOINT=args.endpoint, CUDA_HOME=args.toolkit,
                       PATH=f"{args.toolkit}/bin:"+os.environ.get("PATH",""),
                       PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get("PYTHONPATH",""))
        write_json(path/"command.json",{"argv":argv,"cwd":str(ROOT),"checkpoint":str(args.checkpoint),
                   "endpoint":args.endpoint,"toolkit":args.toolkit,"python":args.python,
                   "start_ns":time.time_ns()})
        set_status(root,batch,entry,"not_run","worker starting; incomplete until receipt is validated")
        code=execute_worker(argv,child_env,path,lock_fd)
        (path/"launcher.exit").write_text(str(code)+"\n")
        receipt=read_json(path/"status.json")
        status=receipt["status"]
        reason=receipt.get("reason","")
        if code != 0 and status in ("not_run","passed"):
            status,reason="execution_failed",f"worker exit {code}; inspect launcher.stderr"
        if status=="passed":
            try:
                review_run(task,path/"native")
                seal_evidence(path)
            except (KeyError,ValueError,TypeError,OSError,StopIteration,OverflowError) as exc:
                status,reason="measurement_invalid",str(exc)
        set_status(root,batch,entry,status,reason)
        print(f"{task['task_id']}  {status}",file=sys.stderr,flush=True)
    from .review import review_batch
    return review_batch(root)
