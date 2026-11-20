"""Collect/reconstruct a fixed suite. The atomic unit of completion is a whole wave."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from tilescope.common import device_slot, digest, read_json, sha256, write_json
from .engines import llama_cpp, openai, vllm
from .resource_window import Monitor
from .transport import capture, decoded, fetch, request_bytes, wire_blocks
from .workload import build, native, plan_waves, tokenize_body


def _files(path):
    return {p.name:sha256(p) for p in sorted(path.iterdir()) if p.is_file() and p.name!="hashes.json"}


def producer():
    """The archive can be moved off-device; source hashes remain valid outside a checkout."""
    base=Path(__file__).parent
    sources=["collect.py","resource_window.py","workload.py","transport.py","targets.py","identity.py",
             "engines/openai.py","engines/llama_cpp.py","engines/vllm.py"]
    return {name:sha256(base/name) for name in sources}


def tokenize(target,suite):
    endpoint=target["deployment"]["endpoint"]
    records=[]
    for text in suite["corpus"]["texts"]:
        body=tokenize_body(target,text)
        raw=fetch(endpoint,"/tokenize",body)
        ids=openai.valid_ids(json.loads(raw)["tokens"])
        if not ids:
            raise ValueError("tokenizer returned no source IDs")
        records.append({"request":body,"response_utf8":raw.decode(),"ids":ids})
    return records


def _request_result(path,plan,spec,meta):
    target=plan["target"]
    route,body=build(plan,spec,path.name)
    raw=(path/"request.json").read_bytes()
    if (meta["execution_id"]!=path.name or meta["task_key"]!=spec["key"] or meta["route"]!=route or
            request_bytes(body)!=raw or meta["raw_request_sha256"]!=sha256(path/"request.json") or
            meta["target_fingerprint"]!=target["target_fingerprint"] or meta["suite_sha256"]!=digest(plan["suite"])):
        raise ValueError("request/plan/target identity mismatch")
    if meta["closed_ns"]<=meta["send_ns"]:
        raise ValueError("invalid request clock interval")
    if spec["kind"]=="rejection":
        if meta["http"] not in (400,413,422):
            raise ValueError("invalid prompt did not produce expected client error")
        raw=b"".join(block for _,block in wire_blocks(path/"wire.jsonl",meta["send_ns"],meta["closed_ns"]))
        error=json.loads(raw)["error"]
        if not error or not isinstance(error,(dict,str)):
            raise ValueError("engine rejection missing error details")
        return dict(**spec,status="expected_rejection",http=meta["http"],execution_id=path.name,error=error)
    if meta["http"]!=200:
        raise ValueError(f"engine returned HTTP {meta['http']}")
    events=decoded(path/"wire.jsonl",meta,body["stream"])
    if native(target) and route=="/completion":
        row=llama_cpp.parse(events,body,target["deployment"]["served_model"])
    else:
        row=openai.parse(events,body,strong=not native(target))
        if native(target):
            row["engine"]={"attribution":"unavailable","reason":"OpenAI chat compatibility route"}
        else:
            a,b=(vllm.snapshot(path/name,target) for name in ("before.prom","after.prom"))
            row["engine"]=vllm.phases(a,b,row["input_tokens"],row["output_tokens"],spec["concurrency"])
    if spec["n"] is not None and row["input_tokens"]!=spec["n"]:
        raise ValueError("actual prompt count differs from requested fixed input")
    if row["first_event_ns"] is None or row["terminal_ns"] is None:
        raise ValueError("missing generation or terminal event")
    sent,closed=meta["send_ns"],meta["closed_ns"]
    total=(closed-sent)/1e9
    first=row["first_visible_ns"]
    contract={"same-content":"same-content","fixed":"same-token-budget","concurrency":"same-token-budget"}.get(spec["kind"])
    row.update(spec,execution_id=path.name,attempt=meta["attempt"],target_fingerprint=target["target_fingerprint"],
               send_ns=sent,closed_ns=closed,total_s=total,ttfv_ms=(first-sent)/1e6 if first is not None else None,
               input_sha256=digest(body.get("prompt",body.get("messages"))),comparison_contract=contract,
               request_output_tokens_per_s=row["output_tokens"]/total)
    return row


def attempt(root,plan,spec,index,session):
    target=plan["target"]
    endpoint=target["deployment"]["endpoint"]
    execution="tilescope-"+uuid4().hex
    path=root/"attempts"/execution
    path.mkdir(parents=True)
    route,body=build(plan,spec,execution)
    data=request_bytes(body)
    (path/"request.json").write_bytes(data)
    meta=dict(schema="tilescope.probe.request.v1",execution_id=execution,task_key=spec["key"],attempt=index,
              status="partial",role=spec["role"],session=session,route=route,clock_domain="host_boot_monotonic_ns",
              engine_adapter=target["engine"]["adapter"],target_fingerprint=target["target_fingerprint"],
              suite_sha256=digest(plan["suite"]),raw_request_sha256=hashlib.sha256(data).hexdigest())
    write_json(path/"receipt.json",meta)
    try:
        if not native(target):
            (path/"before.prom").write_bytes(fetch(endpoint,"/metrics"))
        capture(endpoint,route,data,path/"wire.jsonl",meta)
        if not native(target):
            (path/"after.prom").write_bytes(fetch(endpoint,"/metrics"))
        row=_request_result(path,plan,spec,meta)
        write_json(path/"result.json",row)
        meta["status"]="passed"
    except Exception as exc:
        meta["status"]="failed"
        meta["error"]=f"{type(exc).__name__}: {exc}"
    write_json(path/"receipt.json",meta)
    write_json(path/"hashes.json",_files(path))
    if meta["status"]!="passed":
        raise ValueError(f"{spec['key']} failed (retained {path}): {meta['error']}")
    return execution


def _read_plan(root):
    plan=read_json(root/"plan.json")
    if (root/"plan.sha256").read_text().strip()!=sha256(root/"plan.json"):
        raise ValueError("frozen plan changed")
    if plan["schema"]!="tilescope.probe.run.v1" or plan_waves(plan["suite"])!=plan["waves"]:
        raise ValueError("frozen request matrix mismatch")
    t=plan["target"]
    if t["target_fingerprint"]!=digest([t["profile_sha256"],t["identity"]]):
        raise ValueError("archived target is not bound to this deployment identity")
    static={k:v for k,v in t.items() if k not in ("target_fingerprint","profile_sha256","identity")}
    if digest(static)!=t["profile_sha256"]:
        raise ValueError("archived model/engine/deployment identity changed")
    ident,deploy=t["identity"],t["deployment"]
    if (ident["served_model"]!=deploy["served_model"] or ident["checkpoint"]!=deploy["checkpoint"]
            or ident["device_uuid"]!=deploy["host_identity"]
            or any(ident["artifact"]["files"].get(k)!=v for k,v in t["model"]["storage"]["sha256"].items())):
        raise ValueError("profile model, weight or device differs from bound service")
    if len(plan["tokenization"])!=len(plan["suite"]["corpus"]["texts"]) or len(plan["tokenized"])!=len(plan["tokenization"]):
        raise ValueError("missing frozen tokenizer evidence")
    for ix,record in enumerate(plan["tokenization"]):
        if (record["request"]!=tokenize_body(plan["target"],plan["suite"]["corpus"]["texts"][ix]) or
                json.loads(record["response_utf8"])["tokens"]!=plan["tokenized"][ix]):
            raise ValueError("invalid frozen tokenizer response")
    return plan


def _read_committed(root,plan):
    ids=set()
    done={}
    rows=[]
    failures=[]
    for marker in sorted((root/"waves").glob("*.json")):
        w=int(marker.stem)
        if not 0<=w<len(plan["waves"]) or w in done:
            raise ValueError("unplanned or duplicate wave")
        group=read_json(marker)
        specs=plan["waves"][w]
        if len(specs)!=len(group):
            raise ValueError("missing request in committed wave")
        wave_rows=[]
        for spec,execution in zip(specs,group):
            if execution in ids or not execution.startswith("tilescope-") or not execution[10:].isalnum():
                raise ValueError("invalid/duplicate execution")
            ids.add(execution)
            path=root/"attempts"/execution
            if _files(path)!=read_json(path/"hashes.json"):
                raise ValueError("attempt manifest mismatch: "+execution)
            meta=read_json(path/"receipt.json")
            if meta["status"]!="passed":
                raise ValueError("wave contains unsuccessful attempt")
            row=_request_result(path,plan,spec,meta)
            if row!=read_json(path/"result.json"):
                raise ValueError("decoded response differs from receipt")
            if row.get("status")=="expected_rejection":
                failures.append(row)
            else:
                wave_rows.append(row)
        if len(wave_rows)>1:
            overlap=min(r["closed_ns"] for r in wave_rows)-max(r["send_ns"] for r in wave_rows)
            if overlap<=0:
                raise ValueError("scheduled concurrent clients had no actual request overlap")
            for row in wave_rows:
                row["overlap_ms"]=overlap/1e6
        rows.extend(wave_rows)
        done[w]=group
    for path in sorted((root/"attempts").glob("*")):
        if path.name in ids:
            continue
        rec=read_json(path/"receipt.json") if (path/"receipt.json").exists() else {"status":"partial"}
        failures.append({"execution_id":path.name,"task_key":rec.get("task_key"),"status":rec.get("status"),
                         "reason":rec.get("error","uncommitted wave; retained for diagnosis")})
    return done,rows,failures


def load_run(directory):
    root=Path(directory)
    plan=_read_plan(root)
    done,rows,failures=_read_committed(root,plan)
    if len(done)!=len(plan["waves"]):
        raise ValueError("incomplete suite: missing planned waves")
    sessions=sorted((root/"sessions").glob("*"))
    if not sessions or not (sessions[-1]/"end.json").is_file() or read_json(sessions[-1]/"end.json")["status"]!="complete":
        raise ValueError("incomplete collection session")
    windows={}
    for session in sessions:
        readings=[json.loads(line) for line in (session/"resources.jsonl").read_text().splitlines()]
        valid=[r for r in readings if "error" not in r]
        if not valid:
            continue
        ident=plan["target"]["identity"]
        if any(r["pid"]!=int(ident["server_pid"]) or r["start_ticks"]!=int(ident["start_ticks"])
               or r["gpu_mC"]>=85000 or r["mem_available_bytes"]<12*(1<<30) for r in valid):
            raise ValueError("resource window or service identity invalid")
        windows[session.name]=(valid[0]["ns"],valid[-1]["ns"],(session/"end.json").is_file())
    for row in rows:
        meta=read_json(root/"attempts"/row["execution_id"]/"receipt.json")
        start,end,closed=windows[meta["session"]]
        # An externally interrupted client may exit between periodic samples. Keep
        # the observed boundary explicit when a resumed run reuses that request.
        if start>meta["send_ns"] or closed and end<meta["closed_ns"]:
            raise ValueError("request outside resource sampling window")
        row["resource_window"]={"session":meta["session"],"start_ns":start,"last_sample_ns":end,
                                "covers_close":end>=meta["closed_ns"]}
    strict="engine.controlled_serial_phases" in plan["suite"].get("required_capabilities",[])
    if strict and any(r["concurrency"]==1 and r["engine"]["attribution"]=="unavailable" for r in rows):
        raise ValueError("required serial phase attribution is missing")
    if plan["suite"]["suite_id"]=="soak":
        end=read_json(sessions[-1]/"end.json")
        if len(sessions)!=1 or end["elapsed_s"]<plan["suite"]["duration_s"]:
            raise ValueError("incomplete continuous observation")
    return plan,rows,failures


def run(target,suite,out,resume=False):
    root=Path(out)
    with device_slot():
        waves=plan_waves(suite)
        if resume:
            plan=_read_plan(root)
            if plan["target"]!=target or plan["suite"]!=suite or plan["waves"]!=waves or plan["producer"]!=producer():
                raise ValueError("resume identity, suite, protocol or frozen plan changed")
            if suite["suite_id"]=="soak":
                raise ValueError("soak needs one continuous observation")
            done,_,_=_read_committed(root,plan)
        else:
            root.mkdir(parents=True,exist_ok=False)
            records=tokenize(target,suite)
            plan={"schema":"tilescope.probe.run.v1","target":target,"suite":suite,"waves":waves,
                  "tokenization":records,"tokenized":[r["ids"] for r in records],"producer":producer(),
                  "measurement":{"clock":"host_boot_monotonic_ns","total":"client send to close after EOF",
                                 "first_visible":"arrival of complete SSE frame with nonempty content; JSON completion for nonstream",
                                 "resource_interval_s":1,"gpu_temperature":"gpu-thermal sysfs millidegrees Celsius",
                                 "gpc_hz":"/sys/class/devfreq/gpu-gpc-0/cur_freq","memory":"/proc/meminfo MemAvailable",
                                 "process":"server listener PID, start ticks and RSS from /proc/PID/stat"}}
            write_json(root/"plan.json",plan)
            (root/"plan.sha256").write_text(sha256(root/"plan.json")+"\n")
            done={}
        session=f"{time.time_ns()}-{os.getpid()}"
        log=root/"sessions"/session
        log.mkdir(parents=True)
        started=time.monotonic()
        write_json(log/"start.json",{"wall_ns":time.time_ns(),"pid":os.getpid(),"resume":resume})
        attempts={}
        for receipt in (root/"attempts").glob("*/receipt.json"):
            item=read_json(receipt)
            attempts[item["task_key"]]=max(attempts.get(item["task_key"],0),item["attempt"])
        status="failed"
        try:
            with Monitor(target,log/"resources.jsonl") as monitor:
                for ix,wave in enumerate(waves):
                    if ix in done:
                        continue
                    monitor.check()
                    if suite["suite_id"]=="soak":
                        monitor.wait_until(started+ix*suite["interval_s"])
                    with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                        jobs=[pool.submit(attempt,root,plan,s,attempts.get(s["key"],0)+1,session) for s in wave]
                        executions=[job.result() for job in jobs]
                    monitor.check()
                    write_json(root/"waves"/f"{ix:04d}.json",executions)
                if suite["suite_id"]=="soak":
                    monitor.wait_until(started+suite["duration_s"])
            status="complete"
        finally:
            write_json(log/"end.json",{"status":status,"elapsed_s":time.monotonic()-started,"wall_ns":time.time_ns()})
        load_run(root)
        return 0
