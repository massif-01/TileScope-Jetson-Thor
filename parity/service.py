"""Workload plan, request, resume and finite observation on the existing service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from tilescope.common import device_slot, digest, read_json, write_json
from tilescope.environment import DEFAULT_ENDPOINT
from .resources import Sampler
from .service_io import CORPUS, CORPUS_VERSION, identity, inputs, record, decode_events, sha256


def checked_attempt(path):
    meta=read_json(path/"receipt.json")
    body=read_json(path/"request.json")
    if meta["status"]!="passed" or body["request_id"]!=path.name or meta["execution"]!=path.name:
        raise ValueError("incomplete or mismatched attempt")
    if (digest(body)!=meta["body_hash"] or digest(body["prompt"])!=meta["input_hash"]
        or sha256(path/"stream.jsonl")!=meta["raw_hash"]):raise ValueError("request/raw identity changed")
    if "wire_body_sha256" in meta:
        import hashlib
        actual=hashlib.sha256(json.dumps(body,ensure_ascii=False).encode()).hexdigest()
        if actual!=meta["wire_body_sha256"]:raise ValueError("HTTP request byte hash mismatch")
    outputs=decode_events(path/"stream.jsonl",body)
    if outputs!=read_json(path/"output.json"):raise ValueError("stored output differs from raw stream")
    if digest([read_json(path/"before.json"),read_json(path/"after.json")])!=meta["counters_hash"]:
        raise ValueError("engine counter snapshot changed")
    return meta,outputs


def trial(key,n,ix,o,role,round_no,conc=1):
    return dict(key=key,n=n,ix=ix,o=o,role=role,round=round_no,concurrency=conc)


def plan(args):
    if args.mode=="pilot":
        return [[trial("pilot-short",128,0,16,"pilot",0)],[trial("pilot-long",8192,1,128,"pilot",0)]]
    if args.mode=="matrix" or args.mode=="small":
        waves=[]
        for k in range(args.rounds):
            if args.warmup:
                for i in range(args.warmup):
                    waves.append([trial(f"warm-{k}-{i}",128,3,8,"warmup",k)])
            for n in args.lengths:
                for o in args.outputs:
                    for i in range(args.requests):
                        waves.append([trial(f"r{k}-n{n}-o{o}-i{i}",n,i,o,"formal",k)])
        return waves
    if args.mode=="concurrent":
        return [[dict(trial(f"c{c}-n{n}-o{o}-{i}",n,i,o,"formal",0,min(c,args.requests-off)),requested_clients=c)
                 for i in range(off,min(off+c,args.requests))]
                for n in args.lengths for o in args.outputs for c in args.clients for off in range(0,args.requests,c)]
    if args.mode=="comparison":
        return [[trial(f"{role}-{k}-{i}",n,i,16,"warmup" if i==3 else role,k)]
                for k,role in enumerate("BCCB") for n,i in [(128,3),(128,0),(2048,1)]]
    if args.mode=="soak":return [[trial("soak",128,0,16,"soak",0)]]
    raise ValueError(args.mode)


def run_wave(wave, options, root, workload):
    def submit(t):
        return record(options["endpoint"],options["identity"]["model"],workload[f"{t['n']}-{t['ix']}"],
                      t["o"],root/"attempts",t["key"])
    with ThreadPoolExecutor(max_workers=len(wave)) as pool:return list(pool.map(submit,wave))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["pilot","matrix","concurrent","comparison","small","soak"])
    p.add_argument("--out",type=Path,required=True);p.add_argument("--endpoint",default=DEFAULT_ENDPOINT)
    p.add_argument("--rounds",type=int,default=3);p.add_argument("--requests",type=int,default=4)
    p.add_argument("--lengths",type=int,nargs="+");p.add_argument("--outputs",type=int,nargs="+")
    p.add_argument("--clients",type=int,nargs="+",default=[1,2],help="bounded concurrent client counts in concurrent mode")
    p.add_argument("--corpus",type=Path,help="versioned JSON {version, texts} for small, matrix or concurrent mode")
    p.add_argument("--warmup",type=int,default=1,help="warmup requests per serial round")
    p.add_argument("--duration",type=int,default=3600);p.add_argument("--interval",type=float,default=30)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    args.lengths=args.lengths or ([128,2048] if args.mode=="concurrent" else [128,2048,8192])
    args.outputs=args.outputs or ([16] if args.mode=="concurrent" else [16,128])
    corpus={"version":CORPUS_VERSION,"texts":CORPUS}
    if args.corpus:
        if args.mode not in ("small","matrix","concurrent"):p.error("custom corpus is supported for small, matrix or concurrent runs")
        corpus=read_json(args.corpus)
        if (not isinstance(corpus.get("version"),str) or not corpus["version"] or not isinstance(corpus.get("texts"),list)
            or len(corpus["texts"])<4 or not all(isinstance(t,str) and t.strip() for t in corpus["texts"])):
            p.error("provide version and at least four nonempty source texts")
    if any(len(set(v))!=len(v) for v in (args.clients,args.lengths,args.outputs)):p.error("duplicate workload condition")
    if not (1<=args.rounds<=100 and 1<=args.requests<=len(corpus["texts"]) and all(1<=c<=4 for c in args.clients)
            and all(n>0 for n in args.lengths+args.outputs)
            and args.duration>0 and args.interval>=1 and 0<=args.warmup<=20):p.error("invalid workload or pacing")
    with device_slot() as lock_fd:
        root=args.out.resolve(); ident=identity(args.endpoint); waves=plan(args)
        opts=dict(schema="tilescope.service.v1",mode=args.mode,endpoint=args.endpoint,identity=ident,
                  duration=args.duration,interval=args.interval,waves=waves)
        if args.corpus:opts["corpus_identity"]={"version":corpus["version"],"sha256":digest(corpus)}
        if any(t["n"]+t["o"]>ident["max_model_len"] for wave in waves for t in wave):raise ValueError("workload exceeds deployment capacity")
        if args.resume:
            if args.mode=="soak":raise ValueError("a continuous observation needs a new uninterrupted window")
            if read_json(root/"plan.json")!=opts:raise ValueError("service identity or plan differs from prior execution")
        else:
            root.mkdir(parents=True,exist_ok=False)
            write_json(root/"plan.json",opts)
            lengths={t["n"] for wave in waves for t in wave}
            write_json(root/"inputs.json",inputs(args.endpoint,ident["model"],lengths,corpus["texts"],corpus["version"]))
        workload=read_json(root/"inputs.json")
        completed=set()
        for path in sorted((root/"attempts").glob("*")):
            if not (path/"receipt.json").is_file():continue  # interrupted before request submission
            m=read_json(path/"receipt.json")
            if m["status"]=="passed":
                checked_attempt(path)
                if m["key"] in completed:raise ValueError("duplicate completed key")
                completed.add(m["key"])
        session=f"{time.time_ns()}-{os.getpid()}"
        log=root/"sessions"/session;log.mkdir(parents=True)
        sampler=Sampler(); stop=threading.Event(); protection=[]; last_sample={}
        def monitor():
            errors=0
            with (log/"resources.jsonl").open("w",1) as f:
                while not stop.is_set():
                    try:
                        row=sampler.sample();last_sample.update(row);errors=0
                        gpu_temp=row["temperatures_mC"].get("gpu-thermal")
                        if gpu_temp is not None and gpu_temp>=85000:protection.append("GPU temperature >= 85 C")
                        if row["mem"]["MemAvailable"]<12*1024**3:protection.append("MemAvailable <12 GiB")
                        if any(str(row["processes"].get(str(s["pid"]),{}).get("start_ticks")) != s["start_ticks"] for s in ident["servers"]):
                            protection.append("service process not readable")
                    except Exception as exc:
                        errors+=1;row={"time_ns":time.time_ns(),"error":str(exc)}
                        if errors>=3:protection.append("three resource sampling errors")
                    f.write(json.dumps(row)+"\n")
                    if protection:return
                    stop.wait(1)
        started=time.monotonic();thread=threading.Thread(target=monitor,daemon=True);thread.start()
        write_json(log/"start.json",{"wall_ns":time.time_ns(),"pid":os.getpid(),"resume":args.resume})
        bad=0
        try:
            if args.mode=="soak":
                index=0
                while time.monotonic()-started<args.duration and not protection:
                    key=f"soak-{index}";t=dict(waves[0][0],key=key,ix=0)
                    [receipt]=run_wave([t],opts,root,workload)
                    bad=bad+1 if receipt["status"]!="passed" else 0
                    if bad>=3:protection.append("three consecutive failed requests")
                    if not protection:
                        try:
                            from urllib.request import urlopen
                            with urlopen(args.endpoint+"/health",timeout=5) as h:
                                if h.status!=200:protection.append("health endpoint failed")
                        except OSError as e:protection.append("health endpoint: "+str(e))
                    index+=1
                    until=min(started+args.duration,started+index*args.interval)
                    while not protection and time.monotonic()<until:time.sleep(min(.25,until-time.monotonic()) if until>time.monotonic() else 0)
            else:
                last=None
                for wave in waves:
                    todo=[t for t in wave if t["key"] not in completed]
                    if not todo:continue
                    if len(todo)!=len(wave):raise ValueError("partial concurrent wave cannot be replayed as the same concurrency; start a new batch")
                    if protection:break
                    if last is not None and todo[0]["round"]!=last:time.sleep(2)
                    last=todo[0]["round"]
                    if args.mode=="comparison":
                        # Each measurement window starts a fresh client; service process stays online.
                        cmd=[sys.executable,"-m","parity.wave",str(root),json.dumps(todo)]
                        subprocess.run(cmd,check=True,pass_fds=(lock_fd,))
                    else:
                        records=run_wave(todo,opts,root,workload)
                        if any(r["status"]!="passed" for r in records):
                            raise ValueError("failed request; raw attempt preserved")
        finally:
            stop.set();thread.join()
            write_json(log/"end.json",{"wall_ns":time.time_ns(),"elapsed_s":time.monotonic()-started,
                                       "protection":protection,"completed_normally":sys.exc_info()[0] is None})
        if protection:return 2
    return 0

if __name__=="__main__":
    sys.exit(main())
