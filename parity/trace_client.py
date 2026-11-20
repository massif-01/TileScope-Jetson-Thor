"""Bounded existing HTTP contracts against the owned loopback-only trace service."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import random
import threading
import time
import uuid
from pathlib import Path
from urllib.request import urlopen, Request


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n")

def control(root, value):
    temp = root/"control.new"
    temp.write_text(value)
    temp.replace(root/"trace-control")

def run(root, port, panel="main"):
    base = f"http://127.0.0.1:{port}"
    model = json.load(urlopen(base+"/v1/models", timeout=10))["data"][0]["id"]
    root_id = "ts-"+uuid.uuid4().hex[:12]
    directory = root/"http"
    directory.mkdir()
    def call(label, n=64, out=16, *, seed=11, stream=True, behavior=None, text=None, usage=True):
        rid = f"{root_id}-{label}"
        rng = random.Random(seed)
        ids = [rng.randrange(100, 6000) for _ in range(n)]
        if text is not None:
            req=Request(base+"/tokenize",data=json.dumps({"model":model,"prompt":text}).encode(),headers={"Content-Type":"application/json"})
            ids=json.load(urlopen(req,timeout=10))["tokens"]
        body = {"model":model, "request_id":rid, "prompt":ids, "min_tokens":out,
                "max_tokens":out, "temperature":0, "seed":123, "ignore_eos":True,
                "stream":stream, "return_token_ids":True}
        if stream:
            body["stream_options"]={"include_usage":usage}
        record={"run_id":rid, "label":label, "request":body,
                "prompt_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest(),
                "wall_start_ns":time.time_ns(), "start_ns":time.monotonic_ns(),
                "chunks":[], "done":False, "behavior":behavior, "status":"running","source_text":text}
        conn=http.client.HTTPConnection("127.0.0.1",port,timeout=100)
        response=None
        try:
            conn.request("POST","/v1/completions",body=json.dumps(body),
                         headers={"Content-Type":"application/json"})
            if behavior=="cancel_prefill":
                # Wait for this identity's first scheduler row; disconnect without reading output.
                limit=time.monotonic()+10
                while time.monotonic()<limit:
                    found=False
                    for path in root.glob("engine-*.jsonl"):
                        with path.open("rb") as f:
                            f.seek(max(0,path.stat().st_size-200000))
                            found=any(b'"event":"scheduled"' in line and rid.encode() in line for line in f)
                        if found:
                            break
                    if found:
                        break
                    time.sleep(.01)
                record.update(status="disconnected", saw_schedule_before_close=found)
                return
            response=conn.getresponse()
            record["http_status"]=response.status
            if response.status!=200:
                error_body=response.read().decode("utf-8",errors="replace")
                if label.startswith("queue-") and response.status in (429,503):
                    record.update(status="rejected",error_response=error_body)
                    return
                raise RuntimeError(f"HTTP {response.status}: {error_body[:400]}")
            if not stream:
                record["response"]=json.load(response)
                record["status"]="ok"
                return
            visible=0
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                when=time.monotonic_ns()
                data=raw[5:].strip()
                if data==b"[DONE]":
                    record["done"]=True
                    break
                event=json.loads(data.decode("utf-8",errors="strict"))
                record["chunks"].append({"mono_ns":when,"payload":event})
                if any(c.get("text") for c in event["choices"]):
                    visible+=1
                if behavior=="cancel_decode" and visible>=2:
                    record["status"]="disconnected"
                    return
                if behavior=="slow" and visible:
                    time.sleep(.15 if visible==1 else .03)
            if not record["done"]:
                raise ValueError("stream ended before [DONE]")
            record["status"]="ok"
        except Exception as error:
            record.update(status="failed",error=repr(error))
            raise
        finally:
            if record["status"] == "disconnected":
                record["client_close_start_ns"] = time.monotonic_ns()
                if response is not None:
                    response.close()
            conn.close()
            record["end_ns"]=time.monotonic_ns()
            record["wall_end_ns"]=time.time_ns()
            write(directory/(label+".json"),record)
    def metrics():
        lines=urlopen(base+"/metrics",timeout=5).read().decode().splitlines()
        return {l.partition("{")[0]:float(l.split()[-1]) for l in lines if
                l.startswith(("vllm:num_requests_running{","vllm:num_requests_waiting{"))}
    def recover(label):
        checks=[]
        for _ in range(100):
            vals=metrics()
            checks.append({"ns":time.monotonic_ns(),**vals})
            if len(vals)==2 and not any(vals.values()):
                break
            time.sleep(.1)
        http=urlopen(base+"/health",timeout=5).status
        write(directory/(label+"-recovery.json"),{"health":http,"snapshots":checks})
        call(label+"-next",n=32,out=2,seed=170)
    def profiler(route):
        with urlopen(Request(base+"/"+route,data=b""),timeout=90) as resp:
            return resp.status

    if panel=="cache":
        read_plan=json.loads((root/"plan.json").read_text())
        control(root,"events")
        call("cache-prime-3840",n=3840,out=4,seed=908)
        call("cache-repeat-3840",n=3840,out=4,seed=908)
        control(root,"tensors")
        call("checkpoint-numerics",n=64,out=4,seed=71)
        control(root,"events")
        limit=int(read_plan["private_cmd"][read_plan["private_cmd"].index("--max-model-len")+1])
        over_id=root_id+"-over-limit"
        body={"model":model,"request_id":over_id,"prompt":[123]*(limit+1),
              "max_tokens":1,"temperature":0,"stream":False}
        conn=http.client.HTTPConnection("127.0.0.1",port,timeout=20)
        try:
            begin=time.monotonic_ns()
            conn.request("POST","/v1/completions",body=json.dumps(body),
                         headers={"Content-Type":"application/json"})
            response=conn.getresponse()
            write(root/"over-capacity.json",{"request":body,"limit":limit,"http_status":response.status,
                  "response":json.loads(response.read()),"duration_ns":time.monotonic_ns()-begin})
        finally:
            conn.close()
        recover("capacity")
        control(root,"off")
        return
    control(root,"off")
    call("warmup",seed=6,out=8)
    call("unobserved",seed=55,out=16)
    control(root,"events")
    call("observed",seed=55,out=16)
    control(root,"profile")
    profile_steps=[]
    try:
        profile_steps.append(["start_profile",profiler("start_profile"),time.time_ns()])
        call("profiled",seed=55,out=16)
    finally:
        profile_steps.append(["stop_profile",profiler("stop_profile"),time.time_ns()])
        write(root/"profiler-control.json",profile_steps)
    control(root,"tensors")
    call("layer-replay",seed=71,out=4)
    control(root,"events")
    call("chunk-first",n=2304,out=12,seed=90)
    call("chunk-reuse",n=2304,out=12,seed=90)
    call("regular-stream",seed=89)
    call("regular-nonstream",seed=89,stream=False)
    call("utf8-one",out=1,text="请逐字读出这行文字：中文、café、日本語。")
    call("utf8-no-usage",out=4,text="Bonjour 世界。请用中文回答。",usage=False)
    call("cancel-prefill",n=3072,out=80,seed=43,behavior="cancel_prefill")
    recover("prefill-cancel")
    call("cancel-decode",out=80,behavior="cancel_decode")
    recover("decode-cancel")
    call("slow-reader",n=192,out=32,seed=72,behavior="slow")
    recover("slow-reader")
    barrier=threading.Barrier(7)
    def queued(i):
        barrier.wait()
        call(f"queue-{i}",n=256,out=32,seed=80+i)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(queued,i) for i in range(6)]
        barrier.wait()
        with (root/"queue-metrics.jsonl").open("x",buffering=1) as f:
            while not all(task.done() for task in futures):
                f.write(json.dumps({"ns":time.monotonic_ns(),**metrics()})+"\n")
                time.sleep(.06)
        for task in futures:
            task.result()
    recover("queue")
    control(root,"off")

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--panel",choices=("main","cache"),default="main")
    args=parser.parse_args()
    run(args.output,args.port,args.panel)
