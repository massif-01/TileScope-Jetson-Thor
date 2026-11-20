"""Record the actual client response and sampled service state for controlled prompts."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import platform
import subprocess
import time
from urllib.request import Request, urlopen
from uuid import uuid4

from observation.collect import serving_processes
from tilescope.common import digest, sha256, write_json
from tilescope.environment import DEFAULT_MODEL

CORPUS = [
    "A field engineer logs a voltage reading and its unit. Explain how the next engineer can repeat the observation. ",
    "一位值班工程师记录了设备温度、采样时间与传感器位置。请说明如何核对这项记录。 ",
    "Une équipe relève la température et la fréquence toutes les secondes. Décris comment vérifier la mesure. ",
    "A lab runs two numerical implementations on identical input. Describe how to check output and variation. ",
]
CORPUS_VERSION = "tilescope.parity.public-text.v1"
PHASES = ["queue", "prefill", "decode", "inference"]
STATS = {f"request_{phase}_time_seconds_{suffix}" for phase in PHASES for suffix in ("count","sum")}
STATS.update(["num_requests_running","num_requests_waiting","request_success_total","prompt_tokens_total",
              "generation_tokens_total","prefix_cache_hits_total","prefix_cache_queries_total",
              "request_prefill_kv_computed_tokens_sum","request_prefill_kv_computed_tokens_count",
              "spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"])
METRIC = re.compile(r'^vllm:([\w]+)(?:\{([^}]*)\})?\s+([0-9eE.+-]+)$')


def api(endpoint, path, body=None):
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    return Request(endpoint.rstrip("/")+path, data=data, headers={"Content-Type":"application/json"})


def fetch(endpoint, path, body=None):
    with urlopen(api(endpoint,path,body),timeout=20) as res:
        return json.load(res)


def metrics(endpoint):
    values = {}
    with urlopen(api(endpoint,"/metrics"),timeout=6) as response:
        for line in response:
            m = METRIC.fullmatch(line.decode().strip())
            if m and m[1] in STATS:
                values[m[1]] = values.get(m[1],0) + float(m[3])
    return {"time_ns": time.time_ns(),"values": values}


def read_text(path):
    return Path(path).read_text().strip()


def identity(endpoint, checkpoint=DEFAULT_MODEL):
    model, = fetch(endpoint,"/v1/models")["data"]
    root=Path(model["root"]).resolve()
    if root != Path(checkpoint).resolve():
        raise ValueError(f"Wrong served checkpoint: {root}")
    servers=serving_processes(root)
    if not servers:
        raise ValueError("No serving process associated with checkpoint")
    # Include process start ticks to distinguish restarts and PID reuse.
    for proc in servers:
        proc["start_ticks"]=read_text(f"/proc/{proc['pid']}/stat").rsplit(")",1)[1].split()[19]
    gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"],text=True,timeout=6).strip()
    power = subprocess.check_output(["nvpmodel","-q"],text=True,timeout=6).strip()
    clocks = Path("/sys/class/devfreq/gpu-gpc-0")
    conditions = {"gpu":gpu,"power_policy":power,"kernel":platform.release(),
                  "min_hz":int(read_text(clocks/"min_freq")),"max_hz":int(read_text(clocks/"max_freq")),
                  "governor":read_text(clocks/"governor")}
    return {"platform_conditions":conditions,"model":model["id"],"checkpoint":str(root),"max_model_len":model["max_model_len"],
            "service_version":fetch(endpoint,"/version"),"servers":servers,
            "files":{k:sha256(root/k) for k in ("tokenizer.json","tokenizer_config.json","chat_template.jinja",
                      "config.json","model.safetensors.index.json","SHA256SUMS-20260918.txt")},
            "boot_id":read_text("/proc/sys/kernel/random/boot_id"),
            "gpu_governor":read_text("/sys/class/devfreq/gpu-gpc-0/governor")}


def inputs(endpoint, model, lengths, texts=CORPUS, version=CORPUS_VERSION):
    result={}
    for ix,text in enumerate(texts):
        b={"model":model,"prompt":text,"add_special_tokens":False}
        tokenized=fetch(endpoint,"/tokenize",b)
        ids=tokenized["tokens"]
        if not ids or not all(type(i) is int for i in ids):
            raise ValueError("Tokenizer did not return IDs")
        for n in lengths:
            row=(ids*((n+len(ids)-1)//len(ids)))[:n]
            result[f"{n}-{ix}"]={"text":text,"text_tokens":ids,"length":n,"index":ix,
                                  "sha256":digest(row),"source":version}
    return result


def request_body(model, spec, output, execution):
    n=spec["length"]; ids=spec["text_tokens"]
    row=(ids*((n+len(ids)-1)//len(ids)))[:n]
    if digest(row)!=spec["sha256"]:raise ValueError("workload input identity")
    return dict(model=model,prompt=row,request_id=execution,temperature=0,seed=381127,max_tokens=output,
                stream=True,stream_options={"include_usage":True},return_token_ids=True)


def decode_events(path, body):
    """Interpret full raw data lines using their original monotonic arrival times."""
    ids=[]; texts=[]; times=[]; visible=[]; finish=[]; usages=[]; done=False; req_ids=set(); prompt_echoes=[]
    for line in path.read_text().splitlines():
        r=json.loads(line)
        raw=base64.b64decode(r["data_b64"],validate=True).decode("utf-8").strip()
        if not raw or raw.startswith(":"):continue
        if done or not raw.startswith("data:"):raise ValueError("unexpected SSE record")
        content=raw[5:].strip()
        if content=="[DONE]":
            done=True; continue
        obj=json.loads(content);req_ids.add(obj["id"])
        if obj.get("usage"):usages.append(obj["usage"])
        for choice in obj["choices"]:
            if choice["index"]!=0:raise ValueError("wrong choice index")
            if choice.get("prompt_token_ids") is not None:prompt_echoes.append(choice["prompt_token_ids"])
            text=choice["text"];chunk_ids=choice.get("token_ids") or []
            if not all(type(i) is int and i>=0 for i in chunk_ids):raise ValueError("invalid output IDs")
            if text:visible.append(r["ns"])
            if text or chunk_ids:times.append(r["ns"])
            ids.extend(chunk_ids);texts.append(text)
            if choice["finish_reason"] is not None:finish.append(choice["finish_reason"])
    if not done or len(usages)!=1 or len(finish)!=1 or finish[0] not in ("length","stop"):
        raise ValueError("incomplete SSE, usage, or termination")
    if len(req_ids)!=1 or not next(iter(req_ids)).endswith(body["request_id"]):
        raise ValueError("response execution ID mismatch")
    if prompt_echoes != [body["prompt"]]:raise ValueError("server prompt token echo differs from request")
    usage=usages[0]; n=len(body["prompt"])
    if not (usage["prompt_tokens"]==n and 0<len(ids)==usage["completion_tokens"]<=body["max_tokens"]
            and usage["total_tokens"]==n+len(ids)):raise ValueError("token/usage mismatch")
    return {"response_id":next(iter(req_ids)),"output_ids":ids,"text_sha256":hashlib.sha256("".join(texts).encode()).hexdigest(),
            "finish_reason":finish[0],"usage":usage,"chunk_times_ns":times,"first_visible_ns":visible[0] if visible else None}


def record(endpoint, model, spec, output, parent, label):
    execution="tilescope-"+uuid4().hex
    path=parent/execution;path.mkdir(parents=True)
    body=request_body(model,spec,output,execution)
    write_json(path/"request.json",body)
    meta={"execution":execution,"key":label,"input_hash":spec["sha256"],"body_hash":digest(body),
          "wire_body_sha256":hashlib.sha256(api(endpoint,"/v1/completions",body).data).hexdigest(),
          "workload":f"{spec['length']}-{spec['index']}","output_limit":output,"status":"partial"}
    write_json(path/"receipt.json",meta)
    try:
        before=metrics(endpoint);write_json(path/"before.json",before)
        meta["wall_start_ns"]=time.time_ns();meta["start_ns"]=time.monotonic_ns()
        with (path/"stream.jsonl").open("w",1) as f, urlopen(api(endpoint,"/v1/completions",body),timeout=160) as response:
            meta["http"]=response.status
            for raw in response:
                if time.monotonic_ns()-meta["start_ns"]>160_000_000_000:raise TimeoutError("request deadline")
                f.write(json.dumps({"ns":time.monotonic_ns(),"data_b64":base64.b64encode(raw).decode()})+"\n")
        meta.update(end_ns=time.monotonic_ns(),wall_end_ns=time.time_ns())
        after=metrics(endpoint);write_json(path/"after.json",after)
        summary=decode_events(path/"stream.jsonl",body)
        write_json(path/"output.json",summary)
        meta.update(status="passed",raw_hash=sha256(path/"stream.jsonl"),
                    counters_hash=digest([before,after]))
    except Exception as exc:
        meta.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc,"read"): (path/"http-error.txt").write_bytes(exc.read())
    write_json(path/"receipt.json",meta)
    return meta
