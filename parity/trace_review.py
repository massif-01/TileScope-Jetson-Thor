"""Reconstruct identity-bound scheduler, HTTP, CUDA and resource diagnostics offline."""
import argparse
import collections
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

from parity.trace_client import write

def read(p):
    return json.loads(p.read_text())

def lines(p):
    return [json.loads(x) for x in p.read_text().splitlines()]

def distribution(values):
    if not values:
        return {"count":0}
    vals=sorted(values)
    return {"count":len(vals),"min":vals[0],"median":statistics.median(vals),
            "p95_nearest_rank":vals[math.ceil(.95*len(vals))-1],"max":vals[-1]}

def kernel_category(name):
    low=name.lower()
    for group, words in [("projection/GEMM",("gemm","gemv","matmul","cutlass::kernel")),
                         ("attention",("attention","flash_attn","fmha","paged_decode")),
                         ("normalization",("layer_norm","rmsnorm","rms_norm","layernorm")),
                         ("activation",("silu","gelu")),("Mamba/GDN",("mamba","gated_delta","gdn_","conv1d")),
                         ("quantization",("quantize","quantization","dequant")),("sampling",("sampling","topk","softmax")),
                         ("elementwise fusion",("triton_poi_fused","elementwise_kernel"))]:
        if any(w in low for w in words):
            return group
    return "unclassified"

def merged_us(intervals):
    total=0
    end=-math.inf
    for start, stop in sorted(intervals):
        total+=max(0,stop-max(start,end))
        end=max(end,stop)
    return total

def launch_phase_map(trace, records, req):
    rid=req["run_id"]
    rows=[r for r in records if r.get("rid","").startswith("cmpl-"+rid+"-0-") or
          any(x.startswith("cmpl-"+rid+"-0-") for x in r.get("ids",()))]
    schedule=[r for r in rows if r["event"]=="scheduled"]
    routes=[r for r in rows if r["event"]=="dispatch"]
    if len(routes)!=len(schedule) or any(r["logical"]!=c["count"] for r,c in zip(routes,schedule)):
        raise ValueError("schedule and private-worker batch FIFO disagree")
    phases={r["step"]:("prefill" if c["computed"]<c["prompt"] else
                         "decode_target_verify" if c["drafts"] else "decode_target") for r,c in zip(routes,schedule)}
    events=trace["traceEvents"]
    base=trace["baseTimeNanoseconds"]
    marker_info=[]
    for m in events:
        if m.get("cat")!="user_annotation" or not m["name"].startswith("tilescope/"):
            continue
        tag=m["name"].removeprefix("tilescope/")
        matching=[r for r in rows if r["event"]=="worker_stage" and r["pid"]==m["pid"] and r["name"]==tag and
                  0 <= m["ts"]-(r["unix_ns"]-(r["mono_ns"]-r["start_mono_ns"])-base)/1000
                    <= (r["mono_ns"]-r["start_mono_ns"])/1000]
        if len(matching)!=1:
            continue
        step=matching[0]["step"]
        marker_info.append((m,phases[step]+"/"+tag))
    result={}
    for api in events:
        if api.get("cat") not in ("cuda_driver","cuda_runtime"):
            continue
        corr=api.get("args",{}).get("correlation")
        if corr is None:
            continue
        for m,stage in marker_info:
            if api["pid"]==m["pid"] and api["tid"]==m["tid"] and m["ts"]<=api["ts"]<m["ts"]+m["dur"]:
                result[corr]=stage
                break
    return result

def parse_profiles(root, req, records):
    files=list((root/"profiles").rglob("*.json.gz"))+list((root/"profiles").rglob("*.json"))
    if not files:
        raise ValueError("no profiler trace")
    grouped=collections.defaultdict(list)
    intervals=[]
    meta=[]
    stage_intervals=collections.defaultdict(list)
    for p in files:
        with (gzip.open(p,"rt") if p.suffix==".gz" else p.open()) as f:
            trace=json.load(f)
        base=trace.get("baseTimeNanoseconds")
        if base is None:
            raise ValueError(f"profiler trace lacks wall-time origin: {p.name}")
        begin=(req["wall_start_ns"]-base)/1000
        end=(req["wall_end_ns"]-base)/1000
        corr_stage=launch_phase_map(trace,records,req)
        categories=collections.Counter()
        markers=collections.Counter()
        for evt in trace["traceEvents"]:
            if evt.get("ph")!="X":
                continue
            categories[evt.get("cat","")]+=1
            s=evt["ts"]; e=s+evt["dur"]
            if e<begin or s>end:
                continue
            name=evt["name"]
            if name.startswith("tilescope/"):
                markers[name]+=1
            if evt.get("cat","").lower()!="kernel":
                continue
            clipped=(max(s,begin),min(e,end))
            intervals.append(clipped)
            stage_intervals[corr_stage.get(evt.get("args",{}).get("correlation"),"unassigned")].append(clipped)
            grouped[name].append(clipped[1]-clipped[0])
        meta.append({"file":str(p.relative_to(root)),"base_ns":base,"relative_http_window_us":[begin,end],
                     "categories":dict(categories),"request_markers":dict(markers)})
    if not intervals:
        raise ValueError("no GPU kernel intervals intersect the profiled HTTP request")
    total=sum(sum(a) for a in grouped.values())
    wall=(req["wall_end_ns"]-req["wall_start_ns"])/1e3
    union=merged_us(intervals)
    classes=collections.defaultdict(lambda:[0,0])
    gpu=[]
    for name,values in sorted(grouped.items(),key=lambda i:-sum(i[1])):
        cls=kernel_category(name)
        classes[cls][0]+=len(values); classes[cls][1]+=sum(values)
        gpu.append({"name":name,"category":cls,"calls":len(values),"sum_us":sum(values)})
    return {"files":meta,"request":req["run_id"],"http_wall_us":wall,"kernel_sum_us":total,
            "any_gpu_kernel_us":union,"no_observed_gpu_kernel_us":wall-union,
            "overlap_us":total-union,"calls":len(intervals),"kernels":gpu,
            "stages_by_cupti_launch_correlation":[{"stage":stage,"calls":len(spans),
                "kernel_sum_us":sum(b-a for a,b in spans),"any_kernel_us_within_stage":merged_us(spans)}
                for stage,spans in sorted(stage_intervals.items())],
            "category_by_kernel_sum":[{"category":k,"calls":v[0],"sum_us":v[1],"share":v[1]/total}
                for k,v in sorted(classes.items(),key=lambda i:-i[1][1])]}

def consume(request, rows):
    body=request["request"]
    digest=hashlib.sha256(json.dumps(body["prompt"],separators=(",",":")).encode()).hexdigest()
    rid=request["run_id"]
    if rid!=body["request_id"] or digest!=request["prompt_sha256"]:
        raise ValueError("request identity mismatch")
    http_ms=(request["end_ns"]-request["start_ns"])/1e6
    if request["status"]=="rejected":
        if not request["label"].startswith("queue-") or request.get("http_status") not in (429,503):
            raise ValueError("unexpected rejection outside bounded queue panel")
        return {"id":rid,"status":"rejected","http_status":request["http_status"],"http_wall_ms":http_ms,
                "input_tokens":len(body["prompt"]),"output_tokens_received":0,"error_response":request["error_response"]}
    enqueues=[r for r in rows if r["event"]=="enqueue" and re.fullmatch(re.escape("cmpl-"+rid+"-0-")+r"[a-f0-9]{8}",r["rid"])]
    if len(enqueues)>1:
        raise ValueError("multiple engine IDs for one request")
    internal_id=enqueues[0]["rid"] if enqueues else None
    records=[r for r in rows if internal_id and (r.get("rid")==internal_id or internal_id in r.get("ids",[]))]
    commits=[r for r in records if r["event"]=="commit" and r["token_ids"]]
    scheduled=[r for r in records if r["event"]=="scheduled"]
    stream=body["stream"]
    payloads=[e["payload"] for e in request["chunks"]] if stream else [request.get("response",{})]
    for item in payloads:
        if item.get("choices") and item["id"]!="cmpl-"+rid:
            raise ValueError("HTTP response identity mismatch")
    http_tokens=[t for e in payloads for choice in e.get("choices",[]) for t in choice.get("token_ids") or []]
    text="".join(c.get("text","") for e in payloads for c in e.get("choices",[]))
    finish=[c["finish_reason"] for e in payloads for c in e.get("choices",[]) if c.get("finish_reason")]
    usage=next((e["usage"] for e in reversed(payloads) if e.get("usage")),None)
    if request["status"] not in ("ok","disconnected"):
        raise ValueError(f"HTTP status {request['status']}")
    if request["status"]=="ok":
        if not finish or (stream and not request["done"]):
            raise ValueError("missing completion termination")
        if len(http_tokens)!=body["max_tokens"]:
            raise ValueError("token IDs do not match forced finite workload")
        if not stream or body["stream_options"]["include_usage"]:
            if usage is None or (usage["prompt_tokens"],usage["completion_tokens"],usage["total_tokens"])!=(
                    len(body["prompt"]),len(http_tokens),len(body["prompt"])+len(http_tokens)):
                raise ValueError("missing or inconsistent usage")
        elif usage:
            raise ValueError("usage returned with stream usage disabled")
    if request["label"] not in ("warmup","unobserved"):
        if not enqueues or enqueues[0]["prompt_sha256"]!=digest:
            raise ValueError("missing matching engine prompt identity")
        if request["status"]=="ok" and http_tokens!=[t for c in commits for t in c["token_ids"]]:
            raise ValueError("engine commit IDs differ from actual HTTP token IDs")
    result={"id":rid,"internal_id":internal_id,"status":request["status"],"http_wall_ms":http_ms,
            "input_tokens":len(body["prompt"]),"output_tokens_received":len(http_tokens),"finish":finish,
            "usage":usage,
            "http_output_tokens_per_s":len(http_tokens)*1000/http_ms if request["status"]=="ok" else None,
            "http_input_plus_output_tokens_per_s":(len(body["prompt"])+len(http_tokens))*1000/http_ms
                if request["status"]=="ok" else None,"utf8_output_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "output_token_sha256":hashlib.sha256(json.dumps(http_tokens).encode()).hexdigest(),
            "sse_chunk_gaps_ms":distribution([(b["mono_ns"]-a["mono_ns"])/1e6 for a,b in
                                      zip(request["chunks"],request["chunks"][1:])]),
            "first_nonempty_chunk_ms":next(((e["mono_ns"]-request["start_ns"])/1e6 for e in request["chunks"]
                                      if any(c.get("text") for c in e["payload"]["choices"])),None)}
    if scheduled:
        result.update(queue_enqueue_to_first_schedule_ms=(scheduled[0]["mono_ns"]-enqueues[0]["mono_ns"])/1e6,
            prefill_schedules=[r["count"] for r in scheduled if r["computed"]<r["prompt"]],
            cache_stats=next((c["prefill"] for c in commits if c["prefill"]),None),
            target_dispatch=[{"logical":r["logical"],"padded":r["padded"],"mode":r["mode"]} for r in records if r["event"]=="dispatch"],
            graph_replays=dict(collections.Counter(r.get("stage") for r in records if r["event"]=="cuda_graph_replay")),
            actual_draft_shapes=[r["shape"] for r in records if r["event"]=="proposal"],
            scheduled_verify_steps=sum(r["drafts"]>0 for r in scheduled),
            scheduled_draft_tokens=sum(r["drafts"] for r in scheduled),
            target_verify_scheduled_rows=sum(r["count"] for r in scheduled if r["drafts"]),
            acceptance_counted_draft_tokens=sum(c["mtp"]["verified_draft_count"] for c in commits if c["mtp"]),
            accepted_before_stop=sum(c["mtp"]["accepted_before_stop"] for c in commits if c["mtp"]),
            verified_draft_count=sum(c["mtp"]["verified_draft_count"] for c in commits if c["mtp"]),
            external_finishes=[r["status"] for r in records if r["event"]=="external_finish"])
    stats=result.get("cache_stats")
    if stats:
        keys=("num_prompt_tokens","num_computed_tokens","num_cached_tokens",
              "num_local_cached_tokens","num_external_cached_tokens")
        if (any(type(stats[k]) is not int or stats[k]<0 for k in keys) or
                stats["num_prompt_tokens"]!=len(body["prompt"]) or
                stats["num_computed_tokens"]+stats["num_cached_tokens"]!=stats["num_prompt_tokens"] or
                stats["num_local_cached_tokens"]+stats["num_external_cached_tokens"]!=stats["num_cached_tokens"]):
            raise ValueError("invalid computed/cache-hit token accounting for actual prompt")
    if commits:
        first,last=commits[0],commits[-1]
        native_events=[event for r in commits for event in r["queue_events"]]
        q=next((timestamp for kind,timestamp in native_events if kind=="QUEUED"),None)
        scheduled_at=next((timestamp for kind,timestamp in native_events if kind=="SCHEDULED"),None)
        prefill=(first["mono_ns"]-scheduled[0]["mono_ns"])/1e9
        decode=(last["mono_ns"]-first["mono_ns"])/1e9
        rest=sum(len(c["token_ids"]) for c in commits[1:])
        result.update(first_schedule_to_first_commit_ms=prefill*1000,first_to_last_commit_ms=decode*1000,
            native_queue_to_first_schedule_ms=1000*(scheduled_at-q) if q is not None and scheduled_at is not None else None,
            prompt_tokens_per_s=(len(body["prompt"])/prefill if prefill>0 else None),
            computed_prefill_tokens_per_s=(result["cache_stats"]["num_computed_tokens"]/prefill
                                          if prefill>0 and result["cache_stats"] else None),
            subsequent_committed_tokens=rest,subsequent_tokens_per_s=rest/decode if decode>0 else None,
            batch_sizes=[len(c["token_ids"]) for c in commits],
            commit_interval_ms=distribution([(b["mono_ns"]-a["mono_ns"])/1e6 for a,b in zip(commits,commits[1:])]),
            average_subsequent_ms_per_token=decode*1000/rest if rest else None,
            host_submission_ms={name:distribution([(r["mono_ns"]-r["start_mono_ns"])/1e6 for r in records
                if r["event"]=="worker_stage" and r["name"]==name]) for name in (
                    "target_forward","mtp_proposal","sample_or_rejection")})
    return result

def classify_nvmap(during,after):
    if any("error" in doc for doc in (during,after)):
        raise ValueError("nvmap snapshot not readable")
    retained_tables=after["owned_handle_tables"]
    result={}
    for name in ("iovmm","fsi","vpr"):
        live=during["heap"][name]
        post=after["heap"][name]
        if post["own_client_rows"]:
            status="retained"
        elif live["own_client_rows"]:
            status="cleared"
        else:
            status="unknown"
        old_orphans=set(live["orphan_rows"]); new_orphans=set(post["orphan_rows"])
        same_orphans=old_orphans & new_orphans
        orphan_status=("cleared" if not new_orphans else "retained" if same_orphans else "unknown")
        result[name]={"owner_status":status,"visible_during":len(live["own_client_rows"]),
            "visible_after":len(post["own_client_rows"]),"orphan_rows_after":len(post["orphan_rows"]),
            "orphan_row_status":orphan_status,"orphan_rows_during":len(old_orphans),
            "identical_orphan_rows_at_both_samples":len(same_orphans),
            "orphan_attribution":"global, owner unknown" if post["orphan_rows"] else "no visible orphan rows"}
    overall=("retained" if retained_tables or any(v["owner_status"]=="retained" for v in result.values())
             else "cleared" if any(v["owner_status"]=="cleared" for v in result.values()) else "unknown")
    return {"owner_status":overall,"per_heap":result,"handle_tables_after":len(retained_tables)}

def cache_review(root, summary, requests, records):
    rows=summary["requests"]
    a,b=rows["cache-prime-3840"],rows["cache-repeat-3840"]
    if requests["cache-prime-3840"]["prompt_sha256"]!=requests["cache-repeat-3840"]["prompt_sha256"]:
        raise ValueError("cache probes have different prompt identities")
    for r in (a,b):
        stats=r["cache_stats"]
        if stats["num_prompt_tokens"]!=3840 or stats["num_computed_tokens"]+stats["num_cached_tokens"]!=3840:
            raise ValueError("cache hit/miss accounting does not total prompt length")
    limit=read(root/"over-capacity.json")
    if len(limit["request"]["prompt"])!=limit["limit"]+1 or limit["http_status"]!=400:
        raise ValueError("over-limit rejection not witnessed")
    after=read(root/"http/capacity-recovery.json")
    if after["health"]!=200 or rows["capacity-next"]["status"]!="ok":
        raise ValueError("no successful post-rejection recovery")
    cuda={}
    for name in ("before","ready","after"):
        cuda[name]=read(root/f"cuda-free-{name}.json")["result"]
        if cuda[name]["code"] or not (0<cuda[name]["free_bytes"]<=cuda[name]["total_bytes"]):
            raise ValueError("invalid CUDA memory query for "+name)
    heaps={name:read(root/("all-heaps-"+name+".json")) for name in ("before","ready","after")}
    sh=read(root/"shutdown.json")
    if sh["remaining_same_identity"] or not sh["original_service"]:
        raise ValueError("incomplete private process recovery")
    ready=heaps["ready"]
    nvmap_classification=classify_nvmap(ready,heaps["after"])
    page_bytes=lambda h:h["page_pool_available_pages"]*h["host_page_bytes"]
    replay=[r for r in records if r["event"]=="selected_numerical_reference"]
    snapshots=[r for r in records if r["event"]=="selected_local_replay"]
    if len(replay)!=1 or len(snapshots)!=1 or replay[0]["step"]!=snapshots[0]["step"]:
        raise ValueError("two-row numerical reference is not linked to a single actual replay")
    numeric=replay[0]["result"]
    req=requests["checkpoint-numerics"]
    if not numeric["embedding_matches_checkpoint"] or numeric["prompt_prefix_ids"]!=req["request"]["prompt"][:2]:
        raise ValueError("checkpoint embedding does not match actual request tensor")
    for label,dim,limits in (("norm",5120,(.006,.0002,.003)),("ba",96,(.009,.002,.004))):
        m=numeric[label]
        if m["shape"]!=[2,dim] or m["n"]!=2*dim or m["mismatches"] not in range(m["n"]+1):
            raise ValueError("inconsistent layer shape or violation count: "+label)
        if (m["rtol"],m["rms_atol"],m["l2_limit"])!=limits:
            raise ValueError("frozen numerical threshold mismatch")
        expected_pass=m["finite"] and m["mismatches"]==0 and m["relative_l2"]<=limits[2]
        if m["passed"] is not expected_pass:
            raise ValueError("recorded numerical status disagrees with measured error")
        if not math.isclose(m["relative_l2"],m["rmse"]/m["reference_rms"],rel_tol=1e-8,abs_tol=1e-14):
            raise ValueError("relative L2 inconsistent with reference RMS")
        if not (m["rmse"]-1e-14<=m["max_abs"]<=m["rmse"]*math.sqrt(m["n"])+1e-14 and
                -1-1e-12<=m["cosine"]<=1+1e-12):
            raise ValueError("inconsistent max_abs, RMSE or cosine")
        if m["actual_bf16_sha256"]!=snapshots[0][label]["sha256"]:
            raise ValueError("independent error metrics belong to a different actual tensor")
    return {"numerical_reference":numeric,"numerical_all_within_tolerance":all(numeric[k]["passed"] for k in ("norm","ba")),
            "positive_cache_hit_observed":b["cache_stats"]["num_cached_tokens"]>0,
            "cache_hypothesis":"a repeat may retain an aligned 1600-token page after MTP last-page pruning",
            "nvmap_classification":nvmap_classification,"first_prefill":a["cache_stats"],"repeat_prefill":b["cache_stats"],
            "first_chunk_rows":a["prefill_schedules"],"repeat_chunk_rows":b["prefill_schedules"],
            "same_output":a["output_token_sha256"]==b["output_token_sha256"],
            "limit":limit["limit"],"over_limit_http_status":limit["http_status"],"post_rejection_health":after["health"],
            "cuda_free_bytes":{k:v["free_bytes"] for k,v in cuda.items()},
            "cuda_free_before_minus_after_bytes":cuda["before"]["free_bytes"]-cuda["after"]["free_bytes"],
            "heap_orphan_counts":{phase:{k:len(h["orphan_rows"]) for k,h in row["heap"].items()}
                                  for phase,row in heaps.items()},
            "page_pool_bytes_base_pages":{k:page_bytes(v) for k,v in heaps.items()},
            "page_pool_big_pages_raw":{k:v["page_pool_available_big_pages"] for k,v in heaps.items()},
            "owner_rows_after":sum(len(h["own_client_rows"]) for h in heaps["after"]["heap"].values()),
            "owned_handle_tables_after":len(heaps["after"]["owned_handle_tables"]),
            "interpretation":"private PID identities retired; CUDA device water levels include other processes and driver pools"}

def startup(root):
    log=(root/"server.stdout").read_text().splitlines()
    resource=lines(root/"resources.jsonl")
    ready=read(root/"ready.json")
    selected=["Model loading took","torch.compile took","GPU KV cache size:",
              "Graph capturing finished","init engine (profile"]
    lines_of_interest=[{"line":i+1,"message":s} for i,s in enumerate(log) if any(x in s for x in selected)]
    return {"first_owned_process_sample_unix_ns":resource[0]["unix_ns"],"ready_unix_ns":ready["unix_ns"],
            "first_process_sample_to_http_ready_s":(ready["unix_ns"]-resource[0]["unix_ns"])/1e9,
            "min_system_available_bytes":min(r["available_bytes"] for r in resource),
            "cache_source":read(root/"cache-seed.json"),"server_stdout_events":lines_of_interest}

def analyze(root):
    errors={}
    summary={}
    def section(key,fn):
        try:
            summary[key]=fn()
        except (ValueError,KeyError,IndexError,TypeError,OSError) as e:
            errors[key]=str(e)
    records=sorted([r for p in root.glob("engine-*.jsonl") for r in lines(p)],key=lambda r:r["mono_ns"])
    requests={p.stem:read(p) for p in sorted((root/"http").glob("*.json")) if not p.stem.endswith("-recovery")}
    summary["requests"]={}
    for label,req in requests.items():
        try:
            summary["requests"][label]=consume(req,records)
        except (ValueError,KeyError,IndexError,TypeError) as e:
            errors["request:"+label]=str(e)
    section("startup",lambda:startup(root))
    if read(root/"plan.json").get("panel")=="cache":
        def standalone_recovery():
            sh=read(root/"shutdown.json")
            if sh["remaining_same_identity"]:
                raise ValueError("owned process identities still alive")
            return {"classification":classify_nvmap(read(root/"all-heaps-ready.json"),
                                                    read(root/"all-heaps-after.json")),
                    "client_exit":sh["client"],"service_exit":sh["service"],
                    "production_process_after":sh["original_service"]}
        section("recovery",standalone_recovery)
        section("extended",lambda:cache_review(root,summary,requests,records))
        summary["errors"]=errors
        return summary
    def require(ok, msg):
        if not ok:
            raise ValueError(msg)
    def stages():
        rows=summary["requests"]; a=rows["observed"]
        require(a.get("graph_replays",{}).get("target_forward",0)>0,"no actual target graph replay")
        require(a["verified_draft_count"]>0 and a["actual_draft_shapes"],"no linked MTP proposal and verify")
        require(len(rows["chunk-first"]["prefill_schedules"])>1,"long prompt not chunked in observed run")
        reuse=rows["chunk-reuse"]["cache_stats"]
        require(reuse and 0<=reuse["num_cached_tokens"]<=reuse["num_prompt_tokens"],"cache accounting absent")
        return {"graph":a["graph_replays"],"chunk_first":rows["chunk-first"]["prefill_schedules"],
                "repeat_cache":reuse,"mtp_commits":a["batch_sizes"]}
    def length_buckets():
        groups={}
        for label,entry in summary["requests"].items():
            if entry["status"]!="ok" or not entry.get("internal_id"):
                continue
            if label in ("profiled","layer-replay"):
                continue
            kind="concurrent-6" if re.fullmatch(r"queue-[0-5]",label) else (
                "slow-reader" if label=="slow-reader" else "serial-events")
            key=(kind,entry["input_tokens"],entry["output_tokens_received"])
            samples=groups.setdefault(key,{"ids":[],"commit_ms":[],"mean_subsequent_ms_per_token":[]})
            samples["ids"].append(entry["id"])
            cs=[r for r in records if r["event"]=="commit" and r["rid"]==entry["internal_id"] and r["token_ids"]]
            samples["commit_ms"].extend((b["mono_ns"]-a["mono_ns"])/1e6 for a,b in zip(cs,cs[1:]))
            metric=entry.get("average_subsequent_ms_per_token")
            if metric is not None:
                samples["mean_subsequent_ms_per_token"].append(metric)
        return [{"condition":k[0],"prompt_tokens":k[1],"received_output_tokens":k[2],"request_ids":v["ids"],
            "pooled_inter_commit_ms":distribution(v["commit_ms"]),
            "per_request_mean_subsequent_ms_per_token":distribution(v["mean_subsequent_ms_per_token"])}
            for k,v in sorted(groups.items())]
    section("engine_length_buckets",length_buckets)
    section("observed_path",stages)
    section("gpu",lambda:parse_profiles(root,requests["profiled"],records))
    def overhead():
        rows=summary["requests"]
        labels=("unobserved","observed","profiled")
        require(len({requests[k]["prompt_sha256"] for k in labels})==1,"overhead inputs differ")
        times=[rows[k]["http_wall_ms"] for k in labels]
        return {"order":labels,"samples_ms":times,"events_minus_unobserved_ms":times[1]-times[0],
                "profiler_minus_events_ms":times[2]-times[1],"sample_count_each":1,
                "same_output":len({rows[k]["output_token_sha256"] for k in labels})==1}
    section("overhead",overhead)
    def behavior():
        rows=summary["requests"]
        recover={}
        for label in ("prefill-cancel","decode-cancel","slow-reader","queue"):
            doc=read(root/"http"/(label+"-recovery.json")); last=doc["snapshots"][-1]
            require(doc["health"]==200 and last["vllm:num_requests_waiting"]==0 and
                    last["vllm:num_requests_running"]==0 and rows[label+"-next"]["status"]=="ok",
                    f"{label} did not drain and produce a new completion")
            recover[label]=True
        cancellation = {}
        for label in ("cancel-prefill","cancel-decode"):
            req = requests[label]
            rid = rows[label]["internal_id"]
            aborted = [r for r in records if r["event"] == "external_finish"
                       and rid in r["ids"] and r["status"] == "FINISHED_ABORTED"]
            require(req["status"] == "disconnected" and len(aborted) == 1,
                    "expected one request-bound FINISHED_ABORTED for "+label)
            origin = "client_close_start_ns" if "client_close_start_ns" in req else "end_ns"
            elapsed = aborted[0]["mono_ns"] - req[origin]
            # Old clients sampled end_ns just after close. The scheduler can win that race.
            require(elapsed >= 0 or origin == "end_ns", "scheduler abort precedes client close start")
            cancellation[label] = {"internal_id": rid, "origin": origin, "origin_ns": req[origin],
                                   "scheduler_aborted_ns": aborted[0]["mono_ns"],
                                   "to_scheduler_abort_ms": elapsed/1e6 if elapsed >= 0 else None}
        require(requests["cancel-prefill"].get("saw_schedule_before_close") and
                not rows["cancel-prefill"].get("batch_sizes") and rows["cancel-prefill"]["prefill_schedules"],
                "prefill cancellation not localized ahead of first commit")
        require(sum(rows["cancel-decode"].get("batch_sizes",[]))>1,
                "decode cancellation not localized after actual output")
        require(rows["slow-reader"]["status"]=="ok" and requests["slow-reader"]["behavior"]=="slow",
                "missing normal slow-reader completion")
        q=lines(root/"queue-metrics.jsonl")
        require(len([k for k in rows if k.startswith("queue-") and k[6:].isdigit()])==6,"missing queue clients")
        require(rows["utf8-one"]["output_tokens_received"]==1 and rows["utf8-no-usage"]["usage"] is None,"UTF8/usage boundary")
        queue_rows=[rows[f"queue-{i}"] for i in range(6)]
        return {"recoveries":recover,"cancellation":cancellation,"queue_outcomes":dict(collections.Counter(r["status"] for r in queue_rows)),
                "queue_observed":any(r["vllm:num_requests_waiting"]>0 for r in q),
                "queue_peak":max(r["vllm:num_requests_waiting"] for r in q),
                "running_peak":max(r["vllm:num_requests_running"] for r in q),
                "stream_nonstream_same_output":rows["regular-stream"]["output_token_sha256"]==rows["regular-nonstream"]["output_token_sha256"]}
    section("behavior",behavior)
    def layers():
        evidence=[r for r in records if r["event"]=="selected_local_replay"]
        require(len(evidence)==1,"missing or duplicate selected replay")
        r=evidence[0]
        require(r["norm"]["shape"]==[2,5120] and r["ba"]["shape"]==[2,96],"unexpected selected replay shapes")
        return r
    section("selected_layers",layers)
    def resources():
        pre,during,end=[read(root/p) for p in ("nvmap-before.json","nvmap-ready.json","nvmap-after-3s.json")]
        sh=read(root/"shutdown.json")
        require(not sh["remaining_same_identity"] and sh["original_service"],"private cleanup incomplete")
        owner_classification=classify_nvmap(read(root/"nvmap-extended/ready.json"),
                                            read(root/"nvmap-extended/exited.json"))
        return {"classification":owner_classification,"nvmap_ready":during["own_clients"],"nvmap_after":end["own_clients"],
                "orphan_rows_before_after":[len(pre["orphan_rows"]),len(end["orphan_rows"])],
                "page_pool_before_after":[pre["page_pool_raw"],end["page_pool_raw"]],
                "resident_owner_after":sh["remaining_same_identity"],"original_services":sh["original_service"],
                "scope":"nvmap owner rows; CUDA allocator and global driver pools have different ownership"}
    section("recovery",resources)
    summary["errors"]=errors
    return summary

def compare_layers(main,cache):
    left=read(main/"http/layer-replay.json");right=read(cache/"http/checkpoint-numerics.json")
    if (left["request"]["prompt"]!=right["request"]["prompt"] or
            read(main/"plan.json")["model_files"]!=read(cache/"plan.json")["model_files"]):
        raise ValueError("layer runs differ in actual prompt IDs or checkpoint/tokenizer identity")
    def selected(root):
        events=[r for p in root.glob("engine-*.jsonl") for r in lines(p) if r["event"]=="selected_local_replay"]
        if len(events)!=1:
            raise ValueError("expected a single selected layer replay per run")
        return events[0]
    a,b=selected(main),selected(cache)
    equal={key:a[key]==b[key] for key in ("input","norm","ba")}
    if not all(equal.values()):
        raise ValueError("separately replayed layer bytes or statistics differ")
    return {"same_actual_prompt_ids":True,"distinct_request_ids":left["run_id"]!=right["run_id"],
            "main_request":left["run_id"],"numerical_request":right["run_id"],
            "matched_actual_bf16_sha256":{key:a[key]["sha256"] for key in equal}}

def reject_commit_loss(root):
    request=read(root/"http/observed.json")
    rows=sorted([r for path in root.glob("engine-*.jsonl") for r in lines(path)],key=lambda r:r["mono_ns"])
    original=consume(request,rows)
    terminal=next(r for r in reversed(rows) if r["event"]=="commit" and r.get("finish") and
                  r["rid"]==original["internal_id"] and r["token_ids"])
    try:
        consume(request,[r for r in rows if r is not terminal])
    except ValueError as e:
        if str(e)!="engine commit IDs differ from actual HTTP token IDs":
            raise
        return {"rejected":True,"request":request["run_id"],"original_tokens":original["output_tokens_received"],
                "removed_committed_tokens":len(terminal["token_ids"]),"reason":str(e)}
    raise ValueError("terminal commit payload removed but still accepted")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--check-commit-rejection",action="store_true")
    parser.add_argument("--compare-main-layers",type=Path,help="check identical real layer operands in this cache run and an earlier main run")
    args=parser.parse_args()
    if args.compare_main_layers:
        details=compare_layers(args.compare_main_layers,args.directory)
        print(json.dumps(details,ensure_ascii=False))
        return
    if args.check_commit_rejection:
        decision=reject_commit_loss(args.directory)
        print(json.dumps(decision,ensure_ascii=False))
        return
    answer=analyze(args.directory)
    if args.output:
        write(args.output,answer)
    print(json.dumps({"requests":len(answer["requests"]),"sections":list(answer),"errors":answer["errors"]},ensure_ascii=False))
    raise SystemExit(bool(answer["errors"]))
if __name__=="__main__":
    main()
