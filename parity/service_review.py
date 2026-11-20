"""Recalculate request statistics from raw SSE events and engine counters, offline."""
import argparse
from collections import defaultdict
import csv
import json
import math
from itertools import zip_longest
from pathlib import Path
import statistics

from tilescope.common import digest, read_json, write_json
from .service import checked_attempt
from .service_io import PHASES, request_body


def stats(values):
    values=sorted(v for v in values if v is not None)
    if not values:return None
    return {"count":len(values),"mean":statistics.mean(values),"min":values[0],"median":statistics.median(values),
            "p95_nearest_rank":values[math.ceil(.95*len(values))-1],"max":values[-1],
            "sample_sd":statistics.stdev(values) if len(values)>1 else None}


def engine(path, usage, concurrency):
    a,b=(read_json(path/name)["values"] for name in ("before.json","after.json"))
    delta={k:b[k]-a[k] for k in a.keys()&b.keys()}
    single=(concurrency==1 and delta.get("request_success_total")==1
            and delta.get("prompt_tokens_total")==usage["prompt_tokens"]
            and delta.get("generation_tokens_total")==usage["completion_tokens"]
            and all(a.get(k)==b.get(k)==0 for k in ("num_requests_running","num_requests_waiting")))
    if not single:return {"attribution":"unavailable: aggregate counters did not isolate one request"}
    result={"attribution":"single-request delta"}
    for phase in PHASES:
        prefix=f"request_{phase}_time_seconds"
        count=delta.get(prefix+"_count");total=delta.get(prefix+"_sum")
        result[phase+"_s"]=total if count==1 and total is not None and total>=0 else None
    for key in ("request_prefill_kv_computed_tokens_sum","prefix_cache_queries_total","prefix_cache_hits_total",
                "spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"):
        result[key]=delta.get(key)
    return result


def rows(directory):
    directory=Path(directory);plan=read_json(directory/"plan.json");work=read_json(directory/"inputs.json")
    if plan["schema"]!="tilescope.service.v1":raise ValueError("unknown service schema")
    sessions=[read_json(p) for p in sorted((directory/"sessions").glob("*/end.json"))]
    if not sessions or not sessions[-1]["completed_normally"]:raise ValueError("incomplete session")
    if plan["mode"]!="soak" and sessions[-1]["protection"]:raise ValueError("interrupted by safety condition")
    expected={t["key"]:t for w in plan["waves"] for t in w}
    if len(expected)!=sum(map(len,plan["waves"])):raise ValueError("duplicate request key in plan")
    attempts=sorted((directory/"attempts").glob("*")); result=[]; seen=set(); errors=[]
    for path in attempts:
        if not (path/"receipt.json").exists():
            errors.append({"attempt":path.name,"status":"incomplete"});continue
        rec=read_json(path/"receipt.json")
        key=rec["key"]
        if plan["mode"]=="soak" and key.startswith("soak-") and key[5:].isdigit():
            spec=dict(plan["waves"][0][0],key=key)
        else:spec=expected[key]
        if rec["status"]!="passed":
            errors.append({"key":key,"attempt":path.name,"status":rec["status"],"reason":rec.get("error"),"wall_start_ns":rec.get("wall_start_ns")})
            continue
        if key in seen:raise ValueError("duplicate completed request")
        seen.add(key)
        meta,out=checked_attempt(path)
        ref=work[f"{spec['n']}-{spec['ix']}"]
        if request_body(plan["identity"]["model"],ref,spec["o"],path.name)!=read_json(path/"request.json"):
            raise ValueError("recorded request does not match workload/plan")
        chunk=out["chunk_times_ns"]
        if not (meta["start_ns"]<=min(chunk)<=max(chunk)<=meta["end_ns"] and chunk==sorted(chunk)):
            raise ValueError("invalid monotonic time order")
        usage=out["usage"]; phases=engine(path,usage,spec["concurrency"])
        count=usage["completion_tokens"]; total=(meta["end_ns"]-meta["start_ns"])/1e9
        pre=phases.get("prefill_s"); dec=phases.get("decode_s");computed=phases.get("request_prefill_kv_computed_tokens_sum")
        first=out["first_visible_ns"]
        result.append(dict(**spec,execution=path.name,input_sha256=ref["sha256"],output_sha256=digest(out["output_ids"]),
                           text_sha256=out["text_sha256"],usage=usage,finish=out["finish_reason"],total_s=total,
                           output_tokens=count,ttfv_ms=(first-meta["start_ns"])/1e6 if first is not None else None,
                           chunk_gaps_ms=[(b-a)/1e6 for a,b in zip(chunk,chunk[1:])],
                           client_tpot_ms=(meta["end_ns"]-first)/1e6/(count-1) if first and count>1 else None,
                           engine=phases,computed_prefill_tokens_per_s=computed/pre if pre and computed is not None else None,
                           decode_tokens_per_s=(count-1)/dec if dec and count>1 else None,
                           request_output_tokens_per_s=count/total,wall_start_ns=meta["wall_start_ns"],wall_end_ns=meta["wall_end_ns"]))
    if plan["mode"]!="soak" and seen!=set(expected):
        raise ValueError("missing completed requested samples: "+str(sorted(set(expected)-seen)))
    if plan["mode"]=="soak" and not sessions[-1]["protection"] and (
            sessions[-1]["elapsed_s"]<plan["duration"] or not result):raise ValueError("incomplete observation window")
    return plan,sorted(result,key=lambda r:r["wall_start_ns"]),errors,sessions[-1]


def traffic_windows(directory,plan,values):
    by_key={r["key"]:r for r in values}
    if plan["mode"]=="soak":
        waves=[[r] for r in values]
    else:
        waves=[[by_key[t["key"]] for t in wave] for wave in plan["waves"]]
    summary=defaultdict(int)
    windows=[]
    for wave in waves:
        before=min((read_json(Path(directory)/"attempts"/r["execution"]/"before.json") for r in wave),key=lambda s:s["time_ns"])
        after=max((read_json(Path(directory)/"attempts"/r["execution"]/"after.json") for r in wave),key=lambda s:s["time_ns"])
        begin=before["values"];end=after["values"]
        keys=("request_success_total","prompt_tokens_total","generation_tokens_total")
        delta={k:end[k]-begin[k] for k in keys if k in end and k in begin}
        expected=dict(zip(keys,(len(wave),sum(r["n"] for r in wave),sum(r["output_tokens"] for r in wave))))
        if len(delta)!=3:state="insufficient_metrics"
        elif delta!=expected or any(s.get(k)!=0 for s in (begin,end) for k in ("num_requests_running","num_requests_waiting")):
            state="mixed_or_unresolved"
        else:state="counters_match_owned_requests"
        summary[state]+=1
        windows.append({"executions":[r["execution"] for r in wave],"status":state,"expected":expected,"measured":delta})
    return {"counts":dict(summary),"windows":windows,"scope":"vLLM request counters; other processes on the shared GPU remain a possible confounder"}


def interval_results(directory,values,errors):
    starts=[read_json(path)["wall_ns"] for path in (Path(directory)/"sessions").glob("*/start.json")]
    start=min(starts);buckets=defaultdict(lambda:{"good":[],"bad":0})
    for row in values:buckets[(row["wall_start_ns"]-start)//300_000_000_000]["good"].append(row)
    for row in errors:
        if row.get("wall_start_ns") is not None:buckets[(row["wall_start_ns"]-start)//300_000_000_000]["bad"]+=1
    return [{"index":idx,"completed":len(v["good"]),"unsuccessful_attempts":v["bad"],
             "error_fraction":v["bad"]/(len(v["good"])+v["bad"]),
             "total_s":stats([r["total_s"] for r in v["good"]]),"ttfv_ms":stats([r["ttfv_ms"] for r in v["good"]])}
            for idx,v in sorted(buckets.items())]


def review(directory):
    plan,values,errors,session=rows(directory)
    groups=defaultdict(list)
    for r in values:
        if r["role"] not in ("warmup","pilot"):groups[(r["n"],r["o"],r["concurrency"],r.get("requested_clients",r["concurrency"]),r["role"])].append(r)
    summary=[]
    for (n,o,c,requested,role),group in groups.items():
        items={"n":n,"o":o,"concurrency":c,"requested_clients":requested,"role":role}
        for key in ("ttfv_ms","total_s","client_tpot_ms","output_tokens","computed_prefill_tokens_per_s",
                    "decode_tokens_per_s","request_output_tokens_per_s"):
            items[key]=stats([r[key] for r in group])
        items["chunk_gap_ms"]=stats([v for r in group for v in r["chunk_gaps_ms"]])
        cache_bins=defaultdict(list)
        for row in group:
            ev=row["engine"]
            cache_bins[(ev.get("prefix_cache_hits_total"),ev.get("request_prefill_kv_computed_tokens_sum"))].append(row)
        items["cache_strata"]=[{"prefix_hit_tokens":key[0],"computed_kv_tokens":key[1],"requests":len(vals),
                                "ttfv_ms":stats([r["ttfv_ms"] for r in vals]),"total_s":stats([r["total_s"] for r in vals])}
                               for key,vals in cache_bins.items()]
        items["engine_phase_s"]={k:stats([r["engine"].get(k+"_s") for r in group]) for k in PHASES}
        items["rounds"]=sorted({r["round"] for r in group})
        items["by_round"]=[]
        for rnd in items["rounds"]:
            part=[r for r in group if r["round"]==rnd]
            items["by_round"].append({"round":rnd,"count":len(part),"total_s":stats([r["total_s"] for r in part]),
                "ttfv_ms":stats([r["ttfv_ms"] for r in part]),"decode_tokens_per_s":stats([r["decode_tokens_per_s"] for r in part]),
                "computed_prefill_tokens_per_s":stats([r["computed_prefill_tokens_per_s"] for r in part])})
        repeated=defaultdict(set)
        for r in group:repeated[r["input_sha256"]].add((r["output_sha256"],r["finish"]))
        items["different_repeat_outputs"]=sum(len(v)>1 for v in repeated.values())
        items["changed_inputs"]=[]
        for input_hash,variants in repeated.items():
            if len(variants)<2:continue
            matches=[r for r in group if r["input_sha256"]==input_hash]
            tokens=[read_json(Path(directory)/"attempts"/r["execution"]/"output.json")["output_ids"] for r in matches]
            first=next((i+1 for i,col in enumerate(zip_longest(*tokens)) if len(set(col))>1),None)
            items["changed_inputs"].append({"input_sha256":input_hash,"first_differing_output_id_position":first,
                "executions":[{"id":r["execution"],"round":r["round"],"output_sha256":r["output_sha256"],
                    "finish":r["finish"],"tokens":r["output_tokens"],"prefix_hit_tokens":r["engine"].get("prefix_cache_hits_total")}
                    for r in matches]})
        summary.append(items)
    return {"schema":"tilescope.service.review.v1","mode":plan["mode"],"rows":values,
            "groups":summary,"unsuccessful_attempts":errors,"window":session,
            "traffic":traffic_windows(directory,plan,values),"five_minute_request_bins":interval_results(directory,values,errors)}


def compare(b,c):
    """B/C may name different service builds. All non-treatment identities must match."""
    for key in ("input_sha256","output_sha256","text_sha256","finish","n","o","concurrency"):
        if len({r[key] for r in b+c})!=1:return {"status":"incompatible","reason":key}
    cache={r["engine"].get("prefix_cache_hits_total") for r in b+c}
    if None in cache or len(cache)!=1:return {"status":"incompatible","reason":"cache state unavailable or differs"}
    if len({digest(r["usage"]) for r in b+c})!=1:return {"status":"incompatible","reason":"usage"}
    if len({r["execution"] for r in b+c})!=len(b+c):return {"status":"incompatible","reason":"reused execution"}
    comparisons={}
    for k in ("total_s","ttfv_ms","prefill_s","decode_s","queue_s"):
        field=lambda r:r[k] if k in r else r["engine"].get(k)
        before=[field(r) for r in b];after=[field(r) for r in c]
        if not all(x is not None and x>0 for x in before+after):continue
        change=100*(statistics.mean(after)/statistics.mean(before)-1)
        aligned=len(b)==len(c)>=2 and all((y/x-1)*change>0 for x,y in zip(before,after))
        comparisons[k]={"B":before,"C":after,"change_pct":change,
                        "observation":"same direction, >=5% observed" if aligned and abs(change)>=5 else "no resolved shift",
                        "stable_version_effect":"undetermined"}
    return {"status":"compatible","metrics":comparisons}


def deployment_conditions(plan):
    from urllib.parse import urlsplit
    identity=plan["identity"]
    origin=urlsplit(plan["endpoint"])
    if not identity.get("platform_conditions"):raise ValueError("incomplete platform or deployment configuration")
    operational=[]
    for service in identity["servers"]:
        operational.append({k:v for k,v in service["config"].items() if k!="--port"})
    return {"transport":[origin.scheme,origin.hostname],"servers":operational,
            **{k:identity[k] for k in ("files","checkpoint","model","max_model_len","gpu_governor",
                                       "boot_id","platform_conditions")}}


def comparison(directory, other=None):
    a,rows_a,_,_=rows(directory)
    if other is None:
        if a["mode"]!="comparison":raise ValueError("B-C-C-B batch required")
        b=a
        bs=[r for r in rows_a if r["role"]=="B"];cs=[r for r in rows_a if r["role"]=="C"]
    else:
        b,rows_b,_,_=rows(other)
        # Version is the declared factor. Process identities may differ after a restart.
        if deployment_conditions(a)!=deployment_conditions(b):raise ValueError("incompatible platform, workload or deployment configuration")
        bs=[r for r in rows_a if r["role"]=="formal"];cs=[r for r in rows_b if r["role"]=="formal"]
    key=lambda r:(r["n"],r["ix"],r["o"],r["concurrency"],r.get("requested_clients",r["concurrency"]))
    if not bs or {key(r) for r in bs}!={key(r) for r in cs}:raise ValueError("incomplete matched comparison")
    result=[]
    for idx in sorted({key(r) for r in bs}):
        x=[r for r in bs if key(r)==idx];y=[r for r in cs if key(r)==idx]
        if len(x)!=len(y):raise ValueError("unmatched round counts")
        result.append({"workload":idx,"B_version":a["identity"]["service_version"],
                       "C_version":b["identity"]["service_version"],
                       "same_service_processes":a["identity"]["servers"]==b["identity"]["servers"],
                       **compare(x,y)})
    return result


if __name__=="__main__":
    cli=argparse.ArgumentParser();cli.add_argument("directory",type=Path);cli.add_argument("--out",type=Path)
    cli.add_argument("--compare",nargs="?",const="internal")
    args=cli.parse_args()
    result=review(args.directory)
    if args.compare:result["comparison"]=comparison(args.directory,None if args.compare=="internal" else args.compare)
    if args.out:write_json(args.out,result)
    else:print(json.dumps(result,ensure_ascii=False,allow_nan=False))
