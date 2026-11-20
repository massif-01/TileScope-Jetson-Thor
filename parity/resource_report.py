"""Describe observer watermarks, request brackets and measured rail energy by window."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path
from .service_review import stats


def resource_report(directory, requests):
    samples=[]
    by_session=[]
    for path in sorted((Path(directory)/"sessions").glob("*/resources.jsonl")):
        observations=[json.loads(line) for line in path.read_text().splitlines()]
        observations=[r for r in observations if "mem" in r]
        if observations:by_session.append(observations);samples.extend(observations)
    if not samples:raise ValueError("missing resource observations")
    samples.sort(key=lambda r:r["time_ns"])
    times=[r["time_ns"] for r in samples]
    rails=defaultdict(float);coverage=defaultdict(float)
    bins=defaultdict(list)
    for series in by_session:
        for a,b in zip(series,series[1:]):
            dt=(b["time_ns"]-a["time_ns"])/1e9
            if not 0<dt<=2.5:continue
            for name in a["rail_readings"].keys()&b["rail_readings"].keys():
                p=a["rail_readings"][name]["mW"];q=b["rail_readings"][name]["mW"]
                if None not in (p,q):
                    rails[name]+=(p+q)*dt/2000;coverage[name]+=dt
        for r in series:bins[(r["time_ns"]-times[0])//300_000_000_000].append(r)
    def summary(data):
        def range_of(vals):
            vals=[v for v in vals if v is not None]
            return {"first":vals[0],"last":vals[-1],"min":min(vals),"max":max(vals)} if vals else None
        result=dict(samples=len(data),first_ns=data[0]["time_ns"],last_ns=data[-1]["time_ns"],
                    mem_available_bytes=range_of([r['mem']['MemAvailable'] for r in data]),
                    cuda_free_bytes=range_of([r.get('cuda',{}).get('free_bytes') for r in data]),
                    gpu_mC=range_of([r['temperatures_mC'].get('gpu-thermal') for r in data]),
                    gpc_hz=range_of([r['gpc_hz'] for r in data]),
                    gpu_busy_pct=range_of([r['nvml'].get('gpu_util_percent') for r in data]))
        envelopes=[]
        for a,b in zip(data,data[1:]):
            if b["time_ns"]-a["time_ns"]>2_500_000_000:continue
            old=[int(n) for n in a["cpu_ticks"].split()[1:9]]
            new=[int(n) for n in b["cpu_ticks"].split()[1:9]]
            delta=[j-i for i,j in zip(old,new)]
            if sum(delta)>0:
                envelopes.append(100*(sum(delta)-delta[3]-delta[4])/sum(delta))
        result["system_active_cpu_percent"]=range_of(envelopes)
        result["processes"]={}
        for pid in set().union(*(set(r['processes']) for r in data)):
            prs=[r['processes'][pid] for r in data if pid in r['processes'] and 'rss_bytes' in r['processes'][pid]]
            if not prs:continue
            result["processes"][pid]={"name":prs[0]['name'],"start_ticks":sorted({p['start_ticks'] for p in prs}),
                **{k:range_of([p[k] for p in prs]) for k in ('rss_bytes','fd','threads','cpu_ticks')}}
            rates=[]
            for a,b in zip(data,data[1:]):
                first=a["processes"].get(pid,{})
                last=b["processes"].get(pid,{})
                dt=(b["time_ns"]-a["time_ns"])/1e9
                if (0<dt<=2.5 and "cpu_ticks" in first and "cpu_ticks" in last and first["start_ticks"]==last["start_ticks"]):
                    rates.append(100*(last["cpu_ticks"]-first["cpu_ticks"])/a["ticks_per_second"]/dt)
            result["processes"][pid]["cpu_percent_one_core"]=range_of(rates)
        return result
    windows=[]
    for req in requests:
        l=bisect_left(times,req["wall_start_ns"]);h=bisect_right(times,req["wall_end_ns"])
        before=samples[l-1] if l and req['wall_start_ns']-times[l-1]<=2_500_000_000 else None
        after=samples[h] if h<len(times) and times[h]-req['wall_end_ns']<=2_500_000_000 else None
        windows.append({"execution":req["execution"],"inside":summary(samples[l:h]) if h>l else None,
                        "before":summary([before]) if before else None,"after":summary([after]) if after else None})
    return {"summary":summary(samples),"five_minute_bins":[{"index":i,**summary(v)} for i,v in sorted(bins.items())],
            "request_watermarks":windows,"integrated_rail_joules":dict(rails),"rail_observed_seconds":dict(coverage),
            "interpretation":"Trapezoidal integration per rail across <=2.5 s sample gaps; entire instrumented window including background. No per-request attribution."}
