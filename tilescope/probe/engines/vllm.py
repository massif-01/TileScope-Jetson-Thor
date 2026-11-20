"""Select one model and one engine before applying the controlled-window checks."""
import json
import math
import re

from parity.service_io import METRIC, PHASES, STATS

LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def snapshot(path, target):
    deployment=target["deployment"]
    expected = deployment.get("metrics_model",deployment["served_model"])
    engine = str(deployment.get("metrics_engine","0"))
    sums, seen = {}, set()
    for line in path.read_text().splitlines():
        match = METRIC.fullmatch(line)
        if not match or match[1] not in STATS:
            continue
        labels = {k:json.loads('"'+v+'"') for k,v in LABEL.findall(match[2] or "")}
        if labels.get("model_name")!=expected or labels.get("engine")!=engine:
            continue
        # request_success_total is partitioned by finish_reason. No other
        # free dimensions or replicas may be silently summed.
        filters=deployment.get("metrics_labels",{})
        if any(labels.get(k)!=v for k,v in filters.items()):
            continue
        dimensions={"model_name","engine",*filters}
        if match[1]=="request_success_total":
            dimensions.update(("finished_reason","finish_reason"))
        if labels.keys()-dimensions:
            continue
        signature=(match[1],tuple(sorted(labels.items())))
        if signature in seen:
            raise ValueError("duplicate metrics series")
        seen.add(signature)
        value=float(match[3])
        if not math.isfinite(value) or value<0:
            raise ValueError("invalid vLLM metric")
        sums[match[1]]=sums.get(match[1],0)+value
    return sums


def phases(before,after,n,o,clients):
    a,b=before,after
    delta = {k:b[k]-a[k] for k in a.keys() & b.keys()}
    for k in a.keys()-b.keys():
        delta[k]=None
    single=(clients==1 and delta.get("request_success_total")==1 and
            delta.get("prompt_tokens_total")==n and delta.get("generation_tokens_total")==o and
            all(a.get(k)==b.get(k)==0 for k in ("num_requests_running","num_requests_waiting")))
    if not single:
        return {"attribution":"unavailable","reason":"counter scope, idle queue, success or token deltas did not isolate one serial request"}
    result={"attribution":"single-request delta"}
    for phase in PHASES:
        prefix=f"request_{phase}_time_seconds"
        total=delta.get(prefix+"_sum")
        if delta.get(prefix+"_count")!=1 or total is None or total<0:
            return {"attribution":"unavailable","reason":"phase histogram did not isolate one observation: "+phase}
        result[phase+"_s"]=total
    for k in ("request_prefill_kv_computed_tokens_sum","prefix_cache_queries_total","prefix_cache_hits_total",
              "spec_decode_num_drafts_total","spec_decode_num_draft_tokens_total","spec_decode_num_accepted_tokens_total"):
        result[k]=delta.get(k)
    return result
