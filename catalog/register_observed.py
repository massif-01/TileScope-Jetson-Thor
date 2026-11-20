"""Bind a finite set of BA projection replay configurations to an actual runner trace."""
import argparse
import hashlib
import json
from pathlib import Path


def make_records(summary, evidence, evidence_sha256):
    candidates = {}
    for record in summary["observation_records"]:
        label, phase, rows = record["key"]
        if label != "bf16_ba" or phase not in {"prefill","decode_target","decode_target_verify"}:
            continue
        if type(rows) is not int or not 1 <= rows <= 2048:
            raise ValueError("observed BA row count outside adapter bounds")
        if not record["module"].endswith("layers.0.linear_attn.in_proj_ba") or not record["request_id"]:
            raise ValueError("BA observation is not bound to the selected layer and request")
        x, y, w = record["actual_input"], record["actual_output"], record["actual_weight"]
        if isinstance(y, list):
            y = y[0]
        for tensor, shape in ((x,[rows,5120]), (w,[96,5120]), (y,[rows,96])):
            if tensor["dtype"] != "torch.bfloat16" or tensor["shape"] != shape or tensor["stride"] != [shape[1],1]:
                raise ValueError(f"BA observed layout or dtype outside replay contract: {tensor}")
        scheduler = record["scheduler"]
        if scheduler["phase"] != phase or not scheduler["prompt_token_hash"]:
            raise ValueError("unbound target-forward stage")
        key = (phase, rows)
        if key not in candidates:
            candidates[key] = dict(
                task_id="bf16-ba-"+phase.replace("_","-")+"-"+str(rows), adapter="bf16_ba", rows=rows,
                groups=["new","gemm","observed"],
                source=dict(kind="observed_runtime", stage=phase, evidence=evidence,
                            evidence_sha256=evidence_sha256, request_id=record["request_id"], step=record["step"],
                            observation_key=record["key"], observed_module=record["module"],
                            prompt_token_hash=scheduler["prompt_token_hash"],
                            scheduled=scheduler["scheduled"], speculative_tokens=scheduler["speculative_tokens"],
                            detail="Private eager vLLM target forward; shape/layout observed, activations synthesized. Production graph layout unconfirmed."))
    prefill = sorted(k for k in candidates if k[0]=="prefill")
    decode = sorted((k for k in candidates if k[0].startswith("decode_target")), key=lambda k:(k[0]!="decode_target_verify",k[1]))
    if not prefill or not decode:
        raise ValueError("need at least one measured prefill and one measured decode target shape")
    # Keep stable role IDs so the CLI examples continue to select measured workloads.
    choices = [("bf16-ba-small", decode[0]), ("bf16-ba-medium", prefill[0])]
    if prefill[-1] != prefill[0]:
        choices.append(("bf16-ba-large", prefill[-1]))
    result = []
    for task_id, key in choices:
        item = dict(candidates[key], task_id=task_id)
        if task_id == "bf16-ba-small":
            item["groups"] = item["groups"] + ["edge"]
        result.append(item)
    for label, adapter, task_id, xin, yout in [
        ("input_norm","gemma_rmsnorm","rmsnorm-regular",5120,5120),
        ("mlp_gate","swiglu","swiglu-regular",34816,17408),
    ]:
        rows = prefill[0][1]
        key = [label,"prefill",rows]
        found = next((r for r in summary["observation_records"] if r["key"] == key), None)
        if found is None:
            continue
        for tensor,size in [(found["actual_input"],xin),(found["actual_output"],yout)]:
            if tensor != {"shape":[rows,size],"stride":[size,1],"dtype":"torch.bfloat16"}:
                raise ValueError(f"{label} observed layout mismatch")
        if len(found["input_args"])!=1:
            raise ValueError(f"{label} input fusion boundary differs")
        if label=="input_norm" and found["actual_weight"] != {"shape":[5120],"stride":[1],"dtype":"torch.bfloat16"}:
            raise ValueError("unexpected RMSNorm checkpoint weight layout")
        sched = found["scheduler"]
        if not found["request_id"] or sched["phase"]!="prefill":
            raise ValueError("unbound non-GEMM stage")
        src=dict(candidates[prefill[0]]["source"], request_id=found["request_id"], step=found["step"],
                 observation_key=key,observed_module=found["module"],prompt_token_hash=sched["prompt_token_hash"],
                 scheduled=sched["scheduled"],speculative_tokens=sched["speculative_tokens"])
        result.append(dict(task_id=task_id,adapter=adapter,rows=rows,groups=["new","nongemm","observed"],source=src))
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("summary", type=Path)
    cli.add_argument("--catalog", type=Path, default=Path(__file__).with_name("tasks.json"))
    cli.add_argument("--evidence-path", required=True, help="portable evidence path in the delivery")
    cli.add_argument("--out", type=Path, required=True, help="write a new catalog")
    args=cli.parse_args()
    data=args.summary.read_bytes()
    entries=make_records(json.loads(data),args.evidence_path,hashlib.sha256(data).hexdigest())
    doc=json.loads(args.catalog.read_text())
    # Replace default calibration cases at identical M; custom diagnostic cases stay explicit.
    defaults = {"bf16-ba-small","bf16-ba-medium","bf16-ba-large"} | {e["task_id"] for e in entries}
    doc["tasks"]=[entry for entry in doc["tasks"] if entry["task_id"] not in defaults
                  and not (entry["adapter"]=="bf16_ba" and entry["source"]["kind"]=="observed_runtime")] + entries
    with args.out.open("x") as file:
        file.write(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
