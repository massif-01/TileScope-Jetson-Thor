"""Offline request/runner evidence checks; no model, GPU or vLLM installation required."""
import argparse
import json
from pathlib import Path

from .collect import summarize


def review(folder):
    requests = {r["id"]: r for r in json.loads((folder / "requests.json").read_text())}
    traces = [json.loads(line) for line in (folder / "runner.jsonl").read_text().splitlines()]
    timing = json.loads((folder / "online-requests.json").read_text())
    if {r["id"] for r in timing} != set(requests):
        raise ValueError("Online request identity mismatch")
    for online in timing:
        spec = requests[online["id"]]
        if (online["request"]["prompt"] != spec["ids"]
            or online["usage"]["prompt_tokens"] != len(spec["ids"])
            or online["usage"]["completion_tokens"] != spec["outputs"]):
            raise ValueError("Observed request tokens differ from controlled workload")
    for step in (r for r in traces if r["event"]=="target_forward"):
        batch, = step["batch"]
        prefill = batch["computed_before"] < batch["prompt_length"]
        expected = "prefill" if prefill else ("decode_target_verify" if batch["speculative_tokens"] else "decode_target")
        if expected != batch["phase"] or not (0 < batch["scheduled"] <= step["input_ids"]["shape"][0]):
            raise ValueError("Forward phase or padding contradicts scheduler metadata")
    complete = {r["request"]: r for r in traces if r["event"]=="completed"}
    for rid, spec in requests.items():
        if complete[rid]["prompt_token_ids"] != spec["ids"] or len(complete[rid]["completion_token_ids"])!=spec["outputs"]:
            raise ValueError("Private observer did not execute the complete request")
    from types import SimpleNamespace
    findings = summarize(SimpleNamespace(output=folder))
    expected = json.loads((folder / "summary.json").read_text())["observation_records"]
    if json.loads(json.dumps(findings)) != expected:
        raise ValueError("Saved observation summary differs from runner hooks")
    status = json.loads((folder / "postflight.json").read_text())["serving_processes"]
    before = json.loads((folder / "deployment.json").read_text())["processes"]
    if status != before or json.loads((folder / "isolated-exit.json").read_text())["returncode"] != 0:
        raise ValueError("Deployment identity or observer exit did not pass")
    return {"requests":len(requests),"target_forwards":sum(r["event"]=="target_forward" for r in traces),
            "operator_hooks":sum(r["event"]=="operator" for r in traces),"unique_observations":len(findings),
            "ba_shapes":[r["key"][1:] for r in findings if r["key"][0]=="bf16_ba"],
            "result":"verified"}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    print(json.dumps(review(p.parse_args().folder),indent=2))
