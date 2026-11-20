"""Second implementation of representative wire timings and group statistics.

Uses base64 blocks, SSE frame boundaries and the archived client monotonic clock.
It deliberately does not call the request parsers, the suite reviewer or its statistics.
"""
import base64
import bisect
import json
import math
from pathlib import Path
import re

from tilescope.common import read_json


def reconstruct(attempt):
    sent = read_json(attempt / "request.json")
    receipt = read_json(attempt / "receipt.json")
    if not sent["stream"]:
        raise ValueError("independent sample requires streaming completion")
    raw, ends, ns = bytearray(), [], []
    for line in (attempt / "wire.jsonl").read_text().splitlines():
        chunk = json.loads(line)
        raw.extend(base64.b64decode(chunk["data_b64"], validate=True))
        ends.append(len(raw))
        ns.append(chunk["ns"])
    first, output_count, count_ids, start = None, None, 0, 0
    terminal = 0
    for delimiter in re.finditer(rb"\r?\n\r?\n", raw):
        fields = bytes(raw[start:delimiter.start()]).decode("utf-8").splitlines()
        start = delimiter.end()
        data = "\n".join(line[5:].removeprefix(" ") for line in fields if line.startswith("data:"))
        if not data:
            continue
        if data == "[DONE]":
            terminal += 1
            continue
        if terminal:
            raise ValueError("data follows the terminal event")
        obj = json.loads(data)
        if "choices" in obj:
            contents = [c.get("text") for c in obj["choices"]]
            count_ids += sum(len(c.get("token_ids") or []) for c in obj["choices"])
            if obj.get("usage") is not None:
                output_count = obj["usage"]["completion_tokens"]
        else:
            contents = [obj["content"]]
            count_ids += len(obj["tokens"])
            if obj["stop"]:
                terminal += 1
                output_count = obj["tokens_predicted"]
        if first is None and any(contents):
            first = ns[bisect.bisect_left(ends, start)]
    if start != len(raw) or terminal != 1 or output_count is None or (count_ids and output_count != count_ids):
        raise ValueError(f"unverifiable raw completion in {attempt.name}")
    begin, end = receipt["send_ns"], receipt["closed_ns"]
    if end <= begin or (first is not None and not begin <= first <= end):
        raise ValueError("independent check: inconsistent client clocks")
    return {"execution_id": attempt.name, "ttfv_ms": None if first is None else (first - begin) * 1e-6,
            "total_s": (end - begin) * 1e-9, "output_tokens": output_count}


def _distribution(values):
    ordered = sorted(v for v in values if v is not None)
    n = len(ordered)
    if not n:
        return None
    median = (ordered[(n-1)//2] + ordered[n//2]) / 2
    mean = sum(ordered) / n
    sd = math.sqrt(sum((x - mean)**2 for x in ordered) / (n - 1)) if n > 1 else None
    return {"median": median, "sample_sd": sd}


def _same(raw, reported, description):
    if (raw is None or reported is None):
        if raw is reported:
            return
    elif math.isclose(raw, reported, rel_tol=1e-12, abs_tol=1e-12):
        return
    raise ValueError(f"{description} mismatch: raw {raw!r}, report {reported!r}")


def verify_study(study_path, report_json_path):
    study_path = Path(study_path)
    study = read_json(study_path)
    report = read_json(report_json_path)
    if report["status"] != "complete" or study["runs"] != [run["directory"] for run in report["runs"]]:
        raise ValueError("independent check requires a complete report matching the study")
    checks = {}
    for run in report["runs"]:
        raw = study_path.parent / run["directory"]
        plan = read_json(raw / "plan.json")
        target = plan["target"]["target_id"]
        if target in checks:
            continue
        formal = [s for w in plan["waves"] for s in w if s["kind"] == "fixed" and s["role"] == "formal"]
        if not formal:
            continue
        ref = formal[0]
        condition = (ref["n"], ref["o"], ref["concurrency"])
        expected = {s["key"] for s in formal if (s["n"], s["o"], s["concurrency"]) == condition}
        samples = []
        committed = {execution for wave in (raw/"waves").glob("*.json") for execution in read_json(wave)}
        keys = set()
        for execution in sorted(committed):
            path = raw / "attempts" / execution
            receipt = read_json(path / "receipt.json")
            if receipt["task_key"] not in expected:
                continue
            if receipt["task_key"] in keys or receipt["status"] != "passed":
                raise ValueError("duplicate or incomplete committed independent sample")
            keys.add(receipt["task_key"])
            samples.append(reconstruct(path))
        if len(samples) != len(expected) or len(samples) < 2:
            raise ValueError("independent group missing raw replicates")
        ids = {r["execution_id"] for r in samples}
        groups = [g for g in run["groups"] if g["kind"] == "fixed" and set(g["executions"]) == ids]
        if len(groups) != 1:
            raise ValueError("independent cohort differs from reported executions")
        group = groups[0]
        rows = {row["execution_id"]: row for row in run["rows"]}
        distribution = {}
        for field in ("ttfv_ms", "total_s", "output_tokens"):
            for sample in samples:
                _same(sample[field], rows[sample["execution_id"]][field], field)
            distribution[field] = _distribution([sample[field] for sample in samples])
            if distribution[field] is None:
                _same(None, group[field], field)
                continue
            for stat, value in distribution[field].items():
                _same(value, group[field][stat], field + "." + stat)
        checks[target] = {"directory": run["directory"], "actual_input_tokens": ref["n"], "output_cap": ref["o"],
                          "samples": samples, "statistics": distribution}
    if set(checks) != set(report["formal_by_target"]):
        raise ValueError("missing independent fixed-workload check for a deployment")
    return {"schema": "tilescope.probe.independent.v1", "status": "complete", "checks": checks}
