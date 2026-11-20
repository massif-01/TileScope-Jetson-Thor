"""Reconstruct one frozen request suite on a CPU from its original transport records."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from parity.service_review import stats
from tilescope.common import read_json

EXACT_COUNTS = {"server_usage", "engine_token_ids"}
CLOCK = "host_boot_monotonic_ns"
CLIENT_FIELDS = {
    "ttfv_ms": ("ms", "send to first nonempty visible content", "wire"),
    "total_s": ("s", "send to client close after complete response", "wire"),
    "request_output_tokens_per_s": ("tokens/s", "generated tokens divided by client total", "wire and engine count"),
    "input_tokens": ("tokens", "actual processed input", "engine"),
    "output_tokens": ("tokens", "actual generated output including the engine stop policy", "engine"),
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def request_rows(path, records):
    result = []
    for original in records:
        row = original.copy()
        total_ns = row["closed_ns"] - row["send_ns"]
        first = row["first_visible_ns"]
        _require(total_ns > 0 and row["send_ns"] <= row["first_event_ns"] <= row["terminal_ns"] <= row["closed_ns"],
                 f"invalid clock order: {row['key']}")
        if first is not None:
            _require(row["first_event_ns"] <= first <= row["terminal_ns"], "visible content timestamp outside response")
        row["total_s"] = total_ns / 1e9
        row["ttfv_ms"] = None if first is None else (first - row["send_ns"]) / 1e6
        for count in ("input_tokens", "output_tokens"):
            _require(type(row[count]) is int and row[count] >= 0, f"invalid {count}")
        if row["role"] == "formal":
            _require(row["count_source"] in EXACT_COUNTS, "formal rate requires exact token counts")
            if row["kind"] in {"fixed", "concurrency"}:
                _require(row["n"] == row["input_tokens"], "processed prompt length differs from frozen size")
            _require(0 < row["output_tokens"] <= row["o"], "unexpected generated length")
        row["request_output_tokens_per_s"] = (
            row["output_tokens"] / row["total_s"] if row["count_source"] in EXACT_COUNTS else None
        )
        attempt = path / "attempts" / row["execution_id"]
        body = read_json(attempt / "request.json")
        if row["kind"] == "same-content":
            prompt = body["prompt"]
            _require(isinstance(prompt, str), "same-content requires original untemplated text")
            row["raw_text_sha256"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        row["evidence"] = f"attempts/{row['execution_id']}"
        result.append(row)
    return result


def group_key(row):
    # Same content can tokenize to different lengths. Its identity is the UTF-8 bytes.
    condition = row["raw_text_sha256"] if row["kind"] == "same-content" else row["n"]
    return row["kind"], condition, row["o"], row["concurrency"], row["count_source"]


def _summary(rows, fields=CLIENT_FIELDS):
    return {field: stats([r.get(field) for r in rows]) for field in fields}


def summarize(rows):
    buckets = defaultdict(list)
    for row in rows:
        if row["role"] == "formal":
            buckets[group_key(row)].append(row)
    groups = []
    for (kind, condition, budget, concurrency, source), sample in sorted(buckets.items()):
        rounds, caches = defaultdict(list), defaultdict(list)
        variants = defaultdict(set)
        phase_keys = set()
        for r in sample:
            rounds[r["round"]].append(r)
            eng = r["engine"]
            caches[(eng.get("prefix_cache_hits_total"), eng.get("request_prefill_kv_computed_tokens_sum"),
                    eng.get("cache_n"))].append(r)
            phase_keys.update(k for k in eng if k.endswith(("_s", "_ms")) and isinstance(eng[k], (int, float)))
            variants[r["input_sha256"]].add((r["text_sha256"], r["finish"], tuple(r.get("output_ids") or [])))
        groups.append({
            "kind": kind, "condition": condition, "output_budget": budget, "concurrency": concurrency,
            "count_source": source, "requests": len(sample), "executions": [r["execution_id"] for r in sample],
            **_summary(sample),
            "rounds": [{"round": rnd, "requests": len(part), **_summary(part, ("ttfv_ms", "total_s"))}
                       for rnd, part in sorted(rounds.items())],
            "cache_strata": [{"vllm_prefix_hit_tokens": key[0], "vllm_computed_tokens": key[1],
                              "llama_cache_n": key[2], "requests": len(part),
                              **_summary(part, ("ttfv_ms", "total_s"))}
                             for key, part in caches.items()],
            "engine_native": {k: stats([r["engine"].get(k) for r in sample]) for k in sorted(phase_keys)},
            "missing_ttfv": sum(r["ttfv_ms"] is None for r in sample),
            "changed_repeat_outputs": sum(len(v) > 1 for v in variants.values()),
            "finish_counts": dict(Counter(r["finish"] for r in sample)),
        })
    return groups


def resources(path):
    snapshots = [json.loads(line) for log in sorted((path/"sessions").glob("*/resources.jsonl"))
                 for line in log.read_text().splitlines()]
    valid = [r for r in snapshots if "error" not in r]
    return {"valid_samples": len(valid), "sampling_errors": len(snapshots)-len(valid),
            **{k: stats([r.get(k) for r in valid]) for k in ("mem_available_bytes","gpc_hz","gpu_mC","rss_bytes")},
            "boundaries": "board temperature, GPC frequency and available system memory; RSS of the bound listener"}


def check_soak(path, suite, rows):
    """Map archived sends to the scheduled monotonic observation slots."""
    if suite["suite_id"] != "soak":
        return None
    session, = sorted((path/"sessions").glob("*"))
    start = read_json(session/"start.json")
    resources = [json.loads(s) for s in (session/"resources.jsonl").read_text().splitlines()]
    first = next(sample for sample in resources if "error" not in sample)
    # The initial resource sample pairs wall and monotonic clocks in the same process.
    origin_ns = first["ns"] - (first["wall_ns"] - start["wall_ns"])
    duration, interval = suite["duration_s"], suite["interval_s"]
    expected = duration // interval
    ordered = sorted(rows, key=lambda r: r["send_ns"])
    offsets = [(r["send_ns"] - origin_ns) / 1e9 for r in ordered]
    _require([int(offset // interval) for offset in offsets] == list(range(expected)) and
             [r["round"] for r in ordered] == list(range(expected)), "soak sends do not cover the frozen time slots")
    return {"scheduled_requests":expected, "duration_s":duration, "interval_s":interval,
            "actual_send_offsets_s":offsets, "inter_send_s":stats([b-a for a,b in zip(offsets,offsets[1:])]),
            "start_clock":"session wall_ns mapped with its first paired resource observation",
            "last_resource_offset_s":(resources[-1]["ns"]-origin_ns)/1e9}


def observed_contract(plan, rows, board):
    from .capabilities import plan as capability_plan
    caps = capability_plan(plan["target"], plan["suite"])["capabilities"]
    phases = [r for r in rows if r["engine"]["attribution"] in ("single-request delta", "request-local native timings")]
    facts = {
        "request.complete": ([r["evidence"] for r in rows], "raw response reconstructed"),
        "request.exact_token_counts": ([r["evidence"] for r in rows if r["count_source"] in EXACT_COUNTS],
                                       "request count reconciled with the response"),
        "engine.controlled_serial_phases": ([r["evidence"] for r in phases if r["concurrency"] == 1],
                                            "attributed serial engine phase"),
        "resources.window": (["sessions"] if board["valid_samples"] else [], "bound resource samples"),
    }
    for cap in caps:
        fact = facts.get(cap["capability_id"])
        if fact and fact[0]:
            cap.update(observed_status="verified", evidence=fact[0], reason=fact[1])
    return caps


def review_run(directory):
    """Invalid strict suites retain diagnostics and never return partial performance rows."""
    from .collect import load_run
    path = Path(directory)
    try:
        plan, records, failures = load_run(path)
        rows = request_rows(path, records)
        target = plan["target"]
        board = resources(path)
        cadence = check_soak(path, plan["suite"], rows)
        caps = observed_contract(plan, rows, board)
        missing = [c["capability_id"] for c in caps if c["required"] and c["observed_status"] != "verified"]
        _require(not missing, f"required capabilities lack valid observations: {missing}")
        metric_contracts = {
            key: {"name": key, "unit": unit, "boundary": boundary, "source": source,
                  "count_basis": "row.count_source" if "token" in key else "elapsed time",
                  "clock_domain": CLOCK, "evidence": "row.evidence"}
            for key, (unit, boundary, source) in CLIENT_FIELDS.items()
        }
        return {"schema": "tilescope.probe.review.v1", "status": "complete", "target": target,
                "suite_id": plan["suite"]["suite_id"], "roles": dict(Counter(r["role"] for r in rows)),
                "failure_count": sum(f["status"] != "expected_rejection" for f in failures),
                "expected_rejections": sum(f["status"] == "expected_rejection" for f in failures),
                "unbound_attempts": sum(not f.get("task_key") and f["status"]!="expected_rejection" for f in failures),
                "other_attempts": failures, "rows": rows, "resources": board, "capabilities": caps,
                "metric_contracts": metric_contracts, "soak_schedule":cadence, "groups": summarize(rows)}
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError, StopIteration) as exc:
        return {"schema": "tilescope.probe.review.v1", "status": "failed", "reason": str(exc),
                "rows": [], "groups": []}
