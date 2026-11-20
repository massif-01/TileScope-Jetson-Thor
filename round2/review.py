"""Recompute accepted results and align service telemetry with replay windows."""
import argparse
import bisect
import json
import math
import statistics
from pathlib import Path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_check(row, l2_limit):
    require(row["passed"] is True, "correctness did not pass")
    for key in ("mismatches", "nonfinite"):
        require(type(row[key]) is int and row[key] == 0, f"invalid correctness count: {key}")
    for key in ("max_abs", "relative_l2", "reference_rms"):
        require(finite_number(row[key]) and row[key] >= 0, f"invalid correctness metric: {key}")
    require(row["reference_rms"] > 0 and row["relative_l2"] <= l2_limit, "numerical gate failed")


def summarize(folder):
    require((folder / "exit").read_text().strip() == "0", f"{folder}: nonzero exit")
    log = rows(folder / "stdout")
    require(not any(r["phase"] == "failure" for r in log), "failure in log")
    inputs = [r for r in log if r["phase"] == "input"]
    require(len(inputs) == 1, "expected one task input")
    data = inputs[0]
    workloads = rows(Path(__file__).with_name("WORKLOADS.jsonl"))
    matches = [w for w in workloads if w["module"] == data["layer"]]
    require(len(matches) == 1, "unknown fixed task")
    workload = matches[0]
    require(data["shape"] == workload["actual_probe_mnk"], "fixed task shape mismatch")
    require(data["sha256"] == workload["input_sha256"] and data["storage"] == workload["buffers"],
            "fixed task input or layout mismatch")
    m, n, k = data["shape"]
    checks_all = [r for r in log if r["phase"] == "correctness"]
    require(len(checks_all) == 12, "expected 12 output checks")
    for check in checks_all:
        validate_check(check, workload["validation"]["relative_l2_max"])
    require(sum(r["phase"] == "summary" for r in log) == 4, "expected four summaries")
    telemetry = rows(folder / "telemetry.jsonl")
    times = [r["unix_ns"] for r in telemetry]
    results = []
    for boundary in ("core", "full"):
        samples = [r for r in log if r["phase"] == "sample" and r["boundary"] == boundary]
        require(len(samples) == 100, "expected 50 sample pairs")
        for i in range(50):
            pair = samples[2*i:2*i+2]
            require([r["name"] for r in pair] == (["tile","native"] if i%2 else ["native","tile"]), "order")
            require(all(r["index"] == i for r in pair), "sample index")
        start = samples[0]["end_ns"] - round(samples[0]["graph_us"]*1000)
        end = samples[-1]["end_ns"]
        left, right = bisect.bisect_right(times, start)-1, bisect.bisect_left(times, end)
        require(0 <= left <= right < len(times), "monitor does not bracket samples")
        obs = telemetry[left:right+1]
        activity = {}
        for key in ("gpu_busy_pct","gpc_hz","temperature_mc","vllm:num_requests_running","vllm:num_requests_waiting",
                    "vllm:prompt_tokens_total","vllm:generation_tokens_total"):
            vals = [r[key] for r in obs if key in r]
            activity[key] = [min(vals), max(vals)] if vals else None
        for name in ("native","tile"):
            checks = [r for r in checks_all if r["name"] == name
                      and r["stage"].startswith(boundary+"_")]
            require({r["stage"] for r in checks} == {boundary+"_"+s for s in ("direct","graph","post")},
                    "missing output check")
            graph_us = [r["graph_us"] for r in samples if r["name"] == name]
            require(all(finite_number(x) and x > 0 for x in graph_us), "invalid timing")
            vals = [x / 16 for x in graph_us]
            median, sd = statistics.median(vals), statistics.stdev(vals)
            logged = next(r for r in log if r["phase"] == "summary" and r["name"] == name and r["boundary"] == boundary)
            expected = dict(median_us=median, stdev_us=sd, mean_us=statistics.mean(vals),
                            min_us=min(vals), max_us=max(vals))
            require(logged["samples"] == len(vals) and
                    all(finite_number(logged[key]) and math.isclose(logged[key], value)
                        for key, value in expected.items()), "aggregate mismatch")
            results.append(dict(run_id=folder.name,boundary=boundary,implementation=name,mnk=[m,n,k],
                                median_us=median,sd_us=sd,min_us=min(vals),max_us=max(vals),
                                tflops=2*m*n*k/median/1e6,max_abs=checks[-1]["max_abs"],rel_l2=checks[-1]["relative_l2"],
                                telemetry_samples=len(obs),telemetry=activity))
    return results


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+", type=Path)
    args=parser.parse_args()
    lines = []
    for folder in args.folders:
        try:
            lines.extend(json.dumps(result, allow_nan=False) for result in summarize(folder))
        except (ValueError, TypeError, KeyError, StopIteration, OSError) as error:
            parser.exit(1, f"{folder}: review rejected ({type(error).__name__}): {error}\n")
    print("\n".join(lines))
