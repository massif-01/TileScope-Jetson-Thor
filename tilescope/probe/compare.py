"""Validate actual workload equivalence before producing any comparison."""
from collections import defaultdict
from pathlib import Path

from parity.service_review import compare as legacy_regression, stats
from tilescope.common import digest, write_json
from .report import rebuild
from .review import EXACT_COUNTS

CONTRACTS = {"same-model-regression", "same-content", "same-token-budget"}


def _legacy_row(row):
    return {
        "input_sha256": row["input_sha256"], "output_sha256": digest(row["output_ids"]),
        "text_sha256": row["text_sha256"], "finish": row["finish"], "n": row["input_tokens"],
        "o": row["o"], "concurrency": row["concurrency"], "engine": row["engine"],
        "usage": [row["input_tokens"], row["output_tokens"]], "execution": row["execution_id"],
        "total_s": row["total_s"], "ttfv_ms": row["ttfv_ms"],
    }


def _regression(runs):
    targets = {digest(r["target"]) for r in runs}
    if len(targets) != 1:
        raise ValueError("strict regression requires a single frozen model, weights, template and deployment identity")
    cohorts = defaultdict(list)
    for run in runs:
        for row in run["rows"]:
            if row["role"] == "formal" and row["kind"] == "fixed":
                if row["count_source"] != "engine_token_ids" or not row["output_ids"]:
                    raise ValueError("strict regression requires actual generated IDs")
                key = (row["n"], row["o"], row["ix"], row["concurrency"], row["input_sha256"])
                cohorts[key].append(row)
    if not cohorts:
        raise ValueError("no fixed workload present for strict repeat comparison")
    results = []
    for key, parts in sorted(cohorts.items()):
        parts.sort(key=lambda row: row["send_ns"])
        if len(parts) < 2:
            raise ValueError("missing second execution for strict regression")
        reference = _legacy_row(parts[0])
        for repeat in parts[1:]:
            check = legacy_regression([reference], [_legacy_row(repeat)])
            results.append({"workload": list(key), "reference": reference["execution"],
                            "repeat": repeat["execution_id"], **check})
    return results


def compare_runs(runs, contract):
    if contract not in CONTRACTS:
        raise ValueError(f"unknown comparison contract {contract}")
    if not runs or any(run["status"] != "complete" for run in runs):
        raise ValueError("comparison requires complete requested runs")
    if contract == "same-model-regression":
        return _regression(runs)
    kind = "same-content" if contract == "same-content" else "fixed"
    by_target = defaultdict(lambda: defaultdict(list))
    other_attempts = defaultdict(list)
    unbound = defaultdict(int)
    for run in runs:
        target = run["target"]["target_id"]
        keyed = {r["key"] for r in run["rows"] if r["role"] == "formal" and r["kind"] == kind}
        if keyed:
            unbound[target] += run.get("unbound_attempts", 0)
        other_attempts[target].extend(a for a in run.get("other_attempts", []) if a.get("task_key") in keyed)
        for row in run["rows"]:
            if row["role"] != "formal" or row["kind"] != kind:
                continue
            if row["count_source"] not in EXACT_COUNTS:
                raise ValueError("estimated token counts cannot enter the precise-count comparison")
            if row["concurrency"] != 1:
                raise ValueError("serial cohort contains concurrent traffic")
            if kind == "fixed" and row["n"] != row["input_tokens"]:
                raise ValueError("same-token-budget input length mismatch")
            condition = row["raw_text_sha256"] if kind == "same-content" else row["input_tokens"]
            by_target[target][condition, row["o"]].append(row)
    if len(by_target) < 2:
        raise ValueError("cross-deployment contract requires at least two targets with matching tasks")
    target_ids = {r["target"]["target_id"] for r in runs}
    if set(by_target) != target_ids:
        raise ValueError("a deployment has no formal requests for this contract")
    keys = set(next(iter(by_target.values())))
    if not all(set(groups) == keys for groups in by_target.values()):
        raise ValueError("actual content or token budgets differ across deployments")
    result = []
    for key in sorted(keys):
        cells = {}
        for target, groups in by_target.items():
            sample = groups[key]
            keys_in_cell = {r["key"] for r in sample}
            other = [a for a in other_attempts[target] if a["task_key"] in keys_in_cell]
            errors = sum(a["status"] in {"failed", "partial"} for a in other)
            attempts = len(sample) + len(other)
            cells[target] = {
                "attempts": attempts, "failed_or_partial_attempts": errors,
                "failed_or_partial_fraction": None if unbound[target] else errors / attempts,
                "unbound_attempts_in_relevant_suites": unbound[target],
                "visible_text_requests": sum(r["ttfv_ms"] is not None for r in sample),
                "requests": len(sample), "count_sources": sorted({r["count_source"] for r in sample}),
                **{field: stats([r[field] for r in sample])
                   for field in ("input_tokens", "output_tokens", "ttfv_ms", "total_s")},
                "executions": [r["execution_id"] for r in sample],
            }
        result.append({"raw_text_sha256" if kind == "same-content" else "actual_input_tokens": key[0],
                       "output_budget": key[1], "deployments": cells})
    return result


def compare_study(study_path, contract, out):
    study = rebuild(study_path)
    result = {"schema": "tilescope.probe.compare.v1", "contract": contract, "status": "failed", "rows": []}
    try:
        if study["status"] != "complete":
            raise ValueError("requested study is incomplete: " + " | ".join(study["errors"]))
        result["rows"] = compare_runs(study["runs"], contract)
        result["status"] = "complete"
        result["targets"] = {r["target"]["target_id"]: r["target"] for r in study["runs"]}
    except ValueError as exc:
        result["reason"] = str(exc)
    write_json(Path(out) / "comparison.json", result)
    return result
