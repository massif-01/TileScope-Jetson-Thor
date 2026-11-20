"""Verify a v2 batch and recompute its public rows from each attempt's raw records."""

import hashlib
import json
import math
from pathlib import Path


class ReviewError(ValueError):
    def __init__(self, message, coverage=()):
        super().__init__(message)
        self.coverage = list(coverage)


def fingerprint(value):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inside(directory, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute(), "结果路径必须是相对路径")
    resolved = (directory / name).resolve()
    require(
        resolved.is_relative_to(directory.resolve()) and resolved != directory.resolve(),
        "结果路径超出批次目录",
    )
    return resolved


def _check_entry(batch_dir, entry, catalog):
    from .tasks import review_run
    from .common import task_contract, verify_evidence

    attempt = inside(batch_dir, entry["attempt"])
    identity = load_json(attempt / "identity.json")
    task = load_json(attempt / "task.json")
    require(task == identity["task"], "任务快照与身份记录不一致")
    require(task["task_id"] == entry["task_id"], "任务 ID 不一致")
    require(task == catalog.get(task["task_id"]), "任务快照与登记目录不一致")
    for key, label in (
        ("task_hash", "task"),
        ("implementation_hash", "implementation"),
        ("condition_hash", "conditions"),
    ):
        payload = task_contract(task) if key == "task_hash" else identity[label]
        require(identity[key] == entry[key] == fingerprint(payload), f"{key} 不匹配")
    require(identity["conditions"]["protocol"] == task["protocol"], "计时协议与任务快照不一致")
    require(
        identity["implementation"]["adapter"] == task["adapter"]
        and identity["implementation"]["descriptions"] == task["implementations"],
        "实现身份与任务快照不一致",
    )
    receipt = load_json(attempt / "status.json")
    status = receipt["status"]
    require(status == entry["status"], "任务退出状态与批次索引冲突")
    coverage = dict(
        task_id=task["task_id"],
        status=status,
        reason=receipt.get("reason") or entry.get("reason", ""),
        attempt=entry["attempt"],
        task=task,
        task_hash=entry["task_hash"],
        implementation_hash=entry["implementation_hash"],
        condition_hash=entry["condition_hash"],
    )
    if status != "passed":
        require(bool(coverage["reason"]), "非通过项缺少原因")
        return coverage, []

    raw = attempt / "native"
    verify_evidence(attempt)
    rows = review_run(task, raw)
    require(isinstance(rows, list) and rows, "校验通过项缺少原始计时")
    # Bind comparisons to actual quantized input/weight bytes, including runs of the legacy tasks.
    inputs, samples = [], []
    for line in (raw / "stdout").read_text(encoding="utf-8").splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            if record.get("phase") == "input":
                inputs.append(record)
            elif record.get("phase") == "sample":
                samples.append(record)
    require(len(inputs) == 1 and bool(inputs[0].get("sha256")), "缺少唯一的实际输入哈希")
    coverage["input_hash"] = fingerprint(inputs[0]["sha256"])
    coverage["samples_hash"] = fingerprint(samples)
    seen = set()
    for row in rows:
        key = (row["boundary"], row["implementation"])
        require(key not in seen, "重复统计同一边界和实现")
        seen.add(key)
        require(
            math.isfinite(row["median_us"])
            and row["median_us"] > 0
            and math.isfinite(row["sd_us"])
            and row["sd_us"] >= 0,
            "统计值无效",
        )
    return coverage, [dict(row, task_id=task["task_id"], attempt=entry["attempt"]) for row in rows]


def review_batch(batch_dir):
    """Return no performance rows on a rejected requested batch (including mixed valid/invalid)."""
    batch_dir = Path(batch_dir).resolve()
    coverage, results, problems = [], [], []
    try:
        batch = load_json(batch_dir / "batch.json")
        require(batch["schema"] == "tilescope.batch.v2", "未知批次格式")
        require(batch["mode"] in ("strict", "survey"), "未知批次模式")
        require(isinstance(batch["tasks"], list) and batch["tasks"], "没有请求的任务")
        require(isinstance(batch["run_id"], str) and batch["run_id"], "缺少运行编号")
        catalog = load_json(batch_dir / "catalog.json")
        require(isinstance(catalog, dict), "缺少冻结的任务目录")
        ids = [entry["task_id"] for entry in batch["tasks"]]
        require(len(set(ids)) == len(ids), "同一批次含重复任务 ID")
        if "selection_sha256" in batch or (batch_dir/"selection.json").exists():
            selection = load_json(batch_dir/"selection.json")
            require(fingerprint(selection) == batch["selection_sha256"] and
                    selection == {"mode": batch["mode"], "task_ids": ids}, "请求任务与冻结的选择记录不符")
        selected = set(ids)
        for entry in batch["tasks"]:
            try:
                item, rows = _check_entry(batch_dir, entry, catalog)
                coverage.append(item)
                if item["status"] == "passed":
                    results.extend(rows)
                elif not (batch["mode"] == "survey" and item["status"] == "unsupported"):
                    problems.append(f"{item['task_id']}: {item['status']} {item['reason']}")
            except (ValueError, OSError, TypeError, KeyError, StopIteration, OverflowError) as exc:
                detail = f"{entry['task_id']}: {type(exc).__name__} {exc}"
                problems.append(detail)
                coverage.append(
                    dict(
                        task_id=entry["task_id"],
                        status="measurement_invalid",
                        reason=detail,
                        attempt=entry.get("attempt", ""),
                    )
                )
        for task_id, task in catalog.items():
            if task_id not in selected:
                coverage.append(
                    dict(
                        task_id=task_id, status="not_run", reason="未在本次选择范围内", attempt="", task=task
                    )
                )
        require(not problems, "；".join(problems))
        return dict(
            schema="tilescope.summary.v2",
            run_id=batch["run_id"],
            mode=batch["mode"],
            status="passed",
            coverage=coverage,
            results=results,
            environment=batch["environment"],
            observation=batch.get("observation"),
            source_directory=str(batch_dir),
        )
    except (OSError, ValueError, TypeError, KeyError, StopIteration, OverflowError) as exc:
        raise ReviewError(str(exc), coverage) from exc
