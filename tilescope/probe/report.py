"""Build a portable Markdown/JSON study report from the listed raw run directories."""
from collections import Counter, defaultdict
import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import quote

from tilescope.common import read_json, write_json
from tilescope.report import md, nfmt
from .review import review_run


def _diagnostic_study(study_path, spec, targets):
    results, errors = [], []
    for item in spec.get("diagnostics", []):
        raw = (study_path.parent / item["directory"]).resolve()
        record = dict(item, status="failed")
        results.append(record)
        try:
            target = targets[item["target_id"]]
            kind = item["kind"]
            if kind == "llama_cuda":
                from tools.multimodel.llama_trace import review
                resolved = read_json(raw / "resolved-target.json")
                if (resolved["profile_sha256"] != target["profile_sha256"] or
                        resolved["identity"]["device_uuid"] != target["identity"]["device_uuid"]):
                    raise ValueError("native diagnostic profile or device drift")
                data = review(raw)
            elif kind in {"dense", "qwen_legacy"}:
                cached = read_json(raw / "depth-plan.json")
                if (cached["target_id"] != target["target_id"] or cached["model"] != target["model"] or
                        cached["target_sha256"] != target["source_hashes"]["target"] or
                        cached["boot_id"] != target["identity"]["boot_id"]):
                    raise ValueError("depth diagnostic profile, model or device boot drift")
                if kind == "dense":
                    from .depth.review import review
                else:
                    from .depth.legacy import review_legacy as review
                data = review(raw)
                if item.get("require_gpu") and not data.get("gpu", {}).get("calls"):
                    raise ValueError("required CUDA trace missing")
                if item.get("require_cancel") and not data.get("cancel", {}).get("engine_terminal_ns"):
                    raise ValueError("required matched cancellation event missing")
            else:
                raise ValueError("unknown diagnostic kind")
            record.update(status="complete", result=data)
        except (ValueError, OSError, TypeError, KeyError, RuntimeError, sqlite3.Error) as exc:
            record["reason"] = str(exc)
            errors.append(f"{item['directory']}: {exc}")
    return results, errors


def rebuild(study_path):
    study_path = Path(study_path)
    spec = read_json(study_path)
    paths = spec["runs"]
    if not isinstance(paths, list) or not paths or len(set(paths)) != len(paths):
        raise ValueError("study needs distinct raw run directories")
    runs, errors = [], []
    processed = set()
    counts = Counter()
    kind_counts = defaultdict(Counter)
    targets = {}
    executions = set()
    for name in paths:
        path = (study_path.parent / name).resolve()
        if path in processed:
            raise ValueError("same run directory appears twice")
        processed.add(path)
        run = review_run(path)
        run["directory"] = name
        if run["status"] == "complete":
            ids = {r["execution_id"] for r in run["rows"]}
            if ids & executions:
                run = dict(status="failed", directory=name, reason="reused request executions", rows=[], groups=[])
            else:
                executions.update(ids)
                target_id = run["target"]["target_id"]
                counts[target_id] += run["roles"].get("formal", 0)
                prior = targets.get(target_id)
                if prior and prior["profile_sha256"] != run["target"]["profile_sha256"]:
                    errors.append(f"{name}: same deployment ID used with a different frozen profile")
                targets[target_id] = run["target"]
                kind_counts[target_id].update(r["kind"] for r in run["rows"] if r["role"] == "formal")
        if run["status"] != "complete":
            errors.append(f"{name}: {run['reason']}")
        runs.append(run)
    expected = spec.get("expected_formal_by_target")
    if expected is not None and dict(counts) != expected:
        errors.append(f"formal coverage mismatch: expected {expected}, observed {dict(counts)}")
    kind_counts = {k: dict(v) for k, v in kind_counts.items()}
    if "expected_formal_kinds_by_target" in spec and kind_counts != spec["expected_formal_kinds_by_target"]:
        errors.append("formal workload categories differ from the frozen study")
    diagnostics, diagnostics_errors = _diagnostic_study(study_path, spec, targets)
    errors.extend(diagnostics_errors)
    return {"schema": "tilescope.probe.report.v1", "status": "partial" if errors else "complete",
            "scope": "listed request suites and diagnostics", "formal_by_target": dict(counts), "expected_formal_by_target": expected,
            "errors": errors, "formal_kinds_by_target": kind_counts, "runs": runs, "diagnostics": diagnostics}


def _table(lines, labels, values):
    lines.extend(("", "| " + " | ".join(labels) + " |", "|" + "---|" * len(labels)))
    for row in values:
        lines.append("| " + " | ".join(md(v) for v in row) + " |")
    lines.append("")


def _value(group, field, statistic="median"):
    result = group[field]
    return nfmt(None if result is None else result[statistic])


def render(study_path, result, out):
    lines = ["# 多模型文本请求测量", "",
             f"从原始响应重建了 {sum(result['formal_by_target'].values())} 条有效正式请求。"
             + ("列出的请求批次复核通过。" if result["status"] == "complete" else "整项请求研究尚未完成。"),
             "", "每条请求保留输入、实际返回的输出 token 数以及引擎结束原因。"
             "首个可见文本和完整响应的耗时从客户端发送请求时起算。"
             "同内容组共享原文，分词结果随部署变化。定长组以实际输入 token 数及输出上限为条件。"]
    for error in result["errors"]:
        lines.extend(("", f"- {error}"))
    for run in result["runs"]:
        directory = (study_path.parent / run["directory"]).resolve()
        link = quote(os.path.relpath(directory / "plan.json", out), safe="/-_.~")
        if run["status"] != "complete":
            lines.extend(("", f"## {run['directory']}", "", f"批次失败，原因见复核诊断。{run['reason']}"))
            continue
        target = run["target"]
        model = target["model"]
        lines.extend(("", f"## {target['target_id']} · {model['display_name']} · {run['suite_id']}", "",
                      f"引擎为 {target['engine']['adapter']}。精度配置为 "
                      f"`{json.dumps(model['precision'], ensure_ascii=False, sort_keys=True)}`。"
                      f"正式请求 {run['roles'].get('formal', 0)} 条。各角色计数 {run['roles']}，"
                      f"未成功尝试 {run['failure_count']} 次，按计划拒绝 {run['expected_rejections']} 次。[冻结计划]({link})"))
        caps = run["capabilities"]
        required = [c for c in caps if c["required"]]
        if required:
            _table(lines, ("本批次要求的能力", "声明", "实测"), [
                (c["capability_id"], c["declared_support"], c["observed_status"]) for c in required])
        board = run["resources"]
        lines.append(f"资源窗口保留 {board['valid_samples']} 次读数，采样错误 {board['sampling_errors']} 次。")
        _table(lines, ("资源", "单位", "最低", "最高"), [
            (name, unit, _value(board, field, "min"), _value(board, field, "max"))
            for name,unit,field in (("GPU 温度","m°C","gpu_mC"),("GPU GPC 时钟","Hz","gpc_hz"),
                                    ("系统可用内存","B","mem_available_bytes"),("服务进程 RSS","B","rss_bytes"))])
        if not run["groups"]:
            continue
        values = []
        for g in run["groups"]:
            condition = str(g["condition"])[:12]
            values.append((g["kind"], condition, g["output_budget"], g["concurrency"], g["requests"],
                           g["count_source"], _value(g, "input_tokens"), _value(g, "output_tokens"),
                           _value(g, "ttfv_ms"), _value(g, "total_s"), _value(g, "total_s", "sample_sd")))
        _table(lines, ("任务", "输入条件", "输出上限", "并发", "样本", "计数来源", "输入中位 token",
                        "输出中位 token", "首文本中位 ms", "总时长中位 s", "总时长样本标准差 s"), values)
        phase_table = []
        for g in run["groups"]:
            for phase, distribution in g["engine_native"].items():
                if distribution:
                    phase_table.append((g["kind"], str(g["condition"])[:12], g["output_budget"], phase,
                                        distribution["count"], nfmt(distribution["median"])))
        if phase_table:
            _table(lines, ("任务", "输入条件", "输出上限", "引擎字段（含单位）", "可归因样本数", "中位数"),
                   phase_table)
    if result["diagnostics"]:
        lines.extend(("", "## 局部数值与 GPU 诊断", ""))
        entries = []
        for d in result["diagnostics"]:
            proof = d.get("result", {})
            local = proof.get("local_numerics", {})
            valid = [k for k in ("norm", "linear", "ba") if local.get(k, {}).get("passed")]
            kernels = proof.get("kernel_count", proof.get("gpu", {}).get("calls"))
            relative = quote(os.path.relpath(study_path.parent/d["directory"], out),safe="/-_.~")
            entries.append((d["target_id"], d["kind"], d["status"], ", ".join(valid) or "—",
                            "—" if kernels is None else kernels, f"[原始目录]({relative}/)"))
        _table(lines, ("部署", "诊断", "复核", "局部数值通过项", "跟踪 kernel 数", "证据"), entries)
        lines.append("诊断窗口单独运行。局部数值范围限于所选层和位置。")
    lines.extend(("", "### 统计口径", "",
                  "各组使用全部通过复核的正式样本。JSON 含逐次数据、每轮分组、缓存读数与结束原因，"
                  "单条样本的样本标准差为空。空值表示缺失。vLLM 阶段来自隔离请求窗口的计数器差分，"
                  "llama.cpp 的 prompt_ms 从引擎开始处理到最后一个 prompt 批次，predicted_ms 从该批次到最后生成。"
                  "首个生成 token 使用最后一个 prompt 批次的 logits。cache_n 和 prompt_n 分别表示此前可复用和本次计算的数量。"
                  "两个引擎各按自身阶段定义展示。"
                  "若部署同时改变了引擎和权重精度，其耗时只说明该部署的表现。[完整复算结果](report.json)", ""))
    return "\n".join(lines)


def report_study(study_path, out):
    study_path, out = Path(study_path), Path(out)
    result = rebuild(study_path)
    write_json(out / "report.json", result)
    (out / "REPORT.md").write_text(render(study_path, result, out), encoding="utf-8")
    return result
