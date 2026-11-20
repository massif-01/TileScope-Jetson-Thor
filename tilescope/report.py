"""Generate offline Chinese reports from reviewed raw batches, using Python's standard library."""

import html
import json
import math
import os
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from .review import ReviewError, load_json, review_batch


STATUS = {
    "passed": "通过",
    "unsupported": "已知不支持",
    "environment_blocked": "环境阻断",
    "invalid_input": "输入无效",
    "correctness_failed": "数值失败",
    "execution_failed": "执行失败",
    "measurement_invalid": "测量无效",
    "not_run": "未测",
}
IMPLEMENTATION = {"native": "传统 CUDA", "tile": "cuTile", "ptx": "PTX", "tileir": "Tile IR"}
BOUNDARY = {"core": "核心计算", "full": "完整操作"}
SOURCE = {"observed_runtime": "运行时观测", "derived_config": "配置推导", "synthetic_diagnostic": "合成诊断"}
COMPARISON_RULE = (
    "两次记录按任务、实际输入、计时条件、实现角色和操作边界匹配。变化率为本次中位数除以"
    "对照中位数再减一。单次进程内的图重放样本共享设备状态，每侧只有一轮独立运行时，"
    "稳定变快或变慢均判为未能区分。实现版本变化单列。"
)
esc = html.escape


def nfmt(value):
    if value is None:
        return "—"
    if abs(value) >= 1:
        return format(value, ",.2f").rstrip("0").rstrip(".")
    return format(value, ".4g")


def md(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def compact(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return str(value)


def relative_link(path, output):
    return quote(os.path.relpath(path, output), safe="/-_.~")


def compare_runs(current, previous):
    by_id = {r["task_id"]: r for r in previous["coverage"] if r["status"] == "passed"}
    cur_tasks = {r["task_id"]: r for r in current["coverage"] if r["status"] == "passed"}
    old = {(r["task_id"], r["boundary"], r["implementation"]): r for r in previous["results"]}
    result = []
    for row in current["results"]:
        key = (row["task_id"], row["boundary"], row["implementation"])
        item, prior = cur_tasks[key[0]], by_id.get(key[0])
        entry = dict(task_id=key[0], boundary=key[1], implementation=key[2])
        reason = ""
        if prior is None or key not in old:
            reason = "对照运行缺少对应任务、边界或实现"
        elif item["task_hash"] != prior["task_hash"]:
            reason = "任务定义变化"
        elif item["input_hash"] != prior["input_hash"]:
            reason = "实际输入哈希变化"
        elif item["condition_hash"] != prior["condition_hash"]:
            reason = "测量条件变化"
        elif item["samples_hash"] == prior["samples_hash"]:
            reason = "相同原始样本，缺第二次独立运行"
        if reason:
            result.append(dict(entry, compatible=False, reason=reason))
            continue
        baseline = old[key]["median_us"]
        change = 100 * (row["median_us"] / baseline - 1)
        result.append(
            dict(
                entry,
                compatible=True,
                previous_us=baseline,
                current_us=row["median_us"],
                change_pct=change,
                implementation_changed=item["implementation_hash"] != prior["implementation_hash"],
                judgement="未能区分稳定变化",
            )
        )
    observed = {(r["task_id"], r["boundary"], r["implementation"]) for r in current["results"]}
    for task_id, boundary, implementation in sorted(old.keys() - observed):
        result.append(
            dict(
                task_id=task_id,
                boundary=boundary,
                implementation=implementation,
                compatible=False,
                reason="本次运行缺少对应任务、边界或实现",
            )
        )
    return result


def _label(row):
    name = IMPLEMENTATION.get(row["implementation"], row["implementation"])
    boundary = BOUNDARY.get(row["boundary"], row["boundary"])
    return f"{row['task_id']} · {boundary} · {name}"


def _plot(rows, kind):
    """Dot plots on separately labeled axes preserve short kernels over a wide latency range."""
    latency = kind == "latency"
    title = "图内每次操作的中位耗时" if latency else "图内样本的耗时波动"
    description = (
        "单位 µs，对数轴，数值越小耗时越短。" if latency else "样本标准差除以中位数，线性轴，单位 %。"
    )
    values = [r["median_us"] if latency else 100 * r["sd_us"] / r["median_us"] for r in rows]
    label_width = max(225, max((len(_label(r)) * 7.2 for r in rows), default=0) + 12)
    x0, x1 = label_width, label_width + 375
    width = x1 + 74
    height = 77 + 28 * len(rows)

    def point(v):
        if latency:
            return x0 + (math.log10(v) - lo) / (hi - lo) * (x1 - x0)
        return x0 + v / hi * (x1 - x0)

    if latency:
        lo = math.floor(math.log10(min(values)))
        hi = max(lo + 1, math.ceil(math.log10(max(values))))
        ticks = [10**i for i in range(lo, hi + 1)]
    else:
        hi = max(1, math.ceil(max(values)))
        ticks = [hi * i / 4 for i in range(5)]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height}" '
        'role="img" font-family="system-ui,sans-serif" font-size="12">',
        f"<title>{title}</title><desc>{description} 详细数字见数据表。</desc>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="7" y="18" font-size="14" fill="#182e39">{title}</text>',
    ]
    for tick in ticks:
        x = point(tick)
        parts.append(
            f'<path stroke="#d6e0e3" d="M{x:.2f},43V{height - 12}"/>'
            f'<text fill="#46545b" x="{x:.2f}" y="37" text-anchor="middle">{tick:g}</text>'
        )
    for i, (row, value) in enumerate(zip(rows, values)):
        y = 60 + 28 * i
        color = "#075c70" if row["implementation"] in ("native", "ptx") else "#944b1a"
        parts.append(
            f'<a href="#metric-{i}"><text fill="#263c48" x="7" y="{y + 4}">{esc(_label(row))}</text>'
            f'<circle cx="{point(value):.2f}" cy="{y}" r="4.5" fill="{color}"/>'
            f'<text x="{width - 9:.0f}" y="{y + 4}" text-anchor="end" fill="{color}">{nfmt(value)}</text></a>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


STYLE = """
*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;background:#f2f5f5;color:#162d37;font:16px/1.65 "Source Sans 3","PingFang SC","Microsoft YaHei",system-ui,sans-serif}
main{max-width:1180px;background:#fff;margin:0 auto;padding:clamp(18px,5vw,64px)}header{border-top:6px solid #075c70}
h1{font-size:clamp(1.8rem,4.5vw,3.2rem);line-height:1.14;margin:.55em 0 .35em;letter-spacing:-.035em}h2{font-size:1.45rem;margin:2em 0 .5em;border-bottom:1px solid #b6c8ca}
p{max-width:85ch;margin:.55em 0}a{color:#07566f;text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #c67838;outline-offset:3px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.92rem;min-width:650px}th,td{text-align:left;border-bottom:1px solid #d8e2e2;padding:8px 10px;vertical-align:top}th{color:#344c55;background:#eff4f3;white-space:nowrap}.coverage td:nth-child(5){white-space:nowrap}
.scroll{max-width:100%;overflow:auto}.metric table{min-width:910px}figure{margin:1em 0}.plot{max-width:100%;overflow-x:auto}.plot svg{width:100%;height:auto;min-width:620px;display:block}
figcaption,.note{color:#415961;font-size:.9rem}code,pre{font:13px/1.55 "IBM Plex Mono",ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f6;padding:.8rem}
summary{cursor:pointer;color:#164958;font-weight:600}details{margin:.6em 0}td,li{overflow-wrap:anywhere}.error{color:#8a3226}nav{display:flex;gap:1em;flex-wrap:wrap;padding:.5em 0;font-size:.92rem}
@media(max-width:620px){main{padding:16px}th,td{padding:6px}h2{font-size:1.25rem}}
@media print{main{padding:0}body{background:#fff}nav{display:none}.scroll,.plot{overflow:visible}.plot svg{min-width:0}}
"""


def _write_files(output, files):
    # Compute and serialize all files before publishing any result from the new batch.
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as scratch:
        for name, content in files.items():
            (Path(scratch) / name).write_text(content, encoding="utf-8")
        for name in files:
            os.replace(Path(scratch) / name, output / name)


def _table(headers, rows, ids=False):
    head = "<tr>" + "".join(f'<th scope="col">{esc(h)}</th>' for h in headers) + "</tr>"
    body = []
    for i, row in enumerate(rows):
        anchor = f' id="metric-{i}"' if ids else ""
        body.append(f"<tr{anchor}>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return (
        '<div class="scroll"><table><thead>'
        + head
        + "</thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
    )


def _md_table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
        + ["| " + " | ".join(md(c) for c in row) + " |" for row in rows]
    )


def _attempt_links(batch_dir, output, item):
    if not item.get("attempt"):
        return []
    folder = (batch_dir / item["attempt"]).resolve()
    if not folder.is_relative_to(batch_dir):
        return []
    candidates = [
        ("任务定义", "task.json"),
        ("身份与条件", "identity.json"),
        ("状态", "status.json"),
        ("命令", "command.json"),
        ("命令", "command.sh"),
        ("命令", "command"),
        ("执行记录", "native/stdout"),
        ("错误与编译日志", "native/stderr"),
        ("遥测", "native/telemetry.jsonl"),
        ("产物", "native/compiler"),
        ("运行参数", "native/command"),
    ]
    return [
        (label, relative_link(folder / path, output))
        for label, path in candidates
        if (folder / path).is_file() or ((folder / path).is_dir() and any((folder / path).iterdir()))
    ]


def _observation_panels(batch_dir, output, observation):
    """Render only recorded model metrics. Internal operation shapes use the isolated runner evidence."""
    if not observation:
        msg = "本批次没有关联受控请求记录。"
        return [f"<p>{msg}</p>"], [msg]
    from .common import ROOT

    label = observation.get("summary", "") if isinstance(observation, dict) else ""
    name = observation.get("evidence", "") if isinstance(observation, dict) else str(observation)
    chunks_html, chunks_md = [], []
    if label:
        chunks_html.append("<p>" + esc(str(label)) + "</p>")
        chunks_md.append(str(label))
    locations = [
        (base / name).resolve()
        for base in (batch_dir, ROOT)
        if name and not Path(name).is_absolute() and (base / name).resolve().is_relative_to(base.resolve())
    ]
    summary = next((p for p in locations if p.is_file()), None)
    if summary is None:
        msg = "模型阶段和请求时序请查阅随包观测记录 " + str(name)
        return chunks_html + [f"<p>{esc(msg)}</p>"], chunks_md + [msg]
    observed = load_json(summary)
    if observed.get("schema") != "tilescope.observation.v1":
        href = relative_link(summary, output)
        return chunks_html + [f'<a href="{esc(href)}">随包观测记录</a>'], chunks_md + [
            f"[随包观测记录]({href})"
        ]

    def panel(title, headers, values):
        chunks_html.extend(
            ["<h3>" + esc(title) + "</h3>", _table(headers, [[esc(str(v)) for v in r] for r in values])]
        )
        chunks_md.extend(["### " + title, _md_table(headers, values)])

    online = summary.parent / observed.get("online_timing", "online-requests.json")
    if online.is_file() and online.resolve().parent == summary.resolve().parent:
        table = []
        for request in load_json(online):
            usage = request.get("usage", {})
            gaps = request.get("nonempty_chunk_gaps_ms", [])
            table.append(
                [
                    request["id"],
                    usage.get("prompt_tokens", "未知"),
                    usage.get("completion_tokens", "未知"),
                    nfmt(request.get("first_nonempty_chunk_ms")),
                    nfmt(request.get("total_ms")),
                    nfmt(statistics.median(gaps)) if gaps else "—",
                ]
            )
        panel(
            "在线部署的客户端时序",
            [
                "受控请求",
                "输入 token",
                "输出 token",
                "首个非空 chunk（ms）",
                "总耗时（ms）",
                "后续非空 chunk 间隔中位数（ms）",
            ],
            table,
        )
    shapes = []
    for r in observed.get("observation_records", []):
        inp = r.get("actual_input") or {}
        key = r.get("key", ["未知", "未知", "未知"])
        shapes.append(
            [
                r.get("request_id", "未知"),
                key[1],
                key[0],
                compact(inp.get("shape", [])),
                inp.get("dtype", "未知"),
                compact(inp.get("stride", [])),
            ]
        )
    panel(
        "独立观察实例中的实际前向张量",
        ["受控请求", "调度阶段", "操作", "输入形状", "dtype", "stride"],
        shapes,
    )
    note = "在线请求时序属于现有服务；形状采自同 checkpoint 的短时 runner。详见采集设置。"
    chunks_html.append("<p>" + note + "</p>")
    chunks_md.append(note)
    links = [
        (p.name, relative_link(p, output))
        for p in (summary, online, summary.parent / "runner.jsonl")
        if p.is_file() and p.resolve().parent == summary.resolve().parent
    ]
    chunks_html.append(
        "<p>" + " · ".join(f'<a href="{esc(h)}">{esc(label)}</a>' for label, h in links) + "</p>"
    )
    chunks_md.append(" · ".join(f"[{label}]({h})" for label, h in links))
    return chunks_html, chunks_md


def _backend_panel(study, output):
    title = "同一源码的 PTX / Tile IR 后端验证"
    contents = ["<h2>" + title + "</h2>"]
    markup = ["## " + title]
    passed = study["status"] == "passed"
    note = (
        "同源 FP32 向量加法诊断在两个后端均通过实际编译与数值校验。"
        if passed
        else "本次同源诊断缺少通过的数据。"
    )
    note += "该固定诊断独立于模型任务的跨实现对照。"
    contents.append("<p>" + esc(note) + "</p>")
    markup.append(note)
    if passed:
        headers = ["后端", "向量长度", "中位 µs", "标准差 µs", "样本数", "有效元素/s"]
        data = [
            [
                r["backend"],
                str(study["shape"][0]),
                nfmt(r["median_us"]),
                nfmt(r["sd_us"]),
                str(r["samples"]),
                nfmt(r["elements_per_s"]),
            ]
            for r in study["results"]
        ]
        contents.append(_table(headers, [[esc(v) for v in row] for row in data]))
        markup.append(_md_table(headers, data))
    else:
        reason = compact(study.get("reason", study["status"]))
        contents.append("<pre>" + esc(reason) + "</pre>")
        markup.append("```text\n" + reason + "\n```")
    directory = Path(study["source_directory"])
    links = [
        (label, relative_link(directory / name, output))
        for label, name in (
            ("实际 target 与样本", "stdout"),
            ("编译和错误日志", "stderr"),
            ("退出码", "exit"),
            ("PTX 产物", "ptx"),
            ("Tile IR 产物", "tileir"),
        )
        if (directory / name).exists()
    ]
    contents.append("<p>" + " · ".join(f'<a href="{esc(h)}">{esc(label)}</a>' for label, h in links) + "</p>")
    markup.append(" · ".join(f"[{label}]({h})" for label, h in links))
    return contents, markup


def _render(batch_dir, output, summary, comparisons, previous_id, error):
    coverage = summary["coverage"]
    rows = summary["results"]
    by_id = {r["task_id"]: r for r in coverage}
    passed = sum(r["status"] == "passed" for r in coverage)
    selected = sum(r.get("reason") != "未在本次选择范围内" for r in coverage)
    title = "Thor 性能探针运行报告"
    lead = (
        "本批次未通过，性能比较已停止。"
        if error
        else f"本批次选择了 {selected} 个计算任务，{passed} 个通过校验，得到 {len(rows)} 组分项测量。"
    )
    html_parts = [
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,">',
        f"<title>{title}</title><style>{STYLE}</style></head><body><main><header>",
        f"<h1>{title}</h1><p>{esc(lead)}</p>",
        f'<p class="note">运行 {esc(summary["run_id"])} · {esc(summary["mode"])}</p></header>',
    ]
    md_parts = [f"# {title}", lead, f"运行 `{summary['run_id']}` · {summary['mode']}"]
    html_parts.append(
        '<nav aria-label="报告章节"><a href="#coverage">任务覆盖</a><a href="#measure">算子测量</a>'
        '<a href="#observe">模型观测</a><a href="#proof">复核材料</a></nav>'
    )
    if error:
        html_parts.append(f'<p class="error">{esc(error)}</p>')
        md_parts.append(error)
    if rows:
        measured = [c["task"] for c in coverage if c["status"] == "passed"]
        precisions = "、".join(sorted({str(t["precision"]) for t in measured}))
        note = (
            f"成功测量涉及 {precisions}。表中数值描述所选实现与本批次工作量；模型阶段及请求时序另见观测记录。"
        )
        html_parts.append(f"<p>{esc(note)}</p>")
        md_parts.append(note)
        pairs = {}
        for row in rows:
            pairs.setdefault((row["task_id"], row["boundary"]), {})[row["implementation"]] = row["median_us"]
        comparable = [
            (values["tile"] / values["native"], task, boundary)
            for (task, boundary), values in pairs.items()
            if "tile" in values and "native" in values
        ]
        if comparable:
            ratio, task, boundary = max(comparable)
            label = BOUNDARY.get(boundary, boundary)
            finding = f"按本批次各任务和边界分别计算 cuTile / 传统 CUDA 中位耗时倍数，最高为 {nfmt(ratio)} 倍（{task}，{label}）。"
            html_parts.append("<p>" + esc(finding) + "</p>")
            md_parts.append(finding)
    html_parts.append('<h2 id="coverage">任务覆盖</h2>')
    md_parts.append("## 任务覆盖")
    cols = ["任务", "精度 / 算子", "逻辑形状", "阶段与来源", "状态", "说明"]
    table = []
    for item in coverage:
        task = item.get("task", {})
        source = task.get("source", {})
        if isinstance(source, dict):
            source = " · ".join(
                str(v) for v in (SOURCE.get(source.get("kind"), source.get("kind")), source.get("stage")) if v
            )
        table.append(
            [
                item["task_id"],
                " / ".join(str(task[k]) for k in ("precision", "operation") if task.get(k)),
                " × ".join(map(str, task.get("shape", []))) or "—",
                source or "见任务快照",
                STATUS.get(item["status"], item["status"]),
                item.get("reason") or "—",
            ]
        )
    html_parts.append(
        '<div class="coverage">' + _table(cols, [[esc(str(v)) for v in row] for row in table]) + "</div>"
    )
    md_parts.append(_md_table(cols, table))

    html_parts.append('<h2 id="measure">算子测量</h2>')
    md_parts.append("## 算子测量")
    files = {}
    if rows:
        note = (
            "耗时取 CUDA Graph 内重复操作的摊销中位数，波动取未裁剪重放样本的标准差。"
            "同一任务两种边界分别统计。TFLOP/s 按任务定义的有效运算量换算。"
        )
        html_parts.append(f"<p>{note}</p>")
        md_parts.append(note)
        cols = [
            "任务",
            "边界",
            "实现",
            "中位 µs",
            "标准差 µs",
            "最小–最大 µs",
            "样本数",
            "有效 TFLOP/s",
            "元素/s",
        ]
        table = []
        for row in rows:
            task = by_id[row["task_id"]]["task"]
            count = row.get("samples", row.get("count", task.get("protocol", {}).get("samples")))
            table.append(
                [
                    row["task_id"],
                    BOUNDARY.get(row["boundary"], row["boundary"]),
                    IMPLEMENTATION.get(row["implementation"], row["implementation"]),
                    nfmt(row["median_us"]),
                    nfmt(row["sd_us"]),
                    f"{nfmt(row['min_us'])}–{nfmt(row['max_us'])}",
                    count if count else "见记录",
                    nfmt(row.get("tflops")),
                    nfmt(row.get("elements_per_second")),
                ]
            )
        for kind, desc in (("latency", "耗时（µs，对数轴）"), ("spread", "波动（样本标准差 / 中位耗时，%）")):
            svg = _plot(rows, kind)
            files[kind + ".svg"] = svg.replace('href="#metric-', 'href="index.html#metric-')
            html_parts.append(
                f'<figure><div class="plot">{svg}</div><figcaption>{desc}。点选图中任务可查看数据行。</figcaption></figure>'
            )
        html_parts.append(
            '<div class="metric">'
            + _table(cols, [[esc(str(v)) for v in row] for row in table], True)
            + "</div>"
        )
        md_parts.extend([_md_table(cols, table), "[耗时图](latency.svg) · [波动图](spread.svg)"])
    else:
        msg = "本次没有可发布的性能数据。请查看任务覆盖和原始退出记录。"
        html_parts.append(f"<p>{msg}</p>")
        md_parts.append(msg)
    if previous_id is not None:
        head = "兼容运行比较"
        md_parts.extend(["## " + head, f"对照运行 `{previous_id}`。{COMPARISON_RULE}"])
        html_parts.extend(
            ["<h2>" + head + "</h2>", f"<p>对照运行 <code>{esc(previous_id)}</code>。{COMPARISON_RULE}</p>"]
        )
        cols = ["任务与边界 / 实现", "对照 µs", "本次 µs", "中位变化", "判断"]
        table = []
        for c in comparisons:
            if c["compatible"]:
                verdict = c["judgement"] + ("；实现版本变化" if c["implementation_changed"] else "")
                table.append(
                    [
                        _label(c),
                        nfmt(c["previous_us"]),
                        nfmt(c["current_us"]),
                        f"{c['change_pct']:+.2f}%",
                        verdict,
                    ]
                )
            else:
                table.append([_label(c), "—", "—", "—", c["reason"]])
        md_parts.append(_md_table(cols, table))
        html_parts.append(_table(cols, [[esc(v) for v in r] for r in table]))
    html_parts.append('<h2 id="observe">现有部署的模型观测</h2>')
    md_parts.append("## 现有部署的模型观测")
    observation_html, observation_md = _observation_panels(batch_dir, output, summary.get("observation"))
    html_parts.extend(observation_html)
    md_parts.extend(observation_md)
    if summary.get("backend_study"):
        backend_html, backend_md = _backend_panel(summary["backend_study"], output)
        html_parts.extend(backend_html)
        md_parts.extend(backend_md)
    html_parts.append('<h2 id="proof">技术与复核材料</h2>')
    md_parts.append("## 技术与复核材料")
    env = json.dumps(summary.get("environment"), ensure_ascii=False, indent=2)
    html_parts.append("<details><summary>设备、软件与运行条件</summary><pre>" + esc(env) + "</pre></details>")
    md_parts.append("### 设备、软件与运行条件\n\n```json\n" + env + "\n```")
    index_link = relative_link(batch_dir / "batch.json", output)
    html_parts.append(
        f'<p><a href="{esc(index_link)}">批次索引</a> · <a href="results.json">可复算数据表</a></p>'
    )
    md_parts.append(f"[批次索引]({index_link}) · [可复算数据表](results.json)")
    for item in coverage:
        if not item.get("attempt"):
            continue
        t = item.get("task", {})
        links = _attempt_links(batch_dir, output, item)
        info = {
            k: t[k]
            for k in (
                "comparison_type",
                "semantics",
                "shape",
                "source",
                "inputs",
                "outputs",
                "validation",
                "implementations",
                "protocol",
            )
            if k in t
        }
        info.update(
            {
                k: item[k]
                for k in ("task_hash", "implementation_hash", "condition_hash", "input_hash")
                if k in item
            }
        )
        info["数值误差、实际内核与样本遥测"] = [
            {
                k: r[k]
                for k in (
                    "boundary",
                    "implementation",
                    "max_abs",
                    "rel_l2",
                    "telemetry",
                    "telemetry_samples",
                    "cuda_activities",
                )
                if k in r
            }
            for r in rows
            if r["task_id"] == item["task_id"]
        ]
        html_parts.append(
            f"<details><summary>{esc(item['task_id'])}</summary><p>"
            + " · ".join(f'<a href="{esc(h)}">{esc(label)}</a>' for label, h in links)
            + "</p><pre>"
            + esc(json.dumps(info, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
        md_parts.append(
            "### " + md(item["task_id"]) + "\n\n" + " · ".join(f"[{label}]({h})" for label, h in links)
        )
    html_parts.append("</main></body></html>")
    files["index.html"] = "\n".join(html_parts)
    files["REPORT.md"] = "\n\n".join(md_parts) + "\n"
    exported = dict(
        summary,
        source_directory=os.path.relpath(batch_dir, output),
        comparison=dict(previous_run_id=previous_id, rule=COMPARISON_RULE, rows=comparisons),
    )
    if "backend_study" in exported:
        study = exported["backend_study"]
        exported["backend_study"] = dict(
            study, source_directory=os.path.relpath(study["source_directory"], output)
        )
    if error:
        exported["error"] = error
    files["results.json"] = json.dumps(exported, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    return files


def generate(batch_dir, output_dir, compare=None, backend=None):
    """Rebuild index.html, REPORT.md and results.json. Return summary or raise after writing a diagnostic."""
    batch_dir = Path(batch_dir).resolve()
    output = Path(output_dir).resolve()
    comparisons = []
    previous_id = None
    error = None
    try:
        summary = review_batch(batch_dir)
    except ReviewError as exc:
        error = str(exc)
        try:
            meta = load_json(batch_dir / "batch.json")
        except (ValueError, OSError):
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        summary = dict(
            schema="tilescope.summary.v2",
            run_id=str(meta.get("run_id", batch_dir.name)),
            mode=str(meta.get("mode", "unknown")),
            status="failed",
            coverage=exc.coverage,
            results=[],
            environment=meta.get("environment"),
            observation=meta.get("observation"),
        )
    if error is None and compare is not None:
        try:
            old = review_batch(compare)
            previous_id = old["run_id"]
            comparisons = compare_runs(summary, old)
        except ReviewError as exc:
            error = f"对照运行 {Path(compare).name} 未通过复核：{exc}"
            summary = dict(summary, status="failed", results=[])
    if backend is not None:
        from same_source.study import review as review_backend

        backend_dir = Path(backend).resolve()
        try:
            study = review_backend(backend_dir)
        except (ValueError, OSError, KeyError, TypeError, StopIteration, OverflowError) as exc:
            study = dict(status="measurement_invalid", reason=f"{type(exc).__name__}: {exc}", results=[])
        summary["backend_study"] = dict(study, source_directory=str(backend_dir))
    files = _render(batch_dir, output, summary, comparisons, previous_id, error)
    _write_files(output, files)
    if not summary["results"]:
        for name in ("latency.svg", "spread.svg"):
            (output / name).unlink(missing_ok=True)
    if error:
        raise ReviewError(error, summary["coverage"])
    return summary
