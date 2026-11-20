"""Rebuild a portable report from explicitly requested Thor measurement directories."""
import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote

from tilescope.common import ROOT,read_json,write_json
from .resource_report import resource_report
from .service_review import comparison,review


def report(args):
    spec=getattr(args,"study",None)
    if spec is not None:
        keys=("hardware","board","calibration","calibration_support","trace","trace_cache","operator","backend","service")
        if any(getattr(args,k,None) for k in keys):raise ValueError("choose a study or explicit component directories")
        cfg=read_json(spec)
        if cfg.get("schema")!="tilescope.parity.study.v1" or set(cfg)-set(keys)-{"schema"}:
            raise ValueError("invalid study configuration")
        for k in keys[:-1]:
            if k in cfg:setattr(args,k,(spec.parent/cfg[k]).resolve())
        args.service=[(spec.parent/name).resolve() for name in cfg.get("service",[])]
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    previous = out/"results.json"
    recognized = previous.is_file() and read_json(previous).get("schema") == "tilescope.parity.report.v1"
    raw={}; failures={}
    def load(name,fn):
        try:raw[name]=fn()
        except (ValueError,KeyError,TypeError,IndexError,OSError,StopIteration,ArithmeticError) as e:
            failures[name]=f"{type(e).__name__}: {e}"
    scopes=[]
    if getattr(args,"operator",None):
        from tilescope.review import review_batch
        def check_operator():
            summary=review_batch(args.operator)
            # The outer report already links to the batch relative to the delivery.
            del summary["source_directory"]
            return summary
        load("operator",check_operator);scopes.append(("原始模型算子批次",args.operator))
    if getattr(args,"backend",None):
        from same_source.study import review as backend_review
        def check_backend():
            item=backend_review(args.backend)
            if item["status"]!="passed":raise ValueError(item)
            return item
        load("backend",check_backend);scopes.append(("同源 Triton 两后端",args.backend))
    if getattr(args,"board",None):
        from parity.board import review as board_review
        load("board",lambda:board_review(args.board));scopes.append(("板级只读传感器与 SMBIOS",args.board))
    if args.hardware:
        from parity.hardware.__main__ import review as hw_review
        load("hardware",lambda:hw_review(args.hardware))
        scopes.append(("硬件采集",args.hardware))
    if args.calibration:
        from parity.calibration.review import aggregate
        def read_calibration():
            result=aggregate(args.calibration)
            if result["rejected"]:raise ValueError(str(result["rejected"]))
            return result
        load("calibration",read_calibration);scopes.append(("全部校准进程",args.calibration))
    if getattr(args,"calibration_support",None):
        from parity.calibration.support_review import library_status,tile_status
        def support():
            folder=args.calibration_support
            return {"library":library_status(folder/"library"),"tile_smoke":tile_status(folder/"extra_smoke")}
        load("calibration_support",support)
        scopes.append(("接口能力与精度 smoke",args.calibration_support))
    if args.trace:
        from parity.trace_review import analyze
        def read_trace():
            result=analyze(args.trace)
            if result["errors"]:raise ValueError(str(result["errors"]))
            return result
        load("trace",read_trace);scopes.append(("私有服务、提交记录与 GPU trace",args.trace))
    if getattr(args,"trace_cache",None):
        from parity.trace_review import analyze,compare_layers
        def cache_results():
            if not args.trace:raise ValueError("supplement requires the main trace for identity comparison")
            result=analyze(args.trace_cache)
            if result["errors"]:raise ValueError(result["errors"])
            result["replay_identity"]=compare_layers(args.trace,args.trace_cache)
            return result
        load("trace_cache",cache_results);scopes.append(("缓存、选层数值与资源补充",args.trace_cache))
    for batch in args.service:
        def read_service(path=batch):
            result=review(path)
            result["resources"]=resource_report(path,result["rows"])
            if result["mode"]=="comparison":result["comparison"]=comparison(path)
            return result
        key="service/"+batch.name
        if key in raw or key in failures:raise ValueError("two batches have the same name")
        load(key,read_service);scopes.append((batch.name,batch))
    if not scopes:raise ValueError("select at least one recorded component")
    def rel(p):return quote(os.path.relpath(p.resolve(),out),safe="/-_.~")
    def val(v):
        if v is None:return "—"
        if type(v) is float:
            if 0<abs(v)<.001:return f"{v:.4g}"
            return f"{v:,.3f}".rstrip("0").rstrip(".")
        return str(v)
    def esc(v):return escape(val(v))
    html=[];md=[];navigation=[]
    def paragraph(s):
        html.append("<p>"+esc(s)+"</p>");md.append(s)
    def section(s):
        anchor=f"section-{len(navigation)+1}"
        navigation.append((anchor,s))
        html.append(f'<h2 id="{anchor}">'+esc(s)+"</h2>");md.append("## "+s)
    def table(head,rows):
        html.append('<div class="scroll"><table'+(' class="wide"' if len(head)>5 else "")+'><thead><tr>'+''.join('<th>'+esc(c)+'</th>' for c in head)+'</tr></thead><tbody>')
        lines=['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']
        for row in rows:
            html.append('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>')
            lines.append('| '+' | '.join(val(c).replace('|','\\|').replace('\n',' ') for c in row)+' |')
        html.append('</tbody></table></div>');md.append('\n'.join(lines))
    def links(pairs):
        html.append("<p>"+' · '.join(f'<a href="{escape(rel(p))}">{esc(t)}</a>' for t,p in pairs)+"</p>")
        md.append(' · '.join(f'[{t}]({rel(p)})' for t,p in pairs))
    title="Thor 探测能力与测量报告"
    html.append("<h1>"+title+"</h1>");md.append("# "+title)
    if failures:
        paragraph("所选记录未通过复核，本报告停止发布性能结果。请通过原始记录检查失败项。")
        table(["范围","失败原因"],failures.items())
        raw={}
    else:
        s=[v for k,v in raw.items() if k.startswith("service/")]
        count=sum(len(v["rows"]) for v in s)
        paragraph(f"所选 {len(raw)} 份采集通过离线复核。服务分项含 {count} 个完整请求（含单列的预热和试跑）。")
        matrix_batch=next((v for v in s if v["mode"]=="matrix"),None)
        if matrix_batch:
            formal=[r for r in matrix_batch["rows"] if r["role"]=="formal"]
            paragraph(f"在当前 Qwen3.8-27B-NVFP4 部署上，串行矩阵覆盖 {len(matrix_batch['groups'])} 个输入/输出组合，"
                      f"完成 {len(formal)} 次正式请求。可以按长度区分首输出、引擎 prefill 和后续生成时间。")
        if "trace" in raw:
            obs=raw["trace"]["requests"]["observed"]
            paragraph(f"在同权重且保留图执行的私有实例中，一个 {obs['input_tokens']}-token 输入生成了 {obs['output_tokens_received']} 个 token；"
                      f"引擎实际分 {len(obs['batch_sizes'])} 批提交。此处的提交时序与客户端收到文本的时刻可分别复核。")
        if "calibration" in raw:
            paragraph(f"合成算子通过 {len(raw['calibration']['groups'])} 组固定格式和规模的校验。"
                      "这些校准结果与真实模型权重探针分别列出，不能直接换算为整模型加速。")
        html.append("<!--SECTIONS-->")
        if "hardware" in raw:
            h=raw["hardware"];section("设备与运行条件")
            table(["设备","计算能力","SM","逻辑 CPU","PCI 设备","网口","已连接显示器","遥测点"],
                  [[h["gpu"],".".join(map(str,h["cc"])),*[h[k] for k in ("sm","cpu_objects","pci_objects","interface_count","connected_displays","telemetry_samples")]]])
            paragraph("设备属性来自查询，数值 smoke 和微架构曲线来自实机执行。商业型号与带宽规格按查询和厂商参照分别记录。")
            if "widest_sweep_ratio" in h:
                peak=h["widest_sweep_ratio"]
                paragraph(f"同一配置的轮中位耗时最大相差 {peak['ratio']:.2f} 倍"
                          f"（{peak['test']}，设置 {peak['setting']}）。")
            if h.get("sweep_coverage"):
                sensors=h["sweep_coverage"]
                if sensors["readable_pairs"]:
                    paragraph(f"微架构实验各配置边界共有 {sensors['boundary_samples']} 个采样点，"
                              f"其中 {sensors['readable_pairs']} 个同时读到 GPU 时钟与温度。")
                elif not sensors["inventory_samples_in_native"]:
                    paragraph("本次微架构实验期间没有同步的 GPU 频率或温度读数。轮间耗时差异的原因未确认。")
            if h.get("numerical_smoke"):
                table(["精度 smoke","逐元素核对数","期望值","最大绝对误差"],
                      [[name,v["count"],v["expected"],v["max_abs_error"]] for name,v in h["numerical_smoke"].items()])
                paragraph(f"此轮实验曲线中，warp 工作率最高点在 {h['observed_peak_work_rate_warps']} 个 warp，"
                          f"最慢的 shared memory 步幅为 {h['observed_slowest_bank_stride_words']} 个 32 位字。"
                          "这些点描述已记录的指令与启动条件；bank 数量、L1 物理容量和商业 SKU 仍需其他证据。")
                links([("原始曲线、精度验证、cluster 及占用率",args.hardware/"native.jsonl")])
            links([("属性与媒体枚举",args.hardware/'hardware.json'),("时序与传感器原始记录",args.hardware/'telemetry.jsonl')])
        if "board" in raw:
            board=raw["board"];memory=board["physical_ram"]
            paragraph(f"只读 SMBIOS 结构中发现 {len(memory['type17'])} 个 Type 17 内存设备描述。BPMP 实际公布的这些时钟可连续读取。")
            table(["时钟对象","最低 Hz","最高 Hz"],[[name,min(s["rates_hz"][name] for s in board["samples"]),
                max(s["rates_hz"][name] for s in board["samples"])] for name,state in board["clock_objects"].items() if state=="available"])
        if "operator" in raw:
            section("模型算子双路径对照")
            task=raw["operator"];table(["任务","边界","实现","样本数","中位 µs","标准差 µs","有效 TFLOP/s"],
                [[v[k] for k in ("task_id","boundary","implementation","samples","median_us","sd_us","tflops")] for v in task["results"]])
            paragraph("此处按同一实际输入、相同边界比较两份固定实现。GEMM 有效吞吐来自实际 2MNK。")
            links([("算子批次索引",args.operator/"batch.json")])
        if "backend" in raw:
            section("同一 Triton 源码的两种后端")
            table(["后端","样本数","中位 µs","标准差 µs"],[[r["backend"],r["samples"],r["median_us"],r["sd_us"]] for r in raw["backend"]["results"]])
        if "calibration" in raw:
            c=raw["calibration"];section("合成 GEMM 校准")
            paragraph("每种格式运行一个固定原生实现。各组是独立进程中位数的分布；模型算子双路径结果另见 v2 报告。")
            table(["输入与输出","M × N × K","独立轮数","中位 µs","轮间标准差 µs","有效 TFLOP/s"],
                  [[r["format"],f"{r['m']} × {r['n']} × {r['k']}",r["rounds"],r["median_us"],r["sd_round_median_us"],r["effective_tflops"]] for r in c["groups"]])
            from parity.calibration.review import exports
            exports(args.calibration,out/"calibration")
            links([(n,out/"calibration"/n) for n in ("groups.csv","groups.tex","scale.svg","rounds.svg")])
            for file,alt in [('scale.svg','规模曲线'),('rounds.svg','独立轮次分布')]:
                # The generated SVG comes from validated numeric results and fixed format identifiers.
                svg=(out/'calibration'/file).read_text()
                html.append('<figure>'+svg+'<figcaption>'+alt+'</figcaption></figure>')
        if "calibration_support" in raw:
            section("库接口与精度支持")
            support=raw["calibration_support"];lib=support["library"]
            paragraph(f"cuBLASLt 按固定描述符测试 {len(lib['cases'])} 项组合。"
                      f"返回可选算法的组合数为 {lib['heuristic_with_candidates']}，空算法实际启动并同步成功数为 {lib['synchronized_null_success']}。"
                      "未返回算法与数值正确、执行成功各自独立；该表没有性能排名。")
            table(["cuTile 小矩阵格式","输出元素数","最大绝对误差","超限元素"],
                  [[r["format"],r["elements"],r["max_abs"],r["mismatches"]] for r in support["tile_smoke"]])
        if s:
            section("受控请求")
            paragraph("输入来自当前 tokenizer。首输出是第一个非空 UTF-8 文本块。引擎阶段仅在一个请求与服务计数增量对应时报告。均值、中位数、样本标准差、极值和各组样本 P95 可从 JSON 或逐请求 CSV 复算。")
        for result in s:
            html.append("<h3>"+esc(result["mode"])+"</h3>");md.append("### "+result["mode"])
            if result["groups"]:
                rows_=[]
                for g in result["groups"]:
                    def med(field):return g[field]["median"] if g[field] else None
                    rows_.append([g["role"],g["n"],g["o"],g["concurrency"],g["total_s"]["count"],
                                  med("output_tokens"),med("ttfv_ms"),med("total_s"),med("computed_prefill_tokens_per_s"),med("decode_tokens_per_s")])
                table(["角色","输入","输出上限","并发","请求数","实际输出中位","首输出 ms","总时长 s","实际 prefill token/s","后续生成 token/s"],rows_)
            if result["mode"]=="matrix":
                changes=[(g["n"],g["o"],len(g["changed_inputs"])) for g in result["groups"] if g["changed_inputs"]]
                if changes:
                    paragraph("相同输入、seed 和输出上限的三轮请求中，以下场景出现输出 ID 或结束状态差异。"
                              "这些样本的流完整性和计数已经核验；调用顺序、缓存状态等因素对分歧的因果作用仍未隔离。")
                    table(["输入 token","输出上限","重复结果有差异的文本数"],changes)
                cache_sections=[]
                for g in result["groups"]:
                    for v in g["cache_strata"]:
                        cache_sections.append([g["n"],g["o"],v["prefix_hit_tokens"],v["computed_kv_tokens"],v["requests"],
                                               v["ttfv_ms"]["median"] if v["ttfv_ms"] else None])
                table(["输入","输出上限","前缀命中 token","实际计算 token","样本数","首输出中位 ms"],cache_sections)
            counters=result["traffic"]["counts"]
            if counters:paragraph("服务前后计数与自有工作量一致的窗口为 "
                +str(counters.get("counters_match_owned_requests",0))+" 个，混合或未能唯一归因的窗口为 "
                +str(counters.get("mixed_or_unresolved",0)+counters.get("insufficient_metrics",0))+" 个。其他设备进程的瞬时 GPU 竞争仍有采样盲区。")
            if result["mode"]=="soak":
                w=result["window"];t=result["resources"]["summary"]
                paragraph(f"观测 {w['elapsed_s']/60:.2f} 分钟，{t['samples']} 个资源采样点，失败尝试 {len(result['unsuccessful_attempts'])} 个，保护触发记录 {w['protection']}。温度、内存、RSS、fd、线程和独立功率轨按窗口记录。")
                if t["cuda_free_bytes"]:
                    paragraph(f"观察器 CUDA 可分配内存首次、最低、最后采样值分别为 "
                              f"{t['cuda_free_bytes']['first']/2**30:.2f}、{t['cuda_free_bytes']['min']/2**30:.2f}、"
                              f"{t['cuda_free_bytes']['last']/2**30:.2f} GiB。")
                table(["进程","PID","RSS 首/末 MiB","fd 首/末","线程首/末"],
                      [[v["name"],pid," / ".join(val(v["rss_bytes"][k]/2**20) for k in ("first","last")),
                        " / ".join(val(v["fd"][k]) for k in ("first","last")),
                        " / ".join(val(v["threads"][k]) for k in ("first","last"))] for pid,v in t["processes"].items()])
                table(["5 分钟段","实际样本","系统最低可用 GiB","GPU 最高 °C"],
                      [[r["index"]+1,r["samples"],r["mem_available_bytes"]["min"]/2**30,
                        r["gpu_mC"]["max"]/1000 if r["gpu_mC"] else None] for r in result["resources"]["five_minute_bins"]])
            if "comparison" in result:
                for row in result["comparison"]:
                    paragraph(f"{row['workload']}，比较状态 {row['status']}。"+row.get("reason","同一部署的四次独立客户端窗口用于 A/A 比较，进程间波动与稳定版本差异留待分别判断。"))
                    if row["status"]=="compatible":
                        table(["指标","B 各轮","C 各轮","均值变化 %","结论"],
                              [[k,", ".join(map(val,v["B"])),", ".join(map(val,v["C"])),v["change_pct"],v["observation"]] for k,v in row["metrics"].items()])
        if "trace" in raw:
            section("提交事件、执行路径与服务行为")
            trace=raw["trace"];paragraph(f"私有实例记录 {len(trace['requests'])} 个客户端请求。后端逐步提交记录与 SSE 到达事件分开保存。配置和不同容量的限制见原始 plan。")
            observed=trace["requests"].get("observed",{})
            if observed:
                paragraph(f"代表请求生成 {observed['output_tokens_received']} 个 token，实际提交批量为 {observed['batch_sizes']}。"
                          f"本次 target forward 和 MTP proposal 的 CUDA Graph replay 计数分别为 "
                          f"{observed['graph_replays'].get('target_forward',0)} 和 {observed['graph_replays'].get('mtp_proposal',0)}。")
                gap=observed["commit_interval_ms"]
                table(["时钟边界","间隔数","中位 ms","样本 P95 ms","最长 ms"],
                      [["引擎实际提交",gap.get("count"),gap.get("median"),gap.get("p95_nearest_rank"),gap.get("max")],
                       ["客户端 SSE",*(observed["sse_chunk_gaps_ms"].get(k) for k in ("count","median","p95_nearest_rank","max"))]])
            if "startup" in trace:
                boot=trace["startup"]
                paragraph(f"自有实例从首次进程采样到 HTTP 就绪用时 {boot['first_process_sample_to_http_ready_s']:.2f} 秒。"
                          "缓存来源和加载、编译日志已分开记录；系统文件页缓存未人为清空。")
            if "gpu" in trace:
                g=trace["gpu"]
                paragraph(f"被跟踪请求的墙钟 {g['http_wall_us']/1000:.2f} ms。GPU kernel 区间并集为 {g['any_gpu_kernel_us']/1000:.2f} ms，"
                          f"窗口内未见 kernel 的区间共 {g['no_observed_gpu_kernel_us']/1000:.2f} ms，其中可能包括 CPU、同步和传输。")
                table(["kernel 分类","调用次数","累计设备 ms","占累计 kernel 时间 %"],
                      [[v["category"],v["calls"],v["sum_us"]/1000,100*v["share"]] for v in g["category_by_kernel_sum"]])
            if "overhead" in trace:
                overhead=trace["overhead"]
                paragraph("按未记录事件、记录事件、再开启 profiler 的顺序，各采集一次同输入请求。实际耗时（ms）依次为 "
                          + "、".join(val(v) for v in overhead["samples_ms"])+"。这三次记录可用于复查观测开销，单轮无法区分开销与运行间波动。")
            cancellations=trace.get("behavior",{}).get("cancellation",{})
            if cancellations:
                table(["取消场景","客户端计时起点","至调度器 FINISHED_ABORTED（ms）"],
                      [[name, "开始关闭连接" if v["origin"]=="client_close_start_ns" else "断连后的客户端结束记录",
                        v["to_scheduler_abort_ms"]] for name,v in cancellations.items()])
                paragraph("终点取同一请求的引擎终止事件。HTTP 健康检查和后续请求用于验证服务恢复。"
                          "已提交 GPU 工作的最后完成时刻尚无逐请求证据。")
            for k in ["recovery","behavior","selected_layers"]:
                if k in trace:
                    value=trace[k]
                    if isinstance(value,dict):
                        data=[[n,a] for n,a in value.items() if isinstance(a,(int,float,bool)) or type(a) is str and len(a)<140]
                        if data:table(["项目","结果"],data)
            links([("生产等价配置与边界",args.trace/'plan.json'),("服务退出与归属",args.trace/'shutdown.json')])
    if "trace_cache" in raw:
        sup=raw["trace_cache"]["extended"]; section("相同前缀与两行选层复算")
        table(["请求","输入 token","缓存 token","实际计算 token"],
              [[name,*(sup[k].get(n) for n in ("num_prompt_tokens","num_cached_tokens","num_computed_tokens"))]
               for name,k in [("第一次","first_prefill"),("相同输入再次请求","repeat_prefill")]])
        numeric=sup["numerical_reference"]
        table(["选层","元素数","最大绝对误差","相对 L2","cosine","超限元素"],
              [[k,*(numeric[k][f] for f in ("n","max_abs","relative_l2","cosine","mismatches"))] for k in ("norm","ba")])
        paragraph("数值参照在同一 checkpoint 的真实两行嵌入上以 CPU FP64 计算。该局部重放有独立输入身份和舍入规则，完整模型与生产融合中间值仍有各自的验证边界。")
    section("来源、边界与复核")
    paragraph("实测平台为当前 Jetson Thor，模型为 Qwen3.8-27B-NVFP4。不同来源的规格、可见性、采样水位和执行时间各有定义。单次请求并发、热缓存算子或一次持续观察均不足以判定其他平台或整模型的稳定提速。")
    links(scopes+[("冻结功能定义",ROOT/"docs/goal-parity/PARITY.md"),("交付状态",ROOT/"STATUS.md"),
                  ("历史算子报告",ROOT/"evidence/v2/reports/final/index.html")])
    detail=ROOT/"parity/coverage.json"
    if detail.exists():
        section("完整功能对照")
        matrix=read_json(detail)
        paragraph("本表记录交付版本的验收范围。当前命令选择的采集目录及其复核状态列在报告首页。")
        table(["功能","实际子项","验证状态","限制"],[[r["id"],r["item"],r["status"],r.get("limit","")] for r in matrix])
        matrix_md=["# 冻结来源与逐项验收","", "状态对应交付版本。实际命令与原始证据依据仓库根目录给出。",""]
        headings=["编号","探测子项与限制","状态","实现及原始记录","复核入口","固定版本来源"]
        matrix_md.append("| "+" | ".join(headings)+" |")
        matrix_md.append("| "+" | ".join("---" for _ in headings)+" |")
        def artifact(path,label):
            target=ROOT/path
            return f"[{label}]({rel(target)})" if target.exists() else label+"（未归档）"
        for item in matrix:
            cells=[item["id"],item["item"]+"。"+item.get("limit",""),item["status"],
                   artifact(item["implementation"],"源码")+" · "+artifact(item["evidence"],"证据"),
                   "`"+item["review"]+"`"," · ".join(f"[{n+1}]({url})" for n,url in enumerate(item["source"]))]
            if item.get("supplementary_evidence"):cells[3]+=" · "+artifact(item["supplementary_evidence"],"补充")
            matrix_md.append("| "+" | ".join(v.replace("|","\\|") for v in cells)+" |")
        (out/"MATRIX.md").write_text("\n".join(matrix_md)+"\n")
        with (out/"coverage.csv").open("w",newline="") as csv_file:
            columns=["id","item","status","implementation","evidence","review","source_commit","limit"]
            writer=csv.DictWriter(csv_file,columns,extrasaction="ignore");writer.writeheader();writer.writerows(matrix)
        links([("可点击逐项验收表",out/"MATRIX.md"),("覆盖 CSV",out/"coverage.csv"),("带字段定位的来源索引",detail)])
    if recognized and "calibration" not in raw:
        for name in ("results.json","groups.csv","groups.tex","scale.svg","rounds.svg","view.csv","view.json"):
            (out/"calibration"/name).unlink(missing_ok=True)
    structured={"schema":"tilescope.parity.report.v1","status":"failed" if failures else "passed",
                "scope":[{"name":t,"directory":os.path.relpath(p.resolve(),out)} for t,p in scopes],
                "failures":failures,"sections":raw}
    write_json(out/"results.json",structured)
    with (out/"requests.csv").open("w",newline="") as f:
        fields=["batch","role","round","execution","input_sha256","output_sha256","n","o","concurrency","output_tokens",
                "ttfv_ms","total_s","client_tpot_ms","request_output_tokens_per_s","computed_prefill_tokens_per_s","decode_tokens_per_s"]
        writer=csv.DictWriter(f,fields,extrasaction="ignore");writer.writeheader()
        for k,res in raw.items():
            if k.startswith("service/"):
                writer.writerows(dict(r,batch=k) for r in res["rows"])
    links([("逐请求 CSV",out/"requests.csv"),("完整数据 JSON",out/"results.json")])
    style="body{background:#f4f6f5;color:#18312f;margin:0;font:16px/1.6 system-ui,sans-serif}main{margin:auto;max-width:1120px;background:white;padding:clamp(14px,4vw,64px)}h1{border-top:5px solid #00675e;padding-top:24px}h2{margin-top:38px;border-bottom:1px solid #b2c6bf}a{color:#00675e}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:6px 10px;border-bottom:1px solid #d6dfda;text-align:left}th{background:#ecf2ee}.scroll{overflow-x:auto}figure{margin:18px 0}svg{max-width:100%;height:auto}"
    style+="h1{font-size:clamp(1.5rem,4vw,2.2rem)}.wide{min-width:700px}main{min-width:0}nav{display:flex;gap:.35em 1em;flex-wrap:wrap;padding:.8em 0;border-block:1px solid #d6dfda}a:focus-visible{outline:2px solid #00675e;outline-offset:3px}pre{overflow-x:auto}td{overflow-wrap:anywhere}"
    jump='<nav aria-label="报告章节">'+"".join(f'<a href="#{a}">{esc(label)}</a>' for a,label in navigation)+"</nav>"
    html=[jump if chunk=="<!--SECTIONS-->" else chunk for chunk in html]
    (out/"index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>'+title+"</title><style>"+style+"</style><main>"+'\n'.join(html)+"</main></html>")
    (out/"REPORT.md").write_text("\n\n".join(md)+"\n")
    return int(bool(failures))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("hardware","board","calibration","trace","operator","backend"):p.add_argument("--"+key,type=Path)
    p.add_argument("--calibration-support",type=Path,help="parent of the library and extra_smoke raw directories")
    p.add_argument("--trace-cache",type=Path,help="supplemental cache, local layer oracle and recovery evidence")
    p.add_argument("--study",type=Path,help="resolve all component paths relative to this study JSON")
    p.add_argument("--service",type=Path,action="append",default=[]);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();sys.exit(report(a))
