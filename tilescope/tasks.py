"""Finite, data-driven workload registry and CPU-only evidence checks."""
import bisect
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import struct

from round2.review import summarize as summarize_v1

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "catalog/tasks.json"
MODEL = "Qwen3.8-27B-NVFP4"
ACTIVATION_SEED = 381127
CHECKPOINT_HASH = "c473512c70eace07e2256fe9fd76596ac03e3295bee7d54cfb72676416afcc05"
WEIGHT_IDENTITY = ROOT / "catalog/tensor-identity.json"
LAYER = "model.language_model.layers.0"
SOURCE_TYPES = {"observed_runtime", "derived_config", "synthetic_diagnostic"}
VLLM = "vLLM 301d986473a0ffc1df563422e01eac4a1efd59e0"
FLASHINFER = "FlashInfer 064d9aa268fe8d2f4d7c9f3c5ca83ecb02fb2c9c"
CUTILE = "cuTile Python 1.4.0 / 1689841f14f44d0d98613727b5c420bfefd252cf"
NEW = {
    "bf16_ba": {
        "operation": "gemm", "module": LAYER + ".linear_attn.in_proj_ba",
        "weights": [LAYER + ".linear_attn.in_proj_b.weight", LAYER + ".linear_attn.in_proj_a.weight"],
        "shape_suffix": [96, 5120], "native": "torch.mm(out=), CUDA BLAS stock dispatch",
        "tile": "BF16 MMA 64x64x32; bounded loads/stores for partial M/N tiles",
        "semantics": "TP=1; concatenate B then A projection weights in preparation. X @ W.T, no bias, FP32 accumulation, BF16 output. Downstream b/a split and recurrent update excluded.",
        "reference": "CPU FP64 X @ W.T before BF16 store; both operands read from the actual BF16 buffers",
        "rtol": 0.009, "rms_atol": 0.002, "relative_l2_max": 0.004,
    },
    "gemma_rmsnorm": {
        "operation": "rmsnorm", "module": LAYER + ".input_layernorm",
        "weights": [LAYER + ".input_layernorm.weight"], "shape_suffix": [5120],
        "native": FLASHINFER + " norm.gemma_rmsnorm(out=); explicit CUDA JIT backend, FLASHINFER_USE_CUDA_NORM=1",
        "tile": "one row per tile, zero-masked 8192-lane FP32 reduction; N=5120",
        "semantics": "No residual. FP32 y=x*rsqrt(mean(x*x)+1e-6)*(1+w); final BF16 store. Weight offset is applied within both measured operations.",
        "reference": "CPU FP64 row mean square, epsilon 1e-6 and checkpoint (1+w) scale",
        "rtol": 0.006, "rms_atol": 0.0002, "relative_l2_max": 0.003,
    },
    "swiglu": {
        "operation": "swiglu", "module": LAYER + ".mlp.act_fn",
        "weights": [], "shape_suffix": [17408],
        "native": VLLM + " torch.ops._C.silu_and_mul; same operator as SiluAndMul.forward_cuda",
        "tile": "4x256 tile, BF16-rounded SiLU(gate) multiplied by up, BF16 store; bounded row tail",
        "semantics": "Packed BF16 gate_up rows; first H gates, last H up values. Full fused BF16(SiLU(gate))*up with BF16 output. Input/output projections excluded.",
        "reference": "CPU FP64 SiLU(gate), rounded to BF16 before FP64 multiplication by up; no weight parameter for this operation",
        "rtol": 0.006, "rms_atol": 0.0002, "relative_l2_max": 0.003,
    },
}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_hash(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


@lru_cache(maxsize=1)
def activation_period():
    """Little-endian finite BF16 pattern, replayable without Torch or a GPU."""
    rng = random.Random(ACTIVATION_SEED)
    words = []
    for _ in range(8191):
        bits = rng.getrandbits(32)
        sign = (bits >> 31) << 15
        exponent = 121 + ((bits >> 8) % 9)
        words.append(sign | (exponent << 7) | (bits & 127))
    return struct.pack(f"<{len(words)}H", *words)


def activation_bytes(elements):
    period = activation_period()
    count, remainder = divmod(2 * elements, len(period))
    return period * count + period[:remainder]


@lru_cache(maxsize=32)
def activation_digest(elements):
    return hashlib.sha256(activation_bytes(elements)).hexdigest()


def contiguous(shape, dtype="torch.bfloat16"):
    stride = [math.prod(shape[i+1:]) for i in range(len(shape))]
    return [dtype, shape, stride]


def resolve_task(task, rows=None):
    """Expand a catalog record into the exact supported replay contract."""
    task_id = task["task_id"]
    adapter = task["adapter"]
    need(isinstance(task_id, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", task_id),
         f"invalid task id {task_id!r}")
    m = task["rows"] if rows is None else rows
    need(type(m) is int and 1 <= m <= 2048, "rows must be an integer from 1 through 2048")
    groups = task.get("groups", [])
    need(isinstance(groups, list) and all(isinstance(x, str) and x for x in groups), "invalid groups")
    source = deepcopy(task["source"])
    need(source["kind"] in SOURCE_TYPES and source.get("evidence") and source.get("stage"),
         "source requires kind, stage and evidence")
    if m != task["rows"]:
        source = dict(kind="synthetic_diagnostic", stage="explicit row override",
                      evidence=source["evidence"], detail=f"Original registered rows: {task['rows']}")
    base = dict(schema=2, task_id=task_id, model=MODEL, adapter=adapter, rows=m, groups=groups,
                source=source, comparison_type="implementation_pair", checkpoint_sha256=CHECKPOINT_HASH,
                layout="row-major contiguous", bias=False, batch=1)
    protocol = dict(samples=50, operations_per_graph=16, direct_warmup=20,
                    paired_graph_warmup_seconds=1.0, order="alternating AB/BA",
                    cache="repeated buffers, warm", timer="CUDA external events inside graph", trim=False)
    if adapter in ("v1_fp8", "v1_nvfp4"):
        need(m == 64, "the archived quantized adapter accepts exactly 64 rows")
        v1_id = adapter.removeprefix("v1_")
        v1 = next(w for w in _jsonl(ROOT/"round2/WORKLOADS.jsonl") if w["id"] == v1_id)
        return dict(base, shape=v1["actual_probe_mnk"], padded_shape=v1["actual_probe_mnk"], operation="gemm",
                    phase=source["stage"], module=v1["module"], precision=v1_id.upper(),
                    inputs=v1["buffers"], outputs=v1["output"], semantics=v1["boundaries"],
                    scales=v1["scales_padding"], quantization=v1["weight_and_activation_rule"],
                    globals=v1["globals"], input_sha256=v1["input_sha256"], validation=v1["validation"],
                    implementations=v1["implementations"], boundaries=["core","full"],
                    protocol=protocol, activation_source=v1["activation_source"])
    need(adapter in NEW, f"unsupported registered adapter: {adapter}")
    spec = NEW[adapter]
    shape = [m, *spec["shape_suffix"]]
    width = shape[-1] * (2 if adapter == "swiglu" else 1)
    weights = json.loads(WEIGHT_IDENTITY.read_text())
    inputs = {"x": contiguous([m, width])}
    hashes = {"x": activation_digest(m * width)}
    for param in spec["weights"]:
        inputs[param] = contiguous(weights[param]["shape"])
        hashes[param] = weights[param]["sha256"]
    return dict(base, shape=shape, padded_shape=shape, phase=source["stage"], operation=spec["operation"],
                module=spec["module"], precision="BF16", inputs=inputs, outputs=contiguous([m, shape[1]]),
                semantics=spec["semantics"], scales=None, input_sha256=hashes, boundaries=["full"],
                activation_source="synthetic deterministic LE BF16 bit pattern, MT seed 381127, period 8191",
                validation={key: spec[key] for key in ("reference","rtol","rms_atol","relative_l2_max")},
                implementations={"native":spec["native"], "tile":CUTILE+"; "+spec["tile"]},
                protocol=protocol, weights=list(spec["weights"]))


def load_catalog(path=None):
    document = json.loads(Path(path or DEFAULT_CATALOG).read_text())
    need(document["schema"] == 2, "unsupported catalog schema")
    catalog = {}
    for raw in document["tasks"]:
        extra = set(raw) - {"task_id","adapter","rows","groups","source"}
        need(not extra, f"unknown task parameters: {sorted(extra)}")
        task = resolve_task(raw)
        need(task["task_id"] not in catalog, "duplicate task id: "+task["task_id"])
        catalog[task["task_id"]] = task
    return catalog


def select_tasks(catalog, selectors):
    if isinstance(selectors, str):
        selectors = [selectors]
    chosen = {}
    for selector in selectors:
        if selector.startswith("@"):
            group = selector[1:]
            matches = [v for v in catalog.values() if group == "all" or group in v["groups"]]
        else:
            matches = [catalog[selector]] if selector in catalog else []
        need(matches, "unknown task or group: "+selector)
        chosen.update((entry["task_id"],entry) for entry in matches)
    need(chosen, "select a task or @group")
    return list(chosen.values())


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.startswith("{")]


def _finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def _telemetry(telemetry, start, end):
    stamps = [r["unix_ns"] for r in telemetry]
    a, b = bisect.bisect_right(stamps, start)-1, bisect.bisect_left(stamps, end)
    need(0 <= a <= b < len(stamps), "telemetry does not bracket measurement")
    window = telemetry[a:b+1]
    activity = {}
    for field in ("gpu_busy_pct","gpc_hz","temperature_mc","vllm:num_requests_running","vllm:num_requests_waiting",
                  "vllm:prompt_tokens_total","vllm:generation_tokens_total"):
        vals = [r[field] for r in window if _finite(r.get(field))]
        activity[field] = [min(vals),max(vals)] if vals else None
    return len(window), activity


def review_run(task, raw_dir):
    """Recompute a task from raw emitted samples; emit no partial results on failure."""
    folder = Path(raw_dir)
    if task["adapter"].startswith("v1_"):
        need(resolve_task(task) == task, "unsupported or altered archived task specification")
        rows = summarize_v1(folder)
        need(all(r["mnk"] == task["shape"] for r in rows), "archived run does not match selected task")
        for row in rows:
            row.update(samples=50,shape=row["mnk"],elements_per_second=None)
        return rows

    need(resolve_task(task) == task, "unsupported or altered task specification")
    need((folder/"exit").read_text().strip() == "0", "nonzero worker exit")
    events = _jsonl(folder/"stdout")
    need(not any(e["phase"] == "failure" for e in events), "declared failure")
    inputs = [r for r in events if r["phase"] == "input"]
    need(len(inputs) == 1, "expected one input record")
    inp = inputs[0]
    need(inp["task_hash"] == canonical_hash(task) and inp["shape"] == task["shape"] and inp["layer"] == task["module"],
         "run task identity mismatch")
    need(inp["storage"] == task["inputs"] and inp["sha256"] == task["input_sha256"]
         and inp["output_storage"] == task["outputs"], "input identity or layout mismatch")
    p = task["protocol"]
    repeat, count = p["operations_per_graph"], p["samples"]
    samples = [r for r in events if r["phase"] == "sample"]
    checks = [r for r in events if r["phase"] == "correctness"]
    setup, = [r for r in events if r["phase"] == "setup"]
    need(setup["operations_per_graph"] == repeat, "recorded repeat differs from timing protocol")
    expected_checks = {(name,"full_"+stage) for name in ("native","tile") for stage in ("direct","graph","post")}
    need(len(checks) == 6 and {(r["name"],r["stage"]) for r in checks} == expected_checks, "missing output check")
    tolerance = task["validation"]
    for r in checks:
        need(r["passed"] is True and r["mismatches"] == r["nonfinite"] == 0
             and type(r["mismatches"]) is int and type(r["nonfinite"]) is int, "failed output")
        need(all(_finite(r[k]) and r[k] >= 0 for k in ("max_abs","relative_l2","reference_rms"))
             and r["reference_rms"] > 0 and r["relative_l2"] <= tolerance["relative_l2_max"],
             "numerical gate failed")
        # For n scalars, RMS(error) <= max(abs(error)) <= sqrt(n) * RMS(error).
        # Conservatively bound binary64 roundoff even for serial summation.
        # The n additions and six scalar operations bound the relative
        # perturbation by gamma=(n+6)*u/(1-(n+6)*u), with u=2**-53.
        count_out = math.prod(task["outputs"][1])
        nu = (count_out + 6) * 2.0**-53
        discount = 1.0 - nu / (1.0 - nu)
        error_rms = r["relative_l2"] * r["reference_rms"]
        l2_norm = math.sqrt(count_out) * error_rms
        need(error_rms * discount <= r["max_abs"] and r["max_abs"] * discount <= l2_norm,
             "max_abs inconsistent with reference RMS, relative L2 and element count")
    expected = [(i,n) for i in range(count) for n in (("tile","native") if i%2 else ("native","tile"))]
    need([(r["index"],r["name"]) for r in samples] == expected, "missing or out-of-order sample")
    need(all(r["boundary"]=="full" and _finite(r["graph_us"]) and r["graph_us"]>0 for r in samples),
         "invalid measurement interval")
    warm, = [r for r in events if r["phase"]=="warmup"]
    need(warm["seconds"] >= p["paired_graph_warmup_seconds"] and warm["grouped_replays"] > 0, "incomplete warmup")
    telemetry = _jsonl(folder/"telemetry.jsonl")
    ts0 = samples[0]["end_ns"] - round(samples[0]["graph_us"]*1000)
    observed, activity = _telemetry(telemetry, ts0, samples[-1]["end_ns"])
    summaries = [r for r in events if r["phase"]=="summary"]
    profiles = [r for r in events if r["phase"]=="profile"]
    need(len(summaries)==2 and len(profiles)==2, "incomplete execution evidence")
    tile_symbol = {"bf16_ba":"ba_projection", "gemma_rmsnorm":"gemma_rmsnorm", "swiglu":"swiglu"}[task["adapter"]]
    need(any((folder/"compiler").glob(tile_symbol+"*.tileirbc")) and any((folder/"compiler").glob(tile_symbol+"*.cubin")),
         "missing Tile IR/cubin artifacts")
    compiler_log = (folder/"stderr").read_text(errors="replace")
    need(any("Invoke tile compiler:" in line and tile_symbol in line and "--gpu-name sm_110" in line
             for line in compiler_log.splitlines()), "no matching Thor compiler invocation")
    results=[]
    for name in ("native","tile"):
        profile, = [r for r in profiles if r["name"]==name]
        need(profile["cuda_activities"], f"no measured {name} CUDA activity")
        if name == "tile":
            need(any(tile_symbol in symbol for symbol in profile["cuda_activities"]), "Tile launch absent in CUDA profile")
        vals = [r["graph_us"]/repeat for r in samples if r["name"]==name]
        median, sd = statistics.median(vals), statistics.stdev(vals)
        expected_stats=dict(median_us=median,mean_us=statistics.mean(vals),stdev_us=sd,min_us=min(vals),max_us=max(vals))
        logged, = [r for r in summaries if r["name"]==name]
        need(logged["boundary"]=="full" and logged["samples"]==count and all(
            _finite(logged[key]) and math.isclose(logged[key],value) for key,value in expected_stats.items()),
             "summary differs from raw samples")
        post = next(r for r in checks if r["name"]==name and r["stage"]=="full_post")
        shape = task["shape"]
        results.append(dict(run_id=folder.name, boundary="full", implementation=name, samples=count,
                            shape=shape, mnk=shape if task["operation"]=="gemm" else None,
                            median_us=median,sd_us=sd,min_us=min(vals),max_us=max(vals),
                            tflops=2*math.prod(shape)/median/1e6 if task["operation"]=="gemm" else None,
                            elements_per_second=shape[0]*shape[1]*1e6/median,
                            max_abs=post["max_abs"],rel_l2=post["relative_l2"],
                            telemetry_samples=observed,telemetry=activity, cuda_activities=profile["cuda_activities"]))
    return results
