"""Replay one checkpoint projection. The RUN directory is unique per process."""
import argparse
import hashlib
import json
import logging
import math
import statistics
import time
import traceback
from pathlib import Path

import torch
import cuda.tile as ct
from safetensors import safe_open
from vllm import _custom_ops as ops
from kernels import fp8_projection, nvfp4_projection

M = 64
SEED = 381127
LAYERS = {"fp8": "3.self_attn.o_proj", "nvfp4": "0.mlp.down_proj"}


def emit(phase, **values):
    print(json.dumps(dict(phase=phase, **values), allow_nan=False), flush=True)


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def unpack4(packed, scales):
    lookup = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                          dtype=torch.float64)
    indices = torch.stack((packed & 15, packed >> 4), dim=-1).long()
    values = lookup[indices].reshape(packed.shape[0], -1, 16)
    return (values * scales.double().unsqueeze(-1)).flatten(1)


def unswizzle(raw, rows, groups):
    blocks = raw.cpu().reshape(-1, groups // 4, 32, 4, 4)
    return blocks.permute(0, 3, 2, 1, 4).reshape(-1, groups)[:rows]


def prepare(kind, checkpoint, stream):
    prefix = "model.language_model.layers." + LAYERS[kind]
    with safe_open(checkpoint / "model.safetensors", framework="pt", device="cpu") as f:
        def load(suffix):
            return f.get_tensor(prefix + "." + suffix)
        weight = load("weight" if kind == "fp8" else "weight_packed")
        scales = load("weight_scale")
        if kind == "nvfp4":
            input_div = load("input_global_scale")
            weight_div = load("weight_global_scale")
    n, packed_k = weight.shape
    k = packed_k if kind == "fp8" else 2 * packed_k
    x = torch.randn((M, k), generator=torch.Generator().manual_seed(SEED)).bfloat16().cuda()
    wt = weight.cuda()
    if kind == "fp8":
        sw = scales.float().cuda()
        def quant():
            return ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
        def native(a, sa, dst):
            torch.ops._C.cutlass_scaled_mm(dst, a, wt.T, sa, sw, None)
        def tile(a, sa, dst):
            ct.launch(stream, (1, n // 64, 1), fp8_projection, (a, wt, sa, sw, dst))
    else:
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
        from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm_out
        inv = input_div.cuda()
        alpha = (inv.reciprocal() * weight_div.cuda().reciprocal()).float()
        sw = swizzle_blockscale(scales.cuda())
        def quant():
            return ops.scaled_fp4_quant(x, inv, is_sf_swizzled_layout=True,
                                        backend="flashinfer-cutlass", padded_n=k)
        def native(a, sa, dst):
            flashinfer_scaled_fp4_mm_out(a, wt.T, sa, sw.T, alpha, dst, torch.bfloat16, False, "cutlass")
        def tile(a, sa, dst):
            ct.launch(stream, (1, n // 64, 1), nvfp4_projection,
                      (a, wt, sa.flatten(), sw.flatten(), alpha, dst))
    a, sa = quant()
    stream.synchronize()
    if kind == "fp8":
        lhs = a.cpu().double() * sa.cpu().double()
    else:
        linear_sa = unswizzle(sa, M, k // 16)
        lhs = unpack4(a.cpu(), linear_sa)
        assert torch.equal(unswizzle(sw, n, k // 16).view(torch.uint8), scales.view(torch.uint8))
        emit("checkpoint_scales", input_divisor=input_div.item(), weight_divisor=weight_div.item(),
             alpha=alpha.item(), activation_scale_min=linear_sa.float().min().item(),
             activation_scale_max=linear_sa.float().max().item())
    ref = torch.empty((M, n), dtype=torch.float64)
    for j in range(0, n, 128):
        w = weight[j:j + 128]
        s = scales[j:j + 128]
        rhs = w.double() * s.double() if kind == "fp8" else unpack4(w, s)
        result = lhs @ rhs.T
        ref[:, j:j + 128] = result if kind == "fp8" else result * alpha.item()
    emit("input", shape=[M,n,k], layer=prefix, seed=SEED,
         sha256={name: digest(t) for name,t in (("x",x),("a",a),("sa",sa),("wt",wt),("sw",sw))},
         storage={name: [str(t.dtype),list(t.shape),list(t.stride())]
                  for name,t in (("x",x),("a",a),("sa",sa),("wt",wt),("sw",sw))})
    return quant, a, sa, {"native": native, "tile": tile}, ref


def verify(output, reference, name, stage):
    actual = output.cpu().double()
    rms = reference.square().mean().sqrt().item()
    err = (actual - reference).abs()
    invalid = ~torch.isfinite(actual)
    mismatch = int(((err > 0.009 * reference.abs() + 0.002 * rms) | invalid).sum())
    rel_l2 = err.square().mean().sqrt().item() / rms
    passed = mismatch == 0 and rel_l2 <= .004
    emit("correctness", name=name, stage=stage, passed=passed, mismatches=mismatch,
         nonfinite=int(invalid.sum()), reference_rms=rms,
         max_abs=float(err.max()) if not invalid.any() else None,
         relative_l2=rel_l2 if math.isfinite(rel_l2) else None)
    if not passed:
        raise ValueError("numerical gate failed")


def run(args):
    torch.set_num_threads(2)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        start = time.perf_counter()
        quant, a, sa, implementations, ref = prepare(args.task, args.checkpoint, stream)
        emit("setup", seconds=time.perf_counter()-start)
        for boundary in ("core", "full"):
            graphs = {}
            for name, kernel in implementations.items():
                if args.only and args.only != name:
                    continue
                dst = torch.full(ref.shape, float("nan"), device="cuda", dtype=torch.bfloat16)
                def launch():
                    aq, scale = (a, sa) if boundary == "core" else quant()
                    kernel(aq, scale, dst)
                stage = "launch_or_compile"
                try:
                    start = time.perf_counter()
                    launch()
                    stream.synchronize()
                    emit("first_launch", name=name, boundary=boundary, seconds=time.perf_counter()-start)
                    stage = "correctness"
                    verify(dst, ref, name, boundary + "_direct")
                    stage = "warmup"
                    for _ in range(20):
                        launch()
                    stream.synchronize()
                    stage = "graph_capture"
                    graph = torch.cuda.CUDAGraph()
                    begin, end = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
                    with torch.cuda.graph(graph, stream=stream):
                        begin.record(stream)
                        for _ in range(16):
                            launch()
                        end.record(stream)
                    dst.fill_(float("nan"))
                    graph.replay()
                    stream.synchronize()
                    stage = "correctness"
                    verify(dst, ref, name, boundary + "_graph")
                    graphs[name] = (graph,begin,end,dst)
                except Exception:
                    emit("failure", name=name, boundary=boundary, stage=stage)
                    traceback.print_exc()
                    raise
            order = list(graphs)
            start = time.perf_counter()
            pairs = 0
            while time.perf_counter()-start < 1.0:
                for name in (order if pairs % 2 == 0 else order[::-1]):
                    graphs[name][0].replay()
                stream.synchronize()
                pairs += 1
            emit("warmup", boundary=boundary, grouped_replays=pairs, seconds=time.perf_counter()-start)
            values = {name: [] for name in order}
            for sample in range(50):
                for name in (order if sample % 2 == 0 else order[::-1]):
                    graph,begin,end,_ = graphs[name]
                    graph.replay()
                    stream.synchronize()
                    us = begin.elapsed_time(end) * 1000
                    if not (math.isfinite(us) and us > 0):
                        raise ValueError("invalid CUDA event time")
                    emit("sample", boundary=boundary, name=name, index=sample, graph_us=us, end_ns=time.time_ns())
                    values[name].append(us/16)
            for name in order:
                verify(graphs[name][3], ref, name, boundary + "_post")
            for name, vals in values.items():
                emit("summary", boundary=boundary, name=name, samples=len(vals), median_us=statistics.median(vals),
                     mean_us=statistics.mean(vals), stdev_us=statistics.stdev(vals), min_us=min(vals), max_us=max(vals))
        # Keep profiling separate from every measured window.
        try:
            from torch.profiler import profile, ProfilerActivity
            for name,kernel in implementations.items():
                if args.only and args.only != name:
                    continue
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    aq,scale = quant()
                    kernel(aq,scale,dst)
                    stream.synchronize()
                names=[e.name for e in prof.events() if str(e.device_type).endswith("CUDA")]
                emit("profile", name=name, cuda_activities=names)
        except Exception as error:
            emit("profile_unavailable", reason=str(error))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=LAYERS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--only", choices=["native","tile"])
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("cuda.tile").setLevel(logging.DEBUG)
    run(parser.parse_args())
