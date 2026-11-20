"""Compare BF16 GEMMs in one CUDA context, with shared inputs and graph timing."""

import argparse
import ctypes as c
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import statistics
import time
from pathlib import Path


SHAPES = [(512, 512, 512), (2048, 2048, 2048), (40, 5120, 2048)]
WARMUP = 20
LAUNCHES = 16
SAMPLES = 50
SEED = 41017


def emit(**record):
    print(json.dumps(record, allow_nan=False), flush=True)


def checked(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation}: bridge returned {status}")


def bridge(path):
    lib = c.CDLL(str(path.resolve()))
    ptr = c.c_void_p
    lib.tilescope_prepare.argtypes = [c.c_int] * 3 + [ptr] * 5 + [c.POINTER(ptr)]
    lib.tilescope_launch.argtypes = [ptr, ptr]
    lib.tilescope_destroy.argtypes = [ptr]
    lib.tilescope_versions.argtypes = [c.POINTER(c.c_int)] * 2
    return lib


def fixture(torch, shape):
    m, n, k = shape
    rng = torch.Generator().manual_seed(SEED)
    a = (torch.randint(-63, 64, (m, k), generator=rng) / 16).to(torch.bfloat16)
    bt = (torch.randint(-63, 64, (n, k), generator=rng) / 16).to(torch.bfloat16)
    return a, bt, a.double() @ bt.double().T


def verify(torch, output, reference, backend):
    actual = output.T.cpu().to(torch.float64)
    finite = torch.isfinite(actual)
    error = torch.where(finite, (actual - reference).abs(), float("inf"))
    mismatch = int((error != 0).sum())
    max_abs = float(error.max())
    emit(phase="correctness", backend=backend, correctness="failed" if mismatch else "passed",
         mismatches=mismatch, nonfinite=int((~finite).sum()),
         max_abs=max_abs if math.isfinite(max_abs) else None, atol=0, rtol=0)
    return mismatch == 0


def compare(torch, ct, kernel, lib, shape, graph_warmup_seconds):
    m, n, k = shape
    emit(phase="configuration", mnk=shape, alpha=1, beta=0, seed=SEED,
         strides={"a": [k, 1], "b": [1, k], "d": [1, m]},
         warmup=WARMUP, launches_per_graph=LAUNCHES, samples=SAMPLES,
         graph_warmup_target_seconds=graph_warmup_seconds,
         tile_mnk=[64, 64, 32], cutlass_tile_mnk=[256, 128, 64],
         cutlass_cluster_mnk=[2, 2, 1])
    torch.set_num_threads(2)
    torch.cuda.init()
    if torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("Expected Thor compute capability 11.0")
    driver, runtime = c.c_int(), c.c_int()
    checked(lib.tilescope_versions(c.byref(driver), c.byref(runtime)), "versions")
    emit(phase="environment", arch=list(torch.cuda.get_device_capability()),
         gpu=torch.cuda.get_device_name(), python=platform.python_version(),
         torch=torch.__version__, torch_cuda_build=torch.version.cuda,
         cuda_tile=importlib.metadata.version("cuda-tile"),
         cuda_driver_api=driver.value, cuda_runtime_api=runtime.value)

    start = time.perf_counter()
    a_cpu, bt_cpu, reference = fixture(torch, shape)
    hashes = {
        name: hashlib.sha256(tensor.view(torch.int16).numpy().tobytes()).hexdigest()
        for name, tensor in (("a", a_cpu), ("bt", bt_cpu))
    }
    cpu_seconds = time.perf_counter() - start
    start = time.perf_counter()
    a, bt = a_cpu.cuda(), bt_cpu.cuda()
    c_zero = torch.zeros((n, m), dtype=torch.float32, device="cuda")
    outputs = {name: torch.full_like(c_zero, float("nan")) for name in ("cutlass", "tile")}
    torch.cuda.synchronize()
    emit(phase="data_preparation", input_bf16_sha256=hashes, cpu_seconds=cpu_seconds,
         transfer_and_allocation_seconds=time.perf_counter() - start)

    stream = torch.cuda.Stream()
    handle = c.c_void_p()
    with torch.cuda.stream(stream):
        start = time.perf_counter()
        checked(lib.tilescope_prepare(m, n, k, a.data_ptr(), bt.data_ptr(), c_zero.data_ptr(),
                                     outputs["cutlass"].data_ptr(), stream.cuda_stream,
                                     c.byref(handle)), "prepare")
        try:
            stream.synchronize()
            emit(phase="setup", backend="cutlass", seconds=time.perf_counter() - start)

            def launch_cutlass():
                checked(lib.tilescope_launch(handle, stream.cuda_stream), "launch cutlass")

            def launch_tile():
                ct.launch(stream, (math.ceil(m / 64), math.ceil(n / 64), 1),
                          kernel, (a, bt, outputs["tile"]))

            launches = {"cutlass": launch_cutlass, "tile": launch_tile}
            correctness = []
            for name, launch in launches.items():
                start = time.perf_counter()
                launch()
                stream.synchronize()
                emit(phase="first_launch", backend=name, seconds=time.perf_counter() - start)
                correctness.append(verify(torch, outputs[name], reference, name))
            if not all(correctness):
                raise RuntimeError("GEMM verification failed")

            for _ in range(WARMUP):
                for launch in launches.values():
                    launch()
            stream.synchronize()

            graphs = {}
            for name, launch in launches.items():
                start = time.perf_counter()
                graph = torch.cuda.CUDAGraph()
                begin = torch.cuda.Event(enable_timing=True, external=True)
                end = torch.cuda.Event(enable_timing=True, external=True)
                with torch.cuda.graph(graph, stream=stream):
                    begin.record(stream)
                    for _ in range(LAUNCHES):
                        launch()
                    end.record(stream)
                # Require the captured graph to write the output on its first replay.
                outputs[name].fill_(float("nan"))
                graph.replay()
                stream.synchronize()
                graphs[name] = graph, begin, end
                emit(phase="graph_setup", backend=name, seconds=time.perf_counter() - start)
                if not verify(torch, outputs[name], reference, name + "_first_graph"):
                    raise RuntimeError("initial graph GEMM verification failed")

            timings = {name: [] for name in graphs}
            order = list(graphs)
            warmup_start = time.perf_counter()
            pairs = 0
            while time.perf_counter() - warmup_start < graph_warmup_seconds:
                for name in (order if pairs % 2 == 0 else order[::-1]):
                    graphs[name][0].replay()
                stream.synchronize()
                pairs += 1
            emit(phase="graph_warmup", pairs=pairs, seconds=time.perf_counter() - warmup_start)

            for sample in range(SAMPLES):
                for name in (order if sample % 2 == 0 else order[::-1]):
                    graph, begin, end = graphs[name]
                    graph.replay()
                    stream.synchronize()
                    us = begin.elapsed_time(end) * 1000
                    if not math.isfinite(us) or us <= 0:
                        raise RuntimeError(f"invalid event duration for {name}: {us}")
                    emit(phase="sample", backend=name, sample=sample, graph_us=us,
                         end_unix_ns=time.time_ns())
                    timings[name].append(us / LAUNCHES)

            final = [verify(torch, outputs[name], reference, name + "_post_graph") for name in graphs]
            if not all(final):
                raise RuntimeError("post-graph GEMM verification failed")
            for name, vals in timings.items():
                median = statistics.median(vals)
                emit(phase="summary", backend=name, count=len(vals), median_us=median,
                     mean_us=statistics.mean(vals), stdev_us=statistics.stdev(vals),
                     min_us=min(vals), max_us=max(vals), median_tflops=2 * m * n * k / median / 1e6)
        finally:
            stream.synchronize()
            checked(lib.tilescope_destroy(handle), "destroy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--shape-index", type=int, choices=range(len(SHAPES)), default=0)
    parser.add_argument("--graph-warmup-seconds", type=float, default=1.0)
    args = parser.parse_args()
    if not math.isfinite(args.graph_warmup_seconds) or args.graph_warmup_seconds < 0:
        parser.error("graph warmup must be finite and nonnegative")

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("cuda.tile").setLevel(logging.DEBUG)
    import torch
    import cuda.tile as ct
    from tile_kernel import bf16_gemm

    compare(torch, ct, bf16_gemm, bridge(args.bridge), SHAPES[args.shape_index],
            args.graph_warmup_seconds)


if __name__ == "__main__":
    main()
