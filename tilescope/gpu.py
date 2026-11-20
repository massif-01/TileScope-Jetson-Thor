"""Run one registered implementation pair. GPU imports are confined to the worker."""
import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from .tasks import ROOT, activation_bytes, canonical_hash, resolve_task, review_run
from .environment import DEFAULT_ENDPOINT, file_text, service_activity


def emit(phase, **fields):
    print(json.dumps(dict(phase=phase, **fields), allow_nan=False), flush=True)


def monitor():
    import pynvml as nv
    nv.nvmlInit()
    device = nv.nvmlDeviceGetHandleByIndex(0)
    temp = next((p.with_name("temp") for p in Path("/sys/class/thermal").glob("thermal_zone*/type")
                 if file_text(p) == "gpu-thermal"), None)
    endpoint = os.environ.get("TILESCOPE_ENDPOINT", DEFAULT_ENDPOINT)
    while True:
        sample = dict(unix_ns=time.time_ns())
        try:
            sample["gpu_busy_pct"] = nv.nvmlDeviceGetUtilizationRates(device).gpu
        except nv.NVMLError as exc:
            sample["nvml_error"] = str(exc)
        gpc = file_text("/sys/class/devfreq/gpu-gpc-0/cur_freq")
        temperature = file_text(temp) if temp else None
        sample.update(gpc_hz=int(gpc) if gpc else None,
                      temperature_mc=int(temperature) if temperature else None)
        sample.update(service_activity(endpoint))
        print(json.dumps(sample, allow_nan=False), flush=True)
        time.sleep(.08)


class CorrectnessError(ValueError):
    pass


def tensor_sha(t):
    import torch
    raw = t.detach().cpu().contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def prepare(task, stream):
    import torch
    import cuda.tile as ct
    from . import kernels

    weights = {}
    if task["weights"]:
        from safetensors import safe_open
        path = Path(os.environ["CHECKPOINT"]) / "model.safetensors"
        with safe_open(path, framework="pt", device="cpu") as file:
            for param in task["weights"]:
                weights[param] = file.get_tensor(param)
    rows, cols = task["inputs"]["x"][1]
    host = torch.frombuffer(bytearray(activation_bytes(rows*cols)), dtype=torch.bfloat16).reshape(rows,cols)
    x = host.cuda()
    tensors = {"x": x, **weights}
    hashes = {key:tensor_sha(value) for key,value in tensors.items()}
    storage = {key:[str(value.dtype),list(value.shape),list(value.stride())] for key,value in tensors.items()}
    if hashes != task["input_sha256"] or storage != task["inputs"]:
        raise ValueError("checkpoint or deterministic activation does not match registered input identity")
    output_shape = task["outputs"][1]
    emit("input", layer=task["module"], shape=task["shape"], sha256=hashes, storage=storage,
         output_storage=task["outputs"], task_hash=canonical_hash(task))
    host_fp64 = host.double()
    if task["adapter"] == "bf16_ba":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        wt_cpu = torch.cat([weights[name] for name in task["weights"]], dim=0)
        wt = wt_cpu.cuda()
        ref = host_fp64 @ wt_cpu.double().T

        def native(dst):
            torch.mm(x, wt.T, out=dst)

        def tile(dst):
            ct.launch(stream, (math.ceil(rows/64), math.ceil(output_shape[1]/64), 1),
                      kernels.ba_projection, (x, wt, dst))
    elif task["adapter"] == "gemma_rmsnorm":
        from flashinfer.norm import gemma_rmsnorm
        w_cpu = weights[task["weights"][0]]
        w = w_cpu.cuda()
        ref = host_fp64 * torch.rsqrt(host_fp64.square().mean(dim=1,keepdim=True) + 1e-6) * (1 + w_cpu.double())

        def native(dst):
            gemma_rmsnorm(x, w, eps=1e-6, out=dst)

        def tile(dst):
            ct.launch(stream, (rows, 1, 1), kernels.gemma_rmsnorm, (x,w,dst))
    elif task["adapter"] == "swiglu":
        from vllm import _custom_ops  # registers the CUDA extension used by SiluAndMul.forward_cuda
        gate, up = host_fp64.chunk(2, dim=1)
        activated = (gate * torch.sigmoid(gate)).to(torch.bfloat16)
        ref = activated.double() * up

        def native(dst):
            torch.ops._C.silu_and_mul(dst, x)

        def tile(dst):
            ct.launch(stream, (math.ceil(rows/4), output_shape[1]//256, 1), kernels.swiglu, (x,dst))
    else:
        raise ValueError(f"unknown native worker adapter {task['adapter']}")
    stream.synchronize()
    return {"native": native, "tile": tile}, ref


def verify(dst, ref, tolerance, name, phase):
    import torch
    actual = dst.cpu().double()
    if tuple(dst.shape) != tuple(ref.shape) or dst.dtype != torch.bfloat16 or not dst.is_contiguous():
        raise CorrectnessError(f"{name}: wrong output shape, dtype, or stride")
    rms = float(ref.square().mean().sqrt())
    err = (actual-ref).abs()
    nonfinite = int((~torch.isfinite(actual)).sum())
    mismatch = int((~torch.isfinite(actual) | (err > tolerance["rtol"]*ref.abs()+tolerance["rms_atol"]*rms)).sum())
    relative_l2 = float(err.square().mean().sqrt()) / rms
    passed = mismatch == 0 and math.isfinite(relative_l2) and relative_l2 <= tolerance["relative_l2_max"]
    emit("correctness", name=name, stage=phase, passed=passed, mismatches=mismatch, nonfinite=nonfinite,
         reference_rms=rms, max_abs=float(err.max()) if not nonfinite else None,
         relative_l2=relative_l2 if math.isfinite(relative_l2) else None)
    if not passed:
        raise CorrectnessError(f"{name}: {phase} exceeds predeclared error tolerance")


def benchmark(task):
    import torch
    torch.set_num_threads(2)
    stream = torch.cuda.Stream()
    protocol = task["protocol"]
    phases = {}
    with torch.cuda.stream(stream):
        start = time.perf_counter()
        launches, ref = prepare(task, stream)
        emit("setup", seconds=time.perf_counter()-start, operations_per_graph=protocol["operations_per_graph"])
        for name, launch in launches.items():
            dst = torch.full_like(ref, float("nan"), device="cuda", dtype=torch.bfloat16)
            start = time.perf_counter()
            launch(dst)
            stream.synchronize()
            emit("first_launch", name=name, boundary="full", seconds=time.perf_counter()-start)
            verify(dst, ref, task["validation"], name, "full_direct")
            for _ in range(protocol["direct_warmup"]):
                launch(dst)
            stream.synchronize()
            start = time.perf_counter()
            graph = torch.cuda.CUDAGraph()
            begin, end = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(2)]
            with torch.cuda.graph(graph, stream=stream):
                begin.record(stream)
                for _ in range(protocol["operations_per_graph"]):
                    launch(dst)
                end.record(stream)
            dst.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            emit("capture_and_first_replay", name=name, seconds=time.perf_counter()-start)
            verify(dst, ref, task["validation"], name, "full_graph")
            phases[name] = (graph,begin,end,dst)

        order = list(phases)
        warm = time.perf_counter()
        pairs = 0
        while time.perf_counter()-warm < protocol["paired_graph_warmup_seconds"]:
            for name in (order if pairs%2 == 0 else order[::-1]):
                phases[name][0].replay()
            stream.synchronize()
            pairs += 1
        emit("warmup", boundary="full", grouped_replays=pairs, seconds=time.perf_counter()-warm)
        values = {name:[] for name in order}
        for sample in range(protocol["samples"]):
            for name in (order if sample%2 == 0 else order[::-1]):
                graph,begin,end,_ = phases[name]
                graph.replay()
                stream.synchronize()
                graph_us = begin.elapsed_time(end)*1000
                if not math.isfinite(graph_us) or graph_us<=0:
                    raise ArithmeticError(f"invalid {name} event time")
                emit("sample", boundary="full", name=name, index=sample, graph_us=graph_us, end_ns=time.time_ns())
                values[name].append(graph_us/protocol["operations_per_graph"])
        for name, data in values.items():
            verify(phases[name][3], ref, task["validation"], name, "full_post")
            emit("summary", boundary="full", name=name, samples=len(data), median_us=statistics.median(data),
                 mean_us=statistics.mean(data), stdev_us=statistics.stdev(data), min_us=min(data), max_us=max(data))

        from torch.profiler import profile, ProfilerActivity
        for name, launch in launches.items():
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as profiler:
                launch(phases[name][3])
                stream.synchronize()
            activities = [e.name for e in profiler.events() if str(e.device_type).endswith("CUDA")]
            emit("profile", name=name, cuda_activities=activities)


def run_stage(task, tf, out):
    raw = out / "native"
    py = os.environ.get("PROBE_PYTHON", sys.executable)
    env = os.environ.copy()
    env.setdefault("CHECKPOINT", "/opt/models/Qwen3.8-27B-NVFP4")
    if task["adapter"].startswith("v1_"):
        if (env.get("TILESCOPE_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/") != DEFAULT_ENDPOINT or
                Path(env.get("CUDA_HOME", "/usr/local/cuda-13.3")).resolve() != Path("/usr/local/cuda-13.3").resolve()):
            return ("unsupported", "archived adapter pins CUDA 13.3 and telemetry at "+DEFAULT_ENDPOINT,
                    "historical protocol")
        cmd = ["bash", str(ROOT/"round2/run.sh"), task["adapter"][3:], str(raw)]
        (out/"gpu-command.json").write_text(json.dumps(cmd)+"\n")
        status = subprocess.run(cmd, env=env).returncode
    else:
        # Unique attempt directories protect older samples on restart.
        raw.mkdir()
        compiler = raw/"compiler"
        compiler.mkdir()
        env.setdefault("CUDA_HOME", "/usr/local/cuda-13.3")
        env["PATH"] = str(Path(env["CUDA_HOME"])/"bin")+":"+env.get("PATH","")
        env.update(CUDA_TILE_CACHE_DIR="off", CUDA_TILE_TEMP_DIR=str(compiler), CUDA_TILE_DUMP_BYTECODE=str(compiler))
        if task["adapter"] == "gemma_rmsnorm":
            env.update(FLASHINFER_USE_CUDA_NORM="1", MAX_JOBS="1",
                       FLASHINFER_WORKSPACE_BASE=str(Path.home()/".cache"/"tilescope"/"norm-build"))
        cmd = [py, "-u", "-m", "tilescope.gpu", "--task-json",str(tf),"--out",str(out),"--worker"]
        (out/"gpu-command.json").write_text(json.dumps(cmd)+"\n")
        with (raw/"telemetry.jsonl").open("w") as tel, (raw/"monitor.stderr").open("w") as monitor_err:
            observer = subprocess.Popen([py,"-u","-c","from tilescope.gpu import monitor; monitor()"],
                                        cwd=ROOT, env=env, stdout=tel, stderr=monitor_err)
            try:
                with (raw/"stdout").open("w") as stdout,(raw/"stderr").open("w") as stderr:
                    try:
                        status = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=360).returncode
                    except subprocess.TimeoutExpired:
                        status = 124
                        print("worker exceeded 360s", file=stderr)
                # Let the external observer acquire one post-measurement point before stopping.
                time.sleep(.35)
            finally:
                observer.terminate()
                observer.wait()
        (raw/"exit").write_text(str(status)+"\n")
    if status != 0:
        reason = (raw/"stderr").read_text(errors="replace")[-2000:] if (raw/"stderr").exists() else str(status)
        if "CorrectnessError" in reason or "numerical gate failed" in reason:
            return "correctness_failed",reason,"correctness"
        if "ModuleNotFoundError" in reason:
            return "environment_blocked",reason,"dependency"
        return "execution_failed",reason,"worker"
    try:
        review_run(task, raw)
    except (OSError,KeyError,TypeError,ValueError,StopIteration) as exc:
        return "measurement_invalid",str(exc),"offline_review"
    return "passed","","complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    task = json.loads(args.task_json.read_text())
    out = args.out.resolve()
    if args.worker:
        logging.basicConfig(level=logging.WARNING)
        logging.getLogger("cuda.tile").setLevel(logging.DEBUG)
        try:
            benchmark(task)
        except Exception as err:
            emit("failure", detail=str(err))
            raise
        return 0
    out.mkdir(parents=True,exist_ok=True)
    previous_status = out/"status.json"
    if (out/"native").exists() or (previous_status.exists()
                                   and json.loads(previous_status.read_text()).get("status") != "not_run"):
        parser.exit(2, "attempt already has execution evidence; choose a new attempt\n")
    try:
        if resolve_task(task) != task:
            raise ValueError("descriptor disagrees with its registered adapter")
        status,reason,stage = run_stage(task, args.task_json.resolve(), out)
    except (ValueError,KeyError) as exc:
        status,reason,stage = "invalid_input",str(exc),"descriptor"
    except (OSError,subprocess.SubprocessError) as exc:
        status,reason,stage = "execution_failed",str(exc),"subprocess"
    (out/"status.json").write_text(json.dumps(dict(status=status,reason=reason,stage=stage),indent=2)+"\n")
    return 0 if status=="passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
