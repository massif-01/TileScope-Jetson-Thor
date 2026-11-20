"""Hardware inventory, repeatable telemetry, bounded smoke and experimental curves."""

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

from tilescope.common import device_slot, sha256, write_json
from . import native_window, system, telemetry

HERE = Path(__file__).resolve().parent
PIN = "481668514756b8decda421290c32dbbc130b897f"


def native(out, repeats, sweeps, lock_fd, sensors, interval):
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.3"))
    with tempfile.TemporaryDirectory(prefix="tilescope-hw-") as tmp:
        binary = Path(tmp) / "experiments"
        commands = [
            (
                "build",
                [
                    str(cuda / "bin/nvcc"),
                    "-std=c++17",
                    "-O3",
                    "-lineinfo",
                    "-arch=sm_110a",
                    str(HERE / "experiments.cu"),
                    "-o",
                    str(binary),
                ],
            ),
            ("sass", [str(cuda / "bin/cuobjdump"), "--dump-sass", str(binary)]),
            ("native", [str(binary), str(repeats), str(sweeps), sensors["gpc_hz"], sensors["gpu_mC"] or ""]),
        ]
        for tag, argv in commands:
            start = time.time_ns()
            monotonic = {}
            output = out / ("native.jsonl" if tag == "native" else tag + ".stdout")
            with (
                output.open("wb") as stdout,
                (out / (tag + ".stderr")).open("wb") as stderr,
            ):
                if tag == "native":
                    code, monotonic = native_window.run(argv, stdout, stderr, lock_fd,
                        out/"native-window.jsonl", sensors, interval)
                else:
                    code = subprocess.run(argv, stdout=stdout, stderr=stderr,
                                          timeout=180, pass_fds=(lock_fd,)).returncode
            write_json(
                out / (tag + "-receipt.json"),
                dict(
                    argv=argv,
                    exit=code,
                    **monotonic,
                    start_unix_ns=start,
                    end_unix_ns=time.time_ns(),
                    source_sha256=sha256(HERE / "experiments.cu"),
                ),
            )
            if code:
                raise ValueError(f"{tag} exited {code}; see {tag}.stderr")
        # Machine instructions are retained in full, compressed; hash the executable before temporary cleanup.
        import gzip

        src = out / "sass.stdout"
        (out / "sass.gz").write_bytes(gzip.compress(src.read_bytes(), mtime=0))
        src.unlink()
        (out / "executable.sha256").write_text(sha256(binary) + "\n")


def capture(args):
    out = args.out
    if out.exists():
        raise ValueError(f"new directory required: {out}")
    out.mkdir(parents=True)
    config = {
        "schema": "tilescope.hardware.v1",
        "mode": args.mode,
        "telemetry_samples": args.samples,
        "telemetry_interval_s": args.interval,
        "micro_repeats": args.repeats,
        "micro_sweeps": args.sweeps,
        "requested_telemetry_duration_s": args.duration,
        "upstream_sha": PIN,
        "started_unix_ns": time.time_ns(),
        "source_sha256": {p.name: sha256(p) for p in HERE.glob("*") if p.is_file()},
    }
    write_json(out / "plan.json", config)
    with device_slot() as lock_fd:
        config["lock_acquired_unix_ns"] = time.time_ns()
        write_json(out / "plan.json", config)
        data = system.collect()
        data["schema"] = config["schema"]
        data["timestamp_unix_ns"] = time.time_ns()
        data["specification_reference"] = json.loads((HERE / "specs.json").read_text())
        attr = data["cuda"]["devices"][0]["attributes"]
        cc = attr["CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR"]["value"]
        cluster = attr["CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH"]["value"]
        data["mechanism_inference"] = {
            "tmem": {
                "kind": "inference_from_cc_and_cluster_attribute",
                "candidate": bool(cc >= 11 and cluster)
                if cc is not None and cluster is not None
                else None,
                "capacity_bytes": None,
                "tcgen05_copy_tested": False,
            },
            "unit_counts_per_sm": {"kind": "unmeasured", "value": None},
            "nvfp4": {
                "kind": "inherited_correctness_and_compiled_path",
                "source": "round2/evidence/runs/nvfp4-02/stdout",
                "cubin_evidence": "round2/evidence/build-proof/fp4-native.sass.gz",
                "upstream_tc_probe_label": "assumed",
            },
        }
        write_json(out / "hardware.json", data)
        with (
            (out / "telemetry.jsonl").open("w") as f,
            (out / "tegrastats.stderr").open("wb") as err,
        ):
            for r in telemetry.collect(args.samples, args.interval, err):
                f.write(json.dumps(r, allow_nan=False) + "\n")
                f.flush()
        if args.mode == "full":
            temperature = next((str(p.with_name("temp")) for p in Path("/sys/class/thermal").glob("thermal_zone*/type")
                                if system.reading(p)["value"] == "gpu-thermal"), None)
            sensors = {"gpc_hz": "/sys/class/devfreq/gpu-gpc-0/cur_freq", "gpu_mC": temperature}
            config["sweep_sensor_paths"] = sensors
            config["native_window_interval_s"] = 0.02
            write_json(out / "plan.json", config)
            native(out, args.repeats, args.sweeps, lock_fd, sensors, config["native_window_interval_s"])
    write_json(
        out / "hashes.json", {p.name: sha256(p) for p in out.iterdir() if p.is_file()}
    )
    result = review(out)
    write_json(out / "summary.json", result)
    (out / "summary.txt").write_text(render(result))
    return result


def review(out):
    hashes = json.loads((out / "hashes.json").read_text())
    expected = {"hardware.json", "plan.json", "telemetry.jsonl", "tegrastats.stderr"}
    plan = json.loads((out / "plan.json").read_text())
    if plan["mode"] == "full":
        if "native_window_interval_s" in plan:
            expected.add("native-window.jsonl")
        expected.update(
            {
                "build.stdout",
                "build.stderr",
                "build-receipt.json",
                "sass.gz",
                "sass.stderr",
                "sass-receipt.json",
                "native.jsonl",
                "native.stderr",
                "native-receipt.json",
                "executable.sha256",
            }
        )
    if set(hashes) != expected or any(
        sha256(out / name) != hashes[name] for name in expected
    ):
        raise ValueError("incomplete or changed raw hardware evidence")
    snapshot = json.loads((out / "hardware.json").read_text())
    ts = [json.loads(l) for l in (out / "telemetry.jsonl").read_text().splitlines()]
    if [r["index"] for r in ts] != list(range(plan["telemetry_samples"])) or any(
        a["monotonic_ns"] >= b["monotonic_ns"] for a, b in zip(ts, ts[1:])
    ):
        raise ValueError("telemetry missing or out of order")
    gpu = snapshot["cuda"]["devices"][0]

    def att(key):
        return gpu["attributes"]["CU_DEVICE_ATTRIBUTE_" + key]["value"]

    ret = {
        "schema": plan["schema"],
        "mode": plan["mode"],
        "gpu": gpu["name"],
        "sm": att("MULTIPROCESSOR_COUNT"),
        "cc": [att("COMPUTE_CAPABILITY_MAJOR"), att("COMPUTE_CAPABILITY_MINOR")],
        "attribute_results": len(gpu["attributes"]),
        "cpu_objects": len(snapshot["cpu"]["per_core"]),
        "pci_objects": len(snapshot["io"]["pcie"]),
        "interface_count": len(snapshot["io"]["net_interfaces"]),
        "connected_displays": sum(
            x["status"]["value"] == "connected"
            for x in snapshot["io"]["drm_connectors"].values()
        ),
        "media_libraries": {
            k: [lib["library"] for lib in info["libraries"] if lib["loaded"]]
            for k, info in snapshot["media"].items()
        },
        "telemetry_samples": len(ts),
        "raw_tegrastats_lines": sum(bool(t["tegrastats"].get("raw")) for t in ts),
        "rail_names": sorted(
            set().union(*(r["tegrastats"].get("rails_mW", {}) for r in ts))
        ),
    }
    ret["spec_comparison"] = {
        "selected_sku": None,
        "actual": {
            "gpu_sm_count": ret["sm"],
            "gpu_l2_cache_bytes": att("L2_CACHE_SIZE"),
        },
        "references": {
            name: {k: fields[k] for k in ("gpu_sm_count", "gpu_l2_cache_bytes")}
            for name, fields in snapshot["specification_reference"]["models"].items()
        },
    }
    ret["telemetry_span_s"] = (ts[-1]["monotonic_ns"] - ts[0]["monotonic_ns"]) / 1e9
    values = {}
    for row in ts:
        for key, record in row["sysfs"]["devfreq_Hz"].items():
            if record["value"] is not None:
                values.setdefault("Hz:" + key, []).append(record["value"])
        for key, rail in row["tegrastats"].get("rails_mW", {}).items():
            values.setdefault("mW:" + key, []).append(rail["instant"])
        for key, val in row["tegrastats"].get("temp_C", {}).items():
            values.setdefault("C:" + key, []).append(val)
    ret["telemetry_min_mean_max"] = {
        k: [min(v), statistics.mean(v), max(v)] for k, v in values.items()
    }
    if plan["mode"] == "full":
        for tag in ("build", "sass", "native"):
            receipt = json.loads((out / (tag + "-receipt.json")).read_text())
            if (
                receipt["exit"] != 0
                or receipt["end_unix_ns"] < receipt["start_unix_ns"]
            ):
                raise ValueError(f"unsuccessful {tag} receipt")
        raw = [
            json.loads(line) for line in (out / "native.jsonl").read_text().splitlines()
        ]
        numerics = {}
        for test, (count, expected_val) in {
            "fp16_mma": (256, 17),
            "bf16_mma": (256, 17),
            "int8_mma": (256, 17),
            "fp8_e4m3_mma": (128, 33),
        }.items():
            records = [
                r for r in raw if r.get("test") == test and r["type"] == "numeric"
            ]
            if (
                len(records) != 1
                or records[0]["values"] != [expected_val] * count
                or records[0]["mismatches"] != 0
            ):
                raise ValueError(f"failed {test} numerical output")
            numerics[test] = {
                "count": count,
                "expected": expected_val,
                "max_abs_error": 0,
            }
        ret["numerical_smoke"] = numerics
        ret["warp"] = next(r for r in raw if r["type"] == "warp")
        ret["cluster"] = next(r for r in raw if r["type"] == "cluster")
        if [
            ret["warp"][k]
            for k in (
                "shuffle",
                "ballot",
                "any",
                "all",
                "fence_and_empty_group_completed",
            )
        ] != [32, 65535, 1, 0, 1]:
            raise ValueError("warp/fence smoke mismatch")
        if ret["cluster"]["supported"] and ret["cluster"]["remote_values"] != [4, 3]:
            raise ValueError("distributed shared result mismatch")
        ret["curves"] = {}
        for test, settings in {
            "warp_work": [
                w
                for w in (1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48)
                if w * 32 <= att("MAX_THREADS_PER_BLOCK")
            ],
            "bank_word_stride": [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128],
            "pointer_chase_KiB": [4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512],
        }.items():
            curve = []
            for setting in settings:
                rows = [
                    r for r in raw if r["type"] == "timing" and r["test"] == test and r["setting"] == setting
                ]
                expected_runs = [
                    (s, i)
                    for s in range(plan.get("micro_sweeps", 1))
                    for i in range(plan["micro_repeats"])
                ]
                if [(r.get("sweep", 0), r["repeat"]) for r in rows] != expected_runs:
                    raise ValueError(f"missing {test} samples")
                ms = [r["elapsed_ms"] for r in rows]
                if not all(math.isfinite(t) and t > 0 for t in ms):
                    raise ValueError("invalid event time")
                med = statistics.median(ms)
                medians = [
                    statistics.median(
                        r["elapsed_ms"] for r in rows if r.get("sweep", 0) == s
                    )
                    for s in range(plan.get("micro_sweeps", 1))
                ]
                pt = dict(
                    setting=setting,
                    n=len(ms),
                    median_ms=med,
                    mean_ms=statistics.mean(ms),
                    sample_sd_ms=statistics.stdev(ms),
                    per_sweep_median_ms=medians,
                )
                if test == "warp_work":
                    pt["warp_iterations_per_ms"] = setting * rows[0]["iterations"] / med
                if test == "pointer_chase_KiB":
                    pt["ns_per_access_including_launch"] = (
                        med * 1e6 / rows[0]["iterations"]
                    )
                curve.append(pt)
            ret["curves"][test] = curve
        window = json.loads((out / "native-receipt.json").read_text())
        in_window = sum(window["start_unix_ns"] <= r["unix_ns"] <= window["end_unix_ns"] for r in ts)
        if "native_window_interval_s" in plan:
            series = [json.loads(line) for line in (out/"native-window.jsonl").read_text().splitlines()]
            ret["native_window"] = native_window.review(series, window, plan["native_window_interval_s"])
        edges = [r for r in raw if r["type"] == "sweep_sensor"]
        if "sweep_sensor_paths" in plan:
            # In file order, every measured group must be bracketed by its own two reads.
            records = [r for r in raw if r["type"] in ("timing", "sweep_sensor")]
            cursor = 0
            for sweep in range(plan["micro_sweeps"]):
                for name, points in ret["curves"].items():
                    for point in points:
                        group = records[cursor:cursor+plan["micro_repeats"]+2]
                        if (len(group) != plan["micro_repeats"]+2 or group[0].get("edge") != "before"
                                or group[-1].get("edge") != "after"
                                or group[0]["unix_ns"] > group[-1]["unix_ns"]
                                or any((r["sweep"], r["test"], r["setting"]) != (sweep, name, point["setting"]) for r in group)
                                or not all(r["type"] == "timing" for r in group[1:-1])):
                            raise ValueError("incomplete or misaligned sweep telemetry")
                        cursor += len(group)
            if cursor != len(records) or any(not window["start_unix_ns"] <= r["unix_ns"] <= window["end_unix_ns"] for r in edges):
                raise ValueError("sweep telemetry outside experiment")
        ret["sweep_coverage"] = {"inventory_samples_in_native": in_window, "boundary_samples": len(edges),
            "readable_pairs": sum(r["gpc_hz"] is not None and r["gpu_mC"] is not None for r in edges)}
        comparisons = [(max(values)/min(values), name, point["setting"], values)
                       for name, points in ret["curves"].items() for point in points
                       if len(values := point["per_sweep_median_ms"]) > 1]
        if comparisons:
            ratio, name, setting, values = max(comparisons)
            ret["widest_sweep_ratio"] = {"ratio": ratio, "test": name, "setting": setting, "medians_ms": values}
        ret["observed_peak_work_rate_warps"] = max(
            ret["curves"]["warp_work"],
            key=lambda point: point["warp_iterations_per_ms"],
        )["setting"]
        ret["observed_slowest_bank_stride_words"] = max(
            ret["curves"]["bank_word_stride"], key=lambda point: point["median_ms"]
        )["setting"]
        chase = ret["curves"]["pointer_chase_KiB"]
        baseline = min(p["ns_per_access_including_launch"] for p in chase[:4])
        ret["first_chase_point_above_baseline_8pct_KiB"] = next(
            (
                p["setting"]
                for p in chase
                if p["ns_per_access_including_launch"] > baseline * 1.08
            ),
            None,
        )
        chain = [r for r in raw if r["type"] == "chain_check"]
        expected_kib = [
            p["setting"] for p in ret["curves"]["pointer_chase_KiB"]
        ] * plan.get("micro_sweeps", 1)
        if [r["KiB"] for r in chain] != expected_kib or any(
            r["expected"] != r["observed"] for r in chain
        ):
            raise ValueError("pointer chain mismatched")
        occupancy = [r for r in raw if r["type"] == "occupancy_api"]
        expected_blocks = {
            (threads, size)
            for threads in (32, 64, 128, 256, 512, 1024)
            for size in (0, 4096, 8192, 16384, 32768, 40960, 49152)
            if threads <= att("MAX_THREADS_PER_BLOCK")
            and size <= att("MAX_SHARED_MEMORY_PER_BLOCK")
        }
        if {
            (r["threads"], r["dynamic_smem_bytes"]) for r in occupancy
        } != expected_blocks or len(occupancy) != len(expected_blocks):
            raise ValueError("missing or duplicate occupancy API rows")
        if any(r["max_active_blocks_per_sm"] < 0 for r in occupancy):
            raise ValueError("invalid occupancy API result")
        ret["occupancy_api_points"] = len(occupancy)
        import gzip
        import re
        from collections import defaultdict

        opcodes = defaultdict(set)
        fn = None
        for line in (
            gzip.decompress((out / "sass.gz").read_bytes()).decode().splitlines()
        ):
            if "Function : " in line:
                fn = line.split("Function : ", 1)[1].strip()
            op = re.match(r"\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?\w+\s+)?([\w.]+)", line)
            if fn and op:
                opcodes[fn].add(op[1])
        ret["observed_SASS"] = {
            fn: sorted(
                op
                for op in ops
                if any(
                    key in op
                    for key in (
                        "MMA",
                        "FENCE",
                        "MEMBAR",
                        "VOTE",
                        "SHFL",
                        "DEPBAR",
                        "WARPGROUP",
                    )
                )
            )
            for fn, ops in opcodes.items()
        }
    return ret


def render(r):
    lines = [
        f"GPU {r['gpu']} CC {'.'.join(map(str, r['cc']))}, {r['sm']} SM",
        f"CPU objects {r['cpu_objects']}; PCI objects {r['pci_objects']}; interfaces {r['interface_count']}; connected displays {r['connected_displays']}",
        f"Telemetry {r['telemetry_samples']} samples; tegrastats {r['raw_tegrastats_lines']} lines; rails {', '.join(r['rail_names'])}",
    ]
    if r["mode"] == "full":
        coverage = r["sweep_coverage"]
        if "widest_sweep_ratio" in r:
            lines.append(f"Largest between-sweep median ratio: {r['widest_sweep_ratio']['ratio']:.2f}x")
        if coverage["boundary_samples"]:
            lines.append(f"Sweep boundaries with both GPU sensors: {coverage['readable_pairs']}/{coverage['boundary_samples']}")
        elif not coverage["inventory_samples_in_native"]:
            lines.append("No concurrent GPU frequency or temperature evidence for these sweeps; cause of variation unknown.")
        lines.append("Numerical tile smoke passed: " + ", ".join(r["numerical_smoke"]))
        lines.append(
            "Experimental curve points: "
            + ", ".join(f"{k}={len(v)}" for k, v in r["curves"].items())
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--mode", choices=["query", "full"], default="query")
    duration = cap.add_mutually_exclusive_group()
    duration.add_argument("--samples", type=int, default=3)
    duration.add_argument(
        "--duration", type=float, help="Requested telemetry duration in seconds"
    )
    cap.add_argument("--interval", type=float, default=1.0)
    cap.add_argument("--repeats", type=int, default=6)
    cap.add_argument(
        "--sweeps",
        type=int,
        default=1,
        help="Complete microbenchmark sweeps within this process",
    )
    rev = sub.add_parser("review")
    rev.add_argument("directory", type=Path)
    for sp in (cap, rev):
        sp.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.action == "capture":
        if (
            args.samples < 1
            or not math.isfinite(args.interval)
            or args.interval <= 0
            or args.repeats < 2
            or args.sweeps < 1
        ):
            parser.error(
                "samples >= 1, interval > 0, repeats >= 2, sweeps >= 1 required"
            )
        if args.duration is not None:
            if not math.isfinite(args.duration) or args.duration <= 0:
                parser.error("duration must be positive and finite")
            args.samples = max(1, math.ceil(args.duration / args.interval))
        result = capture(args)
    else:
        result = review(args.directory)
    print(
        json.dumps(result, indent=2) if args.json else render(result),
        end="\n" if args.json else "",
    )


if __name__ == "__main__":
    main()
