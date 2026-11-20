"""Observe GPU sysfs during the native process, outside its CUDA event intervals."""
import json
from pathlib import Path
import statistics
import subprocess
import time

from observation.collect import memory_available


def run(argv, stdout, stderr, lock_fd, path, sensors, interval):
    def sample(file, edge):
        begin = time.monotonic_ns()
        row = {"edge":edge, "read_start_mono_ns":begin, "unix_ns":time.time_ns(),
               "gpu_mC":None, "gpc_hz":None, "available_bytes":memory_available()}
        try:
            for key, source in sensors.items():
                if source:
                    row[key] = int(Path(source).read_text())
        finally:
            row["read_end_mono_ns"] = time.monotonic_ns()
            file.write(json.dumps(row)+"\n")
            file.flush()
        if row["gpu_mC"] is None or row["gpc_hz"] is None:
            raise ValueError("required GPU telemetry unreadable during native experiment")
        if row["gpu_mC"] >= 85_000 or row["available_bytes"] < 12*1024**3:
            raise ValueError("device protection limit reached during native experiment")

    with path.open("x") as file:
        sample(file, "before_spawn")
        begin = time.monotonic_ns()
        with subprocess.Popen(argv, stdout=stdout, stderr=stderr, pass_fds=(lock_fd,)) as proc:
            try:
                while proc.poll() is None:
                    sample(file, "while_running")
                    if time.monotonic_ns() - begin > 180_000_000_000:
                        raise TimeoutError("native experiment exceeded 180 seconds")
                    try:
                        proc.wait(timeout=interval)
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                end = time.monotonic_ns()
                sample(file, "after_wait")
    return proc.returncode, {"pid":proc.pid, "start_monotonic_ns":begin, "end_monotonic_ns":end,
                             "clock_domain":"host_boot_monotonic_ns",
                             "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def review(rows, receipt, interval):
    begin, end = receipt["start_monotonic_ns"], receipt["end_monotonic_ns"]
    if receipt["clock_domain"] != "host_boot_monotonic_ns" or not receipt["boot_id"] or begin >= end:
        raise ValueError("missing native monotonic window or clock origin")
    if ([r["edge"] for r in rows] != ["before_spawn"]+["while_running"]*(len(rows)-2)+["after_wait"]
            or len(rows)<3 or rows[0]["read_end_mono_ns"]>begin or rows[-1]["read_start_mono_ns"]<end
            or any(not begin<=r["read_start_mono_ns"]<=r["read_end_mono_ns"]<=end for r in rows[1:-1])
            or any(a["read_end_mono_ns"]>b["read_start_mono_ns"] for a,b in zip(rows,rows[1:]))
            or any(type(r["gpc_hz"]) is not int or type(r["gpu_mC"]) is not int or
                   r["gpc_hz"]<0 or r["gpu_mC"]>=85000 or r["available_bytes"]<12*1024**3 for r in rows)):
        raise ValueError("incomplete or invalid telemetry bracketing native execution")
    gaps = [(b["read_start_mono_ns"]-a["read_start_mono_ns"])/1e6 for a,b in zip(rows,rows[1:])]
    return {"samples":len(rows),"during":len(rows)-2,"start_monotonic_ns":begin,"end_monotonic_ns":end,
            "target_interval_ms":interval*1000,"actual_interval_ms_min_median_max":[min(gaps),statistics.median(gaps),max(gaps)],
            "before_margin_ms":(begin-rows[0]["read_end_mono_ns"])/1e6,
            "after_margin_ms":(rows[-1]["read_start_mono_ns"]-end)/1e6}
