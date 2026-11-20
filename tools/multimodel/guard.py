"""Supervise a diagnostic command on the Thor with the frozen host protection limits."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def sample():
    mem = next(line for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    zone, = [p.parent for p in Path("/sys/class/thermal").glob("thermal_zone*/type")
             if p.read_text().strip()=="gpu-thermal"]
    return {"monotonic_ns":time.monotonic_ns(),"mem_available_bytes":int(mem.split()[1])*1024,
            "gpu_temp_mC":int((zone/"temp").read_text())}


def guard(s):
    if s["gpu_temp_mC"]>=85000 or s["mem_available_bytes"]<12*1024**3:
        raise RuntimeError(f"resource protection reached: {s}")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log",required=True,type=Path)
    ap.add_argument("command",nargs=argparse.REMAINDER)
    a=ap.parse_args()
    guard(sample())
    with a.log.open("x",buffering=1) as f:
        proc=subprocess.Popen(a.command,start_new_session=True)
        try:
            while proc.poll() is None:
                s=sample()
                print(json.dumps(s),file=f)
                guard(s)
                time.sleep(1)
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=90)
        raise SystemExit(proc.returncode)
