"""Time-stamped samples from tegrastats and direct Linux counters."""

import ctypes
import re
import shutil
import subprocess
import time

from .system import CPU, SYS, reading


def parse_tegrastats(line):
    ram = re.search(r"RAM (\d+)/(\d+)MB(?: \(lfb (\d+)x(\d+)MB\))?", line)
    emc = re.search(r"EMC_FREQ (\d+)%@(\d+)", line)
    cpu = re.search(r"CPU \[([^\]]*)\]", line)
    engines = dict(
        re.findall(
            r"\b((?:NVENC|NVDEC|NVJPG|PVA|OFA|VIC|GR3D_FREQ)\w*)\s+([^ ]+)", line
        )
    )
    rail_pattern = r"\b([A-Za-z][\w-]*) (\d+)mW/(\d+)mW"
    return {
        "source": "tegrastats stdout",
        "raw": line,
        "ram_MiB_used_total_lfb_count_lfb_MiB": [
            int(x) if x else None for x in ram.groups()
        ]
        if ram
        else None,
        "emc_util_percent": int(emc[1]) if emc else None,
        "emc_clock_MHz": int(emc[2]) if emc else None,
        "cpu_percent_at_MHz": cpu[1].split(",") if cpu else None,
        "engine_raw": engines,
        "temp_C": {a: float(b) for a, b in re.findall(r"([\w.-]+)@(-?[\d.]+)C", line)},
        "rails_mW": {
            name: {"instant": int(x), "reported_average": int(avg)}
            for name, x, avg in re.findall(rail_pattern, line)
        },
    }


def sysfs_sample():
    freqs = SYS / "class/devfreq"
    bpmp = SYS / "kernel/debug/bpmp/debug/clk"
    therm = {}
    for d in sorted((SYS / "class/thermal").glob("thermal_zone*")):
        therm[d.name] = {
            "type": reading(d / "type"),
            "temperature": reading(d / "temp", "m°C", True),
        }
    powers = {}
    for hw in sorted((SYS / "class/hwmon").glob("hwmon*")):
        for channel in sorted(hw.glob("power*_input")):
            powers[str(channel)] = reading(channel, "µW", True)
    for iio in sorted((SYS / "bus/iio/devices").glob("iio:device*")):
        for channel in sorted(iio.glob("*_power")):
            powers[str(channel)] = reading(channel, "driver-reported raw", True)
    meminfo = reading("/proc/meminfo")
    mem = {
        k: int(v)
        for k, v in re.findall(
            r"^(MemTotal|MemAvailable|MemFree|SwapTotal|SwapFree):\s+(\d+) kB",
            meminfo["value"] or "",
            re.M,
        )
    }
    return {
        "cpu_kHz": {
            p.parent.parent.name: reading(p, "kHz", True)
            for p in CPU.glob("cpu[0-9]*/cpufreq/scaling_cur_freq")
        },
        "devfreq_Hz": {
            d.name: reading(d / "cur_freq", "Hz", True) for d in sorted(freqs.glob("*"))
        },
        "debugfs_clock_Hz": {
            name: reading(bpmp / name / "rate", "Hz", True)
            for name in (
                "emc",
                "VIC",
                "vic",
                "ofa",
                "pva",
                "nvenc0",
                "nvenc1",
                "nvdec0",
                "nvdec1",
                "isp",
                "nvjpg",
            )
        },
        "gpu_loading_raw": reading(freqs / "gpu-gpc-0/loading", "driver units", True),
        "thermal": therm,
        "power_channels": powers,
        "power_sources": [
            "/sys/class/hwmon/hwmon*/power*_input",
            "/sys/bus/iio/devices/iio:device*/*_power",
        ],
        "memory": {"source": "/proc/meminfo", "unit": "KiB", "values": mem},
    }


def collect(count, interval, out):
    cmd = ["tegrastats", "--interval", str(max(1, round(interval * 1000)))]
    proc = None
    nvml = None
    handle = ctypes.c_void_p()
    source = "nvmlDeviceGetUtilizationRates(device 0)"
    try:
        lib = ctypes.CDLL("libnvidia-ml.so.1")
        code = lib.nvmlInit_v2()
        if code == 0:
            nvml = lib
            code = lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle))
        reason = f"NVML result {code}" if code else None
    except OSError as error:
        reason = str(error)
    if shutil.which(cmd[0]):
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=out, text=True)
    try:
        if proc:
            import selectors

            sel = selectors.DefaultSelector()
            sel.register(proc.stdout, selectors.EVENT_READ)
        for idx in range(count):
            start = time.monotonic()
            if proc:
                ready = sel.select(interval + 2.0)
                line = proc.stdout.readline().strip() if ready else ""
                record = parse_tegrastats(line)
                record["status"] = "queried" if line else "timeout_or_eof"
            else:
                record = {"status": "command_absent", "source": cmd, "raw": None}
            gpu = {"value": None, "unit": "percent", "source": source}
            if reason:
                gpu.update(status="unavailable", error=reason)
            else:
                util = (ctypes.c_uint * 2)()
                code = nvml.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
                gpu.update(
                    status="queried" if code == 0 else "unavailable",
                    value=util[0] if code == 0 else None,
                    result=code,
                )
            yield {
                "index": idx,
                "unix_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "tegrastats": record,
                "sysfs": sysfs_sample(),
                "gpu_utilization": gpu,
            }
            if not proc:
                time.sleep(max(0, interval - (time.monotonic() - start)))
    finally:
        if nvml:
            nvml.nvmlShutdown()
        if proc:
            sel.close()
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            proc.stdout.close()
