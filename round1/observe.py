"""Read Thor GPU clock and temperature sysfs nodes at approximately 50 Hz."""

import json
import time
from pathlib import Path


def read(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def main():
    devfreq = Path("/sys/class/devfreq")
    gpu_thermal = None
    for kind in Path("/sys/class/thermal").glob("thermal_zone*/type"):
        if kind.read_text().strip() == "gpu-thermal":
            gpu_thermal = kind.with_name("temp")
            break

    fields = {
        "gpc_cur_freq": devfreq / "gpu-gpc-0/cur_freq",
        "nvd_cur_freq": devfreq / "gpu-nvd-0/cur_freq",
        "gpu_temp": gpu_thermal,
    }
    print(json.dumps({"sources": {name: str(path) if path else None for name, path in fields.items()}}),
          flush=True)
    while True:
        timestamp = time.time_ns()
        values = {name: read(path) if path else None for name, path in fields.items()}
        print(json.dumps({"unix_ns": timestamp, **values}), flush=True)
        time.sleep(0.02)


if __name__ == "__main__":
    main()
