"""Board-level samples and the selected local listener's process counters."""
import json
import os
from pathlib import Path
import threading
import time

from parity.resources import value


class Monitor:
    def __init__(self, target, path):
        self.ident=target["identity"]
        self.path=path
        self.pid=int(self.ident["server_pid"])
        self.ticks=int(self.ident["start_ticks"])
        self.gpu_temp=next(p.with_name("temp") for p in Path("/sys/class/thermal").glob("thermal_zone*/type")
                           if p.read_text().strip()=="gpu-thermal")
        self.stop=threading.Event()
        self.ready=threading.Event()
        self.protection=None
        self.thread=threading.Thread(target=self.watch,daemon=True)

    def sample(self):
        stat=(Path("/proc")/str(self.pid)/"stat").read_text().rsplit(")",1)[1].split()
        boot=Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        if int(stat[19])!=self.ticks or self.ident["boot_id"]!=boot:
            raise ValueError("bound server PID or host boot identity changed")
        mem=next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemAvailable:"))
        temp=int(self.gpu_temp.read_text())
        return dict(ns=time.monotonic_ns(),wall_ns=time.time_ns(),gpu_mC=temp,mem_available_bytes=mem,
                    gpc_hz=value("/sys/class/devfreq/gpu-gpc-0/cur_freq"),pid=self.pid,start_ticks=self.ticks,
                    cpu_ticks=int(stat[11])+int(stat[12]),rss_bytes=int(stat[21])*os.sysconf("SC_PAGE_SIZE"))

    def watch(self):
        errors=0
        with self.path.open("x",buffering=1) as file:
            while True:
                try:
                    row=self.sample()
                    errors=0
                    if row["gpu_mC"]>=85000 or row["mem_available_bytes"]<12*(1<<30):
                        self.protection="GPU temperature or available memory crossed frozen protection limit"
                    self.ready.set()
                except (ValueError,OSError,StopIteration) as exc:
                    row={"error":str(exc),"ns":time.monotonic_ns()}
                    errors+=1
                    if errors>=3:
                        self.protection="three consecutive resource-sampling failures"
                file.write(json.dumps(row)+"\n")
                if self.protection:
                    self.ready.set()
                    break
                if self.stop.is_set():
                    break
                self.stop.wait(1)

    def __enter__(self):
        self.thread.start()
        self.ready.wait(8)
        self.check()
        return self

    def check(self):
        if not self.ready.is_set() or self.protection:
            raise RuntimeError(self.protection or "no valid initial resource sample")

    def wait_until(self,deadline):
        while time.monotonic()<deadline:
            self.check()
            time.sleep(min(0.25,max(0,deadline-time.monotonic())))

    def __exit__(self,*_):
        self.stop.set()
        self.thread.join()
        self.check()
