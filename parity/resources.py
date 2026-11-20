"""Low-frequency measurements; power labels and sampling clocks are kept explicit."""
from pathlib import Path
import os
import ctypes
import subprocess
import time

from .service_io import read_text


def value(path):
    try:return int(read_text(path))
    except (ValueError,OSError):return None


class Sampler:
    def __init__(self):
        self.temperature={read_text(p):p.with_name("temp") for p in Path("/sys/class/thermal").glob("thermal_zone*/type")}
        self.rails={}
        for p in Path("/sys/bus/i2c/drivers/ina3221").glob("*/hwmon/hwmon*/in*_label"):
            n=p.name[2:].split("_")[0]
            if p.with_name(f"curr{n}_input").is_file():
                self.rails[read_text(p)]=(p.with_name(f"in{n}_input"),p.with_name(f"curr{n}_input"))
        self.pids={}
        for p in Path("/proc").glob("[0-9]*/comm"):
            try:
                name=read_text(p)
                if name=="vllm" or name.startswith("VLLM::"):
                    self.pids[int(p.parent.name)]=name
            except OSError:continue
        ids=subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],
                           capture_output=True,text=True,timeout=8)
        for pid in ids.stdout.splitlines():
            if pid.strip().isdigit():
                proc=Path("/proc")/pid.strip()
                if proc.exists():self.pids[int(pid)]=read_text(proc/"comm")
        self.cuda=None
        try:
            self.cuda=ctypes.CDLL("libcudart.so.13")
            self.cuda.cudaMemGetInfo.argtypes=[ctypes.POINTER(ctypes.c_size_t)]*2
        except OSError as exc:self.cuda_error=str(exc)
        self.nv=None
        try:
            import pynvml
            pynvml.nvmlInit();self.nv=pynvml;self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        except (ImportError,RuntimeError) as e:self.nv_error=str(e)

    def sample(self):
        mem={w[0].rstrip(":"):int(w[1])*1024 for l in read_text("/proc/meminfo").splitlines()
             if (w:=l.split()) and w[0] in ("MemAvailable:","MemFree:","Cached:")}
        procs={}
        for pid,name in self.pids.items():
            path=Path(f"/proc/{pid}")
            try:
                s=(path/"stat").read_text().rsplit(")",1)[1].split()
                procs[str(pid)]={"name":name,"start_ticks":int(s[19]),"rss_bytes":int(s[21])*os.sysconf("SC_PAGE_SIZE"),
                    "cpu_ticks":int(s[11])+int(s[12]),"threads":int(s[17]),"fd":len(list((path/"fd").iterdir()))}
            except OSError as e:procs[str(pid)]={"error":str(e)}
        board={}
        for key,(v,i) in self.rails.items():
            mv,ma=value(v),value(i);board[key]={"mV":mv,"mA":ma,"mW":mv*ma/1000 if None not in (mv,ma) else None}
        nv={}
        if self.nv:
            try:
                nv["gpu_util_percent"]=self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu
            except self.nv.NVMLError as e:nv["error"]=str(e)
        else:nv["error"]=self.nv_error
        memory={"source":"cudaMemGetInfo on the observer, unified memory"}
        if self.cuda:
            free,total=ctypes.c_size_t(),ctypes.c_size_t()
            rc=self.cuda.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total))
            memory.update(rc=rc,free_bytes=free.value if rc==0 else None,total_bytes=total.value if rc==0 else None)
        else:memory["error"]=self.cuda_error
        return dict(cuda=memory,time_ns=time.time_ns(),mem=mem,processes=procs,nvml=nv,
                    temperatures_mC={k:value(p) for k,p in self.temperature.items()},
                    rail_readings=board,gpc_hz=value("/sys/class/devfreq/gpu-gpc-0/cur_freq"),
                    cpu_ticks=read_text("/proc/stat").splitlines()[0],ticks_per_second=os.sysconf("SC_CLK_TCK"))
