"""Read-only pre/post evidence for the retained service, GPU policy and build identity."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from urllib.request import urlopen
from tilescope.common import sha256, write_json
from tilescope.probe.identity import bind
from tilescope.probe.targets import resolve
from tools.multimodel.guard import sample


def command(*argv):
    r=subprocess.run(argv,capture_output=True,text=True)
    return {"argv":argv,"exit":r.returncode,"stdout":r.stdout.strip(),"stderr":r.stderr.strip()}


def record(profile):
    target=bind(resolve(profile))
    base=Path("/sys/class/devfreq/gpu-gpc-0")
    vllm=importlib.metadata.distribution("vllm")
    env={"target":target,"guard":sample(),"kernel":platform.release(),"l4t":Path("/etc/nv_tegra_release").read_text(),
         "board":Path("/proc/device-tree/model").read_bytes().rstrip(b"\0").decode(),
         "device_query":command("nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"),
         "power":command("nvpmodel","-q"),"clocks":{k:(base/k).read_text().strip() for k in ("min_freq","max_freq","governor","cur_freq")},
         "toolkit":command("/usr/local/cuda-13.3/bin/nvcc","--version"),"profiler":command("nsys","--version"),
         "jetpack_packages":command("dpkg-query","-W","nvidia-l4t-core","nvidia-jetpack"),
         "python":platform.python_version(),"packages":{k:importlib.metadata.version(k) for k in ("vllm","torch","gguf","transformers","safetensors")},
         "runtime":{},"disk":command("df","-B1",str(Path.home()))}
    for name in ("vllm/_version.py","vllm/v1/worker/gpu_model_runner.py","vllm/v1/core/sched/scheduler.py","vllm/compilation/cuda_graph.py"):
        env["runtime"][name]=sha256(vllm.locate_file(name))
    with urlopen(target["deployment"]["endpoint"]+"/health",timeout=4) as response:
        env["service_health"]=response.status
    return env


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    write_json(a.out,record(a.target))
