"""Launch one fresh worker process per shape/format/round under a caller-held device lock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import urllib.request

try:
    from .build import FORMATS
except ImportError:
    from build import FORMATS


def write(path, obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def digest(path):
    with path.open("rb") as src:
        h=hashlib.sha256()
        for block in iter(lambda:src.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def load_plan(plan):
    if set(plan) != {"scope","rounds","seed","warmup","samples","condition","cases"}:
        raise ValueError("invalid plan fields")
    for key,lo,hi in (("rounds",1,20),("seed",0,2**32-1),("warmup",0,500),("samples",2,1000)):
        if type(plan[key]) is not int or not lo <= plan[key] <= hi:
            raise ValueError("out of range: "+key)
    policy=plan["condition"]
    if set(policy)!={"mode","require_governor","require_power_mode_id"} or policy["mode"]!="preserve":
        raise ValueError("condition policy must preserve existing state")
    schedules=[]
    for entry in plan["cases"]:
        fmt=entry["format"]
        if set(entry)!={"format","shapes"} or fmt not in FORMATS:
            raise ValueError("invalid format or case")
        for shape in entry["shapes"]:
            if len(shape)!=3 or not all(type(v) is int and 0<v<=2048 for v in shape):
                raise ValueError("expected three positive dimensions <= 2048")
            m,n,k=shape
            align=(128,128,64) if fmt.startswith("nvfp4") else (8,1,8)
            if any(v%a for v,a in zip(shape,align)) or m*n*k>256**3*8:
                raise ValueError(f"{fmt} requires alignment {align} and M*N*K <= 134217728")
            schedules.append((fmt,m,n,k))
    if not schedules or len(schedules)!=len(set(schedules)):
        raise ValueError("empty or duplicate calibration workload")
    return schedules


def read(path):
    try:
        return Path(path).read_text().strip().replace("\x00"," ")
    except OSError:
        return None


def output(argv):
    try:
        p=subprocess.run(argv,text=True,capture_output=True,timeout=5)
        return {"argv":argv,"exit":p.returncode,"stdout":p.stdout.strip(),"stderr":p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {"argv":argv,"exit":None,"error":str(e)}


def conditions():
    base=Path("/sys/class/devfreq/gpu-gpc-0")
    power=output(["nvpmodel","-q"])
    text=power.get("stdout","").splitlines()
    mode=int(text[-1]) if text and text[-1].strip().isdigit() else None
    return {"clock":{name:read(base/name) for name in ("min_freq","max_freq","governor")},
            "mode_id":mode,"nvpmodel":power}


def enforce(policy, measured):
    for key,current in (("require_governor",measured["clock"]["governor"]),
                        ("require_power_mode_id",measured["mode_id"])):
        if policy[key] is not None and policy[key]!=current:
            raise ValueError(f"required condition mismatch: {key}={policy[key]}, observed {current}")


def sensors():
    base=Path("/sys/class/devfreq")
    sources={"gpu_gpc_hz":base/"gpu-gpc-0/cur_freq","gpu_nvd_hz":base/"gpu-nvd-0/cur_freq"}
    temps=[p.with_name("temp") for p in Path("/sys/class/thermal").glob("thermal_zone*/type") if read(p)=="gpu-thermal"]
    if temps:
        sources["gpu_temperature_mc"]=temps[0]
    return sources


def monitor(folder,stop):
    sources=sensors()
    write(folder/"sensor-sources.json",{k:str(v) for k,v in sources.items()})
    with (folder/"sensors.jsonl").open("w") as f:
        while True:
            f.write(json.dumps(dict(unix_ns=time.time_ns(), **{k:read(v) for k,v in sources.items()}))+"\n")
            if stop.wait(.05):
                break


def activity():
    names={"vllm:num_requests_running","vllm:num_requests_waiting","vllm:generation_tokens_total","vllm:prompt_tokens_total"}
    try:
        with urllib.request.urlopen("http://127.0.0.1:58000/metrics", timeout=2) as response:
            return {l.split("{")[0]:float(l.split()[-1]) for l in response.read().decode().splitlines() if l.split("{")[0] in names}
    except (ValueError,OSError) as e:
        return {"unavailable":str(e)}


def run(plan_path,binaries,out):
    plan=json.loads(plan_path.read_text())
    cases=load_plan(plan)
    config_sha256=digest(plan_path)
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(plan_path,out/"plan.json")
    shutil.copyfile(binaries/"build.json",out/"build.json")
    builds=json.loads((binaries/"build.json").read_text())["builds"]
    before=conditions()
    write(out/"conditions-before.json",before)
    enforce(plan["condition"],before)
    write(out/"environment.json",{"model":read("/proc/device-tree/model"),"l4t":read("/etc/nv_tegra_release"),
          "kernel":platform.release(),"nvidia_smi":output(["nvidia-smi","--query-gpu=name,driver_version,uuid","--format=csv,noheader"]),
          "gpu_processes":output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader"]),
          "service_before":activity()})
    stop=threading.Event()
    thread=threading.Thread(target=monitor,args=(out,stop))
    thread.start()
    attempts=[]
    try:
        for rnd in range(1,plan["rounds"]+1):
            ordered=cases if rnd%2 else cases[::-1]
            for fmt,m,n,k in ordered:
                tag=f"r{rnd:02}-{fmt}-{m}x{n}x{k}"
                dest=out/tag
                dest.mkdir()
                exe=binaries/fmt
                cmd=[str(exe),f"--m={m}",f"--n={n}",f"--k={k}",f"--seed={plan['seed']}",
                     f"--iterations={plan['samples']}",f"--warmup={plan['warmup']}"]
                rc=None
                begin=time.time_ns()
                hashes={}
                proc=None
                try:
                    if builds[fmt]["exit"] or digest(exe)!=builds[fmt]["executable_sha256"]:
                        raise ValueError("missing or mismatched build")
                    with (dest/"stdout").open("w") as sout,(dest/"stderr").open("w") as serr:
                        proc=subprocess.Popen(cmd,cwd=dest,stdout=sout,stderr=serr)
                        try:
                            rc=proc.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                            raise
                    for blob in sorted(dest.glob("input-*.bin")):
                        hashes[blob.stem[6:]]=dict(bytes=blob.stat().st_size,sha256=digest(blob))
                        blob.unlink()
                except (OSError,KeyError,ValueError,subprocess.TimeoutExpired) as error:
                    (dest/"error.txt").write_text(str(error)+"\n")
                receipt=dict(id=tag,round=rnd,format=fmt,m=m,n=n,k=k,argv=cmd,pid=proc.pid if proc else None,
                             exit=rc,begin_ns=begin,end_ns=time.time_ns(),inputs=hashes,
                             config_sha256=config_sha256,binary_sha256=builds.get(fmt,{}).get("executable_sha256"),
                             raw_sha256={p.name:digest(p) for p in (dest/"stdout",dest/"stderr") if p.exists()})
                write(dest/"receipt.json",receipt)
                attempts.append(receipt)
                print(tag,rc,flush=True)
    finally:
        stop.set()
        thread.join()
        after=conditions()
        write(out/"conditions-after.json",after)
        write(out/"finish.json",dict(service_after=activity(), policy_unchanged=before==after, attempts=len(attempts)))
    return int(before!=after or any(a["exit"]!=0 for a in attempts))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--bin",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    raise SystemExit(run(a.config,a.bin.resolve(),a.out.resolve()))
