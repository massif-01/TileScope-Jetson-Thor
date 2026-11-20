"""Supplement a running owned trace with all visible Jetson nvmap heaps."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from parity.trace_client import write
from parity.trace_resources import read

def heaps(pids):
    result={}
    for heap in ("iovmm","fsi","vpr"):
        names=read(heap+"/clients").splitlines()
        rows=[r for r in names[1:] if len(r.split())>=4 and r.split()[-2].isdigit() and int(r.split()[-2]) in pids]
        orphan=read(heap+"/orphan_handles").splitlines()
        result[heap]={"headers":[names[0],orphan[0]],
            "own_client_rows":rows,"system_total":[r for r in names if r.startswith("total")],
            "orphan_rows":[r for r in orphan[1:] if r.strip()]}
    files=subprocess.check_output(["sudo","-n","ls","-1","/sys/kernel/debug/nvmap/handles_by_pid"],text=True).splitlines()
    owners={}
    for pid in sorted(pids):
        if str(pid) in files:
            rows=read("handles_by_pid/"+str(pid)).splitlines()
            owners[str(pid)]={"entries":len(rows),"table":rows}
    return {"unix_ns":time.time_ns(),"owner_pids":sorted(pids),"heap":result,"owned_handle_tables":owners,
            "host_page_bytes":os.sysconf("SC_PAGE_SIZE"),
            "page_pool_available_pages":int(read("pagepool/page_pool_available_pages")),
            "page_pool_available_big_pages":int(read("pagepool/page_pool_available_big_pages")),
            "page_pool_big_page_size":int(read("pagepool/page_pool_big_page_size"))}

def capture(directory):
    dest=directory/"nvmap-extended"
    dest.mkdir()
    owner_pids=set()
    started=False
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        resource=directory/"resources.jsonl"
        if resource.exists() and resource.stat().st_size:
            last=None
            with resource.open() as f:
                for line in f:
                    try:
                        last=json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    owner_pids.update(r["pid"] for r in last["processes"])
            if last and not started:
                write(dest/"initial.json",heaps(owner_pids))
                started=True
        if (directory/"ready.json").exists() and not (dest/"ready.json").exists():
            write(dest/"ready.json",heaps(owner_pids))
        if (directory/"shutdown.json").exists():
            time.sleep(1)
            write(dest/"exited.json",heaps(owner_pids))
            return
        time.sleep(1)
    raise TimeoutError("no private shutdown observed")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    capture(p.parse_args().directory)
