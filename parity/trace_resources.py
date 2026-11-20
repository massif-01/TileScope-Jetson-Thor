"""Narrow, read-only Jetson nvmap accounting for the owned trace service."""
import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

BASE="/sys/kernel/debug/nvmap/"

def read(suffix):
    result=subprocess.run(["sudo","-n","cat",BASE+suffix],capture_output=True,text=True,timeout=20)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout

def snapshot(owners):
    start=time.time_ns()
    rows=read("iovmm/clients").splitlines()
    own=[]
    for line in rows[1:]:
        fields=line.split()
        if len(fields)>=4 and fields[-2].isdigit() and int(fields[-2]) in owners:
            own.append(line)
    alloc_lines=read("iovmm/allocations").splitlines()
    selected=[]
    block=False
    for line in alloc_lines[2:]:
        fields=line.split()
        if fields and fields[0] in ("user","kernel","total"):
            block=len(fields)>=4 and fields[-2].isdigit() and int(fields[-2]) in owners
        if block and fields:
            selected.append(line)
    orphans=read("iovmm/orphan_handles").splitlines()
    pool={f:read("pagepool/"+f).strip() for f in (
        "page_pool_available_pages","page_pool_available_big_pages","page_pool_big_page_size")}
    return {"start_unix_ns":start,"end_unix_ns":time.time_ns(),"owner_pids":sorted(owners),
            "client_header":rows[0],"own_clients":own,"all_clients_total":[r for r in rows if r.startswith("total")],
            "allocation_header":alloc_lines[:2],"own_allocations":selected,
            "orphan_header":orphans[0],"orphan_rows":[r for r in orphans[1:] if r.strip()],
            "all_clients_raw_sha256":hashlib.sha256("\n".join(rows).encode()).hexdigest(),
            "page_pool_raw":pool,"system_page_bytes":os.sysconf("SC_PAGE_SIZE")}

def processes(group, known=()):
    result={}
    for path in Path("/proc").glob("[0-9]*"):
        pid=int(path.name)
        try:
            raw=(path/"stat").read_text()
            state=raw.rsplit(")",1)[1].split()
            if int(state[2])!=group and pid not in known:
                continue
            status=dict(line.split(":",1) for line in (path/"status").read_text().splitlines() if ":" in line)
            result[pid]={"pid":pid,"ppid":int(state[1]),"pgrp":int(state[2]),"start_ticks":int(state[19]),
                         "comm":status["Name"].strip(),"exe":os.readlink(path/"exe"),
                         "rss_kib":int(status.get("VmRSS","0").split()[0]),
                         "thread_count":int(status["Threads"]),"fd_count":len(list((path/"fd").iterdir()))}
        except (PermissionError,FileNotFoundError,ProcessLookupError):
            continue
    return result
