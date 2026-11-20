"""Query visible GPU device handles. Only device-node names and process identities are kept."""
import argparse
import os
import subprocess
from pathlib import Path
import time
from tilescope.common import write_json


def capture():
    result=[];unreadable=0;vanished=0
    for proc in Path("/proc").glob("[0-9]*"):
        try:nodes=list((proc/"fd").iterdir())
        except PermissionError:unreadable+=1;continue
        except FileNotFoundError:vanished+=1;continue
        matching=set()
        for node in nodes:
            try:target=os.readlink(node)
            except OSError:continue
            if target.startswith(("/dev/nvhost","/dev/nvidia","/dev/nvmap","/dev/dri/")):matching.add(target)
        if matching:
            try:
                result.append({"pid":int(proc.name),"comm":(proc/"comm").read_text().strip(),
                               "start_ticks":int((proc/"stat").read_text().rsplit(")",1)[1].split()[19]),"nodes":sorted(matching)})
            except FileNotFoundError:vanished+=1
    return {"time_ns":time.time_ns(),"euid":os.geteuid(),"holders":result,"denied_processes":unreadable,"vanished":vanished,
            "scope":"Process identity and GPU or auxiliary device handles. A resident handle alone does not establish utilization."}

def system_capture():
    def identities():
        records={}
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                ticks=stat.read_text().rsplit(")",1)[1].split()[19]
                records[stat.parent.name]=(ticks,(stat.parent/"comm").read_text().strip())
            except (OSError,IndexError):continue
        return records
    start=time.time_ns();before=identities()
    paths=[f"/proc/{pid}/fd" for pid in before]
    # Read only direct device-link matches; discard unrelated file targets in find itself.
    cmd=["sudo","-n","find",*paths,"-mindepth","1","-maxdepth","1","-type","l","(",
         "-lname","/dev/nvhost*","-o","-lname","/dev/nvidia*","-o","-lname","/dev/nvmap*",
         "-o","-lname","/dev/dri/*",")","-printf","%p\t%l\n"]
    proc=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
    matched={}
    for line in proc.stdout.splitlines():
        src,target=line.split("\t",1);pid=Path(src).parts[2]
        matched.setdefault(pid,set()).add(target)
    after=identities()
    holders=[{"pid":int(pid),"start_ticks":int(before[pid][0]),"comm":before[pid][1],
              "stable_identity":before[pid]==after.get(pid),"nodes":sorted(nodes)} for pid,nodes in matched.items()]
    return {"time_start_ns":start,"time_end_ns":time.time_ns(),"method":"sudo -n find on enumerated /proc/PID/fd; GPU/auxiliary links only",
            "enumerated_pids":len(paths),"query_exit":proc.returncode,"diagnostics":proc.stderr.splitlines(),
            "holders":holders,"scope":"point-in-time handle visibility; PID turnover and per-process activity separate"}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    p.add_argument("--system",action="store_true",help="use already authorized sudo -n for device-link-only system inventory")
    a=p.parse_args()
    from tilescope.common import device_slot
    with device_slot():write_json(a.out,system_capture() if a.system else capture())
