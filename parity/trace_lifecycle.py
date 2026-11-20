"""Handle a termination request while retaining an owned experiment's device lock."""
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

@contextlib.contextmanager
def locked_slot(lock):
    with lock.open("a") as fd:
        # While waiting no private service exists. Default SIGTERM is safe then.
        fcntl.flock(fd,fcntl.LOCK_EX)
        pending=[]
        def request_stop(signum,frame):
            pending.append(signum)
        previous={s:signal.signal(s,request_stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            yield pending
        finally:
            for sig,handler in previous.items():
                signal.signal(sig,handler)

def raise_if_requested(pending):
    if pending:
        raise SystemExit(128+pending[0])

def active_group(pgid):
    result=[]
    for f in Path("/proc").glob("[0-9]*/stat"):
        try:
            state=f.read_text().rsplit(")",1)[1].split()
            if state[0]!="Z" and int(state[2])==pgid and int(state[3])==pgid:
                result.append(int(f.parent.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):
            pass
    return result

def stop_group(proc):
    if proc is None:
        return
    # Each owned Popen uses start_new_session=True. Its PID is both PGID and SID.
    pgid=proc.pid
    for sig,grace in ((signal.SIGTERM,15),(signal.SIGKILL,5)):
        if not active_group(pgid):
            break
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid,sig)
        deadline=time.monotonic()+grace
        while active_group(pgid) and time.monotonic()<deadline:
            time.sleep(.05)
    proc.wait(timeout=2)
    remaining=active_group(pgid)
    if remaining:
        raise RuntimeError(f"owned process group did not exit: {remaining}")

def worker(root):
    with locked_slot(root/"slot.lock") as pending:
        child=subprocess.Popen([sys.executable,"-c",
            "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']);"
            " print(p.pid,flush=True); time.sleep(120)"],stdout=subprocess.PIPE,text=True,start_new_session=True)
        try:
            child_pid=int(child.stdout.readline())
            (root/"ready").write_text(json.dumps([child.pid,child_pid]))
            while True:
                raise_if_requested(pending)
                time.sleep(.02)
        finally:
            stop_group(child)

def verify():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        proc=subprocess.Popen([sys.executable,"-m","parity.trace_lifecycle","--worker",d])
        try:
            limit=time.monotonic()+10
            while not (root/"ready").exists() and time.monotonic()<limit:
                time.sleep(.02)
            first_group=json.loads((root/"ready").read_text())
            before=active_group(first_group[0])
            with (root/"slot.lock").open("a") as witness:
                try:
                    fcntl.flock(witness,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    held=True
                else:
                    held=False
                proc.send_signal(signal.SIGTERM)
                result=proc.wait(timeout=25)
                after=active_group(first_group[0])
                fcntl.flock(witness,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if result!=143 or not held or not set(first_group).issubset(before) or after:
                raise RuntimeError("parent signal did not release the lock after terminating its complete group")
            return {"launcher_pid":proc.pid,"signal":"SIGTERM","exit":result,"owned_pids":first_group,
                    "alive_before":before,"alive_after":after,"lock_held_before":held,"lock_reacquired_after":True}
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

if __name__=="__main__":
    if len(sys.argv)==3 and sys.argv[1]=="--worker":
        worker(Path(sys.argv[2]))
    else:
        print(json.dumps(verify()))
