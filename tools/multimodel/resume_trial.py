"""Interrupt one real fixed-length collection, then finish the same batch."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

from tilescope.common import read_json, sha256, write_json
from tilescope.probe.targets import resolve


def archive_files(paths):
    return {str(f):sha256(f) for path in paths for f in path.glob("*") if f.is_file()}


def verify_batch(target, suite, root, receipt):
    cmd=[sys.executable,"-m","tilescope","probe","run","--target",str(target),"--suite",str(suite),"--out",str(root)]
    with receipt.with_suffix(".stdout").open("x") as out,receipt.with_suffix(".stderr").open("x") as err:
        proc=subprocess.Popen(cmd,stdout=out,stderr=err,start_new_session=True)
        try:
            deadline=time.monotonic()+100
            while time.monotonic()<deadline and proc.poll() is None:
                waves=sorted(root.glob("waves/*.json"))
                if len(waves)>=2:
                    committed=[v for p in waves for v in read_json(p)]
                    pending=[p for p in root.glob("attempts/*") if p.name not in committed and
                             (p/"wire.jsonl").is_file() and (p/"wire.jsonl").stat().st_size>0 and not (p/"result.json").exists()]
                    if pending:
                        proc.send_signal(signal.SIGTERM)
                        break
                time.sleep(.01)
            else:
                raise RuntimeError("no real in-flight interruption observed")
            code=proc.wait(timeout=15)
            if code!=-signal.SIGTERM:
                raise ValueError("unexpected interrupted client status")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=15)
        completed=archive_files([root/"attempts"/i for i in committed])
        interrupted=archive_files(pending)
        # The client socket has closed. Allow the scheduler to observe it before the next serial window.
        time.sleep(3)
        resume=subprocess.run(cmd+["--resume"],stdout=out,stderr=err)
    same=all(sha256(p)==h for p,h in {**completed,**interrupted}.items())
    outcome={"command":cmd,"signal":"SIGTERM","exit":code,"resume_exit":resume.returncode,
             "completed_hashes_before":completed,"partial_hashes_before":interrupted,"preserved_after_resume":same,
             "partial":[p.name for p in pending]}
    write_json(receipt,outcome)
    if not same or resume.returncode:
        raise RuntimeError("resume did not complete and preserve existing evidence")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("target","suite","out","receipt"):
        p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args()
    verify_batch(a.target,a.suite,a.out,a.receipt)
