"""Attest tracked/source bytes in an independently restored configuration challenge."""
import argparse
import json
from pathlib import Path
import subprocess

from tilescope.common import digest, read_json, sha256, write_json


def snapshot(repo):
    def git(*argv):
        return subprocess.check_output(["git","-C",str(repo),*argv])
    tracked=git("ls-files","-z").decode().strip("\0").split("\0")
    return {"commit":git("rev-parse","HEAD").decode().strip(),
            "source_tree_sha256":digest({n:sha256(repo/n) for n in sorted(tracked)}),
            "tracked_dirty":git("diff","--name-only","HEAD").decode().splitlines(),
            "added_files":git("ls-files","--others","--exclude-standard","-z").decode().strip("\0").split("\0") or []}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("step",choices=["before","after"])
    ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--evidence",type=Path,required=True)
    ap.add_argument("--allow",nargs="*",default=[])
    a=ap.parse_args()
    d=snapshot(a.repo)
    d["added_files"]=[n for n in d["added_files"] if n]
    if d["tracked_dirty"] or sorted(d["added_files"])!=sorted(a.allow):
        raise ValueError(f"unexpected files in configuration challenge: {d}")
    if a.step=="after":
        original=read_json(a.evidence/"before.json")
        if (original["commit"],original["source_tree_sha256"])!=(d["commit"],d["source_tree_sha256"]):
            raise ValueError("frozen code changed in challenge")
        for n in a.allow:
            p=Path(n)
            if p.suffix!=".json" or p.parts[:2] not in {("profiles","models"),("profiles","targets")}:
                raise ValueError("challenge permits only additive model and target JSON")
        d["configuration_sha256"]={n:sha256(a.repo/n) for n in a.allow}
        d["config_only"]=True
    write_json(a.evidence/(a.step+".json"),d)
