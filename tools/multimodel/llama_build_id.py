"""Record the local CUDA build and bind the profile to its server and linked libraries."""
import argparse
import json
from pathlib import Path
import subprocess

from tilescope.common import read_json, sha256, write_json


def run(build):
    binary=build/"bin/llama-server"
    def process(argv):
        r=subprocess.run(argv,text=True,capture_output=True,check=True)
        return {"argv":list(map(str,argv)),"stdout":r.stdout,"stderr":r.stderr}
    libs={p.name:sha256(p) for p in sorted(binary.parent.glob("*.so*")) if not p.is_symlink()}
    if not any("ggml-cuda" in k for k in libs):
        raise RuntimeError("CUDA backend binary missing")
    return {"binary":str(binary),"sha256":sha256(binary),"libraries":libs,"version":process([binary,"--version"]),
            "ldd":process(["ldd",binary]),"cmake_cache_sha256":sha256(build/"CMakeCache.txt"),
            "ui_asset_sha256":sha256(build/"ui/apps/chat/dist.tar.gz")}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--build",required=True,type=Path)
    cli.add_argument("--out",required=True,type=Path)
    a=cli.parse_args()
    write_json(a.out,run(a.build.resolve()))
