"""Verify the exact membership and bytes of a locally delivered TileScope ZIP."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def verify(path):
    with zipfile.ZipFile(path) as package:
        files=[s for s in package.namelist() if not s.endswith("/")]
        if len(set(files))!=len(files):raise ValueError("duplicate archive entry")
        checks={}
        for line in package.read("MANIFEST.sha256").decode().splitlines():
            expected,name=line.split("  ",1)
            if name in checks:raise ValueError("duplicate manifest entry")
            checks[name]=expected
        if set(files)!=set(checks)|{"MANIFEST.sha256"}:raise ValueError("manifest and archive membership differ")
        for name,expected in checks.items():
            sha=hashlib.sha256()
            with package.open(name) as f:
                for chunk in iter(lambda:f.read(1024*1024),b""):sha.update(chunk)
            if sha.hexdigest()!=expected:raise ValueError("checksum mismatch: "+name)
        ident=json.loads(package.read("DELIVERY.json"))
        if ident["git_bundle_sha256"]!=checks["TileScope.bundle"]:raise ValueError("Git bundle identity mismatch")
        return {"archive":path.name,"checked":len(checks),"commit":ident["commit"],"verified":True}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path);args=cli.parse_args()
    print(json.dumps(verify(args.archive),indent=2))
