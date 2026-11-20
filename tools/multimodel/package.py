"""Export the exact Git tree and self-contained bundle with a verifiable ZIP manifest."""
import argparse
import hashlib
import json
import re
import posixpath
from pathlib import Path
import subprocess
import tempfile
import zipfile

from parity.verify_delivery import verify


def export(output, challenge, verification):
    branch=subprocess.check_output(["git","branch","--show-current"],text=True).strip()
    commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    with tempfile.TemporaryDirectory() as temp:
        work=Path(temp)
        bundle=work/"TileScope.bundle"
        subprocess.run(["git","bundle","create",str(bundle),branch],check=True)
        subprocess.run(["git","bundle","verify",str(bundle)],check=True)
        tree=work/"tree.zip"
        subprocess.run(["git","archive","--format=zip","-o",str(tree),commit],check=True)
        hashes={}
        with zipfile.ZipFile(output,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as dest, zipfile.ZipFile(tree) as src:
            def add(name,data,mode=0o100644):
                if name in hashes:
                    raise ValueError("duplicate export member "+name)
                hashes[name]=hashlib.sha256(data).hexdigest()
                member=zipfile.ZipInfo(name)
                member.compress_type=zipfile.ZIP_DEFLATED
                member.external_attr=mode << 16
                dest.writestr(member,data)
            for info in src.infolist():
                if not info.is_dir():
                    add("source/"+info.filename,src.read(info),info.external_attr >> 16)
            add("TileScope.bundle",bundle.read_bytes())
            add("configuration-challenge.bundle",challenge.read_bytes())
            for path in sorted(verification.rglob("*")):
                if path.is_file():
                    add("validation/"+path.relative_to(verification).as_posix(),path.read_bytes())
            page=src.read("docs/multimodel/REPORT.md").decode()
            def root_link(match):
                link=match[1]
                if link.startswith(("#","https://","http://")):
                    return match[0]
                return "]("+posixpath.normpath(posixpath.join("source/docs/multimodel",link))+")"
            add("README.md",re.sub(r"\]\(([^)]+)\)",root_link,page).encode())
            add("DELIVERY.json",(json.dumps({"branch":branch,"commit":commit,"git_bundle_sha256":hashes["TileScope.bundle"],
                "challenge_bundle_sha256":hashes["configuration-challenge.bundle"],"acceptance":"source/docs/multimodel/ACCEPTANCE.json",
                "independent_final_checks":"validation/summary.json","entry":"source/docs/multimodel/REPORT.md","study":"source/evidence/multimodel/study.json"},indent=2)+"\n").encode())
            dest.writestr("MANIFEST.sha256","".join(f"{h}  {n}\n" for n,h in sorted(hashes.items())))
    return verify(output)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out",type=Path)
    cli.add_argument("--challenge-bundle",required=True,type=Path)
    cli.add_argument("--verification",required=True,type=Path)
    a=cli.parse_args()
    print(json.dumps(export(a.out,a.challenge_bundle,a.verification),indent=2))
