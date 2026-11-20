"""Import only the files listed in the archived Hub API receipt."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

lock, destination = sys.argv[1:3]
for repo, model in json.loads(Path(lock).read_text()).items():
    directory = Path(destination) / repo.split("/")[-1]
    directory.mkdir(parents=True, exist_ok=True)
    for spec in model["files"]:
        name = spec["rfilename"]
        path = directory / name
        expected = spec.get("lfs", {}).get("sha256", spec["blobId"])
        def verify(p):
            if not p.is_file() or p.stat().st_size != spec["size"]:
                return False
            h = hashlib.sha256() if "lfs" in spec else hashlib.sha1(f"blob {spec['size']}\0".encode())
            with p.open("rb") as source:
                for part in iter(lambda: source.read(4 << 20), b""):
                    h.update(part)
            return h.hexdigest() == expected
        if not verify(path):
            temp = path.with_suffix(path.suffix + ".part")
            url = f"https://huggingface.co/{repo}/resolve/{model['revision']}/{name}"
            subprocess.run(["curl","--fail","--location","--silent","--show-error","--retry","2",
                            "--connect-timeout","12","--speed-time","45","--speed-limit","1024",
                            "--continue-at","-","--output",str(temp),url],check=True)
            if not verify(temp):
                raise ValueError(f"Hub content mismatch for {name}")
            temp.replace(path)
        print(repo, name, expected, flush=True)
