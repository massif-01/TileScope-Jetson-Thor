"""Bind a loopback listener, its build, device and checkpoint to a frozen profile."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen

from tilescope.common import digest, read_json, sha256, write_json
from parity.checkpoint import inspect as inspect_index


def output(*args):
    return subprocess.check_output(args, text=True, timeout=30).strip()


def artifact(model, checkpoint, cache):
    storage = model["storage"]
    root = checkpoint.parent if storage["kind"] == "gguf" else checkpoint
    expected = storage["sha256"]
    if not expected:
        raise ValueError("profile lacks verified weight identity")
    files = {}
    for name, wanted in sorted(expected.items()):
        if name != Path(name).name:
            raise ValueError("invalid weight filename")
        p = root / name
        stat = p.stat()
        stamp = [str(p.resolve()),stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        receipt = cache / (digest(stamp) + ".json")
        checksum = read_json(receipt)["sha256"] if receipt.exists() else sha256(p)
        if checksum != wanted:
            raise ValueError(f"weight identity mismatch: {p}")
        if not receipt.exists():
            write_json(receipt, {"stat":stamp,"sha256":checksum})
        files[name] = checksum
    if storage["kind"] == "gguf":
        from gguf import GGUFReader
        reader = GGUFReader(str(checkpoint))
        keys = ["general.architecture","general.file_type","general.name","tokenizer.chat_template"]
        meta = {k:reader.fields[k].contents() for k in keys}
        template = meta.pop("tokenizer.chat_template")
        return {"files":files,"storage":storage["kind"],"metadata":meta,"tensors":len(reader.tensors),
                "template_sha256":hashlib.sha256(template.encode()).hexdigest()}
    cfg = read_json(root/"config.json")
    if cfg["architectures"] != model["expected_architectures"]:
        raise ValueError("checkpoint architecture mismatch")
    for name in ("config.json","tokenizer.json","tokenizer_config.json","chat_template.jinja",
                 "model.safetensors.index.json","generation_config.json"):
        if (root/name).is_file():
            files[name] = sha256(root/name)
    template = ((root/"chat_template.jinja").read_text() if (root/"chat_template.jinja").exists()
                else read_json(root/"tokenizer_config.json")["chat_template"])
    if storage["kind"] == "safetensors_sharded":
        header = inspect_index(root)
        if set(header["shards"]) != set(expected):
            raise ValueError("shard set differs from profile")
    elif storage["kind"] == "safetensors_single":
        weights = storage["weights"]
        if len(weights)!=1 or weights[0] not in expected:
            raise ValueError("single safetensors requires one pinned weight file")
        name = weights[0]
        p = root/name
        with p.open("rb") as f:
            n, = struct.unpack("<Q",f.read(8))
            if not 2 <= n <= 100_000_000:
                raise ValueError("invalid safetensors header size")
            raw = f.read(n)
        tensors = json.loads(raw)
        tensors.pop("__metadata__",None)
        end = 0
        import math
        widths = {"BF16":2,"F32":4,"F16":2,"I64":8}
        for item in sorted(tensors.values(),key=lambda i:i["data_offsets"][0]):
            start,stop = item["data_offsets"]
            if start != end or stop-start != widths[item["dtype"]]*math.prod(item["shape"]):
                raise ValueError("invalid tensor extent")
            end = stop
        if n+8+end != p.stat().st_size:
            raise ValueError("incomplete safetensors file")
        header = {"tensor_count":len(tensors),"dtypes":dict(Counter(t["dtype"] for t in tensors.values())),
                  "header_sha256":hashlib.sha256(raw).hexdigest()}
        adapter = model.get("deep_adapter")
        if adapter and adapter["id"] == "dense_rms_linear.v1":
            header["selected"] = {k:tensors[adapter[k]] for k in ("norm_weight_key","linear_weight_key")}
    else:
        raise ValueError("unknown weight storage")
    return {"files":files,"storage":storage["kind"],"architecture":cfg["architectures"],
            "template_sha256":digest(template),"header":header}


def bind(target):
    d = target["deployment"]
    cp = Path(d["checkpoint"]).resolve()
    url = d["endpoint"]
    port = urlsplit(url).port
    ss = output("ss","-Hlnpt",f"sport = :{port}")
    pids = set(re.findall(r"pid=(\d+)",ss))
    if len(pids) != 1:
        raise ValueError(f"listener has no unique local owner on port {port}")
    pid = int(pids.pop())
    proc = Path("/proc")/str(pid)
    ticks = int((proc/"stat").read_text().rsplit(")",1)[1].split()[19])
    argv = (proc/"cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if cp not in [Path(s).resolve() for s in argv if s.startswith("/")]:
        raise ValueError("local listener command does not name profile checkpoint")
    if Path(d["runtime_executable"]).resolve() != (proc/"exe").resolve():
        raise ValueError("local listener executable differs from profile")
    def get(route):
        with urlopen(url+route,timeout=10) as response:
            return json.load(response)
    listed = [x for x in get("/v1/models")["data"] if x["id"]==d["served_model"]]
    if len(listed)!=1:
        raise ValueError("served model alias mismatch")
    engine = target["engine"]
    if engine["adapter"] == "vllm.local.v1":
        version = get("/version")["version"]
        if version!=engine["build_pin"] or Path(listed[0]["root"]).resolve()!=cp:
            raise ValueError("vLLM model root/build mismatch")
    else:
        props = get("/props")
        if Path(props["model_path"]).resolve()!=cp:
            raise ValueError("llama.cpp model path mismatch")
        version = {"props_build":props["build_info"],"binary_sha256":sha256(proc/"exe")}
        loaded={Path(line.split()[-1]) for line in (proc/"maps").read_text().splitlines()
                if line.split()[-1].startswith("/") and ".so" in line.split()[-1]}
        expected_libs=engine["libraries_sha256"]
        observed_libs={p.name:sha256(p) for p in loaded if p.name in expected_libs}
        if observed_libs!=expected_libs:
            raise ValueError("loaded llama.cpp libraries differ from frozen CUDA build")
        version["loaded_build_libraries"]=observed_libs
        if version["binary_sha256"] != engine["binary_sha256"]:
            raise ValueError("llama.cpp binary differs from frozen build")
    uuid = output("nvidia-smi","--query-gpu=uuid","--format=csv,noheader")
    if uuid != d["host_identity"]:
        raise ValueError("endpoint GPU identity mismatch")
    store = artifact(target["model"],cp,Path.home()/".cache/tilescope/file-identities")
    ident = {"server_pid":pid,"start_ticks":ticks,"boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
             "device_uuid":uuid,"version":version,"argv":argv,"listener":ss,"artifact":store,
             "engine_replica":"0","checkpoint":str(cp),"served_model":d["served_model"]}
    target["identity"] = ident
    target["target_fingerprint"] = digest([target["profile_sha256"],ident])
    return target
