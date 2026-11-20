"""Resolve static model, engine and workload descriptions without loading GPU libraries."""
from pathlib import Path
import json
from urllib.parse import urlsplit

from tilescope.common import digest, sha256

ENGINES = {"vllm.local.v1", "llama_cpp.native.v1"}


def document(path, name):
    data = json.loads(path.read_bytes())
    if data.get("schema") != f"tilescope.{name}.v1":
        raise ValueError(f"wrong {name} schema in {path}")
    if '"__' in json.dumps(data) or "template_status" in data:
        raise ValueError(f"unresolved design template: {path}")
    return data


def resolve(path):
    path = Path(path).resolve()
    target = document(path, "target")
    hashes = {"target": sha256(path)}
    for kind in ("model", "engine"):
        ref = (path.parent / target[kind + "_ref"]).resolve()
        target[kind] = document(ref, kind)
        hashes[kind] = sha256(ref)
    deployment = target["deployment"]
    endpoint = urlsplit(deployment["endpoint"])
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or not endpoint.port or endpoint.path or endpoint.username:
        raise ValueError("deployment requires an explicit local loopback HTTP port")
    for key in ("runtime_executable", "checkpoint"):
        if not Path(deployment[key]).is_absolute():
            raise ValueError(f"expected absolute {key}")
    if target["engine"]["adapter"] not in ENGINES:
        raise ValueError("unknown engine adapter")
    if deployment["mode"] not in {"attach_existing","owned_instance"}:
        raise ValueError("unknown deployment mode")
    target.update(schema="tilescope.resolved-target.v1", source_hashes=hashes)
    target["profile_sha256"] = digest(target)
    return target


def workload(path):
    path = Path(path).resolve()
    suite = document(path, "suite")
    if "corpus_ref" in suite:
        ref = path.parent / suite["corpus_ref"]
        suite["corpus"] = document(ref, "corpus")
    return suite
