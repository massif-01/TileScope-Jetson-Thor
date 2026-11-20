import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
        name = f.name
    os.replace(name, path)


def seal_evidence(attempt):
    attempt = Path(attempt)
    paths = [attempt / "command.json"]
    paths.extend(p for p in (attempt / "native").rglob("*") if p.is_file())
    write_json(attempt / "evidence-hashes.json",
               {str(p.relative_to(attempt)): sha256(p) for p in sorted(paths)})


def verify_evidence(attempt):
    attempt = Path(attempt).resolve()
    hashes = read_json(attempt / "evidence-hashes.json")
    required = {"command.json", "native/stdout", "native/stderr", "native/exit", "native/telemetry.jsonl"}
    if not required.issubset(hashes):
        raise ValueError("raw evidence manifest incomplete")
    observed = {str(p.relative_to(attempt)) for p in (attempt/"native").rglob("*") if p.is_file()}
    if set(hashes) != observed | {"command.json"}:
        raise ValueError("raw evidence manifest membership mismatch")
    for name, expected in hashes.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"evidence hash mismatch: {name}")


@contextmanager
def device_slot():
    """One project GPU job at a time for the shared Thor login."""
    import fcntl
    lock = Path(os.environ.get("TILESCOPE_LOCK", "~/.cache/tilescope/gpu-0.lock")).expanduser()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        try:
            yield file.fileno()
        finally:
            fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def task_contract(task):
    """Mathematical workload identity, separate from dispatch and timing choices."""
    excluded = {"task_id","adapter","groups","implementations","protocol","comparison_type",
                "source","phase"}
    return {k:v for k,v in task.items() if k not in excluded}
