"""HTTP entity bytes and their host-monotonic observation times, shared by both engines."""
import base64
import http.client
import json
import time
from urllib.parse import urlsplit


def request_bytes(body):
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def connect(endpoint, timeout=160):
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.username or parsed.path.rstrip("/") or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be a local HTTP origin")
    return http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)


def fetch(endpoint, path, body=None):
    conn = connect(endpoint, 20)
    try:
        data = request_bytes(body) if body is not None else None
        conn.request("POST" if data is not None else "GET", path, data, {"Content-Type":"application/json"})
        response = conn.getresponse()
        result = response.read()
        if response.status != 200:
            raise ValueError(f"{path}: HTTP {response.status}: {result[:500]!r}")
        return result
    finally:
        conn.close()


def capture(endpoint, route, data, path, meta):
    conn = connect(endpoint)
    try:
        with path.open("x", buffering=1) as wire:
            meta["wall_start_ns"] = time.time_ns()
            meta["send_ns"] = time.monotonic_ns()
            conn.request("POST", route, data, {"Content-Type":"application/json"})
            response = conn.getresponse()
            meta.update(http=response.status, headers=list(response.getheaders()))
            while block := response.read1(65536):
                ns = time.monotonic_ns()
                wire.write(json.dumps({"ns":ns, "data_b64":base64.b64encode(block).decode()})+"\n")
                if ns - meta["send_ns"] > 160_000_000_000:
                    raise TimeoutError("request exceeded 160s")
    finally:
        conn.close()
        meta["closed_ns"] = time.monotonic_ns()
        meta["wall_end_ns"] = time.time_ns()


def wire_blocks(path, send_ns, closed_ns):
    previous = send_ns
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            ns = row["ns"]
            if type(ns) != int or not previous <= ns <= closed_ns:
                raise ValueError("raw chunk outside request clock interval")
            previous = ns
            yield ns, base64.b64decode(row["data_b64"], validate=True)


def decoded(path, meta, stream):
    blocks = wire_blocks(path, meta["send_ns"], meta["closed_ns"])
    if not stream:
        data = bytearray()
        last = None
        for last, block in blocks:
            data.extend(block)
        if last is None:
            raise ValueError("empty HTTP response")
        yield last, json.loads(data.decode("utf-8", errors="strict"))
        return
    # read1() may split a UTF-8 scalar, an SSE field, or coalesce several events.
    pending = b""
    fields = []
    for ns, block in blocks:
        pending += block
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            line = line.removesuffix(b"\r").decode("utf-8", errors="strict")
            if not line:
                if fields:
                    payload = "\n".join(fields)
                    yield ns, "[DONE]" if payload == "[DONE]" else json.loads(payload)
                    fields.clear()
            elif line.startswith("data:"):
                fields.append(line[5:].removeprefix(" "))
            elif not line.startswith((":","id:","event:","retry:")):
                raise ValueError("unrecognized SSE field")
    if pending or fields:
        raise ValueError("truncated SSE frame")
