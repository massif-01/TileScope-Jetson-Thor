"""One profiled native llama.cpp request; startup is outside the CUDA activity window."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tilescope.common import device_slot, read_json, sha256, write_json
from tilescope.probe.engines import llama_cpp
from tilescope.probe.transport import capture, decoded, request_bytes


def ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])


def execute(out, argv):
    # Exec preserves PID/start_ticks. Capture the server's own startup messages even if nsys detaches.
    with (out / "server.log").open("xb") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        write_json(out / "owned.json", {"pid": os.getpid(), "start_ticks": ticks(os.getpid()), "argv": argv})
        os.execv(argv[0], argv)


def command(out, label, argv, timeout=120, check=True):
    begin = time.monotonic_ns()
    with (out / (label + ".log")).open("xb") as log:
        p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    write_json(out / (label + ".json"), {"argv": argv, "begin_ns": begin, "end_ns": time.monotonic_ns(),
                                         "exit_code": p.returncode})
    if check and p.returncode:
        raise RuntimeError(f"{label} exited {p.returncode}: see {out / (label+'.log')}")


def collect(target, out):
    from guard import guard, sample
    from tilescope.probe.identity import bind
    from tilescope.probe.resource_window import Monitor
    from tilescope.probe.targets import resolve
    profile = resolve(target)
    deploy = profile["deployment"]
    if profile["engine"]["adapter"] != "llama_cpp.native.v1" or deploy["mode"] != "owned_instance":
        raise ValueError("profile must select an owned native llama.cpp instance")
    endpoint = deploy["endpoint"]
    port = urlsplit(endpoint).port
    out.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise ValueError("trace port already in use")
    guard(sample())
    session = "tilescope-" + uuid4().hex
    args = deploy["launch_parameters"]
    argv = [deploy["runtime_executable"], "--verbose", "-m", deploy["checkpoint"], "--alias", deploy["served_model"],
            "--host", "127.0.0.1", "--port", str(port), "-ngl", str(args["gpu_layers"]),
            "-c", str(args["max_context"] * args["max_sequences"]), "-np", str(args["max_sequences"])]
    command(out, "nsys-version", ["nsys", "--version"])
    launch = ["nsys", "launch", "--session-new=" + session, "--trace=cuda", "--cuda-graph-trace=node",
              "--discard-environment=true", sys.executable, str(Path(__file__).resolve()), "_exec", str(out), *argv]
    write_json(out / "launch.json", {"argv": launch, "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
    with (out / "launch.log").open("xb") as log:
        launcher = subprocess.Popen(launch, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 150
            while time.monotonic() < deadline:
                guard(sample())
                if launcher.poll() not in (None, 0):
                    raise RuntimeError(f"nsys launch exited {launcher.returncode}")
                try:
                    with urlopen(endpoint + "/health", timeout=2) as health:
                        if health.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(.5)
            else:
                raise TimeoutError("trace-owned server did not become ready")
            target = bind(profile)
            owned, ident = read_json(out / "owned.json"), target["identity"]
            if (owned["pid"], owned["start_ticks"]) != (ident["server_pid"], ident["start_ticks"]):
                raise ValueError("profile listener is not the process created by this trace")
            write_json(out / "resolved-target.json", target)
            with Monitor(target, out / "resources.jsonl") as monitor:
                monitor.check()
                command(out, "start", ["nsys","start","--session="+session,"--sample=none","--cpuctxsw=none",
                                       "--output="+str(out/"activity")])
                body = dict(model=deploy["served_model"], prompt="A device measures the time required for one request. Explain the first step.",
                            seed=381127, temperature=0, stream=False, return_tokens=True, n_predict=16)
                raw = request_bytes(body)
                (out/"request.json").write_bytes(raw)
                meta = {"role":"diagnostic", "route":"/completion", "request_sha256":sha256(out/"request.json")}
                try:
                    capture(endpoint, "/completion", raw, out/"wire.jsonl", meta)
                finally:
                    write_json(out / "receipt.json", meta)
                    command(out, "stop", ["nsys","stop","--session="+session])
                monitor.check()
        finally:
            command(out, "shutdown", ["nsys","shutdown","--session="+session,"--kill=sigterm"], check=False)
            if (out/"owned.json").exists():
                owned = read_json(out/"owned.json")
                pid = owned["pid"]
                deadline = time.monotonic()+8
                while (Path(f"/proc/{pid}").exists() and ticks(pid)==owned["start_ticks"]
                       and Path(f"/proc/{pid}/exe").exists() and time.monotonic()<deadline):
                    time.sleep(.2)
                if Path(f"/proc/{pid}/exe").exists() and ticks(pid)==owned["start_ticks"]:
                    os.kill(pid, signal.SIGTERM)
                    deadline = time.monotonic()+15
                    while Path(f"/proc/{pid}/exe").exists() and ticks(pid)==owned["start_ticks"] and time.monotonic()<deadline:
                        time.sleep(.2)
                    if Path(f"/proc/{pid}/exe").exists() and ticks(pid)==owned["start_ticks"]:
                        raise RuntimeError(f"trace-owned server PID {pid} has not exited")
            if launcher.poll() is None:
                launcher.wait(timeout=30)
    report, = out.glob("activity*.nsys-rep")
    command(out, "export", ["nsys","export","--type=sqlite","--output="+str(out/"activity.sqlite"),str(report)])
    result = review(out)
    write_json(out/"review.json", result)
    return result


def review(out):
    target = read_json(out/"resolved-target.json")
    ident = target["identity"]
    owned = read_json(out/"owned.json")
    launch = read_json(out/"launch.json")
    if (owned["pid"] != ident["server_pid"] or owned["start_ticks"] != ident["start_ticks"] or
            owned["argv"] != ident["argv"] or launch["boot_id"] != ident["boot_id"] or
            target["deployment"]["host_identity"] != ident["device_uuid"]):
        raise ValueError("CUDA trace owner/device identity mismatch")
    meta = read_json(out/"receipt.json")
    start, stop = read_json(out/"start.json"), read_json(out/"stop.json")
    if (meta["http"] != 200 or meta["request_sha256"] != sha256(out/"request.json") or
            not start["end_ns"] <= meta["send_ns"] < meta["closed_ns"] <= stop["begin_ns"] or
            start["exit_code"] or stop["exit_code"]):
        raise ValueError("native request does not fall inside successful trace window")
    completion = llama_cpp.parse(decoded(out/"wire.jsonl",meta,False),read_json(out/"request.json"),
                                 target["deployment"]["served_model"])
    # Nsight's SQLite stores Linux PID in bits 24..47 of the global process ID.
    # Restrict to the socket-owning server and local CUDA device ordinal 0 on this single-GPU host.
    db = sqlite3.connect((out/"activity.sqlite").resolve().as_uri()+"?mode=ro", uri=True)
    try:
        if not db.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='CUPTI_ACTIVITY_KIND_KERNEL'").fetchone()[0]:
            raise ValueError("no CUPTI kernel activity table; no CUDA execution evidence")
        # Depending on the collector, the activity schema may expose node membership.
        columns = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        graph_nodes = None
        if "graphNodeId" in columns:
            graph_nodes = db.execute("""SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL
                                       WHERE ((globalPid >> 24) & 16777215) = ? AND graphNodeId > 0""",
                                     (owned["pid"],)).fetchone()[0]
        rows = db.execute("""
            SELECT start,end,deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE ((globalPid >> 24) & 16777215) = ? ORDER BY start,end
        """, (owned["pid"],)).fetchall()
    finally:
        db.close()
    if not rows or any(end <= begin or device != 0 for begin,end,device in rows):
        raise ValueError("no valid target-owned CUDA kernel intervals on the single Thor GPU")
    union, right = 0, rows[0][0]
    for begin,end,_ in rows:
        union += max(0, end-max(begin,right))
        right = max(right,end)
    log = (out/"server.log").read_text(errors="replace")
    offload = re.findall(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU",log)
    if not offload or int(offload[-1][0]) <= 0:
        raise ValueError("server log lacks positive GPU layer offload during this launch")
    return {"status":"complete","role":"diagnostic","source":"NVIDIA Nsight Systems CUDA activity",
            "producer_version":(out/"nsys-version.log").read_text().strip(),
            "pid":owned["pid"],"start_ticks":owned["start_ticks"],"boot_id":ident["boot_id"],
            "uuid":ident["device_uuid"],"kernel_count":len(rows),"kernel_union_ns":union,
            "first_kernel_trace_ns":rows[0][0],"last_kernel_trace_ns":right,
            "clock_domain":"Nsight trace relative nanoseconds, separate from host request clock",
            "offloaded_layers":[int(v) for v in offload[-1]], "request":completion,
            "activity_rows_with_graph_node_id":graph_nodes,
            "graph_execution":"observed positive graph node IDs" if graph_nodes else "undetermined",
            "scope":"one native generation in a private server; activity window opened after startup",
            "sqlite_sha256":sha256(out/"activity.sqlite")}


if __name__=="__main__":
    if sys.argv[1:2] == ["_exec"]:
        execute(Path(sys.argv[2]), sys.argv[3:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action",required=True)
    p = commands.add_parser("run")
    p.add_argument("--target",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p = commands.add_parser("review")
    p.add_argument("directory",type=Path)
    args = parser.parse_args()
    if args.action == "run":
        with device_slot():
            print(json.dumps(collect(args.target, args.out.resolve()),indent=2))
    else:
        print(json.dumps(review(args.directory),indent=2))
