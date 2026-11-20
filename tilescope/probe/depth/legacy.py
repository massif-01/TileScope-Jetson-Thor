"""Profile entrance for the accepted Qwen MTP, graph and two-operation oracle."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from parity.trace_client import write
from parity.trace_review import consume, lines, parse_profiles

from .client import guard
from .pins import PINS
from .review import cancellation, raw_request


def collect_legacy(args, target, model, deployment):
    from parity.trace import launch
    d = deployment
    checkpoint = Path(d["checkpoint"])
    with urlopen(d["endpoint"].rstrip("/")+"/v1/models",timeout=5) as r:
        matched = [m for m in json.load(r)["data"] if m["id"] == d["served_model"]]
    if len(matched) != 1 or Path(matched[0]["root"]).resolve() != checkpoint.resolve():
        raise ValueError("legacy attached service/checkpoint mismatch")
    config = json.loads((checkpoint/"config.json").read_text())
    if not set(config["architectures"]).intersection(model["expected_architectures"]):
        raise ValueError("legacy architecture mismatch")
    verify = ("import hashlib,importlib.metadata as m,json,sys; d=m.distribution('vllm'); "
              "print(json.dumps({n:hashlib.sha256(d.locate_file(n.replace('.','/')+'.py').read_bytes())"
              ".hexdigest() for n in sys.argv[1:]}))")
    installed = json.loads(subprocess.check_output([d["runtime_executable"],"-c",verify,*PINS], text=True))
    if installed != PINS:
        raise ValueError(f"unreviewed legacy vLLM hook source: {installed}")
    root = args.out.resolve()
    def client_command(directory):
        write(directory/"depth-plan.json", {"target_id":target["target_id"],"model":model,
               "checkpoint":str(checkpoint), "endpoint":f"http://127.0.0.1:{args.port}",
               "served_model":d["served_model"], "config":config, "vllm_source_sha256":installed,
               "target_sha256":hashlib.sha256(args.target.read_bytes()).hexdigest(),
               "attached_model":matched[0], "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()})
        return [sys.executable, "-m", "tilescope.probe.depth", "legacy-client", "--out", str(directory)]
    launch(argparse.Namespace(output=root, python=Path(d["runtime_executable"]), checkpoint=checkpoint,
           port=args.port, max_len=4096, panel="multimodel", timeout=args.timeout, min_free_gib=12),
           client_command=client_command, guard_tick=guard, sequence=["local replay", "profiled", "cancel", "followup"])
    result = review_legacy(root)
    write(root/"legacy-depth-review.json", result)
    print(json.dumps({"kernel_calls":result["gpu"]["calls"],"graph_replays":result["profile"]["graph_replays"],
                      "drafts_verified":result["profile"]["verified_draft_count"],
                      "cancel":result["cancel"], "norm_and_projection_pass":True}))


def review_legacy(root):
    rows = sorted((r for p in root.glob("engine-*.jsonl") for r in lines(p)), key=lambda r:r["mono_ns"])
    plan = json.loads((root/"depth-plan.json").read_text())
    shutdown = json.loads((root/"shutdown.json").read_text())
    if shutdown["client"] != 0 or shutdown["remaining_same_identity"] or not shutdown["original_service"]:
        raise ValueError("private legacy service cleanup failed")
    observations = [r for r in rows if r["event"]=="instrumentation"]
    if ({r["module"]:r["sha256"] for r in observations} != PINS
            or any(r["sha256"] != PINS[r["module"]] for r in observations)):
        raise ValueError("legacy hook source identity mismatch")
    reqs = {k:raw_request(root,k) for k in ("layer-replay","profile","cancel","followup")}
    if any(r["request"]["model"] != plan["served_model"] for r in reqs.values()):
        raise ValueError("legacy request/checkpoint alias drift")
    derived = {k:consume(r,rows) for k,r in reqs.items()}
    profile = derived["profile"]
    if (not profile["graph_replays"].get("target_forward") or not profile["verified_draft_count"] or
            not profile["actual_draft_shapes"] or not profile["batch_sizes"]):
        raise ValueError("missing executed target Graph/MTP/commit evidence")
    replay = derived["layer-replay"]["internal_id"]
    local = [r for r in rows if r["event"]=="selected_local_replay" and r["ids"] == [replay]]
    refs = [r for r in rows if r["event"]=="selected_numerical_reference" and r["ids"] == [replay]]
    if len(local)!=1 or len(refs)!=1 or refs[0]["step"]!=local[0]["step"]:
        raise ValueError("legacy local reference/observed tensors missing")
    proof = refs[0]["result"]
    if not proof["embedding_matches_checkpoint"] or proof["prompt_prefix_ids"] != reqs["layer-replay"]["request"]["prompt"][:2]:
        raise ValueError("legacy observed inputs do not match checkpoint and request")
    for op, size, tol in (("norm",5120,(.006,.0002,.003)), ("ba",96,(.009,.002,.004))):
        r = proof[op]
        if (r["actual_bf16_sha256"] != local[0][op]["sha256"] or r["shape"]!=[2,size] or r["n"]!=size*2
                or (r["rtol"],r["rms_atol"],r["l2_limit"])!=tol or not r["finite"] or not r["passed"]
                or r["mismatches"]!=0 or not 0<=r["relative_l2"]<=tol[2]
                or not math.isclose(r["relative_l2"],r["rmse"]/r["reference_rms"],rel_tol=1e-8,abs_tol=1e-14)
                or not r["rmse"]-1e-14<=r["max_abs"]<=r["rmse"]*math.sqrt(r["n"])+1e-14):
            raise ValueError("legacy numerical data or limits disagree for "+op)
    return {"target":plan["target_id"],"role":"diagnostic","profile":profile,
            "model":plan["model"]["model_id"],"gpu":parse_profiles(root,reqs["profile"],rows),
            "local_numerics":proof,"shutdown":shutdown,"cancel":cancellation(rows,derived,reqs)}
