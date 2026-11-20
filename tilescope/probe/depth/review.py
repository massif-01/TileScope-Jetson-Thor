"""Recompute dense errors, selected reference rows and identity-bound CUDA evidence on CPU."""
import hashlib
import json
import math
import re
import struct
from pathlib import Path

from parity.trace_review import consume, lines, parse_profiles

THRESHOLDS = {"atol":0.01, "rtol":0.02, "relative_l2_max":0.01}


def statistics(actual, reference):
    if len(actual) == 0 or len(actual) != len(reference):
        raise ValueError("shape/empty error in selected output")
    if not all(math.isfinite(v) for row in (actual, reference) for v in row):
        raise ValueError("nonfinite observed or mathematical output")
    errors = [abs(a - b) for a, b in zip(actual, reference)]
    normalized = [e / (THRESHOLDS["atol"] + THRESHOLDS["rtol"] * abs(r))
                  for e, r in zip(errors, reference)]
    ref2 = math.fsum(r*r for r in reference)
    error2 = math.fsum(e*e for e in errors)
    if ref2 == 0:
        raise ValueError("relative L2 undefined for zero-norm reference")
    rel = math.sqrt(error2/ref2)
    return {"n":len(actual), "all_finite":True, "max_abs":max(errors),
            "relative_l2":rel, "reference_rms":math.sqrt(ref2/len(actual)),
            "error_rms":math.sqrt(error2/len(actual)), "max_normalized_error":max(normalized),
            "violations":sum(v > 1 for v in normalized),
            "passed":max(normalized) <= 1 and rel <= THRESHOLDS["relative_l2_max"]}


def tensor(root, field, shape, dtype):
    path = (root / field["file"]).resolve()
    if not path.is_relative_to((root / "tensors").resolve()):
        raise ValueError("invalid tensor path")
    data = path.read_bytes()
    if (field["shape"] != list(shape) or field["dtype"] != dtype or
            hashlib.sha256(data).hexdigest() != field["sha256"]):
        raise ValueError("tensor identity/shape/type mismatch")
    if dtype == "float64":
        values = [r[0] for r in struct.iter_unpack("<d", data)]
    else:
        values = [struct.unpack("<f", b"\0\0" + r[0])[0] for r in struct.iter_unpack("2s", data)]
    if len(values) != math.prod(shape) or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid tensor byte length or nonfinite values")
    return values


def dense_review(root, req, internal):
    d = json.loads((root / "dense.json").read_text())
    plan = json.loads((root / "depth-plan.json").read_text())
    if (d["thresholds"] != THRESHOLDS or d["positions"] != [0,1] or d["engine_request_id"] != internal or
            d["prompt_ids"] != req["request"]["prompt"][:2] or d["epsilon"] != plan["config"]["rms_norm_eps"] or
            d["model_origin"] != plan["model"]["origin"] or set(d["ops"]) != {"norm", "linear"}):
        raise ValueError("selected numerical evidence identity or contract mismatch")
    hidden, intermediate = plan["config"]["hidden_size"], plan["config"]["intermediate_size"]
    result = {}
    for op, info in d["ops"].items():
        width = hidden if op == "norm" else intermediate
        expected_weight = [hidden] if op == "norm" else [hidden, intermediate]
        adapter = plan["model"]["deep_adapter"]
        if (info["weight_shape"] != expected_weight or info["weight_dtype"] != "bfloat16" or
                not info["loaded_weight_matches"] or not re.fullmatch(r"[0-9a-f]{64}", info["weight_sha256"]) or
                info["device"] != "cuda:0" or info["weight_key"] != adapter[f"{op}_weight_key"] or
                not ("." + info["path"]).endswith("." + adapter[f"logical_{op}_path"])):
            raise ValueError("runtime/checkpoint shape, identity or selected module mismatch")
        x = tensor(root, info["input"], (2, width), "bfloat16")
        y = tensor(root, info["actual"], (2, hidden), "bfloat16")
        ref = tensor(root, info["reference"], (2, hidden), "float64")
        if op == "norm":
            w = tensor(root, info["weight"], (width,), "bfloat16")
            if info["weight"]["sha256"] != info["weight_sha256"]:
                raise ValueError("RMSNorm scale digest mismatch")
            oracle = []
            for off in (0, width):
                row = x[off:off+width]
                inv = 1 / math.sqrt(math.fsum(v*v for v in row)/width + d["epsilon"])
                oracle.extend(v*inv*s for v,s in zip(row,w))
            comparison = zip(oracle, ref)
        else:
            indices = sorted({0, hidden//2, hidden-1})
            if info["sample_weight_indices"] != indices:
                raise ValueError("missing independent projection weight rows")
            weights = tensor(root, info["sample_weights"], (len(indices), width), "bfloat16")
            bias = tensor(root, info["bias"], (hidden,), "bfloat16") if info["bias"] else None
            pairs = []
            for i in (0,1):
                for j, k in enumerate(indices):
                    v = math.fsum(a*b for a,b in zip(x[i*width:(i+1)*width], weights[j*width:(j+1)*width]))
                    pairs.append((v + (bias[k] if bias else 0), ref[i*hidden+k]))
            comparison = pairs
        if any(not math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-13) for a,b in comparison):
            raise ValueError(f"{op} independent reference reconstruction mismatch")
        metrics = statistics(y, ref)
        if metrics != info["metrics"] or not metrics["passed"]:
            raise ValueError(f"{op} recorded error metrics differ or numerical thresholds exceeded")
        result[op] = metrics
    return result


def raw_request(root, label):
    directory = root/"http"
    receipt = json.loads((directory/(label+".json")).read_text())
    if json.loads((directory/(label+".request")).read_bytes()) != receipt["request"]:
        raise ValueError("diagnostic raw request and receipt disagree")
    response = (directory/(label+".response")).read_bytes()
    if receipt["request"]["stream"]:
        payloads = [json.loads(line[5:].strip()) for line in response.splitlines()
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]"]
        if payloads != [c["payload"] for c in receipt["chunks"]]:
            raise ValueError("diagnostic SSE payloads and receipt disagree")
    elif json.loads(response) != receipt["response"]:
        raise ValueError("diagnostic raw response and receipt disagree")
    return receipt


def review(root):
    plan = json.loads((root/"depth-plan.json").read_text())
    modes = json.loads((root/"diagnostics.json").read_text())
    rows = sorted((r for p in root.glob("engine-*.jsonl") for r in lines(p)), key=lambda r:r["mono_ns"])
    run = {}
    raw = {}
    labels = ["dense"] + (["profile"] if modes["profile"] else []) + (["cancel","followup"] if modes["cancel"] else [])
    for label in labels:
        raw[label] = raw_request(root,label)
        req = raw[label]
        if req["request"]["model"] != plan["served_model"]:
            raise ValueError("unexpected diagnostic served model")
        run[label] = consume(req, rows)
    engine_id = run["dense"]["internal_id"]
    worker = [r for r in rows if r["event"] == "dense_observed" and r["ids"] == [engine_id]]
    if len(worker) != 1 or not run["dense"].get("target_dispatch"):
        raise ValueError("missing unique real request/worker observation or dispatch")
    worker = worker[0]
    matched = [r for r in rows if r["event"] == "instrumentation" and
               r["module"] == "vllm.v1.worker.gpu_model_runner" and r["pid"] == worker["pid"]
               and r["start_ticks"] == worker["pid_start_ticks"] and r["boot_id"] == plan["boot_id"]]
    if len(matched) != 1:
        raise ValueError("worker identity mismatch")
    result = {"model":plan["model"]["model_id"], "target_id":plan["target_id"], "role":"diagnostic",
              "engine_worker":matched[0], "requests":run,
              "local_numerics":dense_review(root, raw["dense"], engine_id)}
    if modes["profile"]:
        gpu = parse_profiles(root, raw["profile"], rows)
        stages = gpu["stages_by_cupti_launch_correlation"]
        if not any(r["stage"].endswith("/target_forward") and r["calls"] > 0 for r in stages):
            raise ValueError("CUDA launches not correlated with this worker/request forward markers")
        result["gpu"] = gpu
    if modes["cancel"]:
        result["cancel"] = cancellation(rows, run, raw)
    return result


def cancellation(rows, run, raw):
    req = raw["cancel"]
    rid = run["cancel"]["internal_id"]
    stopped = [r for r in rows if r["event"]=="external_finish" and rid in r["ids"]
               and r["status"]=="FINISHED_ABORTED"]
    if len(stopped) != 1:
        raise ValueError("no unique scheduler abort for canceled request")
    sent, closed = req["client_close_start_ns"], req["end_ns"]
    terminal, follow = stopped[0]["mono_ns"], raw["followup"]["end_ns"]
    if not (sent <= closed <= follow and sent <= terminal < raw["followup"]["start_ns"] < follow):
        raise ValueError("invalid cancel/scheduler/followup times")
    return {"client_close_start_ns":sent, "client_closed_ns":closed,
                        "engine_terminal_ns":terminal, "engine_terminal":stopped[0]["status"],
                        "engine_request_id":rid,"followup_completed_ns":follow,
                        "client_close_to_scheduler_abort_ms":(terminal-sent)/1e6}
