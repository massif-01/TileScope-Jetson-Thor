"""Capture two real first-prefill rows and build the CPU FP64 reference outside timing."""
import hashlib
import json
from pathlib import Path


def attach(model, root, rid, prompt, captures, handles):
    import torch
    plan = json.loads((root / "depth-plan.json").read_text())
    adapter = plan["model"]["deep_adapter"]
    captures.update(engine_request_id=rid, prompt_ids=prompt, checkpoint=plan["checkpoint"], ops={})
    config = json.loads((Path(plan["checkpoint"]) / "config.json").read_text())
    captures["epsilon"] = config["rms_norm_eps"]
    for op in ("norm", "linear"):
        selector = adapter[f"logical_{op}_path"]
        matches = [(name, mod) for name, mod in model.named_modules()
                   if name == selector or name.endswith("." + selector)]
        if len(matches) != 1:
            raise ValueError(f"expected one {op} module for {selector}, found {len(matches)}")
        name, module = matches[0]
        if op == "norm" and module.variance_epsilon != captures["epsilon"]:
            raise ValueError("runtime RMSNorm epsilon differs from checkpoint")
        if op == "linear" and getattr(module, "tp_size", 1) != 1:
            raise ValueError("dense projection requires TP=1")
        key = adapter[f"{op}_weight_key"]
        entry = {"path":name, "class":type(module).__module__ + "." + type(module).__name__,
                 "weight_key":key, "runtime_weight":module.weight, "bias":getattr(module, "bias", None)}
        captures["ops"][op] = entry

        def observe(module, args, result, *, entry=entry):
            if "output" in entry or len(args) == 0 or any(arg is not None for arg in args[1:]):
                raise ValueError("unexpected multiple calls, fused residual or extra tensor inputs")
            if isinstance(result, tuple):
                result, bias = result
                if bias is not None:
                    raise ValueError("deferred projection bias outside selected module")
            x = args[0]
            if (x.ndim != 2 or result.ndim != 2 or x.shape[0] < 2 or result.shape[0] != x.shape[0]
                    or x.dtype != torch.bfloat16 or result.dtype != torch.bfloat16
                    or not x.is_cuda or not result.is_cuda):
                raise ValueError("expected real 2D CUDA BF16 inputs and outputs")
            entry.update(input=x[:2].detach().to("cpu"), output=result[:2].detach().to("cpu"),
                         device=str(result.device), input_stride=list(x.stride()),
                         output_stride=list(result.stride()))

        handles.append(module.register_forward_hook(observe))


def archive(root, captures):
    import torch
    from safetensors import safe_open
    from .review import statistics, THRESHOLDS

    checkpoint = Path(captures["checkpoint"])
    plan = json.loads((root / "depth-plan.json").read_text())
    shard = plan["model"]["storage"]["weights"]
    if len(shard) != 1:
        raise ValueError("dense adapter requires single-file BF16 checkpoint")
    document = {"schema":"tilescope.dense-observation.v1", "engine_request_id":captures["engine_request_id"],
                "prompt_ids":captures["prompt_ids"], "positions":[0, 1], "checkpoint_file":shard[0],
                "model_origin":plan["model"]["origin"], "epsilon":captures["epsilon"],
                "boundary":"actual eager vLLM module forward, diagnostic request only",
                "thresholds":THRESHOLDS, "producer_device":torch.cuda.get_device_name(), "ops":{}}
    data = root / "tensors"
    data.mkdir()

    def raw(tensor):
        return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()

    def write_tensor(tensor, name):
        blob = raw(tensor)
        path = data / (name + ".bin")
        with path.open("xb") as file:
            file.write(blob)
        return {"file":str(path.relative_to(root)), "dtype":str(tensor.dtype).removeprefix("torch."),
                "shape":list(tensor.shape), "sha256":hashlib.sha256(blob).hexdigest()}

    with safe_open(str(checkpoint / shard[0]), framework="pt", device="cpu") as weights:
        for op, item in captures["ops"].items():
            x, actual, key = item["input"], item["output"], item["weight_key"]
            w = weights.get_tensor(key)
            loaded = item["runtime_weight"].detach().cpu()
            if w.dtype != torch.bfloat16 or loaded.dtype != w.dtype or not torch.equal(loaded, w):
                raise ValueError(f"loaded {key} identity/layout differs from checkpoint BF16")
            ref_x = x.double()
            info = {k:item[k] for k in ("path", "class", "weight_key", "device", "input_stride", "output_stride")}
            info.update(weight_shape=list(w.shape), weight_dtype="bfloat16",
                        weight_sha256=hashlib.sha256(raw(w)).hexdigest(), loaded_weight_matches=True)
            if op == "norm":
                if w.shape != (x.shape[1],) or item["bias"] is not None:
                    raise ValueError("unsupported standard RMSNorm shape/bias")
                ref = ref_x * (ref_x.square().mean(dim=-1, keepdim=True) + captures["epsilon"]).rsqrt() * w.double()
                info["weight"] = write_tensor(w, "norm-weight")
            else:
                if w.shape != (actual.shape[1], x.shape[1]):
                    raise ValueError("projection is not out_features × in_features")
                ref = ref_x @ w.double().T
                bias_key = key.removesuffix("weight") + "bias"
                has_bias = bias_key in weights.keys()
                if has_bias != (item["bias"] is not None):
                    raise ValueError("runtime bias presence differs from checkpoint")
                info["bias"] = None
                if has_bias:
                    bias = weights.get_tensor(bias_key)
                    if bias.dtype != torch.bfloat16 or not torch.equal(bias, item["bias"].detach().cpu()):
                        raise ValueError("loaded projection bias differs from checkpoint")
                    ref += bias.double()
                    info["bias"] = write_tensor(bias, "linear-bias")
                indices = sorted({0, w.shape[0] // 2, w.shape[0]-1})
                info.update(sample_weight_indices=indices, sample_weights=write_tensor(w[indices], "linear-weight-sample"))
            info.update(input=write_tensor(x, op+"-input"), actual=write_tensor(actual, op+"-output"),
                        reference=write_tensor(ref, op+"-fp64"),
                        metrics=statistics(actual.double().flatten().tolist(), ref.flatten().tolist()))
            document["ops"][op] = info
    with (root / "dense.json").open("x") as file:
        json.dump(document, file, indent=2, allow_nan=False)
