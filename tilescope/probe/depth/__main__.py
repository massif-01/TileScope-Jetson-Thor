"""Isolated diagnostics, run under the owner's device lifecycle."""
import argparse
import hashlib
import json
from pathlib import Path

from parity.trace_client import write


def target_definition(path):
    target = json.loads(path.read_text())
    d = target["deployment"]
    model = json.loads((path.parent/target["model_ref"]).read_text())
    engine = json.loads((path.parent/target["engine_ref"]).read_text())
    if engine["adapter"] != "vllm.local.v1":
        raise ValueError("unsupported engine for this vLLM depth adapter")
    return target, model, d


def prepare(args):
    target, model, deploy = target_definition(args.target)
    if model["deep_adapter"]["id"] != "dense_rms_linear.v1" or deploy["mode"] != "owned_instance":
        raise ValueError("expected dense adapter and isolated owned deployment")
    a = model["deep_adapter"]
    if (a["norm_semantics"] != "standard_rmsnorm" or a["epsilon_source"] != "config.rms_norm_eps" or
            a["rows"] != 2 or model["storage"]["kind"] != "safetensors_single"):
        raise ValueError("unsupported RMSNorm/projection semantics or storage")
    ckpt = Path(deploy["checkpoint"])
    config_bytes = (ckpt/"config.json").read_bytes()
    config = json.loads(config_bytes)
    if not set(model["expected_architectures"]).intersection(config["architectures"]):
        raise ValueError("checkpoint architecture disagrees with profile")
    root = args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    plan = {"target_id":target["target_id"], "model":model, "checkpoint":str(ckpt),
            "endpoint":deploy["endpoint"], "served_model":deploy["served_model"],
            "target_sha256":hashlib.sha256(args.target.read_bytes()).hexdigest(),
            "config":config, "config_sha256":hashlib.sha256(config_bytes).hexdigest(),
            "boot_id":Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    with (root/"depth-plan.json").open("x") as f:
        json.dump(plan, f, indent=2, allow_nan=False)
    (root/"trace-control").write_text("off")
    print(json.dumps({"TILESCOPE_TRACE_DIR":str(root),
                      "prepend_PYTHONPATH":[str(Path(__file__).resolve().with_name("hook")),
                                            str(Path(__file__).resolve().parents[3])],
                      "torch_profiler_dir":str(root/"profiles")}))


def legacy(args):
    target, model, d = target_definition(args.target)
    if model["deep_adapter"]["id"] != "qwen38_legacy.v1" or d["mode"] != "attach_existing":
        raise ValueError("expected legacy Qwen profile")
    from .legacy import collect_legacy
    collect_legacy(args, target, model, d)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare","collect","review","legacy","legacy-client","legacy-review","checkpoint-review"):
        p = sub.add_parser(name)
        p.add_argument("--out",type=Path,required=True)
        if name in ("prepare","legacy"):
            p.add_argument("--target",type=Path,required=True)
        if name == "checkpoint-review":
            p.add_argument("--checkpoint",type=Path,required=True)
        if name == "collect":
            p.add_argument("--profile",action="store_true")
            p.add_argument("--cancel",action="store_true")
        if name == "legacy":
            p.add_argument("--port",type=int,required=True)
            p.add_argument("--timeout",type=int,default=1200)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "legacy":
        legacy(args)
    elif args.command == "legacy-client":
        from .client import collect
        collect(args.out.resolve(), True, True, legacy=True)
    elif args.command == "legacy-review":
        from .legacy import review_legacy
        print(json.dumps(review_legacy(args.out.resolve()),ensure_ascii=False))
    elif args.command == "checkpoint-review":
        from .checkpoint_review import check
        print(json.dumps(check(args.out.resolve(),args.checkpoint.resolve()),ensure_ascii=False))
    elif args.command == "collect":
        from .client import collect
        from tilescope.common import device_slot
        with device_slot():
            collect(args.out.resolve(), args.profile, args.cancel)
    else:
        from .review import review
        result = review(args.out.resolve())
        write(args.out/"depth-review.json",result)
        print(json.dumps({"target":result["target_id"],"numerics":result["local_numerics"],
                          "profiled_kernel_calls":result.get("gpu",{}).get("calls"),
                          "cancel":result.get("cancel")},ensure_ascii=False))


if __name__ == "__main__":
    main()
