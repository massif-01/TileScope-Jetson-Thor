"""Separate the deployment's implemented adapters from the evidence a selected suite can produce."""
from tilescope.common import digest

BASIC = {"request.complete","request.exact_token_counts","identity.local_target","engine.controlled_serial_phases","resources.window"}
DENSE = {"model.standard_rmsnorm","model.dense_linear"}
REQUEST_SUITES = {"new-target","legacy-regression","same-content","bounded-concurrency","protocol-pilot",
                  "chat-pair","soak","rejection-recovery"}


def plan(target, suite):
    required = set(suite["required_capabilities"])
    deep = target["model"].get("deep_adapter") or {}
    engine = target["engine"]
    is_vllm = engine["adapter"] == "vllm.local.v1"
    pinned = is_vllm and engine["build_pin"] == "0.21.1rc1.dev104+g301d98647"
    dense = pinned and deep.get("id") == "dense_rms_linear.v1"
    legacy = pinned and deep.get("id") == "qwen38_legacy.v1"
    supported = BASIC | (DENSE if dense else set())
    if legacy:
        supported |= {"model.qwen_legacy","engine.mtp","engine.graph"}
    if dense or legacy or engine["adapter"] == "llama_cpp.native.v1":
        supported.add("gpu.execution")
    name = suite["suite_id"]
    available = set()
    if name in REQUEST_SUITES:
        available = set(BASIC)
        if not is_vllm and name == "chat-pair":
            available.discard("engine.controlled_serial_phases")
    elif name == "deep-dense" and dense:
        available = DENSE | {"identity.local_target","gpu.execution"}
    caps = []
    for key in sorted(supported | required | DENSE | {"engine.mtp","gpu.execution"}):
        implemented = key in supported
        verified = key=="identity.local_target" and "identity" in target
        status = "verified" if verified else "not_run" if implemented else "unsupported"
        reason = ("bound local process and checkpoint" if verified else
                  "requires measurement" if key in available else
                  "implemented in a separate diagnostic" if implemented else
                  f"no compatible adapter for {deep.get('id')} on {engine['build_pin']}")
        if key=="engine.mtp" and not legacy and not deep.get("id")=="qwen38_legacy.v1":
            status,reason = "not_applicable","no speculative decoding configured"
        caps.append({"capability_id":key,"declared_support":"supported" if implemented else "unsupported",
                     "observed_status":status,"required":key in required,
                     "target_fingerprint":target.get("target_fingerprint"),"implementation":engine["adapter"],
                     "evidence":["identity"] if verified else [],"reason":reason})
    return {"requested":sorted(required),"available":sorted(available),"missing_required":sorted(required-available),
            "optional_missing":[c["capability_id"] for c in caps if not c["required"] and c["observed_status"]=="unsupported"],
            "identity_grade":"verified_local_artifact_and_process" if "identity" in target else "unbound",
            "suite_sha256":digest(suite),"capabilities":caps}
