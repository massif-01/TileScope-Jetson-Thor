"""Run in a private, in-process Qwen engine; inspect actual target-model forwards.

Only tensor metadata are logged. Module hooks and runner wrappers live solely in
this temporary process. The online service is neither instrumented nor reloaded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def token_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def tensor_info(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple)):
        return [tensor_info(item) for item in value]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads((args.output / "requests.json").read_text())
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    import faulthandler
    # Bound private initialization in shared Jetson unified memory.
    ceiling_bytes = 32 * 1024**3
    torch.cuda.set_per_process_memory_fraction(ceiling_bytes / torch.cuda.get_device_properties(0).total_memory)
    faulthandler.dump_traceback_later(75, repeat=True)
    from vllm import LLM, SamplingParams

    options = dict(model=str(args.checkpoint), tensor_parallel_size=1, max_num_seqs=4,
                   enable_chunked_prefill=True, enable_prefix_caching=True,
                   speculative_config={"method": "mtp", "num_speculative_tokens": 3},
                   attention_backend="FLASHINFER", enforce_eager=True,
                   max_model_len=512, max_num_batched_tokens=2048,
                   kv_cache_memory_bytes=3*1024**3, gpu_memory_utilization=.3,
                   enable_flashinfer_autotune=False,
                   skip_mm_profiling=True, limit_mm_per_prompt={"image": 0, "video": 0},
                   distributed_executor_backend="uni")
    (args.output / "isolated-options.json").write_text(json.dumps(options, indent=2))
    llm = LLM(**options)
    faulthandler.cancel_dump_traceback_later()
    engine = llm.llm_engine
    runner = engine.model_executor.driver_worker.worker.model_runner
    selected = {
        "layers.0.linear_attn.in_proj_ba": "bf16_ba",
        "layers.0.input_layernorm": "input_norm",
        "layers.0.mlp.act_fn": "mlp_gate",
        "layers.3.self_attn.o_proj": "fp8_o",
        "layers.0.mlp.down_proj": "nvfp4_down",
    }
    trace = (args.output / "runner.jsonl").open("x", buffering=1)
    state = {"step": 0, "active": False}

    def emit(event, **fields):
        trace.write(json.dumps({"event": event, "unix_ns": time.time_ns(), "step": state["step"],
                                "request": state.get("request"), **fields}) + "\n")

    def hook(name, label):
        def callback(module, inputs, output):
            if state["active"]:
                emit("operator", module=name, label=label, cls=type(module).__name__,
                     inputs=tensor_info(inputs), output=tensor_info(output),
                     weight=tensor_info(getattr(module, "weight", None)),
                     eps=getattr(module, "variance_epsilon", None))
        return callback

    handles = []
    for name, module in runner.model.named_modules():
        for suffix, label in selected.items():
            if name.endswith(suffix):
                handles.append(module.register_forward_hook(hook(name, label)))
                emit("registered", module=name, label=label, cls=type(module).__name__)
    if len(handles) != len(selected):
        raise RuntimeError(f"Expected {len(selected)} target modules, found {len(handles)}")
    execute = runner.execute_model
    forward = runner._model_forward

    def execute_traced(scheduler_output, *a, **kw):
        state["step"] += 1
        state["scheduled"] = dict(scheduler_output.num_scheduled_tokens)
        state["drafts"] = {key: len(ids) for key,ids in scheduler_output.scheduled_spec_decode_tokens.items()}
        return execute(scheduler_output, *a, **kw)

    def forward_traced(*a, **kw):
        b = runner.input_batch
        batch = []
        for i, rid in enumerate(b.req_ids):
            computed = int(b.num_computed_tokens_cpu[i])
            prompt = int(b.num_prompt_tokens[i])
            drafts = state["drafts"].get(rid, 0)
            phase = ("prefill" if computed < prompt else
                     "decode_target_verify" if drafts else "decode_target")
            batch.append(dict(internal_id=rid, computed_before=computed, prompt_length=prompt,
                              prompt_token_hash=token_hash(runner.requests[rid].prompt_token_ids),
                              scheduled=state["scheduled"][rid], speculative_tokens=drafts,
                              phase=phase))
        emit("target_forward", batch=batch, input_ids=tensor_info(kw.get("input_ids")),
             positions=tensor_info(kw.get("positions")))
        state["active"] = True
        try:
            return forward(*a, **kw)
        finally:
            state["active"] = False

    runner.execute_model = execute_traced
    runner._model_forward = forward_traced
    try:
        for spec in request:
            state["request"] = spec["id"]
            out = llm.generate({"prompt_token_ids": spec["ids"]},
                               SamplingParams(temperature=0, seed=spec["seed"],
                                              max_tokens=spec["outputs"], min_tokens=spec["outputs"],
                                              ignore_eos=True), use_tqdm=False)[0]
            emit("completed", internal_id=out.request_id, prompt_token_ids=out.prompt_token_ids,
                 completion_token_ids=list(out.outputs[0].token_ids),
                 cached_tokens=getattr(out, "num_cached_tokens", None))
    finally:
        for h in handles:
            h.remove()
        trace.close()
        engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
