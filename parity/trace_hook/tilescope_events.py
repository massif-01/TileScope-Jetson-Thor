"""Read-only private scheduler/runner evidence for the pinned vLLM 301d9864 source."""
import contextlib
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

root = Path(os.environ["TILESCOPE_TRACE_DIR"])
state = {"mode": "off", "ids": [], "step": 0, "stage": None}
handles = {}

def mode():
    return (root / "trace-control").read_text().strip()

def emit(event, **fields):
    pid = os.getpid()
    if pid not in handles:
        handles[pid] = (root / f"engine-{pid}.jsonl").open("a", buffering=1)
    handles[pid].write(json.dumps({"event": event, "pid": pid, "mono_ns": time.monotonic_ns(),
                                   "unix_ns": time.time_ns(), **fields}, separators=(",", ":")) + "\n")

def digest(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()

def install(module):
    emit("instrumentation", module=module.__name__, file=module.__file__,
         sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    if module.__name__.endswith(".scheduler"):
        scheduler(module.Scheduler)
    elif module.__name__.endswith(".gpu_model_runner"):
        worker(module.GPUModelRunner)
    else:
        graph(module)

def scheduler(cls):
    original_add, original_after, original_update = cls.add_request, cls._update_after_schedule, cls.update_from_output
    def add(self, request, *args, **kwargs):
        if mode() != "off":
            emit("enqueue", rid=request.request_id, prompt_len=request.num_prompt_tokens,
                 prompt_sha256=digest(request.prompt_token_ids or []))
        return original_add(self, request, *args, **kwargs)

    def after(self, output, *args, **kwargs):
        if output.num_scheduled_tokens and mode() != "off":
            # vLLM updates num_computed_tokens in this very call; read before it.
            for rid, count in output.num_scheduled_tokens.items():
                req = self.requests[rid]
                emit("scheduled", rid=rid, computed=req.num_computed_tokens,
                     prompt=req.num_prompt_tokens, count=count,
                     drafts=len(output.scheduled_spec_decode_tokens.get(rid, ())))
        return original_after(self, output, *args, **kwargs)

    def update(self, output, model_output, *args, **kwargs):
        if not output.num_scheduled_tokens or mode() == "off":
            return original_update(self, output, model_output, *args, **kwargs)
        accepted = {}
        for rid, draft in output.scheduled_spec_decode_tokens.items():
            index = model_output.req_id_to_index.get(rid)
            sample = model_output.sampled_token_ids
            if draft and index is not None and sample and sample[index]:
                accepted[rid] = {"verified_draft_count": len(draft),
                                 "accepted_before_stop": len(sample[index])-1}
        result = original_update(self, output, model_output, *args, **kwargs)
        for group in result.values():
            for item in group.outputs:
                emit("commit", rid=item.request_id, token_ids=list(item.new_token_ids),
                     finish=None if item.finish_reason is None else str(item.finish_reason),
                     queue_events=[(e.type.name,e.timestamp) for e in item.events or ()],
                     prefill=dataclasses.asdict(item.prefill_stats) if item.prefill_stats else None,
                     mtp=accepted.get(item.request_id))
        return result
    original_finish=cls.finish_requests
    def finish(self,request_ids,finished_status,*a,**kw):
        result=original_finish(self,request_ids,finished_status,*a,**kw)
        if result and mode()!="off":
            emit("external_finish", ids=[item[0] for item in result], status=finished_status.name)
        return result
    cls.finish_requests=finish
    cls.add_request, cls._update_after_schedule, cls.update_from_output = add, after, update

def tensor_meta(t):
    return {"shape":list(t.shape), "stride":list(t.stride()), "dtype":str(t.dtype)}

def tensor_sample(t):
    x = t.detach().contiguous().cpu()
    import torch
    b = x.view(torch.uint8).numpy().tobytes()
    a = x.float()
    return {**tensor_meta(t), "sha256":hashlib.sha256(b).hexdigest(), "mean":a.mean().item(),
            "rms":a.square().mean().sqrt().item(), "first8":a.flatten()[:8].tolist()}

def stage_call(tag, method, self, a, kw):
    import torch
    prev = state["stage"]
    state["stage"] = tag
    begin = time.monotonic_ns()
    cm = (torch.profiler.record_function("tilescope/"+tag) if state["mode"]=="profile"
          else contextlib.nullcontext())
    try:
        with cm:
            return method(self,*a,**kw)
    finally:
        emit("worker_stage", name=tag, start_mono_ns=begin, ids=state["ids"], step=state["step"])
        state["stage"] = prev

def worker(cls):
    execute, dispatch, forward = cls.execute_model, cls._determine_batch_execution_and_padding, cls._model_forward
    propose, sample = cls.propose_draft_token_ids, cls._sample
    def run(self, output, *a, **kw):
        state.update(mode=mode(), ids=list(output.num_scheduled_tokens), step=state["step"]+1)
        return execute(self,output,*a,**kw)
    def route(self,*a,**kw):
        answer=dispatch(self,*a,**kw)
        if state["ids"] and state["mode"]!="off":
            emit("dispatch",ids=state["ids"],step=state["step"],mode=answer[0].name,
                 logical=kw.get("num_tokens"),padded=answer[1].num_tokens,
                 uniform=answer[1].uniform)
        return answer
    def target(self,*a,**kw):
        if not state["ids"] or state["mode"]=="off":
            return forward(self,*a,**kw)
        out = stage_call("target_forward",forward,self,a,kw)
        target_id=state["ids"][0] if len(state["ids"])==1 else ""
        designated=any("-"+label+"-0-" in target_id for label in ("checkpoint-numerics","layer-replay"))
        if state["mode"]=="tensors" and designated and not state.get("sampled"):
            index=self.input_batch.req_id_to_index[target_id]
            if int(self.input_batch.num_computed_tokens_cpu[index])!=0:
                return out
            ids = kw.get("input_ids")
            pre = kw.get("inputs_embeds")
            if ids is not None or pre is not None:
                # Isolated diagnostic only. Use two actual prefill rows from the executed batch.
                model = self.get_model().language_model.model
                x = pre[:2] if pre is not None else model.embed_tokens(ids[:2])
                norm = model.layers[0].input_layernorm
                proj = model.layers[0].linear_attn.in_proj_ba
                y=norm(x)
                z=proj(y)[0]
                emit("selected_local_replay", ids=state["ids"],step=state["step"],
                     row_source="runner inputs_embeds" if pre is not None else "runner input_ids -> actual embedding",
                     model_input=tensor_meta(pre if pre is not None else ids),
                     input=tensor_sample(x),norm=tensor_sample(y),ba=tensor_sample(z),
                     norm_weight=tensor_meta(norm.weight),ba_weight=tensor_meta(proj.weight),
                     eps=norm.variance_epsilon, rows=2)
                from tilescope_reference import proof
                try:
                    result=proof(root,self,target_id,x,y,z)
                except Exception as err:
                    emit("selected_numerical_error",step=state["step"],ids=state["ids"],error=repr(err))
                else:
                    emit("selected_numerical_reference",step=state["step"],ids=state["ids"],result=result)
                state["sampled"]=True
        return out
    def draft(self,*a,**kw):
        if state["mode"]=="off" or not state["ids"]:
            return propose(self,*a,**kw)
        out=stage_call("mtp_proposal",propose,self,a,kw)
        emit("proposal", ids=state["ids"], step=state["step"],
             shape=list(out.shape) if hasattr(out,"shape") else [len(x) for x in out],
             representation="tensor_shape" if hasattr(out,"shape") else "list_row_lengths")
        return out
    def sampling(self,*a,**kw):
        if state["mode"]=="off" or not state["ids"]:
            return sample(self,*a,**kw)
        return stage_call("sample_or_rejection",sample,self,a,kw)
    cls.execute_model,cls._determine_batch_execution_and_padding,cls._model_forward=run,route,target
    cls.propose_draft_token_ids,cls._sample=draft,sampling

def graph(module):
    # Actual replay API invocation, including graphs nested inside a compiled model.
    import torch
    original=torch.cuda.CUDAGraph.replay
    def replay(self,*a,**kw):
        out=original(self,*a,**kw)
        if state["mode"]!="off" and state["ids"]:
            emit("cuda_graph_replay", ids=state["ids"], step=state["step"],
                 stage=state["stage"], graph_id=id(self))
        return out
    torch.cuda.CUDAGraph.replay=replay
