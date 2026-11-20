"""Source-gated observations for the private vLLM engine. No online-service injection."""
import hashlib
from pathlib import Path

from parity.trace_hook import tilescope_events as events

from .pins import PINS


def install(module):
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest != PINS[module.__name__]:
        raise RuntimeError(f"unreviewed vLLM observer target {module.__name__} {digest}")
    events.emit("instrumentation", module=module.__name__, file=module.__file__, sha256=digest,
                start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    if module.__name__.endswith(".scheduler"):
        events.scheduler(module.Scheduler)
    elif module.__name__.endswith(".cuda_graph"):
        events.graph(module)
    else:
        worker(module.GPUModelRunner)


def worker(cls):
    execute = cls.execute_model
    dispatch = cls._determine_batch_execution_and_padding
    forward = cls._model_forward

    def run(self, output, *args, **kwargs):
        events.state.update(mode=events.mode(), ids=list(output.num_scheduled_tokens),
                            step=events.state["step"] + 1)
        return execute(self, output, *args, **kwargs)

    def route(self, *args, **kwargs):
        result = dispatch(self, *args, **kwargs)
        if events.state["ids"] and events.state["mode"] != "off":
            events.emit("dispatch", ids=events.state["ids"], step=events.state["step"],
                        mode=result[0].name, logical=kwargs.get("num_tokens"),
                        padded=result[1].num_tokens, uniform=result[1].uniform)
        return result

    def target(self, *args, **kwargs):
        ids = events.state["ids"]
        mode = events.state["mode"]
        if not ids or mode == "off":
            return forward(self, *args, **kwargs)
        handles = []
        captures = {}
        try:
            if mode == "dense" and len(ids) == 1:
                rid = ids[0]
                index = self.input_batch.req_id_to_index[rid]
                if int(self.input_batch.num_computed_tokens_cpu[index]) == 0:
                    from .numerics import attach
                    prompt = list(self.requests[rid].prompt_token_ids[:2])
                    observed = kwargs["input_ids"][:2].detach().cpu().tolist()
                    if len(prompt) != 2 or observed != prompt:
                        raise ValueError("worker inputs differ from designated prefill rows")
                    attach(self.get_model(), events.root, rid, prompt, captures, handles)
            output = events.stage_call("target_forward", forward, self, args, kwargs)
            if captures:
                from .numerics import archive
                archive(events.root, captures)
                events.emit("dense_observed", ids=ids, step=events.state["step"],
                            pid_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()[19]))
            return output
        finally:
            for handle in handles:
                handle.remove()

    cls.execute_model = run
    cls._determine_batch_execution_and_padding = route
    cls._model_forward = target
