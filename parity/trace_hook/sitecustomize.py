"""Install private-service observation after each selected vLLM module loads."""
import importlib.abc
import importlib.machinery
import os
import sys

if os.getenv("TILESCOPE_TRACE_DIR"):
    class Observer(importlib.abc.MetaPathFinder):
        modules = {"vllm.v1.core.sched.scheduler", "vllm.v1.worker.gpu_model_runner",
                   "vllm.compilation.cuda_graph"}

        def find_spec(self, fullname, path=None, target=None):
            if fullname not in self.modules:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is None:
                return None
            original = spec.loader.exec_module
            def observed(module):
                original(module)
                from tilescope_events import install
                install(module)
            spec.loader.exec_module = observed
            return spec

    sys.meta_path.insert(0, Observer())
