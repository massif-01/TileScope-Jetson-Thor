"""Install observers only when explicitly enabled in an owned Python process."""
import importlib.abc
import importlib.machinery
import os
import sys


if os.environ.get("TILESCOPE_TRACE_DIR"):
    class Observer(importlib.abc.MetaPathFinder):
        modules = {"vllm.v1.core.sched.scheduler", "vllm.v1.worker.gpu_model_runner",
                   "vllm.compilation.cuda_graph"}

        def find_spec(self, fullname, path, target=None):
            if fullname not in self.modules:
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec:
                original = spec.loader.exec_module

                def load(module):
                    original(module)
                    from tilescope.probe.depth.worker import install
                    install(module)

                spec.loader.exec_module = load
            return spec

    sys.meta_path.insert(0, Observer())
