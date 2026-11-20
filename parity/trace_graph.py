"""Preserve limited code/metadata from the private compile output before cache cleanup."""
import argparse
import hashlib
from pathlib import Path
from parity.trace_client import write
import json

def extract(root):
    seed=set(json.loads((root/"cache-seed.json").read_text())["deployment_compile_directories"])
    data=[]
    for path in (root/"private-cache/vllm/torch_compile_cache").glob("*/rank_0_0/backbone/computation_graph.py"):
        cache_id=path.parents[2].name
        code=path.read_text().splitlines()
        matched=[]
        for i,line in enumerate(code):
            if not (100<len(line)<2000):
                continue
            if not any(tag in line for tag in (
                    "modules_0_modules_linear_attn_modules_in_proj_ba_parameters_weight_",
                    "modules_0_modules_input_layernorm_parameters_weight_", "rms_norm.default(embedding,",
                    "torch.nn.functional.embedding(long,")):
                continue
            if line.lstrip().startswith("def "):
                continue
            for n in [i-1,i]:
                if n>=0 and len(code[n])<2000:
                    matched.append({"line":n+1,"text":code[n]})
            if len(matched)>=24:
                break
        data.append({"path":str(path.relative_to(root)),"compiler_cache_id":cache_id,
            "was_seeded":cache_id in seed,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"selected":matched})
    write(root/"compiled-boundaries.json",data)
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path);extract(p.parse_args().run)
