"""CPU FP64 comparison of two real prefill rows with on-disk checkpoint weights."""
import hashlib
import json
from pathlib import Path

def proof(root, runner, rid, x, y, z):
    import torch
    from safetensors import safe_open
    requested=list(runner.requests[rid].prompt_token_ids[:2])
    index=int(runner.input_batch.req_id_to_index[rid])
    if int(runner.input_batch.num_computed_tokens_cpu[index])!=0 or x.shape!=(2,5120):
        raise ValueError("diagnostic must observe the first two logical text prefill rows")
    plan=json.loads((root/"plan.json").read_text())
    checkpoint=Path(plan["checkpoint"])
    idx=json.loads((checkpoint/"model.safetensors.index.json").read_text())["weight_map"]
    layer="model.language_model.layers.0"
    keys=["model.language_model.embed_tokens.weight",layer+".input_layernorm.weight",
          layer+".linear_attn.in_proj_b.weight",layer+".linear_attn.in_proj_a.weight"]
    def weight(key, ids=None):
        with safe_open(str(checkpoint/idx[key]),framework="pt",device="cpu") as sf:
            if ids is None:
                return sf.get_tensor(key)
            return torch.cat([sf.get_slice(key)[i:i+1] for i in ids],dim=0)
    embedding=weight(keys[0],requested)
    scale,b,a=(weight(k) for k in keys[1:])
    ba=torch.cat((b,a),dim=0)
    xc,yc,zc=(v.detach().cpu() for v in (x,y,z))
    def sha(t):
        return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    ref_input=embedding.double()
    ref_norm=(ref_input*(ref_input.square().mean(dim=1,keepdim=True)+1e-6).rsqrt()
              *(1+scale.double()))
    def compare(actual, reference, rtol, rms_atol, rel_max):
        aa,bb=actual.double().flatten(),reference.flatten()
        diff=aa-bb
        finite=bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
        rms=bb.square().mean().sqrt().item()
        rmse=diff.square().mean().sqrt().item()
        l2=(diff.norm()/bb.norm()).item()
        mismatches=int((diff.abs()>rtol*bb.abs()+rms_atol*rms).sum())
        return {"n":aa.numel(),"shape":list(actual.shape),"finite":finite,"max_abs":diff.abs().max().item(),
            "relative_l2":l2,"cosine":(aa.dot(bb)/(aa.norm()*bb.norm())).item(),
            "rmse":rmse,"reference_rms":rms,"mismatches":mismatches,
            "rtol":rtol,"rms_atol":rms_atol,"l2_limit":rel_max,
            "actual_bf16_sha256":sha(actual),"reference_bf16_sha256":sha(reference.to(torch.bfloat16)),
            "reference_fp64_first8":bb[:8].tolist(),"actual_first8":aa[:8].tolist(),
            "passed":bool(finite and mismatches==0 and l2<=rel_max)}
    return {"request":rid,"prompt_prefix_ids":requested,"computed_before":0,
        "checkpoint_tensors":[{"key":k,"file":idx[k]} for k in keys],
        "checkpoint_subset_sha256":{k:sha(t) for k,t in zip(keys,(embedding,scale,b,a))},
        "embedding_matches_checkpoint":bool(torch.equal(xc,embedding)),
        "reference_boundary":"CPU FP64 pre-store mathematical result vs actual BF16; BA uses observed BF16 norm as operand",
        "norm":compare(yc,ref_norm,.006,.0002,.003),
        "ba":compare(zc,yc.double()@ba.double().T,.009,.002,.004)}
