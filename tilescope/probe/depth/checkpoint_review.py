"""Recompute every projection/reference element directly from the pinned checkpoint on CPU."""
from array import array
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from .review import raw_request, tensor


def check(root, checkpoint):
    plan = json.loads((root/"depth-plan.json").read_text())
    evidence = json.loads((root/"dense.json").read_text())
    model = plan["model"]
    cfg = (checkpoint/"config.json").read_bytes()
    if hashlib.sha256(cfg).hexdigest() != plan["config_sha256"]:
        raise ValueError("checkpoint config digest mismatch")
    filename = evidence["checkpoint_file"]
    if model["storage"]["weights"] != [filename] or Path(filename).name != filename:
        raise ValueError("unexpected dense checkpoint name")
    lut = [struct.unpack("<f",struct.pack("<I",i<<16))[0] for i in range(65536)]

    def decode(blob):
        words = array("H")
        words.frombytes(blob)
        if sys.byteorder != "little":
            words.byteswap()
        return [lut[k] for k in words]

    path = checkpoint/filename
    with path.open("rb") as f:
        h = hashlib.sha256()
        while block := f.read(8<<20):
            h.update(block)
        digest = h.hexdigest()
        if digest != model["storage"]["sha256"][filename]:
            raise ValueError("independent checkpoint content SHA-256 mismatch")
        f.seek(0)
        n, = struct.unpack("<Q",f.read(8))
        if not 2 <= n <= 100_000_000:
            raise ValueError("invalid safetensors header length")
        index = json.loads(f.read(n))
        def weight(key, shape):
            meta = index[key]
            start, end = meta["data_offsets"]
            if (meta["dtype"] != "BF16" or meta["shape"] != shape or
                    end-start != 2*math.prod(shape) or not 0<=start<end<=path.stat().st_size-8-n):
                raise ValueError(f"invalid checkpoint tensor shape, dtype or range for {key}")
            f.seek(8+n+start)
            return f.read(end-start)
        cfg = json.loads(cfg)
        hidden = cfg["hidden_size"]
        intermediate = cfg["intermediate_size"]
        epsilon = cfg["rms_norm_eps"]
        result = {}
        for op in ("norm","linear"):
            row = evidence["ops"][op]
            width = hidden if op == "norm" else intermediate
            shape = [hidden] if op == "norm" else [hidden,width]
            key = model["deep_adapter"][f"{op}_weight_key"]
            blob = weight(key, shape)
            if hashlib.sha256(blob).hexdigest() != row["weight_sha256"]:
                raise ValueError("independent operator weight digest mismatch")
            inp = tensor(root,row["input"],[2,width],"bfloat16")
            ref = tensor(root,row["reference"],[2,hidden],"float64")
            truth = [[],[]]
            if op == "norm":
                scales = decode(blob)
                for i in (0,1):
                    x = inp[i*width:(i+1)*width]
                    inv = 1/math.sqrt(math.fsum(a*a for a in x)/width+epsilon)
                    truth[i].extend(a*inv*b for a,b in zip(x,scales))
            else:
                bias_key = key.removesuffix("weight")+"bias"
                bias = decode(weight(bias_key,[hidden])) if bias_key in index else None
                x = [inp[:width],inp[width:]]
                for j in range(hidden):
                    w = decode(blob[j*width*2:(j+1)*width*2])
                    for i in (0,1):
                        truth[i].append(math.fsum(a*b for a,b in zip(x[i],w))+(bias[j] if bias else 0))
            pairs = list(zip(truth[0]+truth[1],ref))
            if any(not math.isclose(a,b,rel_tol=2e-12,abs_tol=1e-13) for a,b in pairs):
                raise ValueError(f"full CPU checkpoint reference mismatch for {op}")
            result[op] = {"elements":len(pairs),"key":key,"shape":shape,
                          "max_abs_independent_fp64_vs_saved_fp64":max(abs(a-b) for a,b in pairs)}
    return {"schema":"tilescope.dense-checkpoint-rebuild.v1","checkpoint_sha256":digest,
            "request":raw_request(root,"dense")["run_id"],"operators":result}
