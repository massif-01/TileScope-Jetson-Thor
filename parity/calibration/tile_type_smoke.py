"""Numerical cuTile smoke for the frozen partner C++ BF16/FP8 E4M3/E5M2 API cases."""
import argparse
import hashlib
import json
import logging
from pathlib import Path

import torch
import cuda.tile as ct

from round1.tile_kernel import bf16_gemm


def main(fmt):
    torch.set_num_threads(2)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("cuda.tile").setLevel(logging.DEBUG)
    dtype={"bf16":torch.bfloat16,"e4m3":torch.float8_e4m3fn,"e5m2":torch.float8_e5m2}[fmt]
    n=1024
    rows=torch.arange(n,dtype=torch.int32)[:,None]
    cols=torch.arange(n,dtype=torch.int32)[None,:]
    # All inputs exactly representable in all three formats, all expected integer sums fit in FP32.
    a=(((rows*3+cols*2)%7)-3).to(dtype)
    bt=(((rows+cols*5)%5 + (cols%3)) -3).to(dtype)
    ref=a.double() @ bt.double().T
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x,b=a.cuda(),bt.cuda()
        dt=torch.full((n,n),float("nan"),device="cuda",dtype=torch.float32)
        ct.launch(stream,(n//64,n//64,1),bf16_gemm,(x,b,dt))
        stream.synchronize()
        err=(dt.cpu().T.double()-ref).abs()
    bad=int((~torch.isfinite(err) | (err!=0)).sum())
    print(json.dumps(dict(format=fmt,shape=[n,n,n],implementation="round1.tile_kernel.bf16_gemm, cuTile MMA",
                          inputs={key:hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
                                  for key,t in (("A",a),("B_transposed",bt))},
                          reference="full unscaled FP64 host GEMM on exact submitted inputs",elements=n*n,
                          max_abs=float(err.max()) if bool(torch.isfinite(err).all()) else None,
                          mismatches=bad,passed=bad==0)),flush=True)
    return int(bad!=0)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("format",choices=("bf16","e4m3","e5m2"))
    raise SystemExit(main(p.parse_args().format))
