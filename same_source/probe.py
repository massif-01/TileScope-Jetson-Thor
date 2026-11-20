"""One fixed FP32 vector-add, compiled by the same Triton release to its two backends."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
import triton
import triton.language as tl
from triton.runtime.driver import driver


@triton.jit
def vector_add(a, b, out, BLOCK: tl.constexpr):
    i = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out+i, tl.load(a+i) + tl.load(b+i))


def emit(phase, **fields):
    print(json.dumps(dict(phase=phase, **fields),allow_nan=False),flush=True)


def run(out):
    os.environ["TRITON_TILEIR_RUNTIME_FALLBACK"] = "0"
    n, block = 65536, 256
    h = torch.arange(n,dtype=torch.int32)
    lhs = ((h % 71)-35).float()/8
    rhs = ((h % 53)-26).float()/16
    expected = lhs+rhs
    a,b = lhs.cuda(),rhs.cuda()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    emit("input", shape=[n], sha256={key:hashlib.sha256(x.numpy().tobytes()).hexdigest()
         for key,x in (("a",lhs),("b",rhs))}, block=block, num_warps=4, num_stages=1,
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         triton=triton.__version__, triton_path=triton.__file__)
    graphs={}
    with torch.cuda.stream(stream):
        for backend,asmkey in (("nvidia","ptx"),("tileir","tileir")):
            os.environ.update(TRITON_DEFAULT_BACKEND=backend,ENABLE_TILE="1" if backend=="tileir" else "0")
            dst=torch.full_like(a,float("nan"))
            start=time.perf_counter()
            def launch():
                return vector_add[(n//block,)](a,b,dst,BLOCK=block,num_warps=4,num_stages=1)
            try:
                compiled=launch()
                stream.synchronize()
                target=compiled.metadata.target
                assert target.backend==("cuda" if backend=="nvidia" else "tileir") and asmkey in compiled.asm
                folder=out/asmkey
                folder.mkdir()
                artifacts={}
                for key in ("ttir",asmkey,"cubin"):
                    data=compiled.asm[key]
                    data=data if isinstance(data,bytes) else data.encode()
                    (folder/("vector_add."+key)).write_bytes(data)
                    artifacts[key]=hashlib.sha256(data).hexdigest()
                emit("backend",backend=backend,actual=str(target),driver=type(driver.active).__name__,
                     metadata=str(compiled.metadata),name=compiled.name,artifacts=artifacts,
                     first_launch_seconds=time.perf_counter()-start)
                assert torch.equal(dst.cpu(),expected)
                emit("correctness",backend=backend,stage="direct",max_abs=0)
                for _ in range(20): launch()
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                first,last=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(2)]
                with torch.cuda.graph(graph,stream=stream):
                    first.record(stream)
                    for _ in range(16): launch()
                    last.record(stream)
                dst.fill_(float("nan"))
                graph.replay()
                stream.synchronize()
                assert torch.equal(dst.cpu(),expected)
                emit("correctness",backend=backend,stage="graph",max_abs=0)
                graphs[backend]=(graph,first,last,dst)
            except Exception as exc:
                emit("failure",backend=backend,exception=type(exc).__name__,reason=str(exc))
                traceback.print_exc()
                return 1
        order=list(graphs)
        begin=time.perf_counter()
        count=0
        while time.perf_counter()-begin < 1:
            for name in order if count%2==0 else order[::-1]: graphs[name][0].replay()
            stream.synchronize(); count+=1
        emit("warmup",pairs=count,seconds=time.perf_counter()-begin)
        values={n:[] for n in order}
        for i in range(50):
            for name in order if i%2==0 else order[::-1]:
                graph,first,last,_=graphs[name]
                graph.replay(); stream.synchronize()
                us=first.elapsed_time(last)*1000
                emit("sample",backend=name,index=i,graph_us=us,end_ns=time.time_ns())
                values[name].append(us/16)
        for name in order:
            assert torch.equal(graphs[name][3].cpu(),expected)
            emit("correctness",backend=name,stage="post",max_abs=0)
            emit("summary",backend=name,median_us=statistics.median(values[name]),
                 stdev_us=statistics.stdev(values[name]),samples=len(values[name]))
    return 0


if __name__=="__main__":
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    sys.exit(run(parser.parse_args().out))
