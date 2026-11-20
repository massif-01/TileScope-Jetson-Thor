"""Validate the diagnostic independently from the model implementation pairs."""
import json
import math
import hashlib
import struct
from pathlib import Path
import statistics

from tilescope.common import sha256


def review(directory):
    d=Path(directory)
    code=int((d/"exit").read_text())
    rows=[json.loads(l) for l in (d/"stdout").read_text().splitlines() if l.startswith("{")]
    if code:
        errors=[r for r in rows if r["phase"]=="failure"]
        return {"status":"execution_failed","reason":errors or (d/"stderr").read_text()[-2000:],"results":[]}
    inputs = [r for r in rows if r["phase"]=="input"]
    if len(inputs)!=1:
        raise ValueError("missing or duplicate same-source input")
    inp=inputs[0]
    if (inp["shape"],inp["block"],inp["num_warps"],inp["num_stages"])!=([65536],256,4,1):
        raise ValueError("same-source workload/configuration mismatch")
    if sha256(d/"kernel-source.py")!=inp["source_sha256"]:
        raise ValueError("same-source source checksum mismatch")
    for key,mod,offset,div in (("a",71,35,8),("b",53,26,16)):
        data=b"".join(struct.pack("<f",(i%mod-offset)/div) for i in range(65536))
        if hashlib.sha256(data).hexdigest()!=inp["sha256"][key]:
            raise ValueError("same-source input checksum mismatch")
    if not any(r["phase"]=="warmup" and r["pairs"]>0 and r["seconds"]>=1 for r in rows):
        raise ValueError("missing paired graph warmup")
    stages={r["backend"]:r for r in rows if r["phase"]=="backend"}
    if any(r["phase"]=="failure" for r in rows) or set(stages)!={"nvidia","tileir"}:
        raise ValueError("both verified backend receipts required")
    sample=[r for r in rows if r["phase"]=="sample"]
    seq=[(i,b) for i in range(50) for b in (("tileir","nvidia") if i%2 else ("nvidia","tileir"))]
    if [(r["index"],r["backend"]) for r in sample]!=seq:
        raise ValueError("missing or reordered same-source samples")
    results=[]
    for role,key,target in (("nvidia","ptx","cuda"),("tileir","tileir","tileir")):
        backend=stages[role]
        if (backend["actual"]!=f"GPUTarget(backend='{target}', arch=110, warp_size=32)"
                or backend["driver"]!=("CudaDriver" if role=="nvidia" else "TileIRDriver")
                or backend["name"]!="vector_add" or set(backend["artifacts"])!={"cubin","ttir",key}):
            raise ValueError("backend artifact kind mismatch")
        for suffix,h in backend["artifacts"].items():
            if sha256(d/key/("vector_add."+suffix))!=h:
                raise ValueError("backend artifact checksum mismatch")
        checks=[r for r in rows if r["phase"]=="correctness" and r["backend"]==role]
        if [r["stage"] for r in checks]!=["direct","graph","post"] or any(r["max_abs"]!=0 for r in checks):
            raise ValueError("same-source output failed validation")
        vals=[r["graph_us"]/16 for r in sample if r["backend"]==role]
        if any(not math.isfinite(v) or v<=0 for v in vals):
            raise ValueError("invalid graph duration")
        summaries=[r for r in rows if r["phase"]=="summary" and r["backend"]==role]
        if (len(summaries)!=1 or summaries[0]["samples"]!=len(vals)
                or not math.isclose(summaries[0]["median_us"],statistics.median(vals))
                or not math.isclose(summaries[0]["stdev_us"],statistics.stdev(vals))):
            raise ValueError("same-source summary mismatch")
        results.append({"backend":role,"samples":50,"operations_per_graph":16,
                        "median_us":statistics.median(vals),"sd_us":statistics.stdev(vals),
                        "min_us":min(vals),"max_us":max(vals),"elements_per_s":65536e6/statistics.median(vals)})
    return {"schema":"tilescope.same_source.v1","status":"passed","comparison_type":"same_source_backend",
            "task_id":"triton-add-f32","shape":[65536],"block":256,"results":results}
