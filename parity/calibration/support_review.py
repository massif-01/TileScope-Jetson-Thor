"""Review single-sided cuBLASLt API queries and independent cuTile type smoke evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from .build import sha
from .run import write


def require(value, message):
    if not value:
        raise ValueError(message)


def library_status(folder):
    require((folder/"exit").read_text().strip()=="0","cuBLASLt probe process failed")
    events=[json.loads(line) for line in (folder/"stdout").read_text().splitlines()]
    header,*cases=events
    require(header["mnk"]==[1024,1024,1024] and header["workspace_bytes"]==64<<20
            and header["version"]>0,"cuBLASLt scope mismatch")
    names=["C32F outF VEC16","C32F outF VEC32","C16F outH VEC16","C16F outH VEC32",
           "C32F outH VEC16","C32F outBF VEC16","C16F outBF VEC16","C32F_F16 outF VEC16",
           "C32F_FBF outF VEC16","C16F outF VEC16"]
    require([r["case"] for r in cases]==names,"incomplete cuBLASLt configurations")
    for case in cases:
        require(all(type(case[key]) is int and case[key]>=-1 for key in
            ("setup_status","heuristic_status","returned_algorithms","null_algorithm_status","launch_sync_status")),
            "invalid raw API status")
        require(case["setup_status"]>=0 and case["returned_algorithms"] in (0,1),"invalid query result")
        if case["setup_status"]!=0:
            require(case["heuristic_status"]==case["null_algorithm_status"]==case["launch_sync_status"]==-1,
                    "call recorded after descriptor setup rejection")
        else:
            require(case["heuristic_status"]>=0 and case["null_algorithm_status"]>=0,"missing algorithm probe")
            require((case["null_algorithm_status"]==0)==(case["launch_sync_status"]>=0),"inconsistent launch/synchronize record")
    return dict(**header,cases=cases,raw_sha256=sha(folder/"stdout"),
                heuristic_with_candidates=sum(c["heuristic_status"]==0 and c["returned_algorithms"]>0 for c in cases),
                synchronized_null_success=sum(c["null_algorithm_status"]==0 and c["launch_sync_status"]==0 for c in cases))


def tile_status(folder):
    table=[]
    for fmt in ("bf16","e4m3","e5m2"):
        case=folder/fmt
        require((case/"exit").read_text().strip()=="0",f"{fmt} smoke process failed; inspect stderr")
        row=json.loads((case/"stdout").read_text())
        require(row["format"]==fmt and row["shape"]==[1024]*3 and row["elements"]==1024**2,
                f"{fmt} task identity mismatch")
        require(row["passed"] is True and type(row["mismatches"]) is int and row["mismatches"]==row["max_abs"]==0,
                f"{fmt} full numerical reference failed")
        require(set(row["inputs"])=={"A","B_transposed"} and all(re.fullmatch("[0-9a-f]{64}",v) for v in row["inputs"].values()),
                f"{fmt} missing input identities")
        compiler=case/"compiler"
        files=[p for ext in ("tileirbc","cubin") for p in compiler.glob("*bf16_gemm*."+ext)]
        require({p.suffix for p in files}=={".tileirbc",".cubin"},f"{fmt} missing compiled Tile IR or cubin")
        logs=(case/"stderr").read_text(errors="replace")
        require(any("Invoke tile compiler:" in line and "sm_110" in line and "bf16_gemm" in line for line in logs.splitlines()),
                f"{fmt} actual compiler invocation missing")
        table.append(dict(**row,compiler_sha256={p.name:sha(p) for p in sorted(files)}))
    return table


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--smoke",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    result=dict(library=library_status(args.library),tile_type_smoke=tile_status(args.smoke))
    write(args.out,result)
    print("10 cuBLASLt descriptor configurations and 3 full numerical type smokes reviewed")
