"""Run the two profile-binding rejection cases, then collect a valid basic pilot on that listener."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

from tilescope.common import read_json, write_json
from tilescope.probe.review import review_run
from tilescope.probe.targets import resolve, workload


def execute(root,case,action,target,suite,dest,expected_error=None):
    args=[sys.executable,"-m","tilescope","probe",action,"--target",str(target),"--suite",str(suite),"--out",str(dest)]
    with (root/(case+".stdout")).open("xb") as out,(root/(case+".stderr")).open("xb") as err:
        result=subprocess.run(args,stdout=out,stderr=err)
    rec={"argv":args,"exit_code":result.returncode,"stdout":case+".stdout","stderr":case+".stderr"}
    write_json(root/(case+".json"),rec)
    if bool(result.returncode)!=(expected_error is not None):
        raise ValueError(f"{case}: unexpected exit {result.returncode}")
    if expected_error and expected_error not in (root/(case+".stderr")).read_text() and (
            not dest.is_file() or expected_error not in dest.read_text()):
        raise ValueError(f"{case}: refusal did not identify the injected error")
    return rec


def trial(profile,pilot,out):
    out=out.resolve();out.mkdir(parents=True,exist_ok=False)
    target=resolve(profile);suite=workload(pilot)
    raw_target=read_json(profile)
    raw_target.update(model_ref="model.json",engine_ref="engine.json")
    write_json(out/"target.json",raw_target)
    write_json(out/"engine.json",target["engine"])
    write_json(out/"corpus.json",suite.pop("corpus"))
    suite["corpus_ref"]="corpus.json"
    suite["required_capabilities"]=["request.complete","request.exact_token_counts","identity.local_target","resources.window"]
    write_json(out/"basic.json",suite)
    strict=deepcopy(suite)
    strict["required_capabilities"] += ["model.standard_rmsnorm","model.dense_linear"]
    write_json(out/"required-deep.json",strict)
    wrong=deepcopy(target["model"])
    name=next(iter(wrong["storage"]["sha256"]))
    wrong["storage"]["sha256"][name]="0"*64
    write_json(out/"bad-weight.json",wrong)
    write_json(out/"bad-target.json",dict(raw_target,model_ref="bad-weight.json"))
    model=deepcopy(target["model"])
    model["deep_adapter"]={"id":"tilescope.unsupported.test.v999"}
    write_json(out/"model.json",model)
    mismatch=execute(out,"wrong-weight","check",out/"bad-target.json",out/"basic.json",
                     out/"bad-check.json","weight identity mismatch")
    required=execute(out,"required-unknown","check",out/"target.json",out/"required-deep.json",
                     out/"required-check.json","model.standard_rmsnorm")
    missing=read_json(out/"required-check.json")["missing_required"]
    if set(missing)!={"model.standard_rmsnorm","model.dense_linear"}:
        raise ValueError("strict depth plan failed for a different reason")
    checked=execute(out,"optional-unknown","check",out/"target.json",out/"basic.json",out/"basic-check.json")
    if not {"model.standard_rmsnorm","model.dense_linear"}<=set(read_json(out/"basic-check.json")["optional_missing"]):
        raise ValueError("missing optional depth was not disclosed")
    ran=execute(out,"basic-pilot","run",out/"target.json",out/"basic.json",out/"raw")
    replay=review_run(out/"raw")
    if replay["status"]!="complete" or len(replay["rows"])!=2:
        raise ValueError("basic pilot failed offline reconstruction")
    write_json(out/"result.json",dict(status="verified",target_id=target["target_id"],wrong_weight_exit=mismatch["exit_code"],
               strict_exit=required["exit_code"],missing_required=missing,optional_exit=checked["exit_code"],
               basic_run_exit=ran["exit_code"],valid_requests=[r["execution_id"] for r in replay["rows"]]))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",type=Path,required=True)
    p.add_argument("--suite",type=Path,default=Path("workloads/suites/protocol-pilot.json"))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    trial(a.target,a.suite,a.out)
