"""Profile entry point; offline review does not import a model or GPU runtime."""
import argparse
import json
from pathlib import Path
from tilescope.common import write_json
from .targets import resolve, workload
from .capabilities import plan


def dispatch(argv):
    cli = argparse.ArgumentParser(prog="tilescope probe")
    sub = cli.add_subparsers(dest="action",required=True)
    for action in ("check","run"):
        p = sub.add_parser(action)
        p.add_argument("--target",required=True,type=Path)
        p.add_argument("--suite",required=True,type=Path)
        p.add_argument("--out",required=True,type=Path)
        if action=="run":
            p.add_argument("--resume",action="store_true")
        else:
            p.add_argument("--offline",action="store_true",help="validate only the static plan")
    p=sub.add_parser("review")
    p.add_argument("directory",type=Path)
    for action in ("report","compare"):
        p=sub.add_parser(action)
        p.add_argument("--study",type=Path,required=True)
        p.add_argument("--out",type=Path,required=True)
        if action=="compare":
            p.add_argument("--contract",required=True,choices=["same-content","same-token-budget","same-model-regression"])
    p=sub.add_parser("independent",help="recalculate representative raw streams and statistics")
    p.add_argument("--study",required=True,type=Path)
    p.add_argument("--report",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=cli.parse_args(argv)
    if a.action=="independent":
        from .independent import verify_study
        result=verify_study(a.study,a.report)
        write_json(a.out,result)
        print(a.out)
        return int(result["status"]!="complete")
    if a.action in {"check","run"}:
        target=resolve(a.target)
        suite=workload(a.suite)
        if not getattr(a,"offline",False):
            from .identity import bind
            bind(target)
        result=plan(target,suite)
        if a.action=="check":
            write_json(a.out,{"resolved_target":target,**result})
            print(a.out)
            return int(bool(result["missing_required"]))
        if result["missing_required"]:
            raise ValueError(f"required capabilities unavailable: {result['missing_required']}")
        if suite["suite_id"]=="deep-dense":
            from tilescope.common import device_slot, read_json
            from .depth.client import collect
            from .depth.review import review, THRESHOLDS
            plan_path=a.out/"depth-plan.json"
            if not plan_path.is_file():
                raise ValueError("--out must be the depth directory prepared when starting the owned instance")
            prepared=read_json(plan_path)
            if (prepared["target_sha256"]!=target["source_hashes"]["target"] or prepared["model"]!=target["model"] or
                    prepared["boot_id"]!=target["identity"]["boot_id"] or suite["rows"]!=2 or
                    any(suite["tolerances"][k]!=v for k,v in THRESHOLDS.items())):
                raise ValueError("prepared identity or numeric contract differs from requested suite")
            with device_slot():
                collect(a.out.resolve(),suite.get("profile",False),suite.get("cancel",False))
            write_json(a.out/"depth-review.json",review(a.out.resolve()))
            print(a.out/"depth-review.json")
            return 0
        from .collect import run
        run(target,suite,a.out,resume=a.resume)
        return 0
    if a.action=="review":
        from .review import review_run
        result=review_run(a.directory)
    elif a.action=="report":
        from .report import report_study
        result=report_study(a.study,a.out)
    else:
        from .compare import compare_study
        result=compare_study(a.study,a.contract,a.out)
    print(json.dumps({"status":result["status"],"failures":result.get("reason") or result.get("errors") or result.get("failures")},ensure_ascii=False))
    return int(result["status"]!="complete")
