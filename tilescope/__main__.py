"""Unified TileScope operator-probe CLI. Offline commands use the Python standard library."""
import argparse
import json
import os
from pathlib import Path
import sys

from .common import ROOT, device_slot, write_json
from .environment import CUDA, DEFAULT_ENDPOINT, DEFAULT_MODEL, DEFAULT_PYTHON, inspect


def add_selection(p):
    p.add_argument("tasks", nargs="*", help="task IDs or @group")
    p.add_argument("--catalog",type=Path,help="alternate schema-2 catalog")
    p.add_argument("--rows",type=int,help="explicit replay row override for a single supported adapter")


def add_device(p):
    p.add_argument("--python", default=os.getenv("PROBE_PYTHON",DEFAULT_PYTHON))
    p.add_argument("--checkpoint", default=os.getenv("CHECKPOINT",DEFAULT_MODEL))
    p.add_argument("--endpoint", default=os.getenv("TILESCOPE_ENDPOINT",DEFAULT_ENDPOINT))
    p.add_argument("--toolkit",default=os.getenv("CUDA_HOME",CUDA))
    p.add_argument("--min-free-mib",type=int,default=0,help="operator-requested minimum CUDA and system free MiB")


def dump(obj):
    print(json.dumps(obj,indent=2,allow_nan=False,ensure_ascii=False))


def main():
    if sys.argv[1:2]==["probe"]:
        from .probe.cli import dispatch
        return dispatch(sys.argv[2:])
    if sys.argv[1:2]==["parity"]:
        from parity.cli import dispatch
        return dispatch(sys.argv[2:])
    from .tasks import load_catalog, resolve_task, select_tasks
    cli=argparse.ArgumentParser(prog="python3 -m tilescope")
    cmds=cli.add_subparsers(dest="action",required=True)
    cmds.add_parser("probe",help="configurable text-model measurement and offline review")
    cmds.add_parser("parity",help="hardware, calibration, model service observations and reports")
    p=cmds.add_parser("list",help="list or inspect supported task definitions");add_selection(p)
    p.add_argument("--json",action="store_true")
    p=cmds.add_parser("doctor",help="device, dependencies, activity and resource checks");add_device(p);add_selection(p)
    p.add_argument("--out",type=Path)
    p=cmds.add_parser("observe",help="controlled model request and runner shape observation");add_device(p)
    p.add_argument("--out",required=True,type=Path)
    p=cmds.add_parser("run",help="serial measurement of a task or group");add_selection(p);add_device(p)
    p.add_argument("--out",required=True,type=Path);p.add_argument("--resume",action="store_true")
    p.add_argument("--mode",choices=["strict","survey"],default="strict")
    p.add_argument("--observation",type=Path,help="bind a captured summary.json to this batch")
    p=cmds.add_parser("backend",help="bounded same-source Triton PTX/Tile IR diagnostic");add_device(p)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--site",type=Path,required=True)
    p=cmds.add_parser("backend-review",help="verify isolated same-source diagnostic evidence offline")
    p.add_argument("directory",type=Path)
    p=cmds.add_parser("review",help="recalculate a complete batch offline");p.add_argument("batch",type=Path)
    p=cmds.add_parser("report",help="build offline HTML and Markdown");p.add_argument("batch",type=Path)
    p.add_argument("--out",required=True,type=Path);p.add_argument("--compare",type=Path)
    p.add_argument("--backend",type=Path,help="attach a separately verified same-source study")
    args=cli.parse_args()
    if hasattr(args,"min_free_mib") and args.min_free_mib<0:
        raise ValueError("min-free-mib must be nonnegative")
    if args.action in ("run","list","doctor"):
        catalog=load_catalog(args.catalog)
        selected=select_tasks(catalog,args.tasks or ["@all"])
        if args.rows is not None:
            if len(selected)!=1:
                raise ValueError("row override requires exactly one task")
            selected=[resolve_task(selected[0],args.rows)]
            catalog=dict(catalog,**{selected[0]["task_id"]:selected[0]})
    if args.action=="list":
        if args.json:
            dump(selected)
        else:
            for t in selected:
                print(t["task_id"],t["precision"],t["operation"],"×".join(map(str,t["shape"])),
                      t["source"]["kind"],"@"+",@".join(t["groups"]),sep="\t")
    elif args.action=="doctor":
        with device_slot():
            info=inspect(args.python,args.checkpoint,args.endpoint,args.min_free_mib*1024**2,args.toolkit,selected)
        info["requested_tasks"]=[t["task_id"] for t in selected]
        if args.out:
            write_json(args.out,info)
        dump(info)
        return 2 if info["blocked"] else 0
    elif args.action=="observe":
        import subprocess
        with device_slot():
            return subprocess.call([sys.executable,"-m","observation.collect","--output",str(args.out.resolve()),
                                    "--endpoint",args.endpoint,"--checkpoint",str(args.checkpoint),
                                    "--python",args.python],cwd=ROOT)
    elif args.action=="run":
        from .runner import run_batch
        with device_slot() as lock_fd:
            summary=run_batch(args,catalog,selected,lock_fd)
        dump({"status":summary["status"],"run_id":summary["run_id"],"result_rows":len(summary["results"])})
    elif args.action=="backend":
        import subprocess
        if not (args.site/"triton/__init__.py").is_file():
            raise ValueError("isolated Triton-to-Tile-IR --site missing")
        with device_slot():
            code=subprocess.call(["bash",str(ROOT/"same_source/run.sh"),str(args.out.resolve()),
                                  str(args.site.resolve()),args.python],cwd=ROOT)
        from same_source.study import review
        result=review(args.out)
        write_json(args.out/"result.json",result)
        dump(result)
        return code
    elif args.action=="backend-review":
        from same_source.study import review
        result=review(args.directory)
        dump(result)
        return 0 if result["status"]=="passed" else 1
    elif args.action=="review":
        from .review import review_batch
        dump(review_batch(args.batch))
    elif args.action=="report":
        from .report import generate
        generate(args.batch,args.out,compare=args.compare,backend=args.backend)
        print(args.out/"index.html")
    return 0


if __name__=="__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted; incomplete records retained. Use run --resume with the same selection.",file=sys.stderr)
        sys.exit(130)
    except (ValueError,KeyError,OSError,TypeError) as error:
        print(f"TileScope rejected: {error}",file=sys.stderr)
        sys.exit(1)
