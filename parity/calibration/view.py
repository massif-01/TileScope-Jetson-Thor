"""Filter individual calibration outcomes without making an invalid round eligible for comparison."""
import csv
try:
    from .run import load_plan,write
except ImportError:
    from run import load_plan,write


def grouped_view(plan,results,status,group_by):
    valid={row["id"]:row for row in results["process_summaries"]}
    reasons={row["id"]:row["reason"] for row in results["rejected"]}
    complete={(g["format"],g["m"],g["n"],g["k"]) for g in results["groups"]}
    rows=[]
    for case in load_plan(plan):
        fmt,m,n,k=case
        workload=f"{fmt}-{m}x{n}x{k}"
        for rnd in range(1,plan["rounds"]+1):
            tag=f"r{rnd:02}-{workload}"
            row=valid.get(tag)
            failure=reasons.get(tag) or reasons.get(workload)
            ok=row is not None and failure is None
            state="passed" if ok else "failed"
            if status!="all" and status!=state:
                continue
            rows.append(dict(id=tag,status=state,format=fmt,shape=f"{m}x{n}x{k}",round=rnd,
                             workload=workload,complete_workload=case in complete,
                             raw_sample_count=row["us"]["n"] if ok else None,
                             round_median_us=row["us"]["median"] if ok else None,
                             exclusion_reason=failure if failure else (None if ok else "missing attempt")))
    groups={}
    for row in rows:
        key=str(row[group_by])
        group=groups.setdefault(key,dict(key=key,passed=0,failed=0,attempts=[]))
        group[row["status"]]+=1
        group["attempts"].append(row)
    return dict(batch_status="failed" if results["rejected"] else "passed",
                batch_rejection_count=len(results["rejected"]),
                filter=status,group_by=group_by,selected_attempts=len(rows),groups=list(groups.values()))


def save_view(out,view):
    write(out/"view.json",view)
    keys=["id","status","format","shape","round","workload","complete_workload",
          "raw_sample_count","round_median_us","exclusion_reason"]
    with (out/"view.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["group"]+keys)
        writer.writeheader()
        for group in view["groups"]:
            writer.writerows(dict(group=group["key"],**row) for row in group["attempts"])
