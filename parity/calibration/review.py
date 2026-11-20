"""Recompute native calibration from per-process inputs, checks and untrimmed events."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
from xml.sax.saxutils import escape

try:
    from .run import load_plan,write,digest,enforce
except ImportError:
    from run import load_plan,write,digest,enforce


def read(path):
    return json.loads(path.read_text())


def require(value, reason):
    if not value:
        raise ValueError(reason)


def stats(values):
    return {"n":len(values),"median":st.median(values),"mean":st.mean(values),"sd":st.stdev(values) if len(values)>1 else None,
            "min":min(values),"max":max(values)}


def checked(run,plan,config_sha256,build,fmt,m,n,k,rnd,tag):
    rec=read(run/"receipt.json")
    require(rec["id"]==tag and rec["exit"]==0 and rec["round"]==rnd
            and (rec["format"],rec["m"],rec["n"],rec["k"])==(fmt,m,n,k),"run identity or process exit")
    require(rec["end_ns"]>rec["begin_ns"] and type(rec["pid"]) is int,"process/time identity")
    require(rec["config_sha256"]==config_sha256 and rec["binary_sha256"]==build["executable_sha256"],
            "plan or binary identity differs from worker receipt")
    require(build["exit"]==0 and re.fullmatch(r"[a-f0-9]{64}",build["executable_sha256"]),"compiler identity")
    require(rec["raw_sha256"]=={file:digest(run/file) for file in ("stdout","stderr")},"raw hash mismatch")
    events=[json.loads(line) for line in (run/"stdout").read_text().splitlines()]
    expected_stages=["identity","correctness","correctness","setup"]+["sample"]*plan["samples"]+["correctness"]
    require([e["phase"] for e in events]==expected_stages,"sample count or event ordering")
    ident,first,graph,setup=events[:4]
    require(ident["mnk"]==[m,n,k] and ident["format"]==fmt and ident["sms"]>0 and ident["sm"]==110
            and ident["cuda_runtime"]>0 and ident["driver_api"]>0
            and all(ident[key]==plan[key] for key in ("seed","samples","warmup")),"worker configuration")
    expected={"A","B"} if fmt.startswith("bf16") else {"A","B","SFA","SFB"}
    if fmt=="nvfp4-fp4":
        expected.add("normalization")
    inputs=rec["inputs"]
    require(set(inputs)==expected and all(re.fullmatch(r"[a-f0-9]{64}",v["sha256"]) and v["bytes"]>0 for v in inputs.values()),
            "input or quantization identity")
    size=lambda count: count*2 if fmt.startswith("bf16") else count//2
    require(inputs["A"]["bytes"]==size(m*k) and inputs["B"]["bytes"]==size(n*k),"packed extent mismatch")
    require([e["stage"] for e in (first,graph,events[-1])]==["direct","graph","post"]
            and all(e["passed"] is True for e in (first,graph,events[-1])),"numerical reference failed")
    require(setup["operations_per_graph"]==1 and all(math.isfinite(setup[t]) and setup[t]>=0 for t in ("first_ms","initialize_ms","capture_ms")),"setup evidence")
    samples=events[4:-1]
    require([s["i"] for s in samples]==list(range(plan["samples"])),"sample indices")
    us=[s["us"] for s in samples]
    require(all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in us),"invalid CUDA event duration")
    return dict(id=tag,format=fmt,m=m,n=n,k=k,round=rnd,inputs=inputs,raw=rec["raw_sha256"]["stdout"],
                pid=rec["pid"],begin_ns=rec["begin_ns"],end_ns=rec["end_ns"],us=stats(us),
                effective_tflops=2*m*n*k/st.median(us)/1e6,setup=setup)


def aggregate(folder):
    plan=read(folder/"plan.json")
    cases=load_plan(plan)
    config_sha256=digest(folder/"plan.json")
    builds=read(folder/"build.json")["builds"]
    before,after=(read(folder/name) for name in ("conditions-before.json","conditions-after.json"))
    enforce(plan["condition"],before)
    require(before==after and read(folder/"finish.json")["policy_unchanged"] is True, "condition policy changed")
    rows=[]
    rejected=[]
    grouped={}
    for fmt,m,n,k in cases:
        group=[]
        for rnd in range(1,plan["rounds"]+1):
            tag=f"r{rnd:02}-{fmt}-{m}x{n}x{k}"
            try:
                group.append(checked(folder/tag,plan,config_sha256,builds[fmt],fmt,m,n,k,rnd,tag))
            except (KeyError,TypeError,ValueError,OSError) as e:
                rejected.append(dict(id=tag,reason=str(e)))
        if len(group)==plan["rounds"]:
            identity={json.dumps(r["inputs"],sort_keys=True) for r in group}
            if len(identity)!=1 or len({(r["pid"],r["begin_ns"]) for r in group})!=len(group) or len({r["raw"] for r in group})!=len(group):
                rejected.append(dict(id=f"{fmt}-{m}x{n}x{k}",reason="round inputs differ or process evidence repeated"))
                continue
            desc=stats([r["us"]["median"] for r in group])
            grouped[fmt,m,n,k]=dict(format=fmt,m=m,n=n,k=k,work_flop=2*m*n*k,rounds=len(group),events_per_round=plan["samples"],
                                    median_us=desc["median"],mean_round_median_us=desc["mean"],sd_round_median_us=desc["sd"],
                                    min_round_median_us=desc["min"],max_round_median_us=desc["max"],
                                    effective_tflops=2*m*n*k/desc["median"]/1e6)
        rows.extend(group)
    return dict(scope=plan["scope"],config_sha256=config_sha256,ranked=False,unit="us per graph replay, one GEMM",
                process_summaries=rows,groups=list(grouped.values()),rejected=rejected)


def svg(rows,path):
    width,height=900,420
    left,top,right,bottom=90,54,865,335
    colors=["#0b6973","#a24516","#5260a3","#846110"]
    fmts=sorted({r["format"] for r in rows})
    points=[(r["work_flop"],r["median_us"]) for r in rows]
    if not points:
        path.unlink(missing_ok=True)
        return
    xs=sorted({x for x,_ in points})
    lo,hi=map(math.log2,(min(xs),max(xs)))
    ymax=max(r["max_round_median_us"] for r in rows)*1.07
    X=lambda x:left+(math.log2(x)-lo)/max(1,hi-lo)*(right-left)
    Y=lambda y:bottom-y/ymax*(bottom-top)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Native GEMM calibration by format and operation count">',
         '<rect width="100%" height="100%" fill="white"/>','<g font-family="sans-serif" font-size="12" fill="#252525">',
         '<text x="20" y="23" font-size="17">Native GEMM calibration · median across independent round medians</text>']
    for i in range(6):
        y=ymax*i/5
        svg.append(f'<path d="M {left} {Y(y)} H {right}" stroke="#ddd"/><text x="{left-8}" y="{Y(y)+4}" text-anchor="end">{y:.2f}</text>')
    for x in xs:
        label=f"{x/1e6:.2f}M" if x>=1_000_000 else f"{x:,}"
        svg.append(f'<text x="{X(x)}" y="{bottom+18}" text-anchor="middle">{label}</text>')
    for j,fmt in enumerate(fmts):
        group=sorted((r for r in rows if r["format"]==fmt),key=lambda r:r["work_flop"])
        color=colors[j]
        line=' '.join(f'{X(r["work_flop"]):.2f},{Y(r["median_us"]):.2f}' for r in group)
        svg.append(f'<polyline points="{line}" fill="none" stroke="{color}"/>')
        for r in group:
            x=X(r["work_flop"])
            svg.append(f'<path d="M{x} {Y(r["min_round_median_us"])} V{Y(r["max_round_median_us"])}" stroke="{color}"/>'
                       f'<circle cx="{x}" cy="{Y(r["median_us"])}" r="3" fill="{color}"><title>{r["m"]} × {r["n"]} × {r["k"]}, {r["median_us"]:.3f} µs</title></circle>')
        svg.append(f'<text x="{left+j*183}" y="400" fill="{color}">{escape(fmt)}</text>')
    svg+=['<text x="14" y="49">µs</text><text x="385" y="376">2 × M × N × K (log scale)</text>','</g></svg>']
    path.write_text(''.join(svg)+'\n')


def round_plot(result,path):
    rows=result["groups"]
    if not rows:
        path.unlink(missing_ok=True)
        return
    data={(r["format"],r["m"],r["n"],r["k"]):[] for r in rows}
    for r in result["process_summaries"]:
        key=tuple(r[k] for k in ("format","m","n","k"))
        if key in data:
            data[key].append(r)
    colors=("#126d77","#b94f22","#5c58a1","#97680a")
    rounds=max(r["rounds"] for r in rows)
    legend_rows=(rounds+7)//8
    extra=16*(legend_rows-1)
    width,height=940,105+extra+35*len(rows)
    tone=lambda i:colors[i] if rounds<=4 else f"hsl({(i*137)%360},65%,37%)"
    xmax=max(r["max_round_median_us"] for r in rows)
    xs=lambda x:265+615*x/xmax
    drawing=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Independent process round medians">',
             '<rect width="100%" height="100%" fill="white"/><g font-family="sans-serif" font-size="13">',
             '<text x="12" y="22" font-size="17">Independent process medians by GEMM shape and output format (µs)</text>']
    for i in range(rounds):
        x,y=15+110*(i%8),42+16*(i//8)
        drawing.append(f'<circle cx="{x}" cy="{y-4}" r="3" fill="{tone(i)}"/><text x="{x+8}" y="{y}">Round {i+1}</text>')
    for j,(key,samples) in enumerate(data.items()):
        y=65+extra+j*35
        label=f"{key[0]}   {key[1]} × {key[2]} × {key[3]}"
        drawing.append(f'<text x="9" y="{y+4}">{escape(label)}</text><path d="M265 {y} H880" stroke="#ddd"/>')
        for i,row in enumerate(samples):
            val=row["us"]["median"]
            offset=(i-(len(samples)-1)/2)*min(4,20/len(samples))
            drawing.append(f'<circle cx="{xs(val):.2f}" cy="{y+offset:.2f}" r="3" fill="{tone(i)}"><title>round {row["round"]}, {val:.4f} µs</title></circle>')
    for i in range(6):
        x=xmax*i/5
        drawing.append(f'<text x="{xs(x)}" y="{height-15}" text-anchor="middle">{x:.2f}</text>')
    drawing.append('</g></svg>')
    path.write_text("".join(drawing)+"\n")


def exports(folder,out):
    res=aggregate(folder)
    out.mkdir(parents=True,exist_ok=True)
    write(out/"results.json",res)
    columns=["format","m","n","k","rounds","events_per_round","median_us","mean_round_median_us","sd_round_median_us","min_round_median_us","max_round_median_us","effective_tflops"]
    with (out/"groups.csv").open("w",newline="") as f:
        wr=csv.DictWriter(f,fieldnames=columns,extrasaction="ignore")
        wr.writeheader()
        wr.writerows(res["groups"])
    with (out/"groups.tex").open("w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\nFormat & M & N & K & Rounds & Median ($\\mu$s) & TFLOP/s \\\\\n")
        for r in res["groups"]:
            f.write("{format} & {m} & {n} & {k} & {rounds} & {median_us:.4f} & {effective_tflops:.4f} ".format(**r)+r"\\"+"\n")
        f.write("\\end{tabular}\n")
    svg(res["groups"],out/"scale.svg")
    round_plot(res,out/"rounds.svg")
    return res


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--status",choices=("all","passed","failed"),default="all",help="filter the process-outcome view only")
    p.add_argument("--group-by",choices=("format","shape","round","workload"),default="format")
    a=p.parse_args()
    result=exports(a.runs,a.out)
    try:
        from .view import grouped_view,save_view
    except ImportError:
        from view import grouped_view,save_view
    view=grouped_view(read(a.runs/"plan.json"),result,a.status,a.group_by)
    save_view(a.out,view)
    print(f'{len(result["groups"])} complete groups, {len(result["rejected"])} exclusions')
    raise SystemExit(int(bool(result["rejected"])))
