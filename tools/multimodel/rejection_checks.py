"""Repeat E12's ten bounded rejection cases on copies of archived real evidence (CPU only)."""
import argparse
import base64
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from unittest.mock import patch

from tilescope.common import digest, read_json, sha256, write_json
from tilescope.probe import collect
from tilescope.probe.compare import compare_runs
from tilescope.probe.depth.review import dense_review, raw_request
from tilescope.probe.review import review_run
from tools.multimodel.llama_trace import review as cuda_review


def require(condition,message):
    if not condition:
        raise AssertionError(message)


def rejects(call,message):
    try:
        call()
    except ValueError as exc:
        require(message in str(exc),f"wrong rejection: {exc}")
        return str(exc)
    raise AssertionError("expected rejection: "+message)


def strict(directory,message):
    result=review_run(directory)
    require(result["status"]=="failed" and result["rows"]==[] and result["groups"]==[] and message in result["reason"],
            f"bad evidence reached the report or wrong rejection: {result.get('reason')}")
    return {"status":result["status"],"rows":0,"groups":0,"reason":result["reason"]}


def seal_plan(root,plan):
    write_json(root/"plan.json",plan)
    (root/"plan.sha256").write_text(sha256(root/"plan.json")+"\n")


def seal_attempt(path):
    write_json(path/"hashes.json",collect._files(path))


def wire(path):
    return [(r["ns"],base64.b64decode(r["data_b64"])) for line in (path/"wire.jsonl").read_text().splitlines()
            if (r:=json.loads(line))]


def emit(path,chunks):
    with (path/"wire.jsonl").open("w") as out:
        for ns,data in chunks:
            out.write(json.dumps({"ns":ns,"data_b64":base64.b64encode(data).decode()})+"\n")
    seal_attempt(path)


def replace_wire(path,old,new):
    chunks=wire(path)
    raw=b"".join(data for _,data in chunks)
    require(old in raw,"mutation does not apply to this recording")
    return emit(path,[(chunks[-1][0],raw.replace(old,new,1))])


class Cases:
    def __init__(self,a):
        self.a=a
        self.base=collect.load_run(a.request)[0]
        require(self.base["suite"]["suite_id"]=="protocol-pilot","use a real vLLM protocol-pilot suite")

    def live_binding(self):
        directory=self.a.binding
        if directory is None:
            return {"status":"not_requested"}
        bound=read_json(directory/"result.json")
        require(bound["wrong_weight_exit"]!=0 and bound["strict_exit"]!=0 and bound["optional_exit"]==0
                and bound["basic_run_exit"]==0,"live profile refusal exit states changed")
        require("weight identity mismatch" in (directory/"wrong-weight.stderr").read_text(),
                "wrong-weight refusal has no matching live diagnostic")
        required=read_json(directory/"required-check.json")
        basic=read_json(directory/"basic-check.json")
        require(set(required["missing_required"])=={"model.standard_rmsnorm","model.dense_linear"} and
                basic["missing_required"]==[] and set(required["missing_required"])<=set(basic["optional_missing"]),
                "live capability gates changed")
        plan,rows,_=collect.load_run(directory/"raw")
        require(plan["target"]["model"]["deep_adapter"]["id"]=="tilescope.unsupported.test.v999" and
                [r["execution_id"] for r in rows]==bound["valid_requests"] and len(rows)==2,
                "unknown-depth profile has no matching valid basic stream/nonstream pilot")
        return bound

    @contextmanager
    def copy(self,source=None):
        with tempfile.TemporaryDirectory(prefix="tilescope-reject-") as tmp:
            root=Path(tmp)/"run"
            shutil.copytree(source or self.a.request,root)
            yield root

    def stream(self,root):
        execution,=read_json(root/"waves"/"0000.json")
        path=root/"attempts"/execution
        require(read_json(path/"request.json")["stream"],"expected first case streaming")
        return path

    def case1(self):
        with self.copy() as root:
            p=read_json(root/"plan.json"); t=p["target"]
            weights=t["model"]["storage"]["sha256"]
            name=next(iter(weights))
            weights[name]="0"*64
            t["profile_sha256"]=digest({k:v for k,v in t.items() if k not in {"profile_sha256","identity","target_fingerprint"}})
            t["target_fingerprint"]=digest([t["profile_sha256"],t["identity"]])
            seal_plan(root,p)
            return {"mutation":"different expected weight SHA; frozen-plan hashes rebuilt",
                    "filename":name,"outcome":strict(root,"differs from bound service"),"live":self.live_binding()}

    def case2(self):
        target=self.base["target"]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model=deepcopy(target["model"])
            model["deep_adapter"]={"id":"unsupported.test.v999"}
            write_json(root/"model.json",model)
            write_json(root/"engine.json",target["engine"])
            write_json(root/"target.json",dict(schema="tilescope.target.v1",model_ref="model.json",engine_ref="engine.json",
                       target_id=target["target_id"],deployment=target["deployment"]))
            outcomes=[]
            for required,expect in ((["request.complete"],0),(["model.standard_rmsnorm"],1)):
                write_json(root/"suite.json",{"schema":"tilescope.suite.v1","suite_id":"protocol-pilot","required_capabilities":required})
                result=subprocess.run([sys.executable,"-m","tilescope","probe","check","--offline",
                                       "--target",str(root/"target.json"),"--suite",str(root/"suite.json"),
                                       "--out",str(root/"out.json")],capture_output=True,text=True)
                require(result.returncode==expect,result.stderr or "capability exit status mismatch")
                snap=read_json(root/"out.json")
                cap=next(c for c in snap["capabilities"] if c["capability_id"]=="model.standard_rmsnorm")
                require(cap["observed_status"]=="unsupported" and snap["missing_required"]==([] if expect==0 else required),
                        "unsupported depth is hidden or basic plan blocked")
                outcomes.append({"required":required,"exit_code":result.returncode,"missing_required":snap["missing_required"],
                                 "deep_status":cap["observed_status"]})
            return {"offline_plans":outcomes,"baseline_valid_requests":len(collect.load_run(self.a.request)[1]),
                    "live":self.live_binding()}

    def case3(self):
        with self.copy() as root:
            (root/"waves"/"0001.json").unlink()
            missing=strict(root,"missing planned waves")
        with self.copy() as root:
            path=self.stream(root)
            replace_wire(path,b"data: [DONE]",b": deleted terminal")
            terminal=strict(root,"incomplete response")
        return {"missing_wave":missing,"removed_terminal_resealed":terminal}

    def case4(self):
        with self.copy() as root:
            path=self.stream(root)
            body=read_json(path/"request.json")
            replace_wire(path,body["request_id"].encode(),b"tilescope-wrong-request")
            wrong_id=strict(root,"identity mismatch")
        with self.copy() as root:
            path=self.stream(root)
            chunks=wire(path)
            raw=b"".join(data for _,data in chunks)
            lines=raw.splitlines(keepends=True)
            found=False
            for i,line in enumerate(lines):
                if line.startswith(b"data: {"):
                    obj=json.loads(line[5:])
                    if obj.get("usage") is not None:
                        obj["usage"]["completion_tokens"]-=1
                        obj["usage"]["total_tokens"]-=1
                        lines[i]=b"data: "+json.dumps(obj).encode()+b"\n"
                        found=True
            require(found,"missing actual usage coverage")
            emit(path,[(chunks[-1][0],b"".join(lines))])
            counts=strict(root,"IDs and server usage differ")
        return {"wrong_response_id":wrong_id,"output_ids_vs_usage":counts}

    def case5(self):
        with self.copy() as root:
            path=self.stream(root);chunks=wire(path)
            # Split into bytes with the exact original observation time. Delimiters,
            # JSON strings and UTF-8 scalars may all straddle HTTP reads.
            emit(path,[(ns,bytes([byte])) for ns,block in chunks for byte in block])
            checked=review_run(root)
            require(checked["status"]=="complete","valid stream fragmentation rejected")
            fragmented=len(checked["rows"])
            emit(path,chunks+[(chunks[-1][0],b'data: "\xe6')])
            truncated=strict(root,"truncated SSE")
        with self.copy() as root:
            path=self.stream(root); chunks=wire(path)
            emit(path,chunks+[(chunks[-1][0],b"data: [DONE]\n\n")])
            duplicate=strict(root,"after terminal")
        result={"valid_fragmented_requests":fragmented,"truncated":truncated,"duplicate_terminal":duplicate}
        if self.a.native:
            with self.copy(self.a.native) as root:
                path=self.stream(root); chunks=wire(path)
                raw=b"".join(b for _,b in chunks)
                lines=raw.splitlines(keepends=True)
                for i,line in enumerate(lines):
                    if line.startswith(b"data:") and (obj:=json.loads(line[5:]))["stop"]:
                        obj["tokens"]=[1];lines[i]=b"data: "+json.dumps(obj).encode()+b"\n"
                        break
                else:raise ValueError("no real native final record to mutate")
                emit(path,[(chunks[-1][0],b"".join(lines))])
                result["native_cumulative_replay"]=strict(root,"repeated generated data")
        return result

    def case6(self):
        with self.copy() as root:
            plan=read_json(root/"plan.json"); path=self.stream(root)
            after=path/"after.prom"; model=plan["target"]["deployment"]["served_model"]
            data=after.read_text()
            require(f'model_name="{model}"' in data,"requested model label absent")
            after.write_text(data.replace(f'model_name="{model}"','model_name="another-model"').replace('engine="0"','engine="1"'))
            # Preserve a fully consistent unattributed client measurement. Only
            # the explicitly mandatory phase gate should reject this suite.
            row=collect._request_result(path,plan,plan["waves"][0][0],read_json(path/"receipt.json"))
            require(row["engine"]["attribution"]=="unavailable" and row["total_s"]>0,"wrong attribution")
            write_json(path/"result.json",row);seal_attempt(path)
            return {"unattributed_client_total_s":row["total_s"],"required_phase":strict(root,"required serial phase")}

    def case7(self):
        result={}
        with self.copy() as root, patch.dict(os.environ,{"TILESCOPE_LOCK":str(root/"local-test.lock")}):
            for typ in ("profile","engine","protocol"):
                t=deepcopy(self.base["target"])
                if typ=="profile":
                    t["profile_sha256"]="0"*64
                elif typ=="engine":
                    t["engine"]["build_pin"] += "-different"
                else:
                    t["engine"]["primary_protocol"] += "-different"
                result[typ]=rejects(lambda:collect.run(t,self.base["suite"],root,resume=True),"resume identity")
        if self.a.resume_receipt:
            receipt=read_json(self.a.resume_receipt)
            batch=self.a.resume_receipt.parent/Path(receipt["command"][-1]).name
            _,rows,failures=collect.load_run(batch)
            require(receipt["signal"]=="SIGTERM" and receipt["exit"]==-15 and receipt["resume_exit"]==0,
                    "unexpected interrupted/restarted client exit status")
            before={**receipt["completed_hashes_before"],**receipt["partial_hashes_before"]}
            require(before and receipt["partial"] and receipt["completed_hashes_before"],"no real before/after evidence")
            for remote_path,expected in before.items():
                parts=Path(remote_path).parts
                local=batch.joinpath(*parts[parts.index("attempts"):])
                require(sha256(local)==expected,"old completed/partial bytes changed after resume")
            retries=[]
            for old in receipt["partial"]:
                partial=read_json(batch/"attempts"/old/"receipt.json")
                require(any(x["execution_id"]==old for x in failures),"original partial attempt wasn't retained")
                retried,=[r for r in rows if r["key"]==partial["task_key"]]
                require(retried["attempt"]>partial["attempt"] and retried["execution_id"]!=old,"retry reused attempt identity")
                retries.append({"partial":old,"completed":retried["execution_id"],"new_attempt":retried["attempt"]})
            result["live"]={"signal":receipt["signal"],"interrupted_exit":receipt["exit"],"resume_exit":receipt["resume_exit"],
                            "original_hashes_recomputed":len(before),"retries":retries}
        return result

    def case8(self):
        root=self.a.dense
        req=raw_request(root,"dense")
        doc=read_json(root/"dense.json")
        internal=doc["engine_request_id"]
        require(all(r["passed"] for r in dense_review(root,req,internal).values()),"source dense evidence failed")
        with tempfile.TemporaryDirectory() as tmp:
            altered=Path(tmp)
            shutil.copytree(root/"tensors",altered/"tensors")
            shutil.copyfile(root/"depth-plan.json",altered/"depth-plan.json")
            d=deepcopy(doc)
            d["ops"]["linear"]["metrics"]["max_abs"] += 1
            write_json(altered/"dense.json",d)
            inconsistent=rejects(lambda:dense_review(altered,req,internal),"recorded error metrics")
            d=deepcopy(doc)
            actual=d["ops"]["linear"]["actual"]
            data=altered/actual["file"]
            with data.open("r+b") as f:f.write(bytes.fromhex("8046"))  # finite BF16 16384
            actual["sha256"]=sha256(data)
            write_json(altered/"dense.json",d)
            over=rejects(lambda:dense_review(altered,req,internal),"thresholds exceeded")
            return {"resealed_tensor_sha256":actual["sha256"],"max_abs_vs_rms":inconsistent,"false_passed":over}

    def case9(self):
        root=self.a.gpu
        verified=cuda_review(root)
        with tempfile.TemporaryDirectory() as tmp:
            altered=Path(tmp)
            for name in ("resolved-target.json","owned.json","launch.json","receipt.json","start.json","stop.json",
                         "request.json","wire.jsonl","server.log","nsys-version.log"):
                shutil.copyfile(root/name,altered/name)
            db=altered/"activity.sqlite"
            # Keep the real launch/response/offload logs, replace only the CUDA
            # activity table with a schema lacking kernel evidence.
            with sqlite3.connect(db) as con:
                con.execute("CREATE TABLE METADATA(note TEXT)")
            missing=rejects(lambda:cuda_review(altered),"no CUPTI kernel activity")
            shutil.copyfile(root/"activity.sqlite",db)
            with sqlite3.connect(db) as con:
                con.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL")
            none=rejects(lambda:cuda_review(altered),"no valid target-owned CUDA")
            return {"real_owned_kernels":verified["kernel_count"],"missing_activity":missing,"empty_activity":none}

    def case10(self):
        observed=[review_run(p) for p in self.a.comparison]
        require(len(observed)>=2 and all(x["status"]=="complete" for x in observed),"expected real same-content suites")
        compare_runs(observed,"same-content")
        cross=rejects(lambda:compare_runs(observed,"same-model-regression"),"single frozen model")
        row=next(r for r in observed[0]["rows"] if r["kind"]=="same-content")
        row["count_source"]="local_tokenizer_estimate"
        basis=rejects(lambda:compare_runs(observed,"same-content"),"estimated token counts")
        return {"cross_model_strict_regression":cross,"estimated_vs_exact":basis}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",type=Path,required=True,help="real vLLM protocol-pilot run")
    p.add_argument("--binding",type=Path,help="live wrong-weight/unknown-depth trial directory")
    p.add_argument("--resume-receipt",type=Path,help="SIGTERM/restart receipt beside its archived batch")
    p.add_argument("--native",type=Path,help="real native protocol-pilot")
    p.add_argument("--dense",type=Path,help="actual dense diagnostic directory")
    p.add_argument("--gpu",type=Path,help="actual llama.cpp Nsight diagnostic directory")
    p.add_argument("--comparison",type=Path,nargs="+",help="two real same-content suite directories")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--cases",type=int,nargs="+",default=list(range(1,11)),choices=range(1,11))
    a=p.parse_args()
    suite=Cases(a)
    outcomes={}
    evidence={1:"binding",2:"binding",5:"native",7:"resume_receipt",8:"dense",9:"gpu",10:"comparison"}
    for n in a.cases:
        try:
            need=evidence.get(n)
            missing=need is not None and getattr(a,need) is None
            if missing and n in (8,9,10):
                outcomes[str(n)]={"status":"not_run","missing_input":need}
                continue
            detail=getattr(suite,f"case{n}")()
            outcomes[str(n)]={"status":"partial" if missing else "verified","detail":detail}
            if missing:
                outcomes[str(n)]["missing_input"]=need
        except Exception as error:
            outcomes[str(n)]={"status":"failed","reason":f"{type(error).__name__}: {error}"}
    complete=all(r["status"]=="verified" for r in outcomes.values())
    full=set(a.cases)==set(range(1,11))
    failed=any(r["status"]=="failed" for r in outcomes.values())
    inputs={k:[str(p) for p in v] if isinstance(v,list) else str(v) for k,v in vars(a).items()
            if k not in ("out","cases") and v is not None}
    report={"scope":"CPU rejection checks using isolated copies of device evidence","status":"complete" if complete and full
            else "failed" if failed else "partial","cases":outcomes,"inputs":inputs}
    write_json(a.out,report)
    print(json.dumps({"status":report["status"],"cases":{k:v["status"] for k,v in outcomes.items()}}))
    return int(failed or full and not complete)


if __name__=="__main__":
    sys.exit(main())
