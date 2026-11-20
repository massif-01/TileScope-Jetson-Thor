"""Failure modes which could silently produce a plausible service result."""
import base64
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
from parity.service_review import compare,rows,comparison,deployment_conditions
from parity.report import report
from types import SimpleNamespace
from tilescope.common import ROOT,digest,read_json,sha256,write_json


@unittest.skipUnless((ROOT / "evidence/parity/service/service-pilot").is_dir(), "requires private device capture")
class ServiceEvidence(unittest.TestCase):
    pilot=ROOT/"evidence/parity/service/service-pilot"
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.root=Path(self.tmp.name)/"pilot"
        shutil.copytree(self.pilot,self.root)
        self.attempt=next((self.root/"attempts").iterdir())
    def tearDown(self):self.tmp.cleanup()

    def test_actual_pilot_and_missing_request(self):
        _,r,_,_=rows(self.root)
        self.assertEqual([x["n"] for x in r],[128,8192])
        shutil.rmtree(self.attempt)
        with self.assertRaisesRegex(ValueError,"missing completed"):
            rows(self.root)

    def mutate_stream(self,fn):
        p=self.attempt/"stream.jsonl"; lines=p.read_text().splitlines()
        changed=False
        for i,line in enumerate(lines):
            record=json.loads(line)
            raw=base64.b64decode(record["data_b64"]).decode().strip()
            if not raw.startswith("data: {"):continue
            event=json.loads(raw[6:])
            if fn(event):
                record["data_b64"]=base64.b64encode(("data: "+json.dumps(event)+"\n").encode()).decode()
                lines[i]=json.dumps(record);changed=True;break
        self.assertTrue(changed)
        p.write_text("\n".join(lines)+"\n")
        m=read_json(self.attempt/"receipt.json");m["raw_hash"]=sha256(p);write_json(self.attempt/"receipt.json",m)

    def test_resealed_server_input_mismatch(self):
        def change(e):
            for c in e["choices"]:
                if c.get("prompt_token_ids"):
                    c["prompt_token_ids"][0]+=1
                    return True
        self.mutate_stream(change)
        with self.assertRaisesRegex(ValueError,"prompt token echo"):rows(self.root)

    def test_comparison_blocks_power_policy_change(self):
        other=Path(self.tmp.name)/"alternate"
        shutil.copytree(self.root,other)
        for path,policy in [(self.root,"70W"),(other,"different")]:
            config=read_json(path/"plan.json")
            config["identity"]["platform_conditions"]={"power_policy":policy}
            write_json(path/"plan.json",config)
        with self.assertRaisesRegex(ValueError,"deployment configuration"):
            comparison(self.root,other)

    def test_version_is_an_allowed_comparison_factor(self):
        a=read_json(self.root/"plan.json")
        a["identity"]["platform_conditions"]={"power_policy":"70W"}
        b=json.loads(json.dumps(a))
        b["identity"]["service_version"]={"version":"synthetic-next"}
        b["identity"]["servers"][0]["config"]["--port"]="58101"
        b["endpoint"]=a["endpoint"].replace("58000","58101")
        self.assertEqual(deployment_conditions(a),deployment_conditions(b))
        b["identity"]["servers"][0]["config"]["--tensor-parallel-size"]="2"
        self.assertNotEqual(deployment_conditions(a),deployment_conditions(b))

    def test_rejected_report_contains_no_valid_latency_rows(self):
        output=Path(self.tmp.name)/"report"
        parms=SimpleNamespace(out=output,hardware=None,calibration=None,trace=None,service=[self.root])
        self.assertEqual(report(parms),0)
        shutil.rmtree(self.attempt)
        self.assertEqual(report(parms),1)
        result=read_json(output/"results.json")
        self.assertEqual(result["sections"],{})
        self.assertEqual(len((output/"requests.csv").read_text().splitlines()),1)

    def test_resealed_usage_contradicts_returned_ids(self):
        def change(e):
            if e.get("usage"):
                e["usage"]["completion_tokens"]+=1
                return True
        self.mutate_stream(change)
        with self.assertRaisesRegex(ValueError,"token/usage"):rows(self.root)

    def test_declared_comparison_factor_and_known_direction(self):
        rows0=rows(self.root)[1]
        item=rows0[0]
        b=[dict(item,execution="b1"),dict(item,execution="b2")]
        c=[dict(item,execution="c1",total_s=item["total_s"]*1.2),dict(item,execution="c2",total_s=item["total_s"]*1.2)]
        res=compare(b,c)
        self.assertEqual(res["metrics"]["total_s"]["observation"],"same direction, >=5% observed")
        c[0]["output_sha256"]="different"
        self.assertEqual(compare(b,c)["reason"],"output_sha256")
        self.assertEqual(compare(b,b)["status"],"incompatible")
        c=[dict(item,execution="c1"),dict(item,execution="c2")]
        self.assertEqual(compare(b,c)["metrics"]["total_s"]["observation"],"no resolved shift")
