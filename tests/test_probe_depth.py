"""The raw tensors and fixed thresholds determine the verdict, including after resealing."""
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from tilescope.probe.depth.review import THRESHOLDS, dense_review, statistics


class NumericalReview(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root/"tensors").mkdir()
        self.request = {"request":{"prompt":[1,2,3]}}
        adapter = {f"{op}_weight_key":op+".weight" for op in ("linear","norm")}
        for op in ("linear","norm"):
            adapter["logical_"+op+"_path"] = op
        self.plan = {"config":{"rms_norm_eps":0.01, "hidden_size":2, "intermediate_size":3},
                     "model":{"origin":{"revision":"fixture"}, "deep_adapter":adapter}}
        (self.root/"depth-plan.json").write_text(json.dumps(self.plan))
        self.doc = {"thresholds":THRESHOLDS, "positions":[0,1], "engine_request_id":"engine-test",
                    "epsilon":0.01, "model_origin":self.plan["model"]["origin"], "prompt_ids":[1,2],"ops":{}}
        for op in ("linear","norm"):
            width = 3 if op == "linear" else 2
            x = list(range(1,width*2+1))
            field = self.tensor(op+"-x",x,[2,width])
            extra = {}
            if op == "norm":
                ref = []
                for i in (0,2):
                    inv = 1/math.sqrt(sum(e*e for e in x[i:i+2])/2+0.01)
                    ref.extend(e*inv for e in x[i:i+2])
                w = self.tensor("norm-w",[1,1],[2])
                extra["weight"] = w
            else:
                ref = [x[0], x[2], x[3], x[5]]
                w = self.tensor("linear-w",[1,0,0,0,0,1],[2,3])
                extra.update(sample_weight_indices=[0,1], sample_weights=w, bias=None)
            actual = self.tensor(op+"-y",ref,[2,2])
            refs = self.tensor(op+"-r",ref,[2,2],"float64")
            y = [struct.unpack("<f",b"\0\0"+a[0])[0] for a in struct.iter_unpack("2s",(self.root/actual["file"]).read_bytes())]
            self.doc["ops"][op] = {**extra,"path":"model."+op,"device":"cuda:0","weight_key":op+".weight",
                "weight_shape":w["shape"],"weight_dtype":"bfloat16","weight_sha256":w["sha256"],
                "loaded_weight_matches":True,"input":field,"actual":actual,"reference":refs,"metrics":statistics(y,ref)}

    def tensor(self, name, values, shape, dtype="bfloat16"):
        data = (b"".join(struct.pack("<f",v)[2:] for v in values) if dtype == "bfloat16" else
                struct.pack("<"+"d"*len(values),*values))
        rel = "tensors/"+name+".bin"
        (self.root/rel).write_bytes(data)
        return {"file":rel,"dtype":dtype,"shape":shape,"sha256":hashlib.sha256(data).hexdigest()}

    def review(self):
        (self.root/"dense.json").write_text(json.dumps(self.doc))
        return dense_review(self.root, self.request, "engine-test")

    def test_full_rows_and_independent_reference(self):
        self.assertTrue(all(r["passed"] for r in self.review().values()))

    def test_reject_resealed_inconsistent_max_abs(self):
        self.doc["ops"]["linear"]["metrics"]["max_abs"] += 1
        with self.assertRaisesRegex(ValueError,"recorded error metrics"):
            self.review()

    def test_reject_resealed_bad_actual_with_passed_claim(self):
        self.doc["ops"]["linear"]["actual"] = self.tensor("wrong",[9,9,9,9],[2,2])
        with self.assertRaisesRegex(ValueError,"thresholds exceeded"):
            self.review()

    def test_reject_wrong_request_prefix(self):
        self.doc["prompt_ids"] = [3,4]
        with self.assertRaisesRegex(ValueError,"identity or contract"):
            self.review()

    def test_full_checkpoint_by_independent_stdlib_math(self):
        from tilescope.probe.depth.checkpoint_review import check
        self.review()
        ckpt = self.root/"checkpoint"
        ckpt.mkdir()
        cfg = json.dumps(self.plan["config"]).encode()
        (ckpt/"config.json").write_bytes(cfg)
        data = bytearray()
        header = {}
        for op in ("norm","linear"):
            w = self.doc["ops"][op]
            t = w.get("weight") or w["sample_weights"]
            begin = len(data)
            data.extend((self.root/t["file"]).read_bytes())
            header[op+".weight"] = {"shape":w["weight_shape"], "data_offsets":[begin,len(data)], "dtype":"BF16"}
        h = json.dumps(header).encode()
        full = struct.pack("<Q",len(h))+h+data
        name = "model.safetensors"
        (ckpt/name).write_bytes(full)
        self.plan.update(config_sha256=hashlib.sha256(cfg).hexdigest())
        self.plan["model"]["storage"] = {"weights":[name],"sha256":{name:hashlib.sha256(full).hexdigest()}}
        self.doc["checkpoint_file"] = name
        (self.root/"depth-plan.json").write_text(json.dumps(self.plan))
        (self.root/"dense.json").write_text(json.dumps(self.doc))
        http = self.root/"http"
        http.mkdir()
        req, resp = {"stream":False}, {"id":"fixture"}
        (http/"dense.request").write_text(json.dumps(req))
        (http/"dense.response").write_text(json.dumps(resp))
        (http/"dense.json").write_text(json.dumps({"request":req,"response":resp,"run_id":"fixture"}))
        result = check(self.root, ckpt)
        self.assertEqual(result["operators"]["linear"]["elements"],4)
        (ckpt/name).write_bytes(full+b"x")
        with self.assertRaisesRegex(ValueError,"content SHA-256"):
            check(self.root,ckpt)
