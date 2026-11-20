import json
import shutil
import tempfile
import unittest
from pathlib import Path

from parity.calibration.run import load_plan
from tilescope.common import seal_evidence
from tilescope.review import ReviewError,review_batch

ROOT = Path(__file__).resolve().parents[1]


class CalibrationTests(unittest.TestCase):
    def test_out_of_range_alignment(self):
        plan=json.loads((ROOT/"parity/calibration/plan.json").read_text())
        self.assertEqual(len(load_plan(plan)),16)
        plan["cases"][2]["shapes"][0][2]=63
        with self.assertRaisesRegex(ValueError,"alignment"):
            load_plan(plan)

    @unittest.skipUnless((ROOT / "evidence/v2/batches/validated").is_dir(), "requires private device capture")
    def test_resealed_impossible_absolute_error(self):
        original=ROOT/"evidence/v2/batches/validated"
        baseline=json.loads((ROOT/"evidence/v2/reports/final/results.json").read_text())["results"]
        self.assertEqual(review_batch(original)["results"],baseline)
        with tempfile.TemporaryDirectory() as scratch:
            batch=Path(scratch)/"batch"
            shutil.copytree(original,batch)
            attempt=next((batch/"attempts").glob("rmsnorm-regular-*"))
            source=attempt/"native/stdout"
            events=list(map(json.loads,source.read_text().splitlines()))
            check=next(x for x in events if x["phase"]=="correctness")
            check["max_abs"]=1e30
            source.write_text("".join(json.dumps(x)+"\n" for x in events))
            seal_evidence(attempt)
            with self.assertRaisesRegex(ReviewError,"max_abs inconsistent"):
                review_batch(batch)

    @unittest.skipUnless((ROOT / "evidence/parity/calibration/formal").is_dir(), "requires private device capture")
    def test_filtered_failure_on_a_copy_of_formal_events(self):
        from parity.calibration.run import digest
        from parity.calibration.review import exports
        from parity.calibration.view import grouped_view
        real=ROOT/"evidence/parity/calibration/formal"
        plan=json.loads((real/"plan.json").read_text())
        with tempfile.TemporaryDirectory() as scratch:
            data=Path(scratch)/"copy"
            shutil.copytree(real,data)
            raw=next(data.glob("r01-bf16-bf16-128x128x128/stdout"))
            receipt=raw.parent/"receipt.json"
            events=list(map(json.loads,raw.read_text().splitlines()))
            next(row for row in events if row.get("stage")=="post")["passed"]=False
            raw.write_text("".join(json.dumps(row)+"\n" for row in events))
            identity=json.loads(receipt.read_text())
            identity["raw_sha256"]["stdout"]=digest(raw)
            receipt.write_text(json.dumps(identity))
            result=exports(data,Path(scratch)/"export")
            view=grouped_view(plan,result,"failed","format")
            self.assertEqual(len(result["groups"]),15)
            self.assertEqual(view["batch_status"],"failed")
            self.assertEqual(view["selected_attempts"],1)
            row=view["groups"][0]["attempts"][0]
            self.assertEqual(row["id"],raw.parent.name)
            self.assertIsNone(row["round_median_us"])
