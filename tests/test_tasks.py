import copy
import json
from pathlib import Path
import tempfile
import unittest

from tilescope.tasks import DEFAULT_CATALOG, ROOT, load_catalog, resolve_task, review_run, select_tasks


class CatalogContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = load_catalog()

    @unittest.skipUnless((ROOT / "round2/evidence/runs").is_dir(), "requires private device capture")
    def test_archived_quantized_pairs_recompute(self):
        for task, run in (("fp8","fp8-01"),("nvfp4","nvfp4-02")):
            results = review_run(self.catalog[task], ROOT / "round2/evidence/runs" / run)
            self.assertEqual({row["boundary"] for row in results},{"core","full"})
            self.assertTrue(all(row["samples"] == 50 for row in results))

    def test_same_adapter_registers_new_legal_size_without_dispatch_edit(self):
        doc = json.loads(DEFAULT_CATALOG.read_text())
        entry = copy.deepcopy(next(t for t in doc["tasks"] if t["adapter"]=="bf16_ba"))
        entry.update(task_id="ba-seven-row-example", rows=7, groups=["test-new"])
        entry["source"] = dict(kind="synthetic_diagnostic", stage="tail fixture", evidence="tests/test_tasks.py")
        doc["tasks"].append(entry)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"catalog.json"
            path.write_text(json.dumps(doc))
            result, = select_tasks(load_catalog(path), ["@test-new"])
            self.assertEqual(result["shape"], [7,96,5120])
            self.assertEqual(result["outputs"][1:], [[7,96],[96,1]])
            self.assertEqual(resolve_task(result), result)

    def test_illegal_sizes_and_ignored_keys_rejected(self):
        for name in ("bf16-ba-small","rmsnorm-regular","swiglu-regular"):
            for value in (False, 0, -1, 2049, 64.5):
                with self.subTest(name=name,value=value), self.assertRaises(ValueError):
                    resolve_task(self.catalog[name],value)
        with self.assertRaisesRegex(ValueError, "64 rows"):
            resolve_task(self.catalog["nvfp4"], 4)
        doc = json.loads(DEFAULT_CATALOG.read_text())
        doc["tasks"][0]["layout"]="arbitrary"
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"catalog.json"
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError,"unknown task parameters"):
                load_catalog(path)

    def test_row_override_demotes_observed_shape(self):
        task=dict(self.catalog["bf16-ba-small"])
        task["source"] = dict(kind="observed_runtime", stage="prefill", evidence="example-record.json")
        changed=resolve_task(task,2)
        self.assertEqual(changed["source"]["kind"], "synthetic_diagnostic")
        self.assertNotEqual(changed["input_sha256"]["x"], task["input_sha256"]["x"])
        self.assertEqual(changed["input_sha256"][task["weights"][0]], task["input_sha256"][task["weights"][0]])


if __name__ == "__main__":
    unittest.main()
