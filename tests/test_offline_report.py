"""Offline integration tests using the retained, actual Thor samples in fresh v2 envelopes."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tilescope.common import ROOT, read_json, seal_evidence, write_json
from tilescope.report import compare_runs, generate
from tilescope.review import ReviewError, review_batch
from tilescope.runner import identities
from tilescope.tasks import load_catalog


def legacy_batch(root, name, selected, condition="test host identity"):
    """A synthetic v2 envelope around original telemetry. Used only to exercise the reader."""
    catalog = load_catalog()
    batch = root / name
    batch.mkdir()
    environment = dict(
        runtime={},
        compiler={},
        device={"label": condition},
        driver="test",
        power_mode={"stdout": "test"},
        endpoint="test",
    )
    document = {
        "schema": "tilescope.batch.v2",
        "run_id": name,
        "mode": "strict",
        "environment": environment,
        "tasks": [],
    }
    write_json(batch / "catalog.json", catalog)
    for task_id, source in selected:
        task = catalog[task_id]
        folder = batch / "attempts" / task_id
        original = ROOT / "round2/evidence/runs" / source
        shutil.copytree(original, folder / "native")
        identity = identities(task, environment)
        write_json(folder / "task.json", task)
        write_json(folder / "identity.json", identity)
        write_json(folder / "status.json", {"status": "passed", "reason": ""})
        write_json(folder / "command.json", {"test_fixture_from": "round2/evidence/runs/" + source})
        seal_evidence(folder)
        document["tasks"].append(
            dict(
                task_id=task_id,
                status="passed",
                attempt="attempts/" + task_id,
                **{k: v for k, v in identity.items() if k.endswith("_hash")},
            )
        )
    write_json(batch / "batch.json", document)
    return batch


@unittest.skipUnless((ROOT / "round2/evidence/runs").is_dir(), "requires private device capture")
class OfflineReport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_real_raw_keeps_v1_statistics_and_supports_comparison(self):
        current = legacy_batch(self.root, "second", [("nvfp4", "nvfp4-02")])
        prior = legacy_batch(self.root, "first", [("nvfp4", "nvfp4-01")])
        summary = generate(current, self.root / "report", compare=prior)
        stored = [
            json.loads(line) for line in (ROOT / "round2/evidence/results.jsonl").read_text().splitlines()
        ]
        expected = [r for r in stored if r["run_id"] == "nvfp4-02"]
        for new, old in zip(summary["results"], expected, strict=True):
            for key in (
                "boundary",
                "implementation",
                "median_us",
                "sd_us",
                "min_us",
                "max_us",
                "tflops",
                "max_abs",
                "rel_l2",
            ):
                self.assertEqual(new[key], old[key])
        published = read_json(self.root / "report/results.json")
        self.assertEqual(len(published["comparison"]["rows"]), 4)
        self.assertTrue(
            all(r["compatible"] and "未能区分" in r["judgement"] for r in published["comparison"]["rows"])
        )
        report = (self.root / "report/index.html").read_text()
        self.assertIn("<svg", report)
        self.assertIn("nvfp4-02", (current / "attempts/nvfp4/command.json").read_text())
        self.assertIn('href="../second/attempts/nvfp4/native/stdout"', report)
        self.assertNotIn("<script src=", report)

    def test_condition_mismatch_has_no_change_rate(self):
        current = review_batch(legacy_batch(self.root, "second", [("fp8", "fp8-01")], condition="different"))
        prior = review_batch(legacy_batch(self.root, "first", [("fp8", "fp8-01")]))
        comparison = compare_runs(current, prior)
        self.assertTrue(all(not r["compatible"] and "change_pct" not in r for r in comparison))

    def test_copied_raw_samples_do_not_count_as_an_independent_run(self):
        one = review_batch(legacy_batch(self.root, "one", [("fp8", "fp8-01")]))
        copy = review_batch(legacy_batch(self.root, "copy", [("fp8", "fp8-01")]))
        self.assertTrue(all("相同原始样本" in r["reason"] for r in compare_runs(copy, one)))

    def test_sealed_tamper_and_false_check_exclude_entire_batch(self):
        batch = legacy_batch(self.root, "both", [("fp8", "fp8-01"), ("nvfp4", "nvfp4-02")])
        output = self.root / "published"
        generate(batch, output)
        last = batch / "attempts/nvfp4"
        source = last / "native/stdout"
        text = source.read_text()
        source.write_text(text.replace('"passed": true', '"passed": "false"', 1))
        for reseal in (False, True):
            if reseal:
                seal_evidence(last)
            with self.assertRaises(ReviewError):
                generate(batch, output)
            self.assertEqual(read_json(output / "results.json")["results"], [])
            self.assertFalse((output / "latency.svg").exists())
            self.assertNotIn('<tr id="metric-', (output / "index.html").read_text())

    def test_survey_separates_declared_unsupported(self):
        batch = legacy_batch(self.root, "survey", [("fp8", "fp8-01"), ("nvfp4", "nvfp4-02")])
        doc = read_json(batch / "batch.json")
        doc["mode"] = "survey"
        receipt = dict(status="unsupported", reason="test diagnostic: adapter rejected format")
        doc["tasks"][1].update(receipt)
        write_json(batch / "batch.json", doc)
        write_json(batch / "attempts/nvfp4/status.json", receipt)
        out = review_batch(batch)
        self.assertEqual({r["task_id"] for r in out["results"]}, {"fp8"})
        self.assertIn("unsupported", {r["status"] for r in out["coverage"]})
        self.assertIn("not_run", {r["status"] for r in out["coverage"]})

    def test_wrong_catalog_and_traversal(self):
        batch = legacy_batch(self.root, "bad", [("fp8", "fp8-01")])
        catalog = read_json(batch / "catalog.json")
        catalog["fp8"]["rows"] = 128
        write_json(batch / "catalog.json", catalog)
        with self.assertRaisesRegex(ReviewError, "登记目录"):
            review_batch(batch)
        doc = read_json(batch / "batch.json")
        doc["tasks"][0]["attempt"] = "../bad"
        write_json(batch / "batch.json", doc)
        with self.assertRaisesRegex(ReviewError, "超出批次"):
            review_batch(batch)

    def test_supplemental_backend_failure_preserves_accepted_implementation_pair(self):
        batch = legacy_batch(self.root, "main", [("fp8", "fp8-01")])
        diagnostic = self.root / "backend"
        diagnostic.mkdir()
        (diagnostic / "exit").write_text("1\n")
        (diagnostic / "stdout").write_text(
            '{"phase":"failure","stage":"compile","reason":"injected diagnostic"}\n'
        )
        (diagnostic / "stderr").write_text("injected diagnostic\n")
        output = self.root / "rendered"
        result = generate(batch, output, backend=diagnostic)
        self.assertEqual(len(result["results"]), 4)
        self.assertEqual(result["backend_study"]["results"], [])
        self.assertIn("injected diagnostic", (output / "REPORT.md").read_text())

    def test_actual_phase_observations_and_client_timing_stay_separate(self):
        batch = legacy_batch(self.root, "main", [("fp8", "fp8-01")])
        observed = batch / "observations"
        sample = dict(
            id="synthetic-test-request",
            usage=dict(prompt_tokens=64, completion_tokens=4),
            total_ms=60,
            first_nonempty_chunk_ms=32,
            nonempty_chunk_gaps_ms=[10, 18],
        )
        write_json(observed / "online-requests.json", [sample])
        record = dict(
            key=["bf16_ba", "decode_target_verify", 4],
            request_id=sample["id"],
            actual_input={"shape": [4, 5120], "stride": [5120, 1], "dtype": "torch.bfloat16"},
        )
        write_json(
            observed / "summary.json",
            dict(
                schema="tilescope.observation.v1",
                online_timing="online-requests.json",
                observation_records=[record],
            ),
        )
        doc = read_json(batch / "batch.json")
        doc["observation"] = {"evidence": "observations/summary.json"}
        write_json(batch / "batch.json", doc)
        generate(batch, self.root / "rendered")
        report = (self.root / "rendered/REPORT.md").read_text()
        self.assertIn("decode_target_verify", report)
        self.assertIn("首个非空 chunk", report)
        self.assertIn("独立观察实例中的实际前向张量", report)


if __name__ == "__main__":
    unittest.main()
