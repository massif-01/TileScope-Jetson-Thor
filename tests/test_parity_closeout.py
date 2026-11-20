import unittest
from unittest.mock import patch

from parity import trace_review
from parity.hardware.__main__ import review
from tilescope.common import ROOT


@unittest.skipUnless((ROOT / "evidence/parity/trace/main").is_dir(), "requires private device capture")
class CloseoutTests(unittest.TestCase):
    def test_sensor_records_stay_separate_from_cuda_timings(self):
        old = review(ROOT / "evidence/parity/hardware/thor-01")
        self.assertEqual(old["sweep_coverage"]["inventory_samples_in_native"], 0)
        self.assertEqual(old["sweep_coverage"]["boundary_samples"], 0)
        self.assertAlmostEqual(old["widest_sweep_ratio"]["ratio"], 4.719013426, places=7)
        new = review(ROOT / "evidence/closeout/checked")
        self.assertEqual(new["sweep_coverage"]["readable_pairs"], 132)
        self.assertEqual(sum(p["n"] for v in new["curves"].values() for p in v), 132)

    def test_health_cannot_replace_the_request_abort(self):
        root = ROOT / "evidence/parity/trace/main"
        valid = trace_review.analyze(root)["behavior"]["cancellation"]
        self.assertAlmostEqual(valid["cancel-prefill"]["to_scheduler_abort_ms"], 1060.883838)
        self.assertAlmostEqual(valid["cancel-decode"]["to_scheduler_abort_ms"], 145.695474)
        original = trace_review.lines

        def wrong_reason(path):
            rows = original(path)
            for r in rows:
                if r.get("event") == "external_finish" and valid["cancel-prefill"]["internal_id"] in r["ids"]:
                    r["status"] = "FINISHED_STOPPED"
            return rows

        with patch.object(trace_review, "lines", wrong_reason):
            result = trace_review.analyze(root)
        self.assertIn("FINISHED_ABORTED", result["errors"]["behavior"])
        self.assertNotIn("behavior", result)
