import json
import unittest

from check_smoke import parse_smoke


class SmokeParserTest(unittest.TestCase):
    def setUp(self):
        self.shape = (512, 512, 512)
        self.result = {
            "benchmark": "bf16_gemm_cutlass", "m": 512, "n": 512, "k": 512,
            "correctness": "passed", "runtime_ms": {"mean": 0.035},
        }

    def test_accepts_single_line_and_multiline_after_logs(self):
        for indent in (None, 2):
            stdout = "Thor benchmark\n  Disposition: Passed\n" + json.dumps(self.result, indent=indent)
            self.assertEqual(parse_smoke(stdout, 0, self.shape), self.result)

    def test_rejects_failed_process_with_valid_json(self):
        with self.assertRaisesRegex(ValueError, "exit code"):
            parse_smoke(json.dumps(self.result), 1, self.shape)

    def test_rejects_skip_and_truncated_json(self):
        for text in ("This example requires a supported GPU", json.dumps(self.result)[:-1]):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                parse_smoke(text, 0, self.shape)

    def test_rejects_failed_correctness(self):
        self.result["correctness"] = "failed"
        with self.assertRaisesRegex(ValueError, "correctness"):
            parse_smoke(json.dumps(self.result), 0, self.shape)

    def test_rejects_wrong_shape_and_unusable_time(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            parse_smoke(json.dumps(self.result), 0, (1, 512, 512))
        self.result["runtime_ms"]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "runtime"):
            parse_smoke(json.dumps(self.result), 0, self.shape)

    def test_rejects_duplicate_results(self):
        line = json.dumps(self.result)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_smoke(f"{line}\n{line}\n", 0, self.shape)


if __name__ == "__main__":
    unittest.main()
