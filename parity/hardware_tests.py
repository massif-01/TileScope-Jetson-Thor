"""Regression checks for source/absence labeling and raw hardware result gates."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from parity.hardware.__main__ import review
from parity.hardware.system import reading
from parity.hardware.telemetry import parse_tegrastats
from tilescope.common import ROOT, sha256

PILOT = ROOT / "evidence/parity/hardware/attempts/pilot-full"


class HardwareTests(unittest.TestCase):
    def test_unavailable_is_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "speed"
            self.assertEqual(reading(p, "Mbit/s", True)["status"], "absent")
            p.write_text("-1\n")
            self.assertIsNone(reading(p, "Mbit/s", True)["value"])
            p.write_text("0\n")
            self.assertEqual(reading(p, "Mbit/s", True)["value"], 0)

    def test_parse_preserves_emc_unit_and_rails(self):
        data = parse_tegrastats(
            "RAM 500/1024MB (lfb 8x4MB) CPU [0%@972,off] EMC_FREQ 0%@665 gpu@40.5C VDD_GPU 0mW/10mW"
        )
        self.assertEqual(
            data["ram_MiB_used_total_lfb_count_lfb_MiB"], [500, 1024, 8, 4]
        )
        self.assertEqual(data["emc_util_percent"], 0)
        self.assertEqual(data["emc_clock_MHz"], 665)
        self.assertEqual(
            data["rails_mW"]["VDD_GPU"], {"instant": 0, "reported_average": 10}
        )
        self.assertIsNone(parse_tegrastats("CPU [off]")["emc_clock_MHz"])

    def test_resealed_wrong_output_missing_point_and_occupancy_are_rejected(self):
        raw = [
            json.loads(line)
            for line in (PILOT / "native.jsonl").read_text().splitlines()
        ]
        for find, expected in [
            (lambda r: r.get("test") == "fp8_e4m3_mma", "failed fp8_e4m3_mma"),
            (
                lambda r: r.get("test") == "pointer_chase_KiB",
                "missing pointer_chase_KiB",
            ),
            (lambda r: r["type"] == "occupancy_api", "occupancy API"),
        ]:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "run"
                shutil.copytree(PILOT, out)
                rows = list(raw)
                i = next(i for i, r in enumerate(rows) if find(r))
                if rows[i]["type"] == "numeric":
                    # Preserve the producer's claimed zero mismatch while corrupting a real element.
                    rows[i] = {**rows[i], "values": [0] + rows[i]["values"][1:]}
                else:
                    del rows[i]
                path = out / "native.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in rows))
                hashes = json.loads((out / "hashes.json").read_text())
                hashes["native.jsonl"] = sha256(path)
                (out / "hashes.json").write_text(json.dumps(hashes))
                with self.assertRaisesRegex(ValueError, expected):
                    review(out)


if __name__ == "__main__":
    unittest.main()
