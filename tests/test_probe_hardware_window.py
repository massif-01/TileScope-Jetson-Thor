import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from parity.hardware.native_window import review, run


class NativeCoverage(unittest.TestCase):
    def test_parent_observations_bracket_both_sides_of_child_window(self):
        window = {"clock_domain":"host_boot_monotonic_ns", "boot_id":"fixture",
                  "start_monotonic_ns":20, "end_monotonic_ns":60}
        def row(edge, time):
            return {"edge":edge, "read_start_mono_ns":time, "read_end_mono_ns":time+2,
                    "gpu_mC":41000, "gpc_hz":315000000, "available_bytes":32*1024**3}
        rows = [row("before_spawn",10), row("while_running",30), row("after_wait",70)]
        self.assertEqual(review(rows,window,.02)["during"],1)
        with self.assertRaisesRegex(ValueError,"bracketing"):
            review([rows[0],rows[1],row("after_wait",50)],window,.02)
        with self.assertRaisesRegex(ValueError,"bracketing"):
            review([rows[0],rows[-1]],window,.02)

    def test_sampler_lifetime_on_a_real_cpu_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            sensors = {"gpc_hz":base/"hz","gpu_mC":base/"temp"}
            for key,value in (("gpc_hz",500000000),("gpu_mC",41000)):
                sensors[key].write_text(str(value))
            (base/"boot").write_text("test-boot")
            map_path = lambda p: base/"boot" if str(p)=="/proc/sys/kernel/random/boot_id" else Path(p)
            with patch("parity.hardware.native_window.Path",side_effect=map_path), \
                 patch("parity.hardware.native_window.memory_available",return_value=32*1024**3), \
                 (base/"lock").open("w") as lock:
                code, receipt = run([sys.executable,"-c","import time; time.sleep(.05)"],
                    subprocess.DEVNULL,subprocess.DEVNULL,lock.fileno(),base/"window",sensors,.01)
            rows = [json.loads(line) for line in (base/"window").read_text().splitlines()]
            result = review(rows,receipt,.01)
            self.assertEqual(code, 0)
            self.assertGreater(result["during"], 0)
