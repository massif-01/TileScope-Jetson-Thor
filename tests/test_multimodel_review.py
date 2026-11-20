"""Targeted failures around multi-model comparison, fragmented bytes and count sources."""
import base64
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from tilescope.common import sha256, write_json
from tools.multimodel.llama_trace import review as review_cuda
from tilescope.probe.compare import compare_runs
from tilescope.probe.independent import reconstruct
from tilescope.probe.review import check_soak, request_rows, summarize


def sample(key, kind="same-content"):
    return dict(execution_id=key, key=key, role="formal", kind=kind, n=128, o=16,
                ix=0, round=0, concurrency=1, stream=True, input_tokens=128, output_tokens=2,
                count_source="engine_token_ids", send_ns=100, first_event_ns=150, first_visible_ns=170,
                terminal_ns=180, closed_ns=200, prompt_ids=[1]*128, output_ids=[2, 3],
                engine={"attribution":"single-request delta","decode_s":1e-8,"prefix_cache_hits_total":0},
                input_sha256="original", text_sha256="answer", finish="length")


class Comparison(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.raw = Path(self.temp.name)

    def runs(self, kind="same-content"):
        runs = []
        for tid in ("A", "B"):
            data = sample(tid, kind)
            if tid == "B":
                data["text_sha256"] = "another answer"
            write_json(self.raw / "attempts" / tid / "request.json", {"prompt":"相同原文"})
            row = request_rows(self.raw, [data])[0]
            runs.append({"status":"complete","target":{"target_id":tid},"rows":[row]})
        return runs

    def test_original_bytes_can_match_with_different_token_lengths_and_answers(self):
        runs = self.runs()
        runs[1]["rows"][0]["input_tokens"] += 3
        output, = compare_runs(runs, "same-content")
        self.assertEqual(output["raw_text_sha256"],hashlib.sha256("相同原文".encode()).hexdigest())
        self.assertEqual(output["deployments"]["B"]["input_tokens"]["median"],131)
        with self.assertRaisesRegex(ValueError, "single frozen model"):
            compare_runs(runs, "same-model-regression")

    def test_mixed_count_basis_and_content_rejected(self):
        runs = self.runs()
        runs[1]["rows"][0]["count_source"] = "local_tokenizer_estimate"
        with self.assertRaisesRegex(ValueError, "estimated token counts"):
            compare_runs(runs, "same-content")
        runs = self.runs()
        runs[1]["rows"][0]["raw_text_sha256"] = "other text"
        with self.assertRaisesRegex(ValueError, "actual content"):
            compare_runs(runs, "same-content")

    def test_missing_actual_length_fails_and_warmups_are_separate(self):
        row = sample("fixed-1", "fixed")
        with self.assertRaisesRegex(ValueError, "length"):
            request_rows(self.raw, [dict(row, input_tokens=127)])
        runs = self.runs("fixed")
        rows = runs[0]["rows"]
        group, = summarize(rows + [dict(rows[0], role="warmup", total_s=987)])
        self.assertEqual(group["requests"],1)
        self.assertEqual(group["total_s"]["median"],1e-7)
        self.assertIsNone(group["total_s"]["sample_sd"])
        self.assertIsNone(group["cache_strata"][0]["vllm_computed_tokens"])

    def test_independent_wire_observation_uses_last_delimiter_byte(self):
        write_json(self.raw / "request.json", {"stream":True})
        write_json(self.raw / "receipt.json", {"send_ns":99, "closed_ns":900})
        wire = ('data: {"choices":[{"text":"你","token_ids":[1,2]}]}\r\n\r\n'
                'data: {"choices":[],"usage":{"completion_tokens":2}}\r\n\r\n'
                'data: [DONE]\r\n\r\n').encode()
        with (self.raw / "wire.jsonl").open("w") as f:
            for n, byte in enumerate(wire):
                f.write(json.dumps(dict(ns=100+n,data_b64=base64.b64encode(bytes([byte])).decode()))+"\n")
        result = reconstruct(self.raw)
        self.assertAlmostEqual(result["ttfv_ms"],(wire.index(b"\r\n\r\n")+4)/1e6)
        self.assertEqual(result["output_tokens"],2)

    def test_soak_detects_a_burst_followed_by_idle_time(self):
        session = self.raw/"sessions"/"s1"
        write_json(session/"start.json", {"wall_ns":1000})
        (session/"resources.jsonl").write_text(json.dumps({"ns":55,"wall_ns":1005})+"\n")
        suite = {"suite_id":"soak","interval_s":15,"duration_s":300}
        rows = [{"round":i,"send_ns":100 + i*15_000_000_000} for i in range(20)]
        self.assertEqual(check_soak(self.raw, suite, rows)["inter_send_s"]["median"],15)
        rows[19]["send_ns"] = rows[18]["send_ns"] + 10
        with self.assertRaisesRegex(ValueError,"time slots"):
            check_soak(self.raw, suite, rows)

    def test_eos_only_has_an_exact_count_and_no_visible_time(self):
        write_json(self.raw/"request.json", {"stream":True})
        write_json(self.raw/"receipt.json", {"send_ns":10, "closed_ns":20})
        wire = ('data: {"choices":[{"text":"","token_ids":[2]}]}\n\n'
                'data: {"choices":[],"usage":{"completion_tokens":1}}\n\n'
                'data: [DONE]\n\n').encode()
        def save(data):
            (self.raw/"wire.jsonl").write_text(json.dumps({"ns":15,"data_b64":base64.b64encode(data).decode()})+"\n")
        save(wire)
        result = reconstruct(self.raw)
        self.assertEqual(result["output_tokens"],1)
        self.assertIsNone(result["ttfv_ms"])
        save(wire.replace(b"data: [DONE]\n\n",b""))
        with self.assertRaisesRegex(ValueError,"unverifiable"):
            reconstruct(self.raw)

    def test_gpu_path_needs_owned_kernel_activity_and_unions_overlap(self):
        owned = {"pid":1234,"start_ticks":5678,"argv":["/example/llama-server"]}
        target = {"deployment":{"host_identity":"GPU-test","served_model":"mocked-gguf"},
                  "identity":{"server_pid":1234,"start_ticks":5678,"boot_id":"boot","argv":owned["argv"],
                              "device_uuid":"GPU-test"}}
        write_json(self.raw/"owned.json", owned)
        write_json(self.raw/"resolved-target.json", target)
        write_json(self.raw/"launch.json", {"boot_id":"boot"})
        write_json(self.raw/"request.json", {"stream":False,"prompt":"one test","n_predict":2})
        write_json(self.raw/"receipt.json", {"http":200,"send_ns":110,"closed_ns":150,
                                            "request_sha256":sha256(self.raw/"request.json")})
        write_json(self.raw/"start.json", {"end_ns":105,"exit_code":0})
        write_json(self.raw/"stop.json", {"begin_ns":160,"exit_code":0})
        obj = {"index":0,"stop":True,"model":"mocked-gguf","truncated":False,"content":"a","tokens":[1],
               "tokens_predicted":1,"tokens_evaluated":2,"stop_type":"eos",
               "timings":{"predicted_n":1,"prompt_n":2,"predicted_ms":3,"prompt_ms":2}}
        write_json(self.raw/"wire.jsonl", {"ns":130,"data_b64":base64.b64encode(json.dumps(obj).encode()).decode()})
        # Transport records have one JSON object per line.
        (self.raw/"wire.jsonl").write_text(json.dumps(json.loads((self.raw/"wire.jsonl").read_bytes()))+"\n")
        (self.raw/"nsys-version.log").write_text("mock nsys")
        (self.raw/"server.log").write_text("offloaded 4/4 layers to GPU")
        with sqlite3.connect(self.raw/"activity.sqlite") as db:
            db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INT,end INT,deviceId INT,globalPid INT)")
            db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (100,110,0,?)",(4321 << 24,))
        with self.assertRaisesRegex(ValueError,"no valid target-owned"):
            review_cuda(self.raw)
        with sqlite3.connect(self.raw/"activity.sqlite") as db:
            db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,0,?)",
                           [(0,10,1234 << 24),(4,12,1234 << 24),(20,22,1234 << 24)])
        verified = review_cuda(self.raw)
        self.assertEqual(verified["kernel_count"], 3)
        self.assertEqual(verified["kernel_union_ns"], 14)

    def test_bad_suite_never_compares_received_subset(self):
        runs = self.runs()
        runs[1]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "complete requested"):
            compare_runs(runs, "same-content")


if __name__ == "__main__":
    unittest.main()
