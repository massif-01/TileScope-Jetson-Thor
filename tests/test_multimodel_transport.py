"""Frozen protocol risks: split bytes, true termination, count identities and metric scope."""
import base64
import json
from pathlib import Path
import tempfile
import unittest

from tilescope.probe.engines import llama_cpp, openai, vllm
from tilescope.probe.transport import decoded


class EngineContract(unittest.TestCase):
    def wire(self, blocks):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path=Path(tmp.name)/"wire.jsonl"
        path.write_text("".join(json.dumps({"ns":i,"data_b64":base64.b64encode(b).decode()})+"\n"
                                for i,b in enumerate(blocks)))
        return decoded(path,{"send_ns":0,"closed_ns":len(blocks)},True)

    def test_utf8_and_frame_split(self):
        text='data: {"text":"测"}\r\n\r\n'.encode()
        result=list(self.wire([bytes([b]) for b in text]))
        self.assertEqual(result,[(len(text)-1,{"text":"测"})])
        for bad in (text[:-1],b'data: "\xe6"\n\n'):
            with self.assertRaises(ValueError):
                list(self.wire([bad]))

    def native_final(self):
        return dict(index=0,stop=True,model="m",tokens=[],content="",stop_type="limit",
                    tokens_predicted=1,tokens_evaluated=2,truncated=False,
                    timings=dict(prompt_n=2,prompt_ms=15,predicted_n=1,predicted_ms=.1))

    def test_native_terminal_and_count(self):
        body={"prompt":[4,3],"n_predict":1,"stream":True}
        partial=dict(index=0,stop=False,model="m",tokens=[33],content="a",tokens_predicted=1)
        events=[(1,partial),(2,self.native_final())]
        self.assertEqual(llama_cpp.parse(events,body,"m")["output_tokens"],1)
        for bad in (events[:1],events+events[-1:],[(1,partial),(2,dict(self.native_final(),tokens=[33]))],
                    [(1,partial),(2,dict(self.native_final(),tokens_predicted=2))]):
            with self.assertRaises(ValueError):
                llama_cpp.parse(bad,body,"m")

    def test_vllm_termination_identity_and_counts(self):
        body={"prompt":[33],"max_tokens":4,"request_id":"abc","stream":True,"model":"m"}
        event={"id":"cmpl-abc","model":"m","choices":[dict(index=0,text="x",prompt_token_ids=[33],token_ids=[1],
                                                            finish_reason="stop")]}
        usage={"id":"cmpl-abc","model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}
        events=[(1,event),(2,usage),(3,"[DONE]")]
        self.assertEqual(openai.parse(events,body,strong=True)["terminal_ns"],3)
        for bad in (events[:-1],events+events[-1:],[(1,dict(event,id="cmpl-other"))]+events[1:],
                    [events[0],(2,dict(usage,usage=dict(usage["usage"],completion_tokens=2,total_tokens=3))),events[2]]):
            with self.assertRaises(ValueError):
                openai.parse(bad,body,strong=True)

    def test_metric_labels_and_mixed_traffic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"metrics"
            path.write_text('vllm:prompt_tokens_total{model_name="chosen",engine="0"} 20\n'
                            'vllm:prompt_tokens_total{model_name="other",engine="0"} 1031\n'
                            'vllm:prompt_tokens_total{model_name="chosen",engine="1"} 8\n')
            self.assertEqual(vllm.snapshot(path,{"deployment":{"served_model":"chosen"}}),{"prompt_tokens_total":20})
        m=dict(num_requests_running=0,num_requests_waiting=0,prompt_tokens_total=0,generation_tokens_total=0,
               request_success_total=0)
        for phase in ("queue","prefill","decode","inference"):
            for suffix in ("count","sum"):
                m[f"request_{phase}_time_seconds_{suffix}"]=0
        measured={k:(1 if k in ("generation_tokens_total","request_success_total") or k.endswith(("_count","_sum")) else v)
                  for k,v in m.items()}
        measured["prompt_tokens_total"]=20
        self.assertEqual(vllm.phases(m,measured,20,1,1)["attribution"],"single-request delta")
        for k,v in (("request_success_total",2),("num_requests_running",1),("request_decode_time_seconds_count",2)):
            self.assertEqual(vllm.phases(m,dict(measured,**{k:v}),20,1,1)["attribution"],"unavailable")
