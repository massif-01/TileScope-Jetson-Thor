"""Regenerate a pinned Git source index from the three local reference checkouts."""
import argparse
import json
from pathlib import Path
import subprocess
from tilescope.common import ROOT, read_json, write_json

SUPPLEMENTS={
  "O01":["tests/orin_perf_preflight_test.py","docs/analysis/decode-context-scaling-2026-09-27/preflight.py"],
  "O02":["tools/reference/qwen36_27b_vllm_greedy.py","tests/fixtures/README.md"],
  "O03":["src/runtime/reference_benchmark.cpp","tests/reference_benchmark_control_test.cpp",
         "docs/analysis/decode-context-scaling-2026-09-27/run_panel.py"],
  "O04":["tools/evaluation/run_vllm_p40_geometry.py","tests/test_vllm_p40_geometry.py"],
  "O06":["tests/test_mtp_soak.py"],
  "O07":["tests/test_run_target_aot_probe_with_nvmap.py","tests/test_classify_target_aot_nvmap_recovery.py"],
  "O08":["docs/analysis/decode-context-scaling-2026-09-27/summarize_profile.py","tests/prefill_route_evidence_test.cpp"],
  "O09":["tests/evalscope_triplet_gate_test.py","tests/service_qualification_test.py"],
  "O10":["tests/openai_protocol_test.cpp"],
  "O11":["tests/service_gateway_contract_test.cpp"],
  "O12":["tools/evaluation/validate_mtp_draft.py","tools/evaluation/validate_mtp_recursive.py",
         "docs/analysis/decode-numerical-repair-2026-09-27/compare.py","src/runtime/reference_oracle.cpp"],
  "O13":["tools/q3x-inspect/main.cpp","tests/safetensors_test.cpp","tests/checkpoint_metadata_test.cpp",
         "src/kernels/sm87/device_probe.cu","tools/evaluation/audit_mtp_install.py"],
  "B01":["probes/","tests/test_bf16.cu","tests/test_fp8.cu","tests/test_fp4_ptx.cu"],"B02":["Makefile"],"H01":["src/thor_probe/include/"],"H10":["src/thor_probe/src/probe_main.cpp","src/communis/spec_reference.cpp","src/thor_probe/src/output/","schema/","tools/validate_json_output.sh"]
}


def build(qwen, probe, bench):
    docs=ROOT/"docs/upstream"
    src=read_json(docs/"thor-inventory.json")["capabilities"]+read_json(docs/"orin-inventory.json")["inventory"]
    pins=read_json(docs/"SOURCE-PINS.json")["references"]
    folders={p.name:p for p in (qwen,probe,bench)}
    index={}
    for ref in pins:
        name=ref["repository"].split("/")[-1];pin=ref["commit"];folder=folders[name]
        cmd=["git","-C",str(folder)]
        commit=subprocess.check_output(cmd+["rev-parse",pin+"^{commit}"],text=True).strip()
        if commit!=pin:raise ValueError(name)
        files={}; extensions={}
        for line in subprocess.check_output(cmd+["ls-tree","-r",pin],text=True).splitlines():
            metadata,path=line.split("\t",1);mode,kind,oid=metadata.split()
            if kind=="blob":files[path]=oid
        for entry in src:
            if entry.get("repo","Qwen3x-Orin")!=name:continue
            chosen=entry.get("paths",entry.get("source_paths",[]))+SUPPLEMENTS.get(entry["id"],[])
            paths={path:oid for path,oid in files.items() if any(path==c or (c.endswith("/") and path.startswith(c)) for c in chosen)}
            if not paths:raise ValueError(f"no source for {entry['id']}")
            extensions[entry["id"]]=paths
        index[name]={"pin":pin,"tree":subprocess.check_output(cmd+["rev-parse",pin+"^{tree}"],text=True).strip(),
                     "tracked_blob_count":len(files),"capability_source_blobs":extensions,
                     "tree_files":files}
    return index

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("qwen",type=Path);p.add_argument("probe",type=Path);p.add_argument("bench",type=Path)
    p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    write_json(a.out,build(a.qwen,a.probe,a.bench))
