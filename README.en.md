# TileScope

[简体中文](README.md) · **English**

**Performance probes for CUDA operators and model serving on NVIDIA Jetson Thor.** TileScope compares equivalent CUDA operator implementations and measures time to first visible output, total latency, throughput, and variability for real text requests. Each run keeps its measurement conditions and raw timings. Request records retain their inputs and termination status, and operator records retain their correctness checks. Both can be reviewed offline.

## What you can measure

| Use case | Measurements | Commands |
| --- | --- | --- |
| Compare operator implementations | Fixed workloads for CUDA Tile and conventional CUDA, with outputs and timings checked against the same shape, precision, and operation boundary. | `list`, `doctor`, `run`, `review` |
| Measure model serving | Text requests to vLLM and llama.cpp, with client timings, actual token counts, termination status, and engine phases where attribution is possible. | `probe check`, `probe run`, `probe review` |
| Review multiple runs | Comparisons for the same content, the same token budget, or regressions on the same model. Reports retain sample counts, distributions, and failures. | `probe report`, `probe compare`, `probe independent` |
| Inspect hardware and execution | Thor hardware queries, runtime conditions, CUDA kernel traces, and numerical checks on selected layers. | `parity hardware`, engine and model diagnostics |

## Verified deployments

Request collection and offline review were verified for all four deployments on a single Thor. The deeper checks depend on the model architecture and the engine's instrumentation.

| Profile | Model and engine | Additional checks |
| --- | --- | --- |
| [T0](profiles/targets/T0.json) | Qwen3.8-27B-NVFP4 · vLLM | Native engine phases, CUDA Graph replay, MTP activity, and local numerical checks |
| [T1](profiles/targets/T1.json) | SmolLM2-1.7B-Instruct BF16 · vLLM | First-layer RMSNorm and linear projection checks, plus GPU kernel tracing |
| [T2](profiles/targets/T2.json) | Qwen3-0.6B BF16 · vLLM | Configuration-only onboarding for a supported architecture, plus first-layer RMSNorm and linear projection checks |
| [T3](profiles/targets/T3.json) | SmolLM2-1.7B-Instruct Q4_K_M GGUF · llama.cpp CUDA | Native prompt and generation phases, plus CUDA activity attributed to the serving process |

The public profiles record the verified model sources and engine builds. Use `probe check` to see which capabilities a workload requires before switching models. For a new engine or model architecture, start with the existing adapter interfaces and validation rules. [Model sources and versions](profiles/sources.lock.json) · [Adapter guide (Chinese)](docs/ADAPTERS.md)

## Quick start

### Check a configuration offline

The offline commands use the Python standard library. Run them from the repository root.

```sh
python3 -m tilescope list --json
python3 -m tilescope probe check --offline \
  --target profiles/targets/T1.json \
  --suite workloads/suites/protocol-pilot.json \
  --out /tmp/tilescope-check.json
```

`list` shows the available operators and shapes. `check --offline` resolves the model, engine, and workload, then reports supported capabilities and `missing_required`. Runtime capabilities and deployment identity are checked on the device.

### Run a pilot on Thor

Prepare the model weights and the matching CUDA inference environment. Copy `profiles/` into the local configuration directory to preserve its relative references. Set the served model name, checkpoint path, executable path, loopback port, and GPU UUID in the target file. The vLLM check verifies the serving version. The llama.cpp check verifies the executable and loaded library hashes.

```sh
mkdir -p local
cp -R profiles local/
```

With T1 configured, start its dedicated instance in one terminal and wait for `local/t1-server/ready.json`.

```sh
python3 -m tilescope.probe.owned \
  --target local/profiles/targets/T1.json --out local/t1-server
```

From the repository root in a second terminal, run the protocol pilot. It sends one streaming and one non-streaming diagnostic request. Once `probe review` returns `"status": "complete"`, choose a measurement workload.

```sh
TARGET=local/profiles/targets/T1.json
SUITE=workloads/suites/protocol-pilot.json
python3 -m tilescope probe check --target "$TARGET" --suite "$SUITE" --out local/t1-check.json
python3 -m tilescope probe run --target "$TARGET" --suite "$SUITE" --out runs/t1-pilot
python3 -m tilescope probe review runs/t1-pilot
```

Use a fresh output directory for the [fixed-length workload](workloads/suites/new-target.json), the [same-content workload](workloads/suites/same-content.json), or the [limited-concurrency workload](workloads/suites/concurrency.json). Stop the dedicated instance with Ctrl+C.

## Build a report from raw data

List the run directories in a study file. Run paths are relative to that file. This example generates Markdown and JSON reports from the pilot above.

```sh
printf '%s\n' '{"runs":["t1-pilot"]}' > runs/pilot-study.json
python3 -m tilescope probe report --study runs/pilot-study.json --out runs/pilot-report
```

- **`plan.json`** records the resolved deployment, model identity, and workload. Check it before repeating a run under the same conditions.
- **`attempts/`** holds requests, raw responses, timings, and exit status. Review checks model identity, token counts, and protocol termination.
- **`sessions/`** holds session-level resource samples for temperature, clock frequency, and memory. Short or interrupted sessions may lack a sample after the request ends.
- **`REPORT.md` and `report.json`** contain samples and summary statistics. Diagnostic, warm-up, and formal samples are labeled, and failed runs retain their reasons.

Same-model regression checks for changes under fixed conditions. Same-content comparisons show how deployments handle identical input. Same-token-budget comparisons hold the input token count and generation cap constant. Reports use actual generated token counts. An EOS-only response can have a total latency and an output token count while its time to first visible output is absent. Interpret native engine phases using the boundaries defined by each engine. Comparisons that vary precision, engine, or cache policy describe differences between deployments as a whole. [Reporting and statistics guide (Chinese)](docs/reporting.md)

## Repository layout

| Path | Contents |
| --- | --- |
| [`catalog/`](catalog/) | Operator tasks, shapes, and validation rules |
| [`profiles/`](profiles/) and [`workloads/`](workloads/) | Model and engine profiles, deployment examples, and reusable workloads |
| [`tilescope/probe/`](tilescope/probe/) | Request collection, protocol review, diagnostics, and comparisons |
| [`parity/`](parity/) | Hardware probes, CUTLASS calibration, and service diagnostics |
| [`tests/`](tests/) | Tests for parsing, numerical limits, and failure handling. Tests that need private device records skip when those records are absent. |

The public repository provides source code for reuse and new experiments. To reproduce the original acceptance review, use the private Git bundle with its accompanying evidence archive. Obtain public model weights from the listed sources. Partner CUDA baselines and third-party notices are retained alongside their source files. [Baseline provenance](parity/calibration/source.json) · [TileGym license](catalog/TileGym-LICENSE.txt) · [thor-probe license](parity/hardware/LICENSE.thor-probe)
