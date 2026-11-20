# TileScope

**NVIDIA Jetson Thor 上的算子与模型服务性能探针。** TileScope 比较等价 CUDA 算子的执行时间，也从实际文本请求中测量首个可见输出、总耗时、吞吐和波动。每批结果都保留测量条件和原始计时。请求可追溯到输入与终止状态，算子可追溯到校验结果，两条路径均可离线复核。

## 可以测什么

| 用途 | 测量内容 | 命令入口 |
| --- | --- | --- |
| 对照算子实现 | CUDA Tile 与传统 CUDA 的固定工作量。按形状、精度和操作边界核对输出与计时。 | `list`、`doctor`、`run`、`review` |
| 测量模型服务 | vLLM 与 llama.cpp 的文本请求。记录客户端时间、实际 token 数、终止状态及可归因的引擎阶段。 | `probe check`、`probe run`、`probe review` |
| 复核多组结果 | 按同内容、同 token 预算或同模型回归查看差异。保留样本量、分布与失败状态。 | `probe report`、`probe compare`、`probe independent` |
| 核查设备与执行 | Thor 硬件查询、运行条件、CUDA kernel 观察与选定层的局部数学校验。 | `parity hardware`、引擎与模型诊断 |

## 已验证的部署

四种部署均在一台 Thor 上验证了请求采集和离线复算。深度检查按实际模型结构和引擎能力开展。

| 配置 | 模型与引擎 | 已覆盖的深度检查 |
| --- | --- | --- |
| [T0](profiles/targets/T0.json) | Qwen3.8-27B-NVFP4 · vLLM | 原生阶段、Graph replay、MTP 提交、局部数值 |
| [T1](profiles/targets/T1.json) | SmolLM2-1.7B-Instruct BF16 · vLLM | 首层 RMSNorm、线性投影及 GPU kernel 跟踪 |
| [T2](profiles/targets/T2.json) | Qwen3-0.6B BF16 · vLLM | 已支持结构的配置接入、首层 RMSNorm 和线性投影 |
| [T3](profiles/targets/T3.json) | SmolLM2-1.7B-Instruct Q4_K_M GGUF · llama.cpp CUDA | 原生 prompt/生成阶段、归属到服务进程的 CUDA 活动 |

公开配置包含已验证的模型来源和引擎版本。换模型前可用 `probe check` 查看所选工作量需要哪些能力。新增引擎或模型结构时，可从现有适配器的接口与校验规则入手。[模型来源与版本](profiles/sources.lock.json) · [适配器约定](docs/ADAPTERS.md)

## 快速开始

### 在普通电脑上检查配置

离线入口使用 Python 标准库，以下命令在仓库根目录执行。

```sh
python3 -m tilescope list --json
python3 -m tilescope probe check --offline \
  --target profiles/targets/T1.json \
  --suite workloads/suites/protocol-pilot.json \
  --out /tmp/tilescope-check.json
```

`list` 列出可选算子和形状。`check --offline` 解析模型、引擎与工作量，输出支持项及 `missing_required`。它给出的是静态配置检查，实机身份和运行能力由设备端核验。

### 在 Thor 上试跑一个部署

准备模型权重与对应的 CUDA 推理环境。把 `profiles/` 复制到本地配置目录，保留相对引用。更新 target 内的模型名、checkpoint 路径、可执行文件路径、loopback 端口及 GPU UUID。vLLM 核对版本，llama.cpp 核对实际服务二进制和加载库的哈希。

```sh
mkdir -p local
cp -R profiles local/
```

以下以 T1 为例。配置完成后，在一个终端启动独立实例，等待 `local/t1-server/ready.json` 出现。

```sh
python3 -m tilescope.probe.owned \
  --target local/profiles/targets/T1.json --out local/t1-server
```

在另一个终端从仓库根目录运行协议试跑。它会发起流式与非流式诊断请求。看到 `probe review` 返回 `"status": "complete"` 后，即可选择需要的正式工作量。

```sh
TARGET=local/profiles/targets/T1.json
SUITE=workloads/suites/protocol-pilot.json
python3 -m tilescope probe check --target "$TARGET" --suite "$SUITE" --out local/t1-check.json
python3 -m tilescope probe run --target "$TARGET" --suite "$SUITE" --out runs/t1-pilot
python3 -m tilescope probe review runs/t1-pilot
```

用新输出目录运行 [定长工作量](workloads/suites/new-target.json)、[同内容工作量](workloads/suites/same-content.json) 或 [有限并发工作量](workloads/suites/concurrency.json)。独立实例用 Ctrl+C 结束。

## 从原始数据生成报告

将批次目录列入研究清单。路径相对于清单所在目录解析。以下示例把上一步的单次试跑生成 Markdown 和 JSON 报告。

```sh
printf '%s\n' '{"runs":["t1-pilot"]}' > runs/pilot-study.json
python3 -m tilescope probe report --study runs/pilot-study.json --out runs/pilot-report
```

- **`plan.json`** 保存解析后的部署、模型身份与工作量。复跑前先看是否使用了相同的配置和条件。
- **`attempts/`** 保存请求、原始响应、计时和退出状态。复核检查模型身份、token 计数和协议终止。
- **`sessions/`** 保存会话级资源采样，适合观察温度、频率和内存变化。短请求或中断会话可能缺少结束后的测点。
- **`REPORT.md` 与 `report.json`** 提供样本和汇总。诊断、预热和正式样本各有标签，失败批次保留原因。

同一模型回归可按固定条件核对变化。同内容测试回答各部署处理同一输入的表现，同 token 预算测试固定输入 token 数和生成上限。同组结果以实际生成 token 数为准，仅输出 EOS 的请求可以有总耗时和输出计数，首个可见文本时间为空。引擎原生阶段按各自定义解读，跨精度、引擎或缓存策略的结果描述整个部署差异。[报告与统计口径](docs/reporting.md)

## 仓库导览

| 路径 | 内容 |
| --- | --- |
| [`catalog/`](catalog/) | 算子任务、形状和校验合同 |
| [`profiles/`](profiles/) 与 [`workloads/`](workloads/) | 模型、引擎、部署示例和可复用工作量 |
| [`tilescope/probe/`](tilescope/probe/) | 请求采集、协议复核、诊断与比较 |
| [`parity/`](parity/) | 硬件、CUTLASS 校准及服务诊断 |
| [`tests/`](tests/) | 解析、数值边界和异常处理测试。需要私有实机记录的用例会在缺少数据时跳过。 |

公开仓库面向源码复用和新实验。历史验收从私有完整 Git bundle 和配套证据包恢复。公开模型权重按来源清单取得，合作方 CUDA 基线及第三方许可保留在对应源码中。[基线出处](parity/calibration/source.json) · [TileGym 许可](catalog/TileGym-LICENSE.txt) · [thor-probe 许可](parity/hardware/LICENSE.thor-probe)
