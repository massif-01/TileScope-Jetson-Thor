# TileScope · Jetson Thor 性能探针

TileScope 在 NVIDIA Jetson Thor 上比较 CUDA Tile 与传统 CUDA 算子，并采集 vLLM 和 llama.cpp 文本请求的延迟、吞吐及波动。请求记录、引擎原生阶段、GPU kernel 活动和局部数值检查各有自己的测量边界。仓库提供探针源码、模型配置、工作量和合成测试。

## 离线查看配置

在仓库根目录执行以下命令。离线检查使用 Python 标准库。

```sh
python3 -m tilescope list --json
python3 -m tilescope probe check --offline --target profiles/targets/T1.json --suite workloads/suites/new-target.json --out /tmp/tilescope-check.json
```

## 在自己的设备运行

`profiles/targets/` 是部署示例。将端口、服务模型名、checkpoint 路径、运行时路径和 GPU UUID 填为本机实际值。llama.cpp 同时核对对应构建的二进制与动态库。运行时请将本机 target 和输出保存在本地。新批次完成后可用 `probe review` 复核原始计时、计数和终止记录。

本项目覆盖原 Qwen 混合量化模型、SmolLM2 BF16、Qwen3-0.6B BF16 和 SmolLM2 GGUF 四个部署。首次验证使用单台设备，结果只适用于实测配置。原始设备日志、带身份信息的样本和开发者审查材料由实验持有人私下保管；依赖这些材料的归档测试需另外提供对应数据。

公开模型与版本见 [来源清单](profiles/sources.lock.json)。合作方 CUDA 基线及第三方许可保留在对应源码和 [基线出处](parity/calibration/source.json) 中。
