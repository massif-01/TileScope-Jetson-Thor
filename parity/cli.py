"""Dispatch parity probes without importing CUDA dependencies in offline commands."""
import argparse
import os
import sys
from tilescope.common import device_slot

COMMANDS={
    "hardware":("parity.hardware",False,"设备与遥测的采集及离线核对"),
    "calibration-build":("parity.calibration.build",True,"构建冻结 CUTLASS 基准"),
    "calibration":("parity.calibration.run",True,"串行执行固定实现的多轮合成基准"),
    "calibration-review":("parity.calibration.review",False,"从原始 GPU 样本导出分组结果"),
    "service":("parity.service",False,"线上请求矩阵、持续观测与恢复"),
    "service-review":("parity.service_review",False,"从原始 HTTP 事件复算及比较"),
    "trace":("parity.trace",False,"生产等价的私有服务跟踪及退出诊断"),
    "trace-review":("parity.trace_review",False,"从私有运行记录复核阶段、提交、设备时间与恢复"),
    "checkpoint":("parity.checkpoint",False,"核对模型索引、所有分片头和载荷字节范围"),
    "board":("parity.board",False,"只读核对 SMBIOS type 17 及 BPMP 媒体/EMC 时钟"),
    "handles":("parity.handles",False,"只读列出可见的 GPU 句柄持有者"),
    "verify":("parity.verify_delivery",False,"核对 ZIP 文件成员、逐文件校验及 Git bundle 身份"),
    "report":("parity.report",False,"生成综合报告及机器可读结果"),
}


def dispatch(argv):
    p=argparse.ArgumentParser(prog="python3 -m tilescope parity",description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    for name,(_,_,help_text) in COMMANDS.items():sub.add_parser(name,help=help_text,add_help=False)
    a,remaining=p.parse_known_args(argv)
    module,locked,_=COMMANDS[a.command]
    cmd=[sys.executable,"-m",module,*remaining]
    if locked and not any(s in ("-h","--help") for s in remaining):
        # Replacing this process keeps its PID and the advisory lock through the kernel job.
        with device_slot() as fd:
            os.set_inheritable(fd,True)
            os.execv(sys.executable,cmd)
    os.execv(sys.executable,cmd)
