"""Build the four frozen CUTLASS instantiations in a private directory, sequentially."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
FORMATS = ("bf16-f32", "bf16-bf16", "nvfp4-bf16", "nvfp4-fp4")
PIN = "b46b16d003484063bca4ed365e44095c4c6ed633"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out, nvcc, cutlass, formats):
    out.mkdir(parents=True, exist_ok=True)
    cutlass = cutlass.resolve()
    pin = subprocess.check_output(["git", "-C", cutlass, "rev-parse", "HEAD"], text=True).strip()
    if pin != PIN:
        raise ValueError(f"experiment CUTLASS {PIN} required, found {pin}")
    src = HERE / "partner"
    bf = (src/"bench_bf16_cutlass.cu").read_text()
    replacement = "using         ElementC    = float;"
    assert bf.count(replacement) == 1
    (out/"bf16_epilogue.cu").write_text(bf.replace(replacement,"using         ElementC    = cutlass::bfloat16_t;"))
    mixed = out/"mixed.cu"
    with (HERE/"mixed.patch").open() as patch:
        subprocess.run(["patch","-s","-o",mixed,src/"bench_nvfp4_fp4_bf16.cu"], stdin=patch, check=True)
    results = {}
    for fmt in formats:
        exe = out/fmt
        args = [str(nvcc), "-v", "-std=c++17", "-O3", "-arch=sm_110a",
                "--expt-relaxed-constexpr", "--expt-extended-lambda", f'-DCAL_FORMAT="{fmt}"',
                f'-DCUTLASS_COMMIT="{pin}"', f"-I{src}", f"-I{HERE}", f"-I{out}",
                *(f"-I{cutlass / folder}" for folder in ("include", "tools/util/include", "examples/common"))]
        if fmt == "nvfp4-bf16":
            source = mixed
        else:
            source = HERE / "fixed.cu"
            included = {"bf16-f32":"bench_bf16_cutlass.cu","bf16-bf16":"bf16_epilogue.cu",
                        "nvfp4-fp4":"bench_nvfp4_fp4.cu"}[fmt]
            args.extend([f'-DCAL_PARTNER="{included}"',f'-DCAL_PACKED={int(fmt=="nvfp4-fp4")}'])
        args.extend([str(source), "-o", str(exe)])
        start = time.time_ns()
        with (out/f"{fmt}.log").open("w") as log:
            rc = subprocess.run(args,stdout=log,stderr=subprocess.STDOUT).returncode
        results[fmt] = dict(argv=args, exit=rc, start_ns=start,end_ns=time.time_ns(),
                            executable_sha256=sha(exe) if rc == 0 else None)
        (out/"build.json").write_text(json.dumps(dict(cutlass=pin, nvcc=subprocess.check_output([nvcc,"--version"],text=True),
            source_sha256={p.name:sha(p) for p in [*src.iterdir(), HERE/"measure.hpp",HERE/"fixed.cu",HERE/"mixed.patch",Path(__file__)]},
            builds=results),indent=2)+"\n")
        print(fmt, rc, flush=True)
    return int(any(v["exit"] for v in results.values()))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True,type=Path)
    p.add_argument("--cutlass",required=True,type=Path)
    p.add_argument("--nvcc",default="/usr/local/cuda-13.3/bin/nvcc")
    p.add_argument("--formats",nargs="+",choices=FORMATS,default=FORMATS)
    a = p.parse_args()
    raise SystemExit(build(a.out.resolve(),a.nvcc,a.cutlass,a.formats))
