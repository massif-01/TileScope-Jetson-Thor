"""Validate the pinned upstream BF16 benchmark's result and process status."""

import argparse
import json
import math
import re
from pathlib import Path


def parse_smoke(stdout, exit_code, shape):
    if exit_code:
        raise ValueError(f"benchmark exit code {exit_code}")
    decoder = json.JSONDecoder()
    results = []
    # The pinned program emits one root object beginning on its own line.
    for match in re.finditer(r"^\{", stdout, re.MULTILINE):
        try:
            result, end = decoder.raw_decode(stdout, match.start())
        except json.JSONDecodeError:
            continue
        if not stdout[end:].partition("\n")[0].strip():
            results.append(result)

    if len(results) != 1 or results[0].get("benchmark") != "bf16_gemm_cutlass":
        raise ValueError("expected exactly one BF16 benchmark JSON object")
    result = results[0]
    if tuple(result.get(dim) for dim in "mnk") != shape:
        raise ValueError("unexpected GEMM shape")
    if result.get("correctness") != "passed":
        raise ValueError("benchmark did not report correctness=passed")
    timings = result.get("runtime_ms")
    mean = timings.get("mean") if isinstance(timings, dict) else None
    if type(mean) not in (int, float) or not math.isfinite(mean) or mean <= 0:
        raise ValueError("missing or invalid runtime mean")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stdout", type=Path)
    parser.add_argument("exit_code", type=Path)
    parser.add_argument("shape", type=int, nargs=3)
    args = parser.parse_args()
    try:
        result = parse_smoke(args.stdout.read_text(), int(args.exit_code.read_text()), tuple(args.shape))
    except ValueError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
