"""Fixed cuTile implementations. The bounds are part of the measured implementation.

RMSNorm and SwiGLU follow NVIDIA TileGym src/tilegym/ops/cutile/rms_norm.py
and silu_and_mul.py at 8ca8b47e05c629406de5eed83daa5f651061d9df, adapted for
the two inference-only output contracts. See catalog/TileGym-LICENSE.txt.
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: MIT
"""
import cuda.tile as ct


@ct.kernel
def ba_projection(x, wt, dst):
    row, col = ct.bid(0), ct.bid(1)
    acc = ct.full((64, 64), 0, dtype=ct.float32)
    for block in range(x.shape[1] // 32):
        a = ct.load(x, (row, block), (64, 32), padding_mode=ct.PaddingMode.ZERO)
        b = ct.load(wt, (col, block), (64, 32), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a, ct.permute(b, (1, 0)), acc)
    ct.store(dst, (row, col), acc.astype(dst.dtype))


@ct.kernel
def gemma_rmsnorm(x, w, dst):
    row = ct.bid(0)
    cols = ct.arange(8192, dtype=ct.int32)
    value = ct.gather(x, (row, cols), mask=cols < 5120, padding_value=0).astype(ct.float32)
    scale = ct.gather(w, cols, mask=cols < 5120, padding_value=0).astype(ct.float32)
    rstd = ct.rsqrt(ct.sum(value * value, axis=0) / 5120.0 + 0.000001)
    out = (value * rstd * (1.0 + scale)).astype(dst.dtype)
    ct.scatter(dst, (row, cols), out, mask=cols < 5120)


@ct.kernel
def swiglu(x, dst):
    row, col = ct.bid(0), ct.bid(1)
    gate = ct.load(x, (row, col), (4, 256), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    up = ct.load(x, (row, col + 68), (4, 256), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    activated = (gate / (1.0 + ct.exp(-gate))).astype(x.dtype)
    acc = activated.astype(ct.float32) * up
    ct.store(dst, (row, col), acc.astype(dst.dtype))
