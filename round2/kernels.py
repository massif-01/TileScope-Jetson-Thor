"""Fixed cuTile references for two projection contracts, using row-major outputs."""
import cuda.tile as ct


@ct.kernel
def fp8_projection(a, wt, sa, sw, dst):
    row, col = ct.bid(0), ct.bid(1)
    acc = ct.full((64, 64), 0, dtype=ct.float32)
    for step in range(a.shape[1] // 64):
        x = ct.load(a, (row, step), (64, 64))
        w = ct.load(wt, (col, step), (64, 64))
        acc = ct.mma(x, ct.permute(w, (1, 0)), acc)
    row_scale = ct.load(sa, (row, 0), (64, 1))
    col_scale = ct.load(sw, (col, 0), (64, 1))
    ct.store(dst, (row, col), (acc * row_scale * ct.permute(col_scale, (1, 0))).astype(dst.dtype))


def scale_fragment(flat, tile, step, k_tiles):
    # Physical order of the vLLM 128 x 4 blockscale swizzle:
    # [row // 128, K-group // 4, row % 32, (row % 128) // 32, K-group % 4].
    rows = tile * 64 + ct.arange(64, dtype=ct.int32)
    base = ((rows // 128 * k_tiles + step) * 32 + rows % 32) * 16 + rows % 128 // 32 * 4
    offsets = ct.reshape(base, (64, 1)) + ct.reshape(ct.arange(4, dtype=ct.int32), (1, 4))
    return ct.gather(flat, offsets)


@ct.kernel
def nvfp4_projection(a, wt, sa, sw, alpha, dst):
    row, col = ct.bid(0), ct.bid(1)
    acc = ct.full((64, 64), 0, dtype=ct.float32)
    steps = a.shape[1] // 32
    for step in range(steps):
        px = ct.reshape(ct.load(a, (row, step), (64, 32)), (2048,))
        pw = ct.reshape(ct.load(wt, (col, step), (64, 32)), (2048,))
        x = ct.reshape(ct.unpack_from_bytes(px, ct.float4_e2m1fn), (64, 64))
        w = ct.reshape(ct.unpack_from_bytes(pw, ct.float4_e2m1fn), (64, 64))
        sx = scale_fragment(sa, row, step, steps)
        sy = scale_fragment(sw, col, step, steps)
        acc = ct.mma_scaled(x, sx, ct.permute(w, (1, 0)), ct.permute(sy, (1, 0)), acc)
    factor = ct.reshape(ct.load(alpha, (0,), (1,)), ())
    ct.store(dst, (row, col), (acc * factor).astype(dst.dtype))
