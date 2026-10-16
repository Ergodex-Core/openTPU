"""RhinoVLA's shape-only cost on openTPU (docs/vla_frequency_plan.md phase 0): the MACs, MXU
cycles and port B chunks of one action chunk from the model's dimensions alone, no port, no
image, in tools/pi05_cost.py's units so the two models compare on the same scale:

  MXU     an MM's advances: N x KB x ceil(M / MCOLS) cycles, M the row group (at most ACT_ROWS)
  port B  D-byte weight chunks: N x KB a row group (the matrix re-streams every row group)

The VPU, QACT and QST are not estimated here (they need the kernels; pi0.5's tally has them at
0.26 / 0.10 / 0.34 s at LANES 64, and RhinoVLA's rows are ~45% of pi0.5's, its vision rows
the same 3 x 256 patches before the merger).

Shapes (arXiv 2606.07383 v4, Table 8 and sections 3.2 / 3.4; the paper's V1, not the V3
branch): ViT 24 layers, 1024 wide, MLP 4096, 16 heads, a 16 x 16 patch grid (256 patches) a
view merged 2 x 2 to 64 tokens, 3 views; Qwen3-VL-2B text model 28 layers, 2048 wide, MLP
6144 (gated: gate, up, down), 16 query heads and 8 KV heads of 128; the action expert 18
layers, 1024 wide, MLP 3072, 16 / 8 heads of 128, 1 state token + 30 noisy action tokens,
attending to the prefix KV of the last 18 backbone layers; the flow steps are not stated
(pi0.5's 10 here, --steps). The prompt's tokens: --prompt (the paper does not state them).

    python tools/rhinovla_shape.py [--mcols 32] [--act-rows 255] [--mhz 200] [--steps 10]
"""
from __future__ import annotations

import argparse

D = 128


def mm(n: int, k: int, m: int, mcols: int, act_rows: int) -> tuple[int, int, int]:
    """One [m, k] @ [k, n] (weights [n, k] int8): (MACs, MXU cycles, port B chunks)."""
    kb = -(-k // D)
    groups = [min(act_rows, m - i) for i in range(0, m, act_rows)]
    cyc = sum(n * kb * -(-g // mcols) for g in groups)
    return m * k * n, cyc, n * kb * len(groups)


def attn(rows: int, kv_rows: int, hq: int, hd: int, mcols: int, act_rows: int):
    """QK^T and PV per head: the scores [rows, kv_rows] and the context [rows, hd], as MMs
    whose "weights" are K / V rows (streamed) and whose stationary rows are Q / P."""
    s = mm(kv_rows, hd, rows, mcols, act_rows)          # scores: N = kv_rows, K = hd
    p = mm(hd, kv_rows, rows, mcols, act_rows)          # context: N = hd, K = kv_rows
    return tuple(hq * (a + b) for a, b in zip(s, p))


def add(acc, x):
    return tuple(a + b for a, b in zip(acc, x))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcols", type=int, default=32)
    ap.add_argument("--act-rows", type=int, default=255)
    ap.add_argument("--mhz", type=float, default=200.0)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--prompt", type=int, default=40, help="prompt tokens in the prefix")
    ap.add_argument("--views", type=int, default=3)
    a = ap.parse_args()
    mc, ar = a.mcols, a.act_rows

    # the vision tower: 256 patches a view through 24 layers (the merger after them)
    vt = (0, 0, 0)
    rows = 256
    for _ in range(24):
        vt = add(vt, mm(3 * 1024, 1024, rows, mc, ar))              # QKV
        vt = add(vt, attn(rows, rows, 16, 64, mc, ar))
        vt = add(vt, mm(1024, 1024, rows, mc, ar))                  # O
        vt = add(vt, mm(4096, 1024, rows, mc, ar))                  # MLP up
        vt = add(vt, mm(1024, 4096, rows, mc, ar))                  # MLP down
    vt = add(vt, mm(2048, 4 * 1024, 64, mc, ar))                    # the 2 x 2 merger + projector
    vision = tuple(a.views * x for x in vt)

    # the prefix: views x 64 tokens + the prompt + 1 state token through the 28-layer backbone;
    # every layer's K / V kept (the expert reads the last 18 layers'); bidirectional over the
    # prefix as pi0.5's
    np_ = a.views * 64 + a.prompt + 1
    pf = (0, 0, 0)
    for li in range(28):
        last = li == 27
        pf = add(pf, mm(2048 + 2 * 1024, 2048, np_, mc, ar))        # Q (16 x 128), K / V (8 x 128)
        if not last:
            pf = add(pf, attn(np_, np_, 16, 128, mc, ar))
            pf = add(pf, mm(2048, 2048, np_, mc, ar))               # O
            pf = add(pf, mm(2 * 6144, 2048, np_, mc, ar))           # gate, up
            pf = add(pf, mm(2048, 6144, np_, mc, ar))               # down

    # one flow step of the expert: 31 rows, 18 layers, attending to the prefix KV of the
    # backbone's last 18 layers (projected by the backbone: no cost here) and to its own
    ex = (0, 0, 0)
    m = 31
    for _ in range(18):
        ex = add(ex, mm(2048 + 2 * 1024, 1024, m, mc, ar))          # Q, K, V (heads of 128)
        ex = add(ex, attn(m, np_ + m, 16, 128, mc, ar))
        ex = add(ex, mm(1024, 2048, m, mc, ar))                     # O
        ex = add(ex, mm(2 * 3072, 1024, m, mc, ar))                 # gate, up
        ex = add(ex, mm(1024, 3072, m, mc, ar))                     # down
    ex = add(ex, mm(1024, 72, m, mc, ar))                           # action in
    ex = add(ex, mm(72, 1024, m, mc, ar))                           # action out

    hz = a.mhz * 1e6
    print(f"RhinoVLA (paper shapes): MCOLS {mc}, ACT_ROWS {ar}, {a.mhz:g} MHz, {a.views} views, "
          f"prefix {np_} rows, expert {m} rows x {a.steps} steps")
    tot = [0, 0, 0]
    for name, (macs, cyc, chunks), k in (("vision", vision, 1), ("prefix", pf, 1),
                                        ("expert_step", ex, a.steps)):
        print(f"{name:12s}{' x%d' % k if k > 1 else '':4s} MACs={macs / 1e9:8.1f}G  "
              f"MXU={cyc / hz:.3f}s  portB={chunks / hz:.3f}s (a chunk a cycle; {chunks * D / 1e9:.2f} GB)")
        for i, v in enumerate((macs, cyc, chunks)):
            tot[i] += k * v
    print(f"chunk        MACs={tot[0] / 1e12:.3f}T  MXU={tot[1] / hz:.3f}s  portB={tot[2] / hz:.3f}s "
          f"({tot[2] * D / 1e9:.1f} GB)  MXU-bound {hz / tot[1]:.2f} Hz")


if __name__ == "__main__":
    main()
