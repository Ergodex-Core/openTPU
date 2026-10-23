"""Split a Gemma 3 decode profile's blocked-MXU cycles by cause, for the layers and the LM
head separately (the LM head: the MM instructions from the first one whose comment names it,
else the last run of MMs with the vocabulary's rows). Usage: python3 tools/mxu_block_split.py PROFILE [more...] (docs/v4/m0-gating-split.md)"""
import sys

from opentpu import lens

for path in sys.argv[1:]:
    doc = lens.load(path)
    for d in doc["profiles"]:
        cyc = d["cycles"]
        sl = d["slices"][0]
        B = sl["buckets"]
        ins = [r for r in d["instrs"] if r[0] == 0]
        prog = d["programs"][0]
        # the LM head's first instruction: an MM whose program comment names the head / logits,
        # else the first MM after the last VOP (the final norm) whose rows are the vocabulary's
        head = None
        for r in ins:
            if r[4] == "MM":
                cm = (prog[r[2]][3] or "").lower()
                if "head" in cm or "logit" in cm or "vocab" in cm:
                    head = r
                    break
        if head is None:
            mms = [r for r in ins if r[4] == "MM"]
            # the vocabulary's MMs: the largest N (detail "Mx1152 . Nx1152^T")
            import re
            def n_of(r):
                m = re.search(r"\. (\d+)x", r[5])
                return int(m.group(1)) if m else 0
            big = max(n_of(r) for r in mms)
            head = next(r for r in mms if n_of(r) == big)
        t_head = head[9]            # its start cycle
        cend = B["c"]
        k_head = next((k for k, c in enumerate(cend) if c >= t_head), len(cend))
        print(f"{d['name']}: {cyc:,} cycles; LM head from cycle {t_head:,} "
              f"(comment {prog[head[2]][3]!r}), bucket {k_head} of {len(cend)}")
        causes = lens.port_b_causes(d)
        print("  port B causes (whole run):")
        for key, label in lens.CAUSES:
            v = causes.get(key, 0)
            if v > 0:
                print(f"    {100 * v / cyc:5.1f}%  {label}")
        for name, k0, k1 in (("whole run", 0, None), ("layers", 0, k_head),
                             ("LM head", k_head, None)):
            bs = lens.blocked_split(B, k0, k1)
            n = sum(B["n"][k0:k1])
            bm = sum(B["bm"][k0:k1]) + sum(B["bd"][k0:k1])
            if bs is None:
                print(f"  {name}: {n:,} cycles, port B busy {100 * bm / n:.1f}%, "
                      f"mb {sum(B['mb'][k0:k1]):,} ({100 * sum(B['mb'][k0:k1]) / n:.1f}%); no split")
                continue
            mb = bs["mb"] or 1
            print(f"  {name}: {n:,} cycles, port B busy {100 * bm / n:.1f}%, "
                  f"mb {bs['mb']:,} ({100 * bs['mb'] / n:.1f}% of the cycles): "
                  f"credits {bs['mbc']:,} ({100 * bs['mbc'] / mb:.1f}%), "
                  f"row {bs['mbr']:,} ({100 * bs['mbr'] / mb:.1f}%), "
                  f"scales {bs['mbs']:,} ({100 * bs['mbs'] / mb:.1f}%), "
                  f"other {bs['mbo']:,} ({100 * bs['mbo'] / mb:.1f}%)"
                  f"  [ms {sum(B['ms'][k0:k1]):,}, fm {sum(B['fm'][k0:k1]):,}, "
                  f"bs {sum(B['bs'][k0:k1]):,}, as {sum(B['as'][k0:k1]):,}, "
                  f"mean FIFO level {sum(B['ff'][k0:k1]) / n:.1f} chunks, while rows late {lens.fifo_level_at(B, 'mbr', k0, k1):.1f}, "
                  f"mx {sum(B['mx'][k0:k1]):,}, am {sum(B['am'][k0:k1]):,}]")
        # the distribution of the chunk FIFO level over the buckets (mean level per bucket)
        import collections
        hist = collections.Counter()
        for k in range(len(B["c"])):
            if B["n"][k]:
                lvl = B["ff"][k] / B["n"][k]
                hist[min(10, int(lvl // 16))] += B["n"][k]
        tot = sum(hist.values()) or 1
        print("  cycles by the bucket's mean FIFO level: " + ", ".join(
            f"{'>=160' if i == 10 else f'{16 * i}-{16 * i + 15}'}: {100 * hist[i] / tot:.1f}%"
            for i in sorted(hist)))
