"""Offline decoder audit and pinned tokenizer-data preparation. Never serving."""
import hashlib
import json
from pathlib import Path

import numpy as np

from opentpu.asr.phonon2_tdt import DecoderShape, weight_shapes, TEXT_STRIDE


def read_weights(checkpoint):
    return checkpoint._read("", weight_shapes())


def text_table(pieces, skipped, vocab):
    """Metaspace always, no CTC grouping; UTF-8 tables independent of inputs."""
    table = np.zeros((vocab, 2, TEXT_STRIDE), np.uint8)
    lengths, visible = np.zeros((vocab, 2), np.float32), np.ones(vocab, np.float32)
    for i in range(vocab):
        if i in skipped:
            visible[i] = 0
            continue
        for following in range(2):
            b = pieces[i].replace("▁", " " if following else "").encode("utf-8")
            if len(b) > TEXT_STRIDE:
                raise ValueError("UTF-8 piece exceeds bounded slot")
            table[i, following, :len(b)] = np.frombuffer(b, np.uint8)
            lengths[i, following] = len(b)
    return table, lengths, visible


def pinned_tokenizer(path, provenance_path):
    raw = Path(path).read_bytes()
    provenance = json.loads(Path(provenance_path).read_text())
    expected = provenance["files"]["base-tokenizer.json"]["sha256"]
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("tokenizer differs from pinned source")
    data = json.loads(raw)
    if data["decoder"] != {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": True}:
        raise ValueError("unsupported tokenizer decoder")
    vocab = data["model"]["vocab"]
    if len(vocab) != 8192 or set(vocab.values()) != set(range(8192)) or any(k.startswith("<0x") for k in vocab):
        raise ValueError("unsupported vocabulary/byte fallback")
    pieces = [None] * 8193
    for piece, index in vocab.items():
        pieces[index] = piece
    pieces[8192] = "<blank>"
    # Includes configured EOS, UNK and PAD and the explicit added blank. Other
    # added control-looking pieces are not blanket-suppressed as in Fermion CUDA.
    skipped = {0, 2, 3, 8192}
    return text_table(pieces, skipped, 8193), pieces, skipped, expected


def load_text_asset(root):
    root = Path(root)
    policy = json.loads((root / "docs/phonon2/tdt_semantic_policy.json").read_text())
    raw = (root / "docs/phonon2/token_text_vocabulary.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != policy["vocabulary_asset_sha256"]:
        raise ValueError("UTF-8 vocabulary asset differs from audited source")
    pieces = json.loads(raw)
    if len(pieces) != 8193 or not all(isinstance(p, str) for p in pieces):
        raise ValueError("invalid vocabulary")
    return text_table(pieces, set(policy["skip_special_ids"]), 8193), pieces, policy


def reference(weights, encoder, valid, shape=DecoderShape(), *, max_steps=None, table=None):
    """Own native Torch eager equations and Transformers cache/control semantics."""
    import torch
    import torch.nn.functional as f
    h, vocab = shape.hidden, shape.vocab
    max_steps = 10 * len(encoder) - 1 if max_steps is None else max_steps
    w = {n: torch.from_numpy(a.dense() if hasattr(a, "dense") else np.array(a, np.float32)) for n, a in weights.items()}
    enc = f.linear(torch.from_numpy(np.array(encoder, np.float32)), w["encoder_projector.weight"], w["encoder_projector.bias"])
    hidden, cell = torch.zeros(2, 1, h), torch.zeros(2, 1, h)
    cached, pending, initialized, frame, seen = torch.zeros(1, h), shape.blank, False, 0, 0
    records, text = [], []
    for _ in range(max_steps):
        if not initialized or pending != shape.blank:
            x = w["decoder.embedding.weight"][pending:pending + 1]
            for layer in range(2):
                g = (f.linear(x, w[f"decoder.lstm.weight_ih_l{layer}"], w[f"decoder.lstm.bias_ih_l{layer}"]) +
                     f.linear(hidden[layer], w[f"decoder.lstm.weight_hh_l{layer}"], w[f"decoder.lstm.bias_hh_l{layer}"]))
                i, forget, candidate, output = g.chunk(4, -1)
                cell[layer] = forget.sigmoid() * cell[layer] + i.sigmoid() * candidate.tanh()
                x = output.sigmoid() * cell[layer].tanh()
                hidden[layer] = x
            cached = f.linear(x, w["decoder.decoder_projector.weight"], w["decoder.decoder_projector.bias"])
            initialized = True
        logits = f.linear(f.relu(enc[frame:frame + 1] + cached), w["joint.head.weight"], w["joint.head.bias"])
        token, duration = int(logits[0, :vocab].argmax()), int(logits[0, vocab:].argmax())
        if token == shape.blank and duration == 0:
            duration = 1
        piece = b""
        if table is not None and token != shape.blank and table[2][token]:
            following = min(seen, 1)
            piece = bytes(table[0][token, following, :int(table[1][token, following])])
            seen += 1
        text.append(piece)
        records.append({"token": token, "duration": duration, "frame": frame, "emit": token != shape.blank,
                        "hidden": hidden.numpy().copy(), "cell": cell.numpy().copy(), "cache": cached.numpy().copy(),
                        "logits": logits.numpy().copy()})
        frame += duration
        pending = token
        if token == shape.eos or frame >= valid:
            break
    status = 2 if records[-1]["token"] == shape.eos else 1 if frame >= valid else 3
    return {"records": records, "status": status, "frame": frame, "text": b"".join(text), "encoder_projected": enc.numpy()}


def read_result(device, plan):
    """Audit/I/O only: collect device-selected UTF-8 segments, never map tokens."""
    def array(name):
        spec = plan.checkpoints[name]
        n = int(np.prod(spec["shape"]))
        a = spec["offset"]
        return device.memory[a:a + n * 4].view("<f4").reshape(spec["shape"]).copy()
    state = array("state")
    steps = int(state[3])
    trace = array("trace")[:steps]
    raw = array("text").view(np.uint8).reshape(plan.max_steps, TEXT_STRIDE)
    lengths = array("text_lengths").astype(int)
    text = b"".join(bytes(row[:length]) for row, length in zip(raw[:steps], lengths[:steps]))
    result = {"state": state, "trace": trace, "text": text, "encoder_projected": array("encoder_projected"),
              "hidden": array("hidden"), "cell": array("cell"), "cache": array("cache")}
    if plan.audit:
        result.update(audit_hidden=array("audit_hidden")[:steps], audit_cache=array("audit_cache")[:steps])
    return result
