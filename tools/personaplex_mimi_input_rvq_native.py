"""Pinned native input-RVQ reference for offline Mimi qualification.

The existing quantizer helper strictly loads all 32 checkpoint codebooks and
activates eight. This adds the two encoder-side input projections and captures
the original Euclidean greedy decisions, without board or serving execution.
"""

import hashlib
import inspect
import json
from pathlib import Path
import struct

import numpy as np

from tools.personaplex_mimi_check import codec_header, native_quantizer


_BASE_CONTRACT_SHA256 = "5342c087a41a95fd899bf62ac92c2c2ad31e8fa09d60788eda74214df5dd4aa1"
_HEADER_SHA256 = "ec30eb84e7a32c11b2913db3d89595da90a422320dea3b66a3b39e16f821cc93"
_PROJECTION_NAMES = (
    "quantizer.rvq_first.input_proj.weight",
    "quantizer.rvq_rest.input_proj.weight",
)
_PROJECTION_SHAPE = (256, 512, 1)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _base_contract(contract):
    # Resolve from the installed existing helper, so this draft remains
    # location-independent when promoted beside that helper.
    repo = Path(inspect.getfile(native_quantizer)).resolve().parents[1]
    path = repo / "tools/data/personaplex/mimi_rvq_contract.json"
    if (contract.get("base_contract_sha256") != _BASE_CONTRACT_SHA256
            or _sha256(path) != _BASE_CONTRACT_SHA256):
        raise ValueError("Pinned output RVQ source/header contract changed")
    base = json.loads(path.read_text())
    if (contract.get("installed_source_sha256") != base["installed_source_sha256"]
            or contract.get("codec_header") != base["codec_header"]
            or contract.get("native_constructor") != base["native_constructor"]
            or contract.get("active_codebooks") != base["active_codebooks"]
            or contract.get("normalization_epsilon") != base["normalization_epsilon"]):
        raise ValueError("Input RVQ contract differs from the pinned quantizer")
    return base


def _checked_projections(codec_path, contract, base):
    path = Path(codec_path).resolve()
    before = codec_header(path, base)
    if (base["codec_header"]["header_sha256"] != _HEADER_SHA256
            or contract.get("input_projection_keys") != list(_PROJECTION_NAMES)):
        raise ValueError("Input RVQ checkpoint header/projection contract differs")
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError("Cached codec header is incomplete")
        size = struct.unpack("<Q", raw_size)[0]
        if size != base["codec_header"]["header_bytes"]:
            raise ValueError("Cached codec header length differs")
        raw = handle.read(size)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != _HEADER_SHA256:
        raise ValueError("Cached codec header changed")
    header = json.loads(raw)
    actual = {name: header.get(name) for name in _PROJECTION_NAMES}
    if actual != contract.get("input_projection_tensors"):
        raise ValueError("Cached encoder input projection metadata differs")
    for name, meta in actual.items():
        if not isinstance(meta, dict):
            raise ValueError("Missing encoder input projection: " + name)
        offsets = meta.get("data_offsets")
        if (meta.get("dtype") != "F32" or tuple(meta.get("shape", ())) != _PROJECTION_SHAPE
                or not isinstance(offsets, list) or len(offsets) != 2
                or type(offsets[0]) is not int or type(offsets[1]) is not int
                or offsets[1] - offsets[0] != 4 * 256 * 512):
            raise ValueError("Encoder input projection shape/offset differs: " + name)
    if codec_header(path, base) != before:
        raise ValueError("Cached codec changed during input projection header check")
    return before


def _array(tensor, shape, name):
    value = tensor.detach().cpu().numpy().copy()
    if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
        raise ValueError("Native input RVQ tensor differs: " + name)
    return np.ascontiguousarray(value)


def native_input_quantizer(codec_path, contract):
    """Return original native quantizer, normalized tables and input matrices.

    Input matrices have shape [semantic/acoustic, 256, 512]. The old helper
    proves normalized embedding_sum/usage tables bit-exact to native centroids.
    """
    base = _base_contract(contract)
    before = _checked_projections(codec_path, contract, base)
    native, tables, _output_projections, native_root = native_quantizer(str(codec_path), base)
    if (contract.get("semantic_codebooks") != 1 or contract.get("active_codebooks") != 8
            or native.num_codebooks != 8):
        raise ValueError("Native input RVQ active codebook split differs")
    projections = np.stack([
        _array(branch.input_proj.weight, _PROJECTION_SHAPE, name)[:, :, 0]
        for branch, name in zip((native.rvq_first, native.rvq_rest), _PROJECTION_NAMES)
    ])
    if projections.shape != (2, 256, 512) or codec_header(codec_path, base) != before:
        raise ValueError("Native input RVQ projections or codec changed")
    native._mimi_input_rvq_contract = _BASE_CONTRACT_SHA256
    return native, tables, projections, native_root


def collect_native_input_quantizer(native, latents):
    """Capture source Euclidean distances and greedy residual decisions.

    The branches project the same 512-wide input independently. Each source
    VectorQuantization layer supplies its own encode/decode, and the diagnostic
    cdist uses the original EuclideanCodebook._quantize expression. Distances
    are raw p=2 values; no equivalence to device FP32 squared-distance math is
    asserted. A new full native encode is checked for exact IDs per frame.
    """
    import torch

    if getattr(native, "_mimi_input_rvq_contract", None) != _BASE_CONTRACT_SHA256:
        raise ValueError("Use native_input_quantizer's pinned quantizer")
    values = np.asarray(latents)
    if (values.dtype != np.float32 or values.ndim != 3 or values.shape[1:] != (1, 512)
            or not 0 < values.shape[0] <= 128 or not np.isfinite(values).all()):
        raise ValueError("Native input RVQ needs finite FP32 [frames,1,512]")
    values = np.ascontiguousarray(values)
    ids, projected, residual_before, centroids, distances, margins = [], [], [], [], [], []
    with torch.no_grad():
        for frame in values:
            source = torch.from_numpy(frame.transpose(1, 0).copy()[None, :, :])
            # source is [1,512,1], matching SplitResidualVectorQuantizer.encode.
            full = native.encode(source).reshape(-1)
            if tuple(full.shape) != (8,) or full.dtype != torch.int64:
                raise ValueError("Native input RVQ did not emit eight IDs")
            frame_ids, frame_projected, frame_residuals = [], [], []
            frame_centroids, frame_distances, frame_margins = [], [], []
            for branch, n_q in ((native.rvq_first, 1), (native.rvq_rest, 7)):
                branch_input = branch.input_proj(source)
                if tuple(branch_input.shape) != (1, 256, 1):
                    raise ValueError("Native input RVQ branch projection shape differs")
                frame_projected.append(_array(branch_input.reshape(256), (256,), "branch input projection"))
                residual = branch_input
                for layer in branch.vq.layers[:n_q]:
                    frame_residuals.append(_array(residual.reshape(256), (256,), "RVQ residual"))
                    codebook = layer._codebook
                    flattened = codebook._reshape_input(layer.project_in(layer._rearrange_input(residual)))
                    dists = torch.cdist(flattened[None], codebook.embedding[None], p=2)[0]
                    if tuple(dists.shape) != (1, 2048):
                        raise ValueError("Native input RVQ distance shape differs")
                    index = layer.encode(residual).reshape(-1)
                    codebook_index = codebook.encode(flattened).reshape(-1)
                    winner = dists.argmin(dim=-1)
                    if (tuple(index.shape) != (1,) or index.dtype != torch.int64
                            or not torch.equal(index, codebook_index)
                            or not torch.equal(index, winner)):
                        raise AssertionError("Native RVQ codebook.encode differs from source cdist argmin")
                    decoded = layer.decode(index.reshape(1, 1))
                    chosen = codebook.decode(codebook_index).reshape(256)
                    if (tuple(decoded.shape) != (1, 256, 1)
                            or not torch.equal(decoded.reshape(256), chosen)
                            or not torch.equal(chosen, codebook.embedding.index_select(0, index).reshape(256))):
                        raise AssertionError("Native RVQ codebook.decode differs from selected centroid")
                    ordered = torch.sort(dists, dim=-1).values
                    margin = ordered[0, 1] - ordered[0, 0]
                    frame_ids.append(int(index[0]))
                    frame_centroids.append(_array(chosen, (256,), "selected centroid"))
                    frame_distances.append(_array(dists.reshape(2048), (2048,), "native distances"))
                    frame_margins.append(float(margin))
                    residual = residual - decoded
            if frame_ids != full.detach().cpu().tolist():
                raise AssertionError("Greedy native layer IDs differ from full native.encode")
            ids.append(frame_ids)
            projected.append(np.stack(frame_projected))
            residual_before.append(np.stack(frame_residuals))
            centroids.append(np.stack(frame_centroids))
            distances.append(np.stack(frame_distances))
            margins.append(frame_margins)
    result = {
        "ids": np.asarray(ids, np.int64),
        "projected": np.stack(projected),
        "residual_before": np.stack(residual_before),
        "selected_centroids": np.stack(centroids),
        "native_distances": np.stack(distances),
        "winner_margin": np.asarray(margins, np.float32),
    }
    if (result["ids"].shape != (len(values), 8)
            or result["projected"].shape != (len(values), 2, 256)
            or any(result[name].shape != (len(values), 8, 256)
                   for name in ("residual_before", "selected_centroids"))
            or result["native_distances"].shape != (len(values), 8, 2048)
            or result["winner_margin"].shape != (len(values), 8)
            or not np.isfinite(result["winner_margin"]).all()):
        raise ValueError("Native input RVQ diagnostic shape/finiteness differs")
    return result
