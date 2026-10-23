"""Qwen3-TTS text resize MLP, lowered entirely to the existing openTPU ISA.

This is a projection kernel, not a text-to-waveform implementation. Both affine
layers (including biases) and SiLU execute on device. Weight rows are sharded;
the intermediate activations are gathered before the second affine layer.
"""
from .. import language as ol
from .lib import silu


@ol.jit
def text_projection(x, fc1, bias1, fc2, bias2, out):
    sid = ol.program_id()
    mid = fc1.shape[0]
    width = fc2.shape[0]
    first = ol.dot(ol.load(x), fc1)
    first = first + ol.load(bias1[sid * mid:(sid + 1) * mid])[None, :]
    hidden = ol.all_gather(silu(first))
    result = ol.dot(hidden, fc2)
    result = result + ol.load(bias2[sid * width:(sid + 1) * width])[None, :]
    ol.store(out[:, sid * width:(sid + 1) * width], result)
