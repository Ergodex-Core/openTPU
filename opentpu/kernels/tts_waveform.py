"""Bounded SnakeBeta and causal waveform operators on the shared FP32/MX ISA."""
import math
from .. import language as ol
from .tts_predictor_precision import activation_pair,corrected_dot

SNAKE_ARGUMENT_LIMIT=262144.0


def sin2_bounded(z):
    # RNE FP32 add/sub yields an integral nearest multiple for |z|<=2**18.
    # pi_hi=201/64 keeps n*pi_hi exact in this domain; pi_lo corrects it.
    absolute=ol.abs(z)
    rounded=absolute*(1/math.pi)+8388608.0
    nearest=rounded-8388608.0
    r=absolute-nearest*3.140625
    r.set(r-nearest*(math.pi-3.140625))
    del absolute,rounded,nearest
    square=r*r
    p=ol.full(r.shape,1/6227020800)
    for coefficient in [-1/39916800,1/362880,-1/5040,1/120,-1/6,1.0]:p.set(p*square+coefficient)
    sine=r*p
    return ol.minimum(ol.maximum(sine*sine,0.),1.)


@ol.jit
def sin2_probe(input,output):
    ol.store(output,sin2_bounded(ol.load(input)))
