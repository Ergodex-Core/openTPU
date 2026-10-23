"""Two factor-2 tokenizer upsamplers and causal ConvNeXt using existing ISA."""
import math
from .. import language as ol
from .tts_predictor_precision import activation_pair,corrected_dot
from .mlp import down_chunk


def gelu_erf_approx(x):
    # Abramowitz-Stegun 7.1.26 erf approximation, evaluated by FP32 VOPs.
    # This is an explicitly tested approximation of source nn.GELU(), not tanh GELU.
    u=ol.abs(x)*(1/math.sqrt(2))
    t=ol.recip(1+u*.3275911)
    p=(((((t*1.061405429-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t)
    erf_abs=1-p*ol.exp2(u*u*(-ol.LOG2E))
    return x*.5+ol.abs(x)*.5*erf_abs


@ol.jit
def gelu_probe(input,output):
    ol.store(output,gelu_erf_approx(ol.load(input)))


@ol.jit
def upsample_pair(m):
    pair=activation_pair(ol.load(m.input),m.identities[m.channels])
    bias=ol.load(m.up_bias)[None,:]
    for phase in ol.static_range(2):
        result=corrected_dot(pair,m.up[phase],m.up_low[phase])
        ol.store(m.up_output[phase:phase+1,:],result+bias)


@ol.jit
def convnext_row(m,phase: int,row_position: int,output_row: int):
    x=ol.load(m.up_output[phase:phase+1,:])
    accum=ol.zeros((1,m.channels))
    for tap in ol.static_range(6):
        previous=row_position+tap-6
        if previous>=0:
            row=ol.load(m.history[previous%6:previous%6+1,:])
            accum.set(accum+row*ol.load(m.dw[tap,:])[None,:])
    depthwise=accum+x*ol.load(m.dw[6,:])[None,:]+ol.load(m.dw_bias)[None,:]
    ol.store(m.history[row_position%6:row_position%6+1,:],x)
    ol.store(m.depthwise,depthwise)
    mean=ol.sum(depthwise,axis=1)*(1/m.channels)
    centered=depthwise-mean[:,None]
    variance=ol.sum(centered*centered,axis=1)*(1/m.channels)
    normalized=centered*ol.rsqrt(variance+1e-6)[:,None]
    normalized=normalized*ol.load(m.norm_weight)[None,:]+ol.load(m.norm_bias)[None,:]
    ol.store(m.normalized,normalized)
    pair=activation_pair(normalized,m.identities[m.channels])
    chunk=down_chunk(4*m.channels,ol.block_size(),'int8')
    result=None
    for start in ol.static_range(0,4*m.channels,chunk):
        stop=start+chunk
        hidden=corrected_dot(pair,m.fc1[start:stop,:],m.fc1_low[start:stop,:])+ol.load(m.fc1_bias[start:stop])[None,:]
        activation=gelu_erf_approx(hidden)
        ol.store(m.hidden[:,start:stop],hidden)
        ol.store(m.activation[:,start:stop],activation)
        term=corrected_dot(activation_pair(activation,m.identities[chunk]),m.fc2[:,start:stop],m.fc2_low[:,start:stop])
        result=term if result is None else result+term
    projected=result+ol.load(m.fc2_bias)[None,:]
    ol.store(m.projected,projected)
    output=x+projected*ol.load(m.gamma)[None,:]
    ol.store(m.output,output)
    if output_row>=0:ol.store(m.final[output_row:output_row+1,:],output)
