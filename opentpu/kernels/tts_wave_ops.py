"""Looped waveform convolution, SnakeBeta and raw [-1,1] output operations."""
from .. import language as ol
from ..compiler import Tensor
from .tts_predictor_precision import activation_pair,corrected_dot
from .tts_waveform import sin2_bounded


def padded_pair(x,m):
    if m.padded!=m.nin:
        pad=ol.zeros((m.batch,m.padded));pad[:,:m.nin].set(x);x=pad
    return activation_pair(x,m.identity)


@ol.jit
def begin_wave(m):
    ol.store(m.guard,ol.zeros((1,)))


@ol.jit
def snake_beta(m):
    for index in ol.range(m.nrows//m.batch):
        start=index*m.batch;x=ol.load(m.input[start:start+m.batch,:])
        z=x*ol.load(m.alpha)[None,:]
        maximum=ol.max(ol.max(ol.abs(z)))
        ol.store(m.guard,ol.maximum(ol.load(m.guard),maximum))
        periodic=sin2_bounded(z)
        result=x+periodic*ol.load(m.inv_beta)[None,:]
        ol.store(m.output[start:start+m.batch,:],result)


@ol.jit
def prepare_window(m,fresh: bool):
    if m.history_rows:
        old=ol.zeros((m.history_rows,m.nin)) if fresh else ol.load(m.history)
        ol.store(m.window[:m.history_rows,:],old)
        del old
    for index in ol.range(m.nrows//m.batch):
        start=index*m.batch
        values=ol.load(m.input[start:start+m.batch,:])
        ol.store(m.window[m.history_rows+start:m.history_rows+start+m.batch,:],values)
    if m.history_rows:
        recent=ol.load(m.window[m.nrows:m.nrows+m.history_rows,:])
        ol.store(m.history,recent)


@ol.jit
def causal_convolution(m):
    for index in ol.range(m.nrows//m.batch):
        start=index*m.batch;result=None
        for tap in ol.static_range(m.kernel):
            offset=start+tap*m.dilation
            x=ol.load(m.window[offset:offset+m.batch,:])
            term=corrected_dot(padded_pair(x,m),m.weights[tap],m.low[tap])
            if result is None:result=term
            else:result.set(result+term)
        value=result+ol.load(m.bias)[None,:]
        if m.residual is not None:
            ol.store(m.branch[start:start+m.batch,:],value)
            value=value+ol.load(m.residual[start:start+m.batch,:])
        ol.store(m.output[start:start+m.batch,:],value)


@ol.jit
def transposed_convolution(m):
    for index in ol.range(m.nrows//m.batch):
        start=index*m.batch
        current=padded_pair(ol.load(m.window[start+1:start+1+m.batch,:]),m)
        previous=padded_pair(ol.load(m.window[start:start+m.batch,:]),m)
        for phase in ol.static_range(m.rate):
            a=corrected_dot(current,m.weights[phase],m.low[phase])
            b=corrected_dot(previous,m.weights[phase+m.rate],m.low[phase+m.rate])
            value=a+b+ol.load(m.bias)[None,:]
            destination=Tensor(m.output.base+(start*m.rate+phase)*(4*m.nout),
                               (m.batch,m.nout),(m.rate*m.nout,1))
            ol.store(destination,value)


@ol.jit
def finish_wave(m):
    for index in ol.range(m.samples//m.batch):
        start=index*m.batch
        raw=ol.load(m.raw[start:start+m.batch,:])
        clamped=ol.minimum(ol.maximum(raw,-1.),1.)
        destination=Tensor(m.waveform.base+start*4,(m.batch,1),(1,1))
        ol.store(destination,clamped)
