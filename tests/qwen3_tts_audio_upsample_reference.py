"""Independent full-sequence ConvTranspose/depthwise/LN/exact-erf source algebra."""
import math
import numpy as np


class AudioUpsampleReference:
    def __init__(self,channels,weights,*,dtype=np.float32):
        self.channels,self.dtype=channels,dtype;self.w={k:np.asarray(v,dtype=dtype) for k,v in weights.items()}

    def sequence(self,inputs):
        x=np.asarray(inputs,dtype=self.dtype);traces=[];d=self.dtype
        for stage in range(2):
            p=f'decoder.upsample.{stage}.';w=self.w[p+'0.conv.weight']
            up=np.empty((2*len(x),self.channels),dtype=d)
            for phase in range(2):up[phase::2]=x@w[:,:,phase]+self.w[p+'0.conv.bias']
            padded=np.pad(up,((6,0),(0,0)));dw=np.zeros_like(up)
            for tap in range(7):dw=dw+padded[tap:tap+len(up)]*self.w[p+'1.dwconv.conv.weight'][:,0,tap]
            dw=dw+self.w[p+'1.dwconv.conv.bias']
            centered=dw-np.mean(dw,axis=-1,keepdims=True)
            normalized=centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+d(1e-6))
            normalized=normalized*self.w[p+'1.norm.weight']+self.w[p+'1.norm.bias']
            hidden=normalized@self.w[p+'1.pwconv1.weight'].T+self.w[p+'1.pwconv1.bias']
            z=hidden*d(1/math.sqrt(2))
            erf=np.fromiter((math.erf(float(v)) for v in z.flat),dtype=d,count=z.size).reshape(z.shape)
            activation=d(.5)*hidden*(d(1)+erf)
            projected=activation@self.w[p+'1.pwconv2.weight'].T+self.w[p+'1.pwconv2.bias']
            output=up+projected*self.w[p+'1.gamma']
            traces.append(dict(up=up,depthwise=dw,normalized=normalized,hidden=hidden,activation=activation,projected=projected,output=output))
            x=output
        return traces
