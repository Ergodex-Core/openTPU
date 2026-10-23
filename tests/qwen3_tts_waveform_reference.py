"""Independent source convolution/transpose/SnakeBeta/full waveform algebra."""
import numpy as np


class WaveformReference:
    def __init__(self,spec,weights,*,dtype=np.float32):
        self.spec,self.dtype=spec,dtype;self.w={k:np.asarray(v,dtype=dtype) for k,v in weights.items()}

    def sequence(self,inputs):
        traces={};arguments={};d=self.dtype
        def snake(x,p):
            alpha=np.exp(self.w[p+'.alpha']);inverse=d(1)/(np.exp(self.w[p+'.beta'])+d(1e-9))
            z=x*alpha;arguments[p]=float(np.max(np.abs(z)))
            sine=np.sin(z);y=x+(sine*sine)*inverse;traces[p]=y.copy();return y
        def conv(x,p,dilation=1,residual=None):
            weight=self.w[p+'.weight'];nout,_,kernel=weight.shape;history=(kernel-1)*dilation
            padded=np.pad(x,((history,0),(0,0)));y=np.zeros((len(x),nout),dtype=d)
            for tap in range(kernel):y=y+padded[tap*dilation:tap*dilation+len(x)]@weight[:,:,tap].T
            y=y+self.w[p+'.bias']
            if residual is not None:traces[p+'.branch']=y.copy();y=y+residual
            traces[p]=y.copy();return y
        def transpose(x,p,rate):
            weight=self.w[p+'.weight'];previous=np.concatenate([np.zeros_like(x[:1]),x[:-1]],axis=0)
            y=np.empty((len(x)*rate,weight.shape[1]),dtype=d)
            for phase in range(rate):y[phase::rate]=x@weight[:,:,phase]+previous@weight[:,:,phase+rate]+self.w[p+'.bias']
            traces[p]=y.copy();return y
        x=conv(np.asarray(inputs,dtype=d),'decoder.decoder.0.conv')
        for stage,rate in enumerate(self.spec.rates):
            p=f'decoder.decoder.{stage+1}.block.';x=snake(x,p+'0');x=transpose(x,p+'1.conv',rate)
            for unit,dilation in enumerate(self.spec.dilations):
                u=p+str(unit+2);residual=x
                x=snake(x,u+'.act1');x=conv(x,u+'.conv1.conv',dilation)
                x=snake(x,u+'.act2');x=conv(x,u+'.conv2.conv',residual=residual)
        x=snake(x,'decoder.decoder.5');raw=conv(x,'decoder.decoder.6.conv').reshape(-1)
        return dict(raw=raw,waveform=np.clip(raw,d(-1),d(1)),traces=traces,max_arguments=arguments)
