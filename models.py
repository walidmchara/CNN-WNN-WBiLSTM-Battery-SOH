"""Research prototypes of the architectures reported in the associated paper.

These modules implement the disclosed component *families*; they are NOT
certified exact reconstructions of unavailable original experiment source.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class MorletBank(nn.Module):
    """Differentiable fixed Morlet convolutional filterbank with scale-specific kernels."""
    def __init__(self, scales=4, kernel_size=15):
        super().__init__()
        assert kernel_size % 2 == 1
        t=torch.arange(kernel_size).float() - kernel_size//2
        kernels=[]
        for scale in torch.linspace(1.0,4.5,scales):
            z=t/scale
            w=(torch.cos(5*z)*torch.exp(-0.5*z.square()))/torch.sqrt(scale)
            w=w-w.mean(); w=w/(w.abs().sum()+1e-8)
            kernels.append(w)
        self.register_buffer('kernel',torch.stack(kernels)[:,None,:])
        self.padding=kernel_size//2
    def forward(self,x):
        # x (batch, steps, vars); return time-resolved averaged wavelet responses
        z=x.mean(-1,keepdim=True).transpose(1,2)
        return F.conv1d(z,self.kernel,padding=self.padding).transpose(1,2)

class AdditiveAttention(nn.Module):
    def __init__(self,d):
        super().__init__(); self.energy=nn.Sequential(nn.Linear(d,d//2),nn.Tanh(),nn.Linear(d//2,1))
    def forward(self,x):
        alpha=torch.softmax(self.energy(x),dim=1)
        return (x*alpha).sum(1)

class CNNWaveletBiLSTM(nn.Module):
    def __init__(self,in_features,hidden=32,dropout=0.1):
        super().__init__()
        self.wavelet=MorletBank(); self.cnn=nn.Sequential(nn.Conv1d(in_features+4,hidden,3,padding=1),nn.ReLU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.ReLU())
        self.bilstm=nn.LSTM(hidden,hidden,batch_first=True,bidirectional=True)
        self.attention=AdditiveAttention(hidden*2)
        self.head=nn.Sequential(nn.Dropout(dropout),nn.Linear(hidden*2,1))
    def forward(self,x):
        z=torch.cat([x,self.wavelet(x)],-1).transpose(1,2)
        h=self.cnn(z).transpose(1,2)
        h,_=self.bilstm(h)
        return self.head(self.attention(h)).squeeze(-1)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,d):
        super().__init__();self.d=d
    def forward(self,x):
        n=x.size(1);i=torch.arange(self.d,device=x.device)
        pos=torch.arange(n,device=x.device,dtype=x.dtype)[:,None]
        freq=torch.exp(-math.log(10000.0)*(2*(i//2))/self.d)
        angles=pos*freq[None,:]
        pe=torch.where((i%2==0)[None,:],torch.sin(angles),torch.cos(angles))
        return x+pe[None,:,:]

class WaveletTransformer(nn.Module):
    def __init__(self,in_features,hidden=32,dropout=0.1):
        super().__init__();self.wavelet=MorletBank();self.input=nn.Linear(in_features+4,hidden);self.pos=SinusoidalPositionalEncoding(hidden)
        layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=dropout,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,num_layers=2)
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))
    def forward(self,x):
        x=self.pos(self.input(torch.cat([x,self.wavelet(x)],-1)))
        return self.head(self.encoder(x)[:,-1,:]).squeeze(-1)

class SparseQueryAttention(nn.Module):
    """Sampled-key top-query sparse attention (compact ProbSparse-style prototype)."""
    def __init__(self,d,heads=4,dropout=0.1):
        super().__init__();assert d%heads==0
        self.heads=heads;self.d=d//heads
        self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.out=nn.Linear(d,d)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        b,l,d=x.size();h=self.heads;k=min(l,max(1,math.ceil(2*math.log(l+1))))
        def split(y):return y.reshape(b,l,h,self.d).transpose(1,2)
        q,kall,v=map(split,(self.q(x),self.k(x),self.v(x)))
        # deterministic sparse key sample; no future targets ever enter the input window
        key_idx=torch.linspace(0,l-1,k,device=x.device).long()
        sample=(q@kall[:,:,key_idx,:].transpose(-1,-2))/math.sqrt(self.d)
        importance=sample.max(-1).values-sample.mean(-1)
        top_idx=importance.topk(k,dim=-1).indices
        idx=top_idx[:,:,:,None].expand(b,h,k,self.d)
        topq=torch.gather(q,2,idx)
        weights=torch.softmax(topq@kall.transpose(-1,-2)/math.sqrt(self.d),dim=-1)
        topval=self.dropout(weights)@v
        context=v.mean(2,keepdim=True).expand(b,h,l,self.d).clone()
        context.scatter_(2,idx,topval)
        context=context.transpose(1,2).contiguous().reshape(b,l,d)
        return self.out(context)

class InformerBlock(nn.Module):
    def __init__(self,d,dropout=0.1):
        super().__init__();self.attn=SparseQueryAttention(d,dropout=dropout)
        self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d,d))
    def forward(self,x):h=self.n1(x+self.attn(x));return self.n2(h+self.ff(h))

class FuzzyModel(nn.Module):
    def __init__(self,in_features,hidden=32,dropout=0.1,attention='informer'):
        super().__init__();self.wavelet=MorletBank()
        self.cnn=nn.Sequential(nn.Conv1d(in_features+4,hidden,3,padding=1),nn.GELU(),nn.Conv1d(hidden,hidden,3,padding=1),nn.GELU())
        self.pos=SinusoidalPositionalEncoding(hidden)
        if attention=='informer':self.encoder=nn.Sequential(InformerBlock(hidden,dropout),InformerBlock(hidden,dropout))
        else:self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(hidden,4,2*hidden,dropout=dropout,batch_first=True),num_layers=2)
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,1))
    def forward(self,x):
        z=torch.cat([x,self.wavelet(x)],-1).transpose(1,2)
        h=self.pos(self.cnn(z).transpose(1,2))
        return self.head(self.encoder(h).mean(dim=1)).squeeze(-1)

class WHDMNet(nn.Module):
    """Multivariate sequence => 2D embedding => dilated conv => linearized attention."""
    def __init__(self,in_features,hidden=32,dropout=0.1):
        super().__init__();self.project=nn.Linear(in_features,hidden)
        self.conv=nn.Sequential(nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=2,dilation=2),nn.GELU())
        self.wavelet=MorletBank();self.wavelet_proj=nn.Linear(4,hidden)
        self.q=nn.Linear(hidden,hidden);self.k=nn.Linear(hidden,hidden);self.v=nn.Linear(hidden,hidden)
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Dropout(dropout),nn.Linear(hidden,1))
    def forward(self,x):
        b,t,f=x.shape;s=math.ceil(math.sqrt(t))
        z=self.project(x);z=F.pad(z,(0,0,0,s*s-t))
        z=self.conv(z.transpose(1,2).reshape(b,-1,s,s)).reshape(b,-1,s*s).transpose(1,2)[:,:t,:]
        z=z+self.wavelet_proj(self.wavelet(x))
        q=F.elu(self.q(z))+1;k=F.elu(self.k(z))+1;v=self.v(z)
        # Linear attention O(T*d^2) with global key-value summary
        kv=(k.transpose(1,2)@v); denom=q@k.sum(dim=1).unsqueeze(-1)+1e-6
        h=(q@kv)/denom
        return self.head(h[:,-1,:]).squeeze(-1)

def make_model(family,in_features,hidden=32,dropout=0.1):
    if family in ('cnn_wavelet_bilstm','fuzzy_bilstm'):return CNNWaveletBiLSTM(in_features,hidden,dropout)
    if family=='wavelet_transformer':return WaveletTransformer(in_features,hidden,dropout)
    if family=='fuzzy_informer':return FuzzyModel(in_features,hidden,dropout,attention='informer')
    if family=='fuzzy_transformer':return FuzzyModel(in_features,hidden,dropout,attention='transformer')
    if family=='whdmnet':return WHDMNet(in_features,hidden,dropout)
    raise ValueError('unknown family: '+family)
