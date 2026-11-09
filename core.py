"""Leakage-aware dataset preparation, FCM and evaluation utilities."""
import json,math,random
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def seed_all(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(torch.get_num_threads(),2))

class FuzzyCMeans:
    def __init__(self,n_clusters=3,m=2.0,seed=42,n_iter=100):
        if m<=1:raise ValueError('m must be > 1')
        self.k=n_clusters;self.m=m;self.seed=seed;self.n_iter=n_iter
    def transform(self,x):
        d2=np.maximum(((x[:,None,:]-self.centers[None,:,:])**2).sum(-1),1e-10)
        inv=np.power(d2,-1.0/(self.m-1));return inv/inv.sum(axis=1,keepdims=True)
    def fit(self,x):
        rng=np.random.default_rng(self.seed)
        self.centers=x[rng.choice(len(x),self.k,replace=False)].copy()
        for _ in range(self.n_iter):
            u=self.transform(x); w=u**self.m
            centers=(w.T@x)/(w.sum(0)[:,None]+1e-9)
            if np.max(np.abs(centers-self.centers))<1e-6:break
            self.centers=centers
        return self

def demo_data(n=240,seed=42,domain='solar_irradiance'):
    rng=np.random.default_rng(seed);t=np.arange(n)
    weather=20+7*np.sin(t/100*2*np.pi)+rng.normal(0,1,n)
    cloud=np.clip(0.4+0.3*np.sin(t/42)+rng.normal(0,.08,n),0,1)
    if domain.startswith('battery'):
        health=1-0.0007*t+0.005*np.sin(t/11)+rng.normal(0,.001,n)
        return pd.DataFrame({'time':t,'capacity':health*2.0,'temperature':weather,'current':0.8+0.1*np.sin(t/7),'target':health})
    sun=np.maximum(0,np.sin((t%24)/24*2*np.pi))
    target=np.clip(900*sun*(1-.6*cloud)+rng.normal(0,15,n),0,None)
    if domain=='pv_power':target=target*0.25
    return pd.DataFrame({'time':t,'temperature':weather,'cloud_cover':cloud,'solar_elevation':sun,'target':target})

def prepare(df,lookback,feature_cols=None,target='target',time_col='time',clusters=0,seed=42):
    df=df.copy()
    if time_col in df:df=df.sort_values(time_col,kind='stable').reset_index(drop=True)
    if target not in df:raise ValueError(f'missing target column {target!r}')
    if feature_cols is None:
        feature_cols=[c for c in df.select_dtypes(include='number').columns if c!=time_col]
    if target not in feature_cols:feature_cols.append(target)
    if len(df)<max(80,lookback+30):raise ValueError('need >= 80 records with enough historical context')
    for c in feature_cols:df[c]=pd.to_numeric(df[c],errors='coerce')
    if df[feature_cols].isna().any().any():
        raise ValueError('missing/non-numeric data; clean or impute before splitting (train-only imputation recommended)')
    X=df[feature_cols].to_numpy(dtype=np.float32)
    y=df[target].to_numpy(dtype=np.float32)
    train_end=int(.70*len(df));valid_end=int(.85*len(df))
    # Statistics only from historical train period.
    mean=X[:train_end].mean(0);std=X[:train_end].std(0);std=np.maximum(std,1e-6)
    ym=float(y[:train_end].mean());ys=max(float(y[:train_end].std()),1e-6)
    X=(X-mean)/std;yn=(y-ym)/ys
    fcm=None
    if clusters:
        fcm=FuzzyCMeans(clusters,seed=seed).fit(X[:train_end])
        U=fcm.transform(X).astype(np.float32)
        X=np.concatenate([X,U],axis=1)
    # Define a clean chronological calibration slice *within* train.
    calib_start=int(.85*train_end)
    windows=np.stack([X[i-lookback:i] for i in range(lookback,len(df))]).astype(np.float32)
    labels=yn[lookback:].astype(np.float32)
    indices=np.arange(lookback,len(df))
    masks={'train':indices<calib_start,'calib':(indices>=calib_start)&(indices<train_end),
           'val':(indices>=train_end)&(indices<valid_end),'test':indices>=valid_end}
    split={name:(torch.tensor(windows[mask]),torch.tensor(labels[mask])) for name,mask in masks.items()}
    meta={'features':feature_cols,'input_features':X.shape[1],'n_rows':len(df),'lookback':lookback,
          'train_end':train_end,'calibration_start':calib_start,'validation_end':valid_end,
          'target_mean':ym,'target_std':ys,'x_mean':mean.tolist(),'x_std':std.tolist(),
          'fcm_centroids':None if fcm is None else fcm.centers.tolist()}
    return split,meta

def metrics(truth,pred,domain='solar_irradiance'):
    truth=np.asarray(truth,dtype=float);pred=np.asarray(pred,dtype=float)
    m={'MAE':float(np.mean(np.abs(truth-pred))),
       'RMSE':float(np.sqrt(np.mean((truth-pred)**2))),
       'R2':float(1-np.sum((truth-pred)**2)/(np.sum((truth-truth.mean())**2)+1e-12))}
    # Irradiance/power MAPE excludes low/zero values to avoid singularity.
    floor=10 if domain in ('solar_irradiance','pv_power') else 1e-4
    ok=np.abs(truth)>floor
    m['MAPE_daylight_or_nonzero_pct']=float(100*np.mean(np.abs((truth[ok]-pred[ok])/truth[ok]))) if ok.any() else None
    m['mape_denominator_threshold']=floor;m['mape_valid_count']=int(ok.sum())
    return m
