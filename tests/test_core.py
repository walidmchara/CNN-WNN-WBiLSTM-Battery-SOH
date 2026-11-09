import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from models import make_model, MorletBank
from core import FuzzyCMeans, prepare, demo_data, metrics

def test_model_shapes_and_backward():
    cfg=json.loads((Path(__file__).parents[1]/'config.json').read_text())
    f=5+cfg['n_clusters']
    model=make_model(cfg['family'],f,hidden=16,dropout=.1)
    x=torch.randn(3,24,f);prediction=model(x)
    assert prediction.shape==(3,)
    prediction.square().mean().backward()
    assert any(v.grad is not None for v in model.parameters())

def test_fcm_probs():
    rng=np.random.default_rng(1);x=rng.normal(size=(45,3))
    f=FuzzyCMeans(3).fit(x);u=f.transform(x)
    assert u.shape==(45,3)
    np.testing.assert_allclose(u.sum(1),1,atol=1e-6)

def test_temporal_splits_and_scaler():
    df=demo_data(180)
    s,m=prepare(df,24,clusters=2)
    assert len(s['train'][0]) and len(s['test'][0])
    assert m['train_end']<m['validation_end']<m['n_rows']
    assert s['test'][0].shape[-1]==len(m['features'])+2

def test_metrics():
    x=metrics([1,2,3],[1,2,3],'battery_soh')
    assert x['MAE']==0 and x['RMSE']==0 and x['R2']==1
