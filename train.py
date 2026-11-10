"""Runnable train/evaluate + optional paper-inspired HPO and residual calibration.

Never confuse the '--smoke' synthetic demo with the article's reported results.
"""
import argparse,json,math,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
from models import make_model
from core import prepare,seed_all,demo_data,metrics
from training_recovery import EpochCheckpoint

def train_one(split, meta, cfg, epochs=None, quiet=False, recovery=None, resume=False):
    seed_all(cfg['seed'])
    net=make_model(cfg['family'],meta['input_features'],cfg['hidden'],cfg['dropout'])
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'])
    loader=DataLoader(TensorDataset(*split['train']),batch_size=cfg['batch_size'],shuffle=True)
    xval,yval=split['val']; epochs=epochs if epochs is not None else cfg['epochs']
    best=math.inf;best_weights=None;history=[];start_epoch=0
    if resume:
        if recovery is None: raise ValueError('Recovery directory is required for resume')
        state=recovery.restore({'model':net,'optimizer':optimizer})
        start_epoch=state['epoch'];best=state['best'];best_weights=state['best_weights'];history=state['history']
        print(f"Resuming after epoch {start_epoch}")
    for epoch in range(start_epoch,epochs):
        net.train();loss_sum=0.0;count=0
        for x,y in loader:
            optimizer.zero_grad();pred=net(x);loss=(pred-y).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);optimizer.step()
            loss_sum+=float(loss.item())*len(x);count+=len(x)
        net.eval()
        with torch.no_grad():v=float((net(xval)-yval).square().mean().item())
        history.append({'epoch':epoch+1,'train_mse':loss_sum/max(count,1),'validation_mse':v})
        if v<best:best=v;best_weights={k:t.detach().clone() for k,t in net.state_dict().items()}
        if recovery is not None:
            recovery.save(epoch+1, {'model':net,'optimizer':optimizer},
                          {'best':best,'best_weights':best_weights,'history':history})
        if not quiet:print(f"epoch={epoch+1} train={history[-1]['train_mse']:.6f} val={v:.6f}")
    net.load_state_dict(best_weights)
    return net,history,best

def predict(net,data):
    net.eval()
    with torch.no_grad():return net(data[0]).cpu().numpy()

def bounds_to_cfg(p,cfg):
    changed=dict(cfg);changed['learning_rate']=float(np.exp(np.log(1e-4)+p[0]*(np.log(3e-3)-np.log(1e-4))))
    changed['dropout']=float(.05+.35*p[1]);return changed

def whale_optimize(fn,n=5,seed=42):
    rng=np.random.default_rng(seed);X=rng.random((min(4,n),2));fitness=np.array([fn(x) for x in X]);used=len(X)
    while used<n:
        best=X[np.argmin(fitness)];t=(used-len(X))/max(1,n-len(X));a=2*(1-t)
        j=used%len(X);r=rng.random(2);A=2*a*r-a;C=2*r
        if rng.random()<.5:proposal=best-A*np.abs(C*best-X[j])
        else:proposal=np.abs(best-X[j])*np.exp(rng.uniform(-1,1))*np.cos(2*np.pi*rng.uniform(-1,1))+best
        proposal=np.clip(proposal,0,1);fit=fn(proposal);used+=1
        if fit<fitness[j]:X[j]=proposal;fitness[j]=fit
    return X[np.argmin(fitness)],float(fitness.min())

def cbo_illustrative(fn,n=5,seed=42):
    # An explicit CBO-*inspired* chaotic global search placeholder; see REPRODUCIBILITY.md.
    rng=np.random.default_rng(seed);p=rng.random(2);best=p.copy();val=fn(p)
    z=.347
    for k in range(1,n):
        z=4*z*(1-z)
        direction=rng.normal(size=2);direction/=np.linalg.norm(direction)+1e-10
        proposal=np.abs((best+z*.7*direction) % 2)
        proposal=np.where(proposal>1,2-proposal,proposal)
        new=fn(proposal)
        if new<val:best,val=proposal,new
    return best,val

def optuna_search(fn,n=5,seed=42):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(lambda trial:fn(np.array([trial.suggest_float('lr_index',0,1),trial.suggest_float('dropout_index',0,1)])),n_trials=n)
    return np.array([study.best_params['lr_index'],study.best_params['dropout_index']]),study.best_value

def residual_regressor(kind,seed):
    if kind=='xgboost':
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=50,max_depth=3,learning_rate=.05,subsample=.8,n_jobs=2,random_state=seed)
    if kind=='lightgbm':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=50,learning_rate=.05,num_leaves=15,verbosity=-1,random_state=seed,n_jobs=2,min_child_samples=3)
    return None

def residual_features(net,xy,raw_pred):
    x=xy[0].cpu().numpy()
    # strictly historical, never future targets (last-observation, window mean/std, base prediction)
    return np.concatenate([raw_pred.reshape(-1,1),x[:,-1,:],x.mean(1),x.std(1)],axis=1)

def run(args):
    cfg=json.loads(Path(__file__).with_name('config.json').read_text())
    for k in ['epochs','lookback','batch_size','hidden','seed']:
        if getattr(args,k,None) is not None:cfg[k]=getattr(args,k)
    if args.smoke:df=demo_data(domain=cfg['domain'],seed=cfg['seed'])
    elif args.csv:df=pd.read_csv(args.csv)
    else:raise SystemExit('Provide --csv PATH for real data or --smoke for a synthetic test')
    selected=args.features.split(',') if args.features else None
    split,meta=prepare(df,cfg['lookback'],feature_cols=selected,target=args.target,time_col=args.time_col,clusters=cfg['n_clusters'],seed=cfg['seed'])
    if any(len(x[0])<1 for x in split.values()):raise ValueError('empty subset: reduce lookback or increase dataset')
    technique=(cfg['optimizer'] if args.search=='paper' else args.search)
    search_report=None
    if technique!='none':
        def objective(p):
            candidate=bounds_to_cfg(p,cfg)
            _,_,mse=train_one(split,meta,candidate,epochs=min(3,cfg['epochs']),quiet=True)
            return mse
        if technique=='woa':p,b=whale_optimize(objective,args.trials,cfg['seed'])
        elif technique=='cbo':p,b=cbo_illustrative(objective,args.trials,cfg['seed'])
        elif technique=='optuna':p,b=optuna_search(objective,args.trials,cfg['seed'])
        else:raise ValueError('unsupported search')
        cfg=bounds_to_cfg(p,cfg);search_report={'algorithm':technique,'trials':args.trials,'validation_mse':float(b),
            'warning':'CBO search is illustrative, not a verified implementation of paper equations' if technique=='cbo' else None}
    if args.resume and technique!='none':
        raise ValueError('Resume with hyperparameter search is not supported; use --search none')
    source_signature = (hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
                        if args.csv and not args.smoke else 'synthetic_smoke_seed_'+str(cfg['seed']))
    recovery = EpochCheckpoint(args.checkpoint_dir or Path(args.out)/'recovery', {
        'training_config':cfg,'source_sha256':source_signature,'smoke':args.smoke,
        'target':args.target,'time_col':args.time_col,'features':args.features,
        'search':technique,
    })
    net,history,_=train_one(split,meta,cfg,recovery=recovery,resume=args.resume)
    mean,scale=meta['target_mean'],meta['target_std']
    ground=split['test'][1].cpu().numpy()*scale+mean
    neural=predict(net,split['test'])*scale+mean
    corrected=neural.copy();residual_fitted=False
    if cfg['residual']!='none':
        cal_y=split['calib'][1].cpu().numpy()*scale+mean
        cal_neural=predict(net,split['calib'])*scale+mean
        model=residual_regressor(cfg['residual'],cfg['seed'])
        # Residual learner is fit ONLY using time-disjoint calibration observations.
        model.fit(residual_features(net,split['calib'],cal_neural),cal_y-cal_neural)
        corrected=neural+model.predict(residual_features(net,split['test'],neural))
        residual_fitted=True
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state_dict':net.state_dict(),'config':cfg,'meta':meta},out/'checkpoint.pt')
    pd.DataFrame(history).to_csv(out/'training_log.csv',index=False)
    pd.DataFrame({'actual':ground,'neural_prediction':neural,'final_prediction':corrected}).to_csv(out/'predictions.csv',index=False)
    result={'paper_doi':cfg['doi'],'run_kind':'synthetic smoke test' if args.smoke else 'user-provided data experiment',
            'model_family':cfg['family'],'config':cfg,'data_meta':meta,
            'neural_metrics':metrics(ground,neural,cfg['domain']),
            'final_metrics':metrics(ground,corrected,cfg['domain']),'residual_model_fit':residual_fitted,'search':search_report,
            'IMPORTANT':'These are newly computed run metrics; NOT published paper results.'}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,3));ax.plot(ground,label='actual');ax.plot(corrected,label='model');ax.legend();ax.set_title('Holdout predictions (not article figure)');fig.tight_layout();fig.savefig(out/'predictions.png',dpi=150);plt.close(fig)
    except ImportError:pass
    print('Wrote outputs:',out,'\n',json.dumps(result['final_metrics'],indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv');ap.add_argument('--smoke',action='store_true');ap.add_argument('--features',help='comma-separated feature column names; target is always appended')
    ap.add_argument('--target',default='target');ap.add_argument('--time-col',default='time');ap.add_argument('--lookback',type=int)
    ap.add_argument('--epochs',type=int);ap.add_argument('--batch-size',type=int);ap.add_argument('--hidden',type=int);ap.add_argument('--seed',type=int)
    ap.add_argument('--search',choices=['paper','none','woa','cbo','optuna'],default='none',help='paper selects optimizer declared in associated paper')
    ap.add_argument('--trials',type=int,default=5);ap.add_argument('--out',default='outputs/latest')
    ap.add_argument('--checkpoint-dir',help='Persistent recovery path (Drive or mounted volume)')
    ap.add_argument('--resume',action='store_true',help='Continue from last completed epoch')
    run(ap.parse_args())
