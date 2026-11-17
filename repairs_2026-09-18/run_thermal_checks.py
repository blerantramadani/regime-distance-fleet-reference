"""Frozen thermal sensitivity analyses; no writes to the source project."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent
sys.path.insert(0, str(SOURCE/'scripts'))
from run_health_twin_baseline import (SOURCE_COLUMNS, TARGETS, AMBIENT_SOURCES,
    make_feature_matrix, fit_ridge, robust_training_keep, predict)
from run_health_twin_regime_peer_v2 import regime_matrix, residual_frame
from run_health_twin_rolling_ablation_v1 import INDEPENDENT_FOLDS, LOCAL_FOLDS
from run_health_twin_trim_distance_ablation import factorial_scores
from run_health_twin_alert_stability_v1 import calibrate_common, apply_common_thresholds


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def fit_frozen(x,y,turbine,train,alpha):
    initial=fit_ridge(x[train],y[train],alpha=10.)
    keep,_=robust_training_keep(y[train]-predict(initial,x[train]),turbine[train])
    model=fit_ridge(x[train][keep],y[train][keep],alpha=alpha)
    _,scales=robust_training_keep(y[train]-predict(model,x[train]),turbine[train])
    return model,scales


def score_dense(frame,scales,bandwidth,turbines):
    """Same leave-one-out trim/weights as source, plus uniform lower median."""
    times=pd.Index(frame.timestamp.unique()).sort_values()
    a=times.get_indexer(frame.timestamp);b=pd.Index(turbines).get_indexer(frame.turbine_id)
    r=np.full((len(times),len(turbines)),np.nan)
    desc=np.full((*r.shape,7),np.nan)
    r[a,b]=frame.residual.to_numpy()
    desc[a,b,:]=frame[[f'regime_{i}' for i in range(7)]].to_numpy()
    candidates=np.isfinite(r[:,None,:]) & ~np.eye(len(turbines),dtype=bool)[None,:,:]
    v=np.where(candidates,r[:,None,:],np.nan)
    peer_scale=np.array([scales[t] for t in turbines])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='All-NaN slice encountered')
        center=np.nanmedian(v,axis=2)
        scale=np.maximum(np.maximum(1.4826*np.nanmedian(abs(v-center[:,:,None]),axis=2),
                        np.nanmedian(np.where(candidates,peer_scale[None,None,:],np.nan),axis=2)),.25)
        keep=candidates & (abs(v-center[:,:,None])<=4.5*scale[:,:,None])
        retained=np.where(keep,v,np.nan)
        midpoint=np.nanmedian(retained,axis=2)
    count=keep.sum(axis=2)
    sorted_values=np.sort(np.where(keep,v,np.inf),axis=2,kind='stable')
    lower=np.take_along_axis(sorted_values,np.maximum((count-1)//2,0)[:,:,None],axis=2)[:,:,0]
    distances=np.sqrt(np.sum((desc[:,:,None,:]-desc[:,None,:,:])**2,axis=3))
    kernel=np.where(keep,-.5*(distances/bandwidth)**2,-np.inf)
    maximum=np.max(kernel,axis=2,keepdims=True)
    weights=np.exp(kernel-np.where(np.isfinite(maximum),maximum,0))
    order=np.argsort(np.where(keep,v,np.inf),axis=2,kind='stable')
    sorted_weights=np.take_along_axis(weights,order,axis=2)
    cumulative=np.cumsum(sorted_weights,axis=2)
    # Source weighted_median uses weights.sum()/2, not cumulative[-1]/2.
    index=np.argmax(cumulative>=weights.sum(axis=2,keepdims=True)/2,axis=2)
    distance=np.take_along_axis(sorted_values,index[:,:,None],axis=2)[:,:,0]
    valid=np.isfinite(r)&(candidates.sum(axis=2)>=3)&(count>=3)
    ti,tu=np.where(valid)
    return pd.DataFrame({'timestamp':times.to_numpy()[ti],'turbine_id':np.array(turbines)[tu],
                         'static_midpoint':r[valid]-midpoint[valid],
                         'static_lower':r[valid]-lower[valid],
                         'distance':r[valid]-distance[valid],
                         'retained_peers':count[valid]})


def unit_checks():
    # Sparse and even/odd counts, residual trim, equal descriptors.
    turbines=[f'T{i:02d}' for i in range(1,7)]
    rows=[]
    for n in [3,4,5,6]:
        for i in range(n):
            rows.append({'timestamp':pd.Timestamp('2015-10-01')+pd.Timedelta(minutes=10*n),
                         'turbine_id':turbines[i],'residual':float(i if n!=4 or i!=3 else 100),
                         **{f'regime_{j}':0. for j in range(7)}})
    f=pd.DataFrame(rows);s={t:1. for t in turbines}
    fast=score_dense(f,s,1.,turbines)
    old=factorial_scores(f,'toy',s,1.)
    for method,col in [('static_trim','static_midpoint'),('distance_trim','distance')]:
        z=old[old.method==method].merge(fast,on=['timestamp','turbine_id'],validate='one_to_one')
        assert len(z)==len(fast)
        np.testing.assert_allclose(z.specific_residual,z[col],atol=1e-12)
    np.testing.assert_allclose(fast.distance,fast.static_lower,atol=1e-12)
    return {'vectorized_vs_original_sparse_even_odd':'passed','uniform_distance_equals_lower_median':'passed'}


def cube_from_scores(s,start,end):
    days=pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='D')
    idx=days.get_indexer(s.timestamp.dt.floor('D'));assert (idx>=0).all()
    cube=np.zeros((len(days),3,3))
    for j,col in enumerate(['static_midpoint','static_lower','distance']):
        val=s[col].to_numpy()
        cube[:,j,0]=np.bincount(idx,minlength=len(days))
        cube[:,j,1]=np.bincount(idx,weights=val,minlength=len(days))
        cube[:,j,2]=np.bincount(idx,weights=val*val,minlength=len(days))
    return days,cube


def effects(stats):
    if (stats[...,0]<=0).any():raise ValueError('Bootstrap draw has zero support')
    sd=np.sqrt(np.maximum(stats[...,2]/stats[...,0]-(stats[...,1]/stats[...,0])**2,0))
    if (sd[...,:2]<=0).any():raise ValueError('Zero baseline spread')
    return 100*(sd[...,2,None]/sd[...,:2]-1)


def boot(cube,block,seed):
    rng=np.random.default_rng(seed);n=len(cube);starts=np.arange(n-block+1)
    draws=[]
    for _ in range(2000):
        selected=[]
        while len(selected)<n:
            start=int(rng.choice(starts));selected.extend(range(start,start+block))
        draws.append(effects(cube[np.array(selected[:n])].sum(axis=0)))
    return np.quantile(np.array(draws),[.025,.5,.975],axis=0),np.mean(np.array(draws)<0,axis=0)


def prepare(data,ambient,turbines):
    x,_=make_feature_matrix(data,turbines,ambient);desc,_=regime_matrix(data,ambient)
    eligible=(data.has_temperature_record & data.active_power_mean.between(0,2350)
              & data.wind_speed_mean.between(4,20) & data.generator_speed_mean.ge(100)
              & data.power_reference_end.ge(2200) & np.isfinite(x).all(axis=1)
              & np.isfinite(desc).all(axis=1)).to_numpy()
    return x,desc,eligible


def rolling(data,out):
    turbines=sorted(data.turbine_id.unique());turbine=data.turbine_id.to_numpy();time=data.timestamp
    metrics=[];intervals=[];daily=[];per_turbine=[];repro=[]
    for protocol,folds,ambient in [('independent',INDEPENDENT_FOLDS,'met_temperature_mean'),
                                  ('local',LOCAL_FOLDS,'local_ambient_temp_mean')]:
        source=SOURCE/f'data/experiment_results/health_twin_trim_distance_ablation_{protocol}_v1'
        old=pd.read_csv(source/'fold_target_method_results.csv');old_ci=pd.read_csv(source/'mechanism_contrasts.csv')
        x,raw,eligible=prepare(data,ambient,turbines)
        for fi,fold in enumerate(folds):
            train=(time.ge(fold['start'])&time.lt(fold['train_end'])).to_numpy()
            test=(time.ge(fold['valid_end'])&time.lt(fold['test_end'])).to_numpy()
            scale=np.std(raw[eligible&train],axis=0);scale=np.where(scale>1e-8,scale,1.)
            descriptor=(raw-np.mean(raw[eligible&train],axis=0))/scale
            for ki,target in enumerate(TARGETS):
                locked=old[(old.fold==fold['fold'])&(old.target==target)].set_index('method')
                alpha=float(locked.loc['distance_trim','normal_model_alpha']);bw=float(locked.loc['distance_trim','selected_distance_bandwidth'])
                y=data[target].to_numpy();usable=eligible&np.isfinite(y)
                model,scales=fit_frozen(x,y,turbine,usable&train,alpha)
                mask=usable&test;residual=y[mask]-predict(model,x[mask])
                frame=residual_frame(data,residual,mask,descriptor)
                s=score_dense(frame,scales,bw,turbines)
                # Actual-data comparison with original slow scoring on first 25 timestamps.
                sample=frame[frame.timestamp.isin(frame.timestamp.unique()[:25])]
                original=factorial_scores(sample,target,scales,bw)
                for old_method,new_method in [('static_trim','static_midpoint'),('distance_trim','distance')]:
                    z=original[original.method==old_method].merge(s,on=['timestamp','turbine_id'],validate='one_to_one')
                    assert len(z)==len(original[original.method==old_method])
                    np.testing.assert_allclose(z.specific_residual,z[new_method],atol=1e-10,rtol=1e-10)
                    assert len(s)==int(locked.loc[old_method,'score_rows'])
                    np.testing.assert_allclose(np.std(s[new_method]),locked.loc[old_method,'specific_std'],atol=1e-9,rtol=1e-9)
                dates,cube=cube_from_scores(s,fold['valid_end'],fold['test_end'])
                observed=effects(cube.sum(axis=0));observed_days=cube[:,0,0]>0
                key={'protocol':protocol,'fold':fold['fold'],'target':target}
                metrics.append({**key,'alpha':alpha,'bandwidth':bw,'rows':len(s),'calendar_days':len(dates),
                                'observed_days':int(observed_days.sum()),'missing_days':';'.join(dates[~observed_days].strftime('%Y-%m-%d')),
                                'even_peer_pct':100*float((s.retained_peers%2==0).mean()),
                                'original_effect_pct':float(observed[0]),'harmonized_effect_pct':float(observed[1])})
                for day,stats in zip(dates,cube):
                    for j,name in enumerate(['static_midpoint','static_lower','distance']):
                        daily.append({**key,'day':str(day.date()),'method':name,'n':int(stats[j,0]),'sum':stats[j,1],'sumsq':stats[j,2]})
                for tur,g in s.groupby('turbine_id'):
                    sd=g[['static_midpoint','static_lower','distance']].std(ddof=0)
                    per_turbine.append({**key,'turbine_id':tur,'rows':len(g),
                        'original_effect_pct':100*(sd.distance/sd.static_midpoint-1),
                        'harmonized_effect_pct':100*(sd.distance/sd.static_lower-1)})
                for bd in [3,7,14]:
                    seed=20260901+10000*fi+1000*ki+100+bd
                    legacy,lp=boot(cube[observed_days],bd,seed)
                    saved=old_ci[(old_ci.fold==fold['fold'])&(old_ci.target==target)&(old_ci.contrast=='distance')&(old_ci.block_days==bd)].iloc[0]
                    np.testing.assert_allclose(legacy[:,0],saved[['ci95_low_pct','bootstrap_median_std_change_pct','ci95_high_pct']].to_numpy(dtype=float),atol=1e-7,rtol=1e-7)
                    corrected,cp=boot(cube,bd,seed) if not observed_days.all() else (legacy,lp)
                    for name,values,prob,j in [('legacy_observed_days',legacy,lp,0),('calendar_original_median',corrected,cp,0),('calendar_harmonized_median',corrected,cp,1)]:
                        intervals.append({**key,'block_days':bd,'variant':name,'effect_pct':float(observed[j]),
                                          'ci_low_pct':float(values[0,j]),'ci_high_pct':float(values[2,j]),'bootstrap_probability_lower':float(prob[j])})
                repro.append({**key,'old_spreads_and_counts':'passed','old_ci_3_7_14_days':'passed','vectorized_sample':'passed'})
                print(f'{protocol} {fold["fold"]} {target}: {observed[0]:.3f}% -> {observed[1]:.3f}%; missing days {int((~observed_days).sum())}',flush=True)
        # Checkpoint results after each protocol.
        for name,rows in [('fold_effects',metrics),('intervals',intervals),('daily_statistics',daily),('per_turbine_effects',per_turbine),('reproduction',repro)]:
            pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)


def common(data,out):
    turbines=sorted(data.turbine_id.unique());turbine=data.turbine_id.to_numpy();time=data.timestamp
    x,_,eligible=prepare(data,'met_temperature_mean',turbines)
    eligible &= time.ge('2015-05-14 12:00').to_numpy()
    source=SOURCE/'data/experiment_results/health_twin_all_peer_alert_stability_v2'
    config=pd.read_csv(source/'selected_configuration.csv').set_index('target')
    raw_period={};score_period={}
    for period,start,end in [('calibration','2015-10-01','2015-11-01'),('test','2015-11-01','2016-01-01')]:
        frames=[]
        # Frozen alert models: train May 14--Aug 31, alpha selected in September.
        for target in TARGETS:
            y=data[target].to_numpy();usable=eligible&np.isfinite(y)
            train=usable&time.lt('2015-09-01').to_numpy()
            model,_=fit_frozen(x,y,turbine,train,float(config.loc[target,'normal_model_alpha']))
            mask=usable&time.ge(start).to_numpy()&time.lt(end).to_numpy()
            frames.append(pd.DataFrame({'target':target,'timestamp':time[mask].to_numpy(),
                                       'turbine_id':turbine[mask],'residual':y[mask]-predict(model,x[mask])}))
        raw=pd.concat(frames,ignore_index=True)
        score=pd.read_csv(source/f'{period}_alert_states.csv.gz',usecols=['target','timestamp','turbine_id','residual'],parse_dates=['timestamp'])
        check=score.merge(raw,on=['target','timestamp','turbine_id'],suffixes=('_saved','_new'),validate='one_to_one')
        assert len(check)==len(score)
        np.testing.assert_allclose(check.residual_saved,check.residual_new,atol=1e-10,rtol=1e-10)
        raw_period[period]=raw;score_period[period]=score
    summaries=[];thresholds=[]
    for variant in ['original_scoreable','scoreable_min4','eligible_min4']:
        states={}
        for period in ['calibration','test']:
            raw=raw_period[period] if variant=='eligible_min4' else score_period[period]
            s=raw.groupby(['target','timestamp'],as_index=False).agg(available_turbines=('turbine_id','nunique'),common_state=('residual','median'))
            if variant!='original_scoreable':s=s[s.available_turbines>=4]
            states[period]=s
        th=calibrate_common(states['calibration']);assert th.estimable.all()
        thresholds.append(th.assign(variant=variant))
        for period in ['calibration','test']:
            a=apply_common_thresholds(states[period],th)
            a.to_csv(out/f'common_{variant}_{period}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
            if variant=='original_scoreable':
                saved=pd.read_csv(source/f'{period}_common_alert_states.csv.gz',parse_dates=['timestamp'])
                z=a.merge(saved,on=['target','timestamp'],validate='one_to_one',suffixes=('_new','_old'))
                assert len(z)==len(a)==len(saved)
                for col in ['common_state','common_centre','common_absolute_deviation_threshold','point_flag','persistent_flag']:
                    np.testing.assert_allclose(z[col+'_new'],z[col+'_old'],atol=1e-10,rtol=1e-10)
            for target,g in a.groupby('target'):
                summaries.append({'variant':variant,'period':period,'target':target,'rows':len(g),'min_turbines':int(g.available_turbines.min()),
                                  'point_flags':int(g.point_flag.sum()),'point_rate_pct':100*g.point_flag.mean(),
                                  'persistent_rate_pct':100*g.persistent_flag.mean(),'threshold':float(g.common_absolute_deviation_threshold.iloc[0])})
    pd.DataFrame(summaries).to_csv(out/'common_summary.csv',index=False)
    pd.concat(thresholds,ignore_index=True).to_csv(out/'common_thresholds.csv',index=False)
    print('Common-state definitions and frozen residual reproduction complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['rolling','common'],required=True)
    args=parser.parse_args();out=ROOT/args.part
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    unit=unit_checks();out.mkdir()
    path=SOURCE/'data/research_layers/health_twin_base_v1.csv.gz'
    manifest=json.loads(path.with_name('health_twin_base_v1_manifest.json').read_text())
    digest=sha(path);assert digest==manifest['output_sha256']
    d=pd.read_csv(path,usecols=SOURCE_COLUMNS,parse_dates=['timestamp']).sort_values(['timestamp','turbine_id'],kind='stable').reset_index(drop=True)
    (rolling if args.part=='rolling' else common)(d,out)
    provenance={'unit_checks':unit,'input_sha256':digest,'runner_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'PROTOCOL.md'),
                'source_scripts_sha256':{p.name:sha(p) for p in sorted((SOURCE/'scripts').glob('*.py'))},
                'outputs':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out/'verification.json').write_text(json.dumps(provenance,indent=2)+'\n')
