"""Apply the same frozen calendar repair to all four original M2 contrasts."""
import json
import numpy as np
import pandas as pd
from run_thermal_checks import (ROOT,SOURCE,SOURCE_COLUMNS,TARGETS,prepare,fit_frozen,
                               predict,residual_frame,factorial_scores,sha)

METHODS=['static_no_trim','static_trim','distance_trim','distance_trim_reliability']
CONTRASTS=[('trim',1,0),('distance',2,1),('reliability',3,2),('end_to_end',3,0)]


def draw(cube,num,den,block,seed):
    rng=np.random.default_rng(seed);n=len(cube);draws=[]
    for _ in range(2000):
        ids=[]
        while len(ids)<n:
            start=int(rng.choice(np.arange(n-block+1)));ids.extend(range(start,start+block))
        stats=cube[np.array(ids[:n])].sum(axis=0)
        assert (stats[:,0]>0).all()
        sd=np.sqrt(np.maximum(stats[:,2]/stats[:,0]-(stats[:,1]/stats[:,0])**2,0))
        draws.append(100*(sd[num]/sd[den]-1))
    return np.quantile(draws,[.025,.5,.975])


if __name__=='__main__':
    out=ROOT/'mechanism_M2';out.mkdir(exist_ok=False)
    p=SOURCE/'data/research_layers/health_twin_base_v1.csv.gz'
    d=pd.read_csv(p,usecols=SOURCE_COLUMNS,parse_dates=['timestamp']).sort_values(['timestamp','turbine_id'],kind='stable').reset_index(drop=True)
    tur=sorted(d.turbine_id.unique());x,raw,eligible=prepare(d,'met_temperature_mean',tur)
    train=d.timestamp.ge('2015-05-14 12:00').to_numpy()&d.timestamp.lt('2015-09-01').to_numpy()
    test=d.timestamp.ge('2015-10-01').to_numpy()&d.timestamp.lt('2015-11-01').to_numpy()
    scale=np.std(raw[eligible&train],axis=0);scale=np.where(scale>1e-8,scale,1.)
    desc=(raw-np.mean(raw[eligible&train],axis=0))/scale
    path=SOURCE/'data/experiment_results/health_twin_trim_distance_ablation_independent_v1'
    locked=pd.read_csv(path/'fold_target_method_results.csv');old_ci=pd.read_csv(path/'mechanism_contrasts.csv')
    results=[];daily=[];dates=pd.date_range('2015-10-01','2015-10-31')
    for ki,target in enumerate(TARGETS):
        lock=locked[(locked.fold=='M2')&(locked.target==target)].set_index('method')
        y=d[target].to_numpy();usable=eligible&np.isfinite(y);mask=usable&test
        model,scales=fit_frozen(x,y,d.turbine_id.to_numpy(),usable&train,float(lock.loc['distance_trim','normal_model_alpha']))
        f=residual_frame(d,y[mask]-predict(model,x[mask]),mask,desc)
        scores=factorial_scores(f,target,scales,float(lock.loc['distance_trim','selected_distance_bandwidth']))
        cube=np.zeros((31,4,3))
        for j,method in enumerate(METHODS):
            s=scores[scores.method==method];v=s.specific_residual.to_numpy();di=dates.get_indexer(s.timestamp.dt.floor('D'))
            assert len(s)==int(lock.loc[method,'score_rows'])
            np.testing.assert_allclose(np.std(v),lock.loc[method,'specific_std'],atol=1e-9)
            cube[:,j,0]=np.bincount(di,minlength=31);cube[:,j,1]=np.bincount(di,weights=v,minlength=31);cube[:,j,2]=np.bincount(di,weights=v*v,minlength=31)
            for day,stats in zip(dates,cube[:,j,:]):daily.append({'target':target,'day':str(day.date()),'method':method,'n':int(stats[0]),'sum':stats[1],'sumsq':stats[2]})
        observed=cube[:,0,0]>0
        assert dates[~observed].strftime('%Y-%m-%d').tolist()==['2015-10-30']
        for ci,(name,num,den) in enumerate(CONTRASTS):
            for bd in [3,7,14]:
                seed=20260901+10000+1000*ki+100*ci+bd
                old=draw(cube[observed],num,den,bd,seed);new=draw(cube,num,den,bd,seed)
                saved=old_ci[(old_ci.fold=='M2')&(old_ci.target==target)&(old_ci.contrast==name)&(old_ci.block_days==bd)].iloc[0]
                np.testing.assert_allclose(old,saved[['ci95_low_pct','bootstrap_median_std_change_pct','ci95_high_pct']].to_numpy(float),atol=1e-7,rtol=1e-7)
                results.append({'target':target,'contrast':name,'block_days':bd,'effect_pct':saved.observed_std_change_pct,
                                'legacy_ci_low_pct':old[0],'legacy_ci_high_pct':old[2],
                                'calendar_ci_low_pct':new[0],'calendar_ci_high_pct':new[2]})
        print('All M2 contrasts corrected:',target,flush=True)
    pd.DataFrame(results).to_csv(out/'intervals.csv',index=False);pd.DataFrame(daily).to_csv(out/'daily_statistics.csv',index=False)
    (out/'verification.json').write_text(json.dumps({'original_60_intervals_reproduced':True,'input_sha256':sha(p),'runner_sha256':sha(__file__),
        'helper_sha256':sha(ROOT/'run_thermal_checks.py'),'outputs':{q.name:sha(q) for q in out.iterdir() if q.is_file()}},indent=2)+'\n')
