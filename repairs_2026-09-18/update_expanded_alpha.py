"""Synchronize frozen expanded-alpha M2 intervals with calendar bootstrap."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
THERMAL=ROOT
sys.path.insert(0,str(THERMAL))
from run_thermal_checks import SOURCE,SOURCE_COLUMNS,TARGETS,prepare,fit_frozen,predict,residual_frame,score_dense,cube_from_scores,boot,effects,sha

out=ROOT/'expanded_alpha_M2';out.mkdir(exist_ok=False)
p=SOURCE/'data/research_layers/health_twin_base_v1.csv.gz'
d=pd.read_csv(p,usecols=SOURCE_COLUMNS,parse_dates=['timestamp']).sort_values(['timestamp','turbine_id'],kind='stable').reset_index(drop=True)
tur=sorted(d.turbine_id.unique());x,raw,eligible=prepare(d,'met_temperature_mean',tur)
train=d.timestamp.ge('2015-05-14 12:00').to_numpy()&d.timestamp.lt('2015-09-01').to_numpy()
test=d.timestamp.ge('2015-10-01').to_numpy()&d.timestamp.lt('2015-11-01').to_numpy()
scale=np.std(raw[eligible&train],axis=0);scale=np.where(scale>1e-8,scale,1.)
desc=(raw-np.mean(raw[eligible&train],axis=0))/scale
path=SOURCE/'data/experiment_results/health_twin_alpha_boundary_sensitivity_independent_v1/fold_target_sensitivity.csv'
locked=pd.read_csv(path);updated=locked.copy();rows=[]
primary_ci=pd.read_csv(THERMAL/'rolling/intervals.csv')
for ki,target in enumerate(TARGETS):
    idx=locked.index[(locked.fold=='M2')&(locked.target==target)][0];old=locked.loc[idx]
    y=d[target].to_numpy();usable=eligible&np.isfinite(y);mask=usable&test
    model,scales=fit_frozen(x,y,d.turbine_id.to_numpy(),usable&train,float(old.expanded_alpha))
    f=residual_frame(d,y[mask]-predict(model,x[mask]),mask,desc)
    s=score_dense(f,scales,float(old.expanded_bandwidth),tur)
    assert len(s)==old.score_rows
    dates,cube=cube_from_scores(s,'2015-10-01','2015-11-01')
    observed=effects(cube.sum(axis=0));np.testing.assert_allclose(observed[0],old.expanded_distance_std_change_pct,atol=1e-9)
    seed=20260901+10000+1000*ki+100+7
    legacy,_=boot(cube[cube[:,0,0]>0],7,seed);new,_=boot(cube,7,seed)
    np.testing.assert_allclose(legacy[[0,2],0],old[['expanded_ci95_low_pct','expanded_ci95_high_pct']].to_numpy(float),atol=1e-7)
    base=primary_ci[(primary_ci.fold=='M2')&(primary_ci.target==target)&(primary_ci.block_days==7)&(primary_ci.variant=='calendar_original_median')].iloc[0]
    updated.loc[idx,['locked_ci95_low_pct','locked_ci95_high_pct']]=[base.ci_low_pct,base.ci_high_pct]
    updated.loc[idx,['expanded_ci95_low_pct','expanded_ci95_high_pct']]=new[[0,2],0]
    rows.append({'target':target,'old_low':float(legacy[0,0]),'old_high':float(legacy[2,0]),'calendar_low':float(new[0,0]),'calendar_high':float(new[2,0]),'effect_pct':float(observed[0])})
pd.DataFrame(rows).to_csv(out/'M2_calendar_comparison.csv',index=False)
updated.to_csv(out/'fold_target_sensitivity_calendar.csv',index=False)
(out/'verification.json').write_text(json.dumps({'original_M2_effects_and_intervals_reproduced':True,'input_sha256':sha(p),'source_results_sha256':sha(path),'runner_sha256':sha(__file__),'helper_sha256':sha(THERMAL/'run_thermal_checks.py'),'outputs':{q.name:sha(q) for q in out.iterdir() if q.is_file()}},indent=2)+'\n')
print(pd.DataFrame(rows).to_string(index=False))
