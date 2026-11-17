"""Section 3.6: median October 99.5th-percentile threshold of each reference in units of the training residual scale."""
from pathlib import Path
import importlib.util, sys, numpy as np, pandas as pd
spec = importlib.util.spec_from_file_location("rd", Path(__file__).resolve().parents[1] / "run_detection.py")
rd = importlib.util.module_from_spec(spec); spec.loader.exec_module(rd)
data = pd.read_csv(rd.rc.PROJECT / "data/research_layers/health_twin_base_v1.csv.gz", usecols=rd.SOURCE_COLUMNS, parse_dates=["timestamp"]).sort_values(["timestamp","turbine_id"], kind="stable").reset_index(drop=True)
amb = rd.AMBIENT_SOURCES["independent_meteorology"]["column"]
turbines = sorted(data.turbine_id.astype(str).unique()); tid = data.turbine_id.astype(str).to_numpy()
X,_ = rd.make_feature_matrix(data, turbines, amb); D,_ = rd.regime_matrix(data, amb)
el = (data.timestamp.ge(rd.MONITORING_START) & data.has_temperature_record & data.active_power_mean.between(0,2350) & data.wind_speed_mean.between(4,20) & data.generator_speed_mean.ge(100) & data.power_reference_end.ge(2200) & np.isfinite(X).all(1) & np.isfinite(D).all(1)).to_numpy()
t = data.timestamp
tr=(t.ge(rd.MONITORING_START)&t.lt(rd.TRAIN_END)).to_numpy(); tu=(t.ge(rd.TRAIN_END)&t.lt(rd.TUNING_END)).to_numpy(); ca=(t.ge(rd.TUNING_END)&t.lt(rd.CALIBRATION_END)).to_numpy(); te=(t.ge(rd.CALIBRATION_END)&t.lt(rd.TEST_END)).to_numpy()
mu=np.mean(D[el&tr],0); sc=np.std(D[el&tr],0); D=(D-mu)/np.where(sc>1e-8,sc,1)
cfg = pd.read_csv(rd.SAVED/"selected_configuration.csv").set_index("target")
rows=[]
for target in rd.TARGETS[:3]:
    y=data[target].to_numpy(float); us=el&np.isfinite(y)
    f=rd.fit_window(X,y,tid,us,tr,tu,ca); bw=float(cfg.loc[target,"selected_distance_bandwidth"])
    m=us&ca
    fr=rd.residual_frame(data,y[m]-rd.predict(f["model"],X[m]),m,D); raw=rd.residual_frame(data,y[m],m,D)
    ct,cv,cs=rd.rc.states(fr,raw,f["residual_scales"],bw,turbines)
    sc_=f["residual_scales"]
    for ref in ["distance","static_trim"]:
        ratio=[]
        for i,tb in enumerate(turbines):
            x=cs[ref][cv[:,i],i]; c=np.median(x); thr=np.quantile(np.abs(x-c),0.995)
            ratio.append(thr/float(sc_[tb]))
        rows.append({"target":target,"ref":ref,"median_thr_over_training_scale":float(np.median(ratio))})
    print(target, flush=True)
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).resolve().parent / "threshold_in_training_scales.csv", index=False)
print(out.round(3).to_string(index=False))
