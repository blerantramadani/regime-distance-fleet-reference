"""Tail diagnostics for Section 3.6: October thresholds, threshold/SD ratio and excess kurtosis per reference (gear oil, generator bearings G and R)."""
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
    out={}
    for name,per in [("cal",ca),("test",te)]:
        m=us&per
        fr=rd.residual_frame(data,y[m]-rd.predict(f["model"],X[m]),m,D); raw=rd.residual_frame(data,y[m],m,D)
        out[name]=rd.rc.states(fr,raw,f["residual_scales"],bw,turbines)
    (ct,cv,cs),(tt,tv,ts)=out["cal"],out["test"]
    for ref in ["distance","static_trim","fleet_median_residual"]:
        sds,thr,kurt,sdt,q995t=[],[],[],[],[]
        for i in range(16):
            x=cs[ref][cv[:,i],i]; c=np.median(x); d=np.abs(x-c)
            sds.append(np.std(x)); thr.append(np.quantile(d,0.995))
            z=(x-x.mean())/x.std(); kurt.append(np.mean(z**4)-3)
            xt=ts[ref][tv[:,i],i]; sdt.append(np.std(xt)); q995t.append(np.quantile(np.abs(xt-c),0.995))
        rows.append({"target":target,"ref":ref,"cal_sd_med":np.median(sds),"cal_thr_med":np.median(thr),"thr_over_sd":np.median(np.array(thr)/np.array(sds)),"excess_kurtosis_med":np.median(kurt),"test_sd_med":np.median(sdt),"test_q995_med":np.median(q995t)})
    print(target, flush=True)
r=pd.DataFrame(rows); print(r.round(3).to_string(index=False))
r.to_csv(Path(__file__).resolve().parent / "diagnostic_tails.csv", index=False)
