"""Additional models with observed-weather and illustrative weather-error validation."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from session1_models import baseline, fit_forecast, interval, metrics
ROOT=Path(__file__).resolve().parent
CONFIGS={
 'seasonal_mean_3d':dict(kind='mean',days=3),
 'seasonal_mean_7d':dict(kind='mean',days=7),
 'calendar_ridge_21d':dict(kind='ridge',days=21,weather=False),
 'weather_ridge_21d':dict(kind='ridge',days=21,weather=True),
 'weather_ridge_60d':dict(kind='ridge',days=60,weather=True),
 'weather_boosting_60d':dict(kind='boost',days=60,weather=True),
 'weather_boosting_120d':dict(kind='boost',days=120,weather=True),
 'weather_extra_trees_60d':dict(kind='extra',days=60,weather=True),
 'weather_random_forest_60d':dict(kind='forest',days=60,weather=True),
 'SARIMA_101_110_24_21d':dict(kind='SARIMA',days=21),
 'seasonal_naive':dict(kind='baseline'),
}


def features(frame,history,weather):
    f=pd.DataFrame(index=frame.index)
    for k in range(1,5):
        f[f'sin{k}']=np.sin(2*np.pi*k*frame.ds.dt.hour/24)
        f[f'cos{k}']=np.cos(2*np.pi*k*frame.ds.dt.hour/24)
    f['weekend']=(frame.ds.dt.dayofweek>=5).astype(int)
    # All target lags are at least 48h: even horizon 48 is known at the forecast origin.
    for lag in [48,72,168]:
        f[f'lag{lag}']=history.reindex(frame.index-lag).to_numpy()
    if weather:
        for c in ['cloud_cover','temperature','wind_speed','pressure','humidity','daylight','air_density']:
            f[c]=frame[c]
        f['wind2']=frame.wind_speed**2
        f['wind3']=frame.wind_speed**3*frame.air_density
        f['clear_day']=frame.daylight*(1-frame.cloud_cover)
        for k in range(1,5):
            for trig in ['sin','cos']:
                f[f'solar_{trig}{k}']=f[f'{trig}{k}']*f.clear_day
    return f


def perturb(frame,seed):
    """Illustrative assumptions, NOT estimated forecast-error distribution."""
    rng=np.random.default_rng(seed); f=frame.copy(); day=np.repeat([1.,1.5],24)
    # A shared error per forecast day plus hourly noise creates temporal dependence.
    def noise(scale):
        return scale*day*(np.repeat(rng.normal(size=2),24)*.7+rng.normal(size=48)*.3)
    f['temperature']+=noise(2)
    f['wind_speed']=np.maximum(0,f.wind_speed+noise(1))
    f['cloud_cover']=np.round(np.clip(f.cloud_cover+noise(.25),0,1)*4)/4
    f['pressure']+=noise(2)
    f['humidity']=np.clip(f.humidity+noise(5),0,100)
    f['air_density']*=((f.pressure/frame.pressure)*(frame.temperature+273.15)/(f.temperature+273.15))
    return f


def predictor(data,origin,cfg):
    history=data.y.iloc[:origin]
    kind=cfg['kind']
    if kind=='baseline':
        p=baseline(history.to_numpy())[0]; return lambda frame:p
    if kind=='mean':
        p=np.tile(history.iloc[-24*cfg['days']:].to_numpy().reshape(-1,24).mean(axis=0),2)
        return lambda frame:p
    if kind=='SARIMA':
        p,_=fit_forecast(history.to_numpy(),cfg); return lambda frame:p
    train=data.iloc[max(168,origin-cfg['days']*24):origin]
    x=features(train,history,cfg['weather'])
    if kind=='ridge': model=make_pipeline(StandardScaler(),Ridge(alpha=100))
    elif kind=='boost': model=HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=15,learning_rate=.06,l2_regularization=10,early_stopping=False,random_state=42)
    elif kind=='extra': model=ExtraTreesRegressor(n_estimators=120,min_samples_leaf=3,max_features=.9,n_jobs=1,random_state=42)
    else: model=RandomForestRegressor(n_estimators=100,min_samples_leaf=3,max_features=.9,n_jobs=1,random_state=42)
    model.fit(x,train.y)
    return lambda frame:np.maximum(model.predict(features(frame,history,cfg['weather'])),0)


def compare_windows(pred, out):
    """Pool weather scenarios within each actual window, never count seeds as windows."""
    rows=[]
    for (scenario,phase,fold,name),g in pred.groupby(['scenario','phase','fold','model']):
        rows.append(dict(scenario=scenario,phase=phase,fold=fold,model=name,
                         start=pd.to_datetime(g.ds).min(),end=pd.to_datetime(g.ds).max(),
                         hours=g.ds.nunique(),scenarios=g.seed.nunique(),
                         **metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    windows=pd.DataFrame(rows)
    keys=['scenario','phase','fold']
    base=windows[windows.model=='seasonal_naive'][keys+['RMSE']].rename(columns={'RMSE':'baseline_RMSE'})
    windows=windows.merge(base,on=keys,validate='many_to_one')
    windows['RMSE_ratio']=windows.RMSE/windows.baseline_RMSE
    windows['improvement_pct']=100*(1-windows.RMSE_ratio)
    windows['beats_baseline']=windows.RMSE < windows.baseline_RMSE
    windows['RMSE_rank']=windows.groupby(keys).RMSE.rank(method='min')
    consistency=windows[windows.phase=='selection'].groupby(['scenario','model']).agg(
        windows=('fold','nunique'),wins=('beats_baseline','sum'),
        mean_RMSE=('RMSE','mean'),median_RMSE_ratio=('RMSE_ratio','median'),
        best_RMSE_ratio=('RMSE_ratio','min'),worst_RMSE_ratio=('RMSE_ratio','max'),
        mean_rank=('RMSE_rank','mean')).reset_index()
    consistency['win_rate']=consistency.wins/consistency.windows
    consistency=consistency.sort_values(['scenario','wins','mean_rank'],ascending=[True,False,True])
    windows.to_csv(out/'window_comparison.csv',index=False)
    consistency.to_csv(out/'window_consistency.csv',index=False)
    for scenario in windows.scenario.unique():
        g=windows[windows.scenario==scenario]
        for metric in ['RMSE','RMSE_ratio']:
            g.pivot(index='model',columns=['phase','start'],values=metric).to_csv(out/f'windows_{scenario}_{metric}.csv')
    return windows,consistency


def run():
    out=ROOT/'results_more'; out.mkdir(exist_ok=True)
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds'])
    future=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds']); future.index=np.arange(len(data),len(data)+48)
    y=data.y.to_numpy(); assert np.isfinite(y).all() and (y>0).all()
    assert data.ds.diff().dropna().eq(pd.Timedelta(hours=1)).all() and data.ds.is_unique
    assert future.ds.tolist()==list(pd.date_range(data.ds.iloc[-1]+pd.Timedelta(hours=1),periods=48,freq='h'))
    scenarios=['observed_weather','weather_stress']
    calibration={(name,s):[] for name in CONFIGS for s in scenarios}
    predictions=[]
    for fold,origin in enumerate(range(len(data)-12*48,len(data),48)):
        phase='calibration' if fold<6 else ('selection' if fold<11 else 'last_window_audit')
        target=data.iloc[origin:origin+48]; actual=target.y.to_numpy()
        variants=[('observed_weather',0,target)]+[('weather_stress',seed,perturb(target,1000*fold+seed)) for seed in [1,2,3]]
        for name,cfg in CONFIGS.items():
            predict=predictor(data,origin,cfg)
            for scenario,seed,frame in variants:
                p=predict(frame)
                assert p.shape==(48,) and np.isfinite(p).all()
                if phase=='calibration': calibration[name,scenario].append(actual-p); continue
                if name=='seasonal_naive': _,lo,hi=baseline(y[:origin])
                else: lo,hi=interval(p,calibration[name,scenario])
                for h in range(48):
                    predictions.append(dict(model=name,scenario=scenario,seed=seed,phase=phase,fold=fold,ds=target.ds.iloc[h],horizon=h+1,actual=actual[h],forecast=p[h],lower_95=lo[h],upper_95=hi[h]))
        print(f'{fold+1}/12 {phase} complete',flush=True)
    pred=pd.DataFrame(predictions)
    compare_windows(pred,out)
    rows=[]
    for (scenario,phase,name),g in pred.groupby(['scenario','phase','model']):
        rows.append(dict(scenario=scenario,phase=phase,model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    summary=pd.DataFrame(rows)
    for (scenario,phase),g in summary.groupby(['scenario','phase']):
        base=g[g.model=='seasonal_naive'].RMSE.iloc[0]
        summary.loc[g.index,'RMSE_ratio']=g.RMSE/base
    summary=summary.sort_values(['phase','scenario','RMSE'])
    # Prioritize robustness under stated weather assumptions, not observed weather alone.
    candidates=summary[(summary.scenario=='weather_stress')&(summary.phase=='selection')]
    winner=candidates.iloc[0].model
    p=predictor(data,len(data),CONFIGS[winner])(future)
    errors=list(calibration[winner,'weather_stress'])
    for _,g in pred[(pred.model==winner)&(pred.scenario=='weather_stress')].groupby(['fold','seed']):
        errors.append(g.actual.to_numpy()-g.forecast.to_numpy())
    if winner=='seasonal_naive': _,lo,hi=baseline(y)
    else: lo,hi=interval(p,errors)
    submission=pd.DataFrame(dict(ds=future.ds.dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy(),forecast=p,lower_95=lo,upper_95=hi))
    assert len(submission)==48 and np.isfinite(submission.iloc[:,1:]).all().all()
    assert (submission.lower_95>=0).all() and (submission.lower_95<=submission.forecast).all() and (submission.forecast<=submission.upper_95).all()
    summary.to_csv(out/'model_comparison.csv',index=False)
    pred.to_csv(out/'validation_predictions.csv',index=False)
    submission.to_csv(out/'selected_model_forecast.csv',index=False)
    report=dict(selected_model=winner,configs=CONFIGS,selection='Lowest pooled RMSE under illustrative weather stress on five reused selection windows',
                weather_stress='Three seeded scenarios per origin; shared daily plus hourly errors. Day-one scales: temperature 2 C, wind 1 m/s, cloud .25, pressure 2 mmHg, humidity 5 percentage points. Day-two scales multiplied by 1.5. Daylight unchanged; density adjusted for temperature/pressure.',
                caveats=['Stress assumptions are illustrative, not measured forecast uncertainty.','Observed-weather validation is optimistic for weather models.','Selection windows were reused after earlier model comparisons; selection optimism is possible.','The last window was already seen; it is an audit, not an untouched test.','Three stress scenarios do not create three independent historical windows.','Empirical nominal 95% intervals do not guarantee future coverage.'])
    (out/'run_summary.json').write_text(json.dumps(report,indent=2))
    print(summary[summary.phase=='selection'].to_string(index=False),flush=True)
    print('SELECTED',winner,flush=True)

if __name__=='__main__': run()
