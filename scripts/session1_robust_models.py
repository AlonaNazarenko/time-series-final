"""Broader rolling comparison; no hidden targets or external data are used."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from session1_models import baseline, metrics, interval
from session1_more_models import predictor, perturb, features, compare_windows
ROOT=Path(__file__).resolve().parent.parent
CONFIGS={
 'seasonal_naive':dict(kind='baseline'),
 'SARIMA_21d':dict(kind='SARIMA',days=21),
 'SARIMA_MA_21d':dict(kind='sarima_new',days=21,order=[1,0,1],seasonal=[0,1,1,24]),
 'SARIMA_ARMA_21d':dict(kind='sarima_new',days=21,order=[1,0,1],seasonal=[1,1,1,24]),
 'SARIMA_diff_21d':dict(kind='sarima_new',days=21,order=[0,1,1],seasonal=[0,1,1,24]),
 'weather_ridge_21d':dict(kind='ridge',days=21,weather=True),
 'weather_boosting_60d':dict(kind='boost',days=60,weather=True),
 'hour_weather_ridge_7d':dict(kind='hour_ridge',days=7,alpha=20),
 'hour_weather_ridge_14d':dict(kind='hour_ridge',days=14,alpha=20),
 'hour_weather_ridge_28d':dict(kind='hour_ridge',days=28,alpha=20),
 'hour_weather_ridge_60d':dict(kind='hour_ridge',days=60,alpha=20),
 'weather_correction_14d':dict(kind='correction',days=14,alpha=20),
 'weather_correction_28d':dict(kind='correction',days=28,alpha=20),
 'residual_boosting_28d':dict(kind='residual_boost',days=28),
 'residual_boosting_60d':dict(kind='residual_boost',days=60),
}
BLENDS={
 'blend_SARIMA75_ridge25':{'SARIMA_21d':.75,'weather_ridge_21d':.25},
 'blend_SARIMA50_ridge50':{'SARIMA_21d':.5,'weather_ridge_21d':.5},
 'blend_SARIMA75_hour25':{'SARIMA_21d':.75,'hour_weather_ridge_14d':.25},
 'blend_SARIMA50_hour50':{'SARIMA_21d':.5,'hour_weather_ridge_14d':.5},
 'blend_SARIMA50_correction50':{'SARIMA_21d':.5,'weather_correction_14d':.5},
 'blend_three':{'SARIMA_21d':.5,'hour_weather_ridge_14d':.25,'residual_boosting_28d':.25},
}


def hour_features(frame):
    f=pd.DataFrame(index=frame.index)
    hour=frame.ds.dt.hour
    for h in range(24):
        gate=(hour==h).astype(float)
        f[f'h{h}']=gate
        f[f'cloud_h{h}']=gate*frame.cloud_cover
        f[f'cloud2_h{h}']=gate*frame.cloud_cover**2
    f['wind']=frame.wind_speed
    f['wind2']=frame.wind_speed**2
    f['wind3density']=frame.wind_speed**3*frame.air_density
    f['temperature']=frame.temperature
    f['humidity']=frame.humidity
    f['pressure']=frame.pressure
    f['daylight']=frame.daylight
    return f


def make_predictor(data,origin,cfg):
    kind=cfg['kind']; hist=data.iloc[:origin]
    if kind not in ['sarima_new','hour_ridge','correction','residual_boost']:
        return predictor(data,origin,cfg),[]
    recent=hist.tail(24*cfg['days'])
    if kind=='sarima_new':
        scale=max(recent.y.std(),1)
        model=SARIMAX(recent.y.to_numpy()/scale,order=cfg['order'],seasonal_order=cfg['seasonal'],
                      enforce_stationarity=False,enforce_invertibility=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            fit=model.fit(disp=False,maxiter=250)
            if not fit.mle_retvals.get('converged',True):
                fit=model.fit(start_params=fit.params,method='powell',disp=False,maxiter=400)
        if not fit.mle_retvals.get('converged',True): raise RuntimeError('Nonconverged SARIMA')
        p=np.maximum(fit.forecast(48)*scale,0)
        return lambda frame:p,[str(w.message) for w in caught]
    if kind in ['hour_ridge','correction']:
        model=make_pipeline(StandardScaler(),Ridge(alpha=cfg['alpha']))
        model.fit(hour_features(recent),recent.y)
        if kind=='hour_ridge': return lambda frame:np.maximum(model.predict(hour_features(frame)),0),[]
        # Learn expected weather effect, then anchor to the actual last day.
        # Both future days use the same known last-day reference; never actual future y.
        reference=pd.concat([hist.tail(24),hist.tail(24)],ignore_index=True)
        fitted_reference=model.predict(hour_features(reference))
        lastday=reference.y.to_numpy()
        return lambda frame:np.maximum(lastday+model.predict(hour_features(frame))-fitted_reference,0),[]
    x=features(recent,hist.y,True)
    model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=10,learning_rate=.05,
                                       l2_regularization=20,early_stopping=False,random_state=42)
    model.fit(x,recent.y.to_numpy()-x.lag48.to_numpy())
    def predict(frame):
        f=features(frame,hist.y,True)
        return np.maximum(f.lag48.to_numpy()+model.predict(f),0)
    return predict,[]


def get_predictions(data,origin,variants,diagnostics):
    allp={}
    for name,cfg in CONFIGS.items():
        predict,notes=make_predictor(data,origin,cfg)
        if notes: diagnostics.append(dict(origin=origin,model=name,notes=notes))
        allp[name]={}
        for scenario,seed,frame in variants:
            p=np.asarray(predict(frame))
            assert p.shape==(48,) and np.isfinite(p).all() and (p>=0).all()
            allp[name][scenario,seed]=p
    for name,weights in BLENDS.items():
        allp[name]={key:sum(w*allp[m][key] for m,w in weights.items()) for key in allp['seasonal_naive']}
    return allp


def run():
    out=ROOT/'results/robust';out.mkdir(exist_ok=True)
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds']);y=data.y.to_numpy()
    future=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds']);future.index=np.arange(len(data),len(data)+48)
    assert (y>0).all() and np.isfinite(y).all() and data.ds.is_unique
    assert data.ds.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    assert future.ds.tolist()==list(pd.date_range(data.ds.iloc[-1]+pd.Timedelta(hours=1),periods=48,freq='h'))
    scenarios=['observed_weather','weather_stress']
    names=list(CONFIGS)+list(BLENDS)
    calibration={(name,s):[] for name in names for s in scenarios}
    predictions=[];diagnostics=[]
    for fold,origin in enumerate(range(len(data)-18*48,len(data),48)):
        phase='calibration' if fold<6 else ('selection' if fold<17 else 'last_window_audit')
        target=data.iloc[origin:origin+48];actual=target.y.to_numpy()
        # Origin-based seeds keep scenarios unchanged if the number of folds changes.
        variants=[('observed_weather',0,target)]+[('weather_stress',seed,perturb(target,origin*10+seed)) for seed in [1,2,3]]
        allp=get_predictions(data,origin,variants,diagnostics)
        for name in names:
            for (scenario,seed),p in allp[name].items():
                if phase=='calibration': calibration[name,scenario].append(actual-p); continue
                if name=='seasonal_naive': _,lo,hi=baseline(y[:origin])
                else: lo,hi=interval(p,calibration[name,scenario])
                for h in range(48):
                    predictions.append(dict(model=name,scenario=scenario,seed=seed,phase=phase,fold=fold,
                        ds=target.ds.iloc[h],horizon=h+1,actual=actual[h],forecast=p[h],lower_95=lo[h],upper_95=hi[h]))
        print(f'{fold+1}/18 {phase}: {target.ds.iloc[0]}',flush=True)
    pred=pd.DataFrame(predictions)
    windows,consistency=compare_windows(pred,out)
    rows=[]
    for (scenario,phase,name),g in pred.groupby(['scenario','phase','model']):
        rows.append(dict(scenario=scenario,phase=phase,model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    summary=pd.DataFrame(rows)
    for (scenario,phase),g in summary.groupby(['scenario','phase']):
        base=g[g.model=='seasonal_naive'].RMSE.iloc[0]
        summary.loc[g.index,'RMSE_ratio']=g.RMSE/base
    summary=summary.sort_values(['phase','scenario','RMSE'])
    candidates=summary[(summary.scenario=='weather_stress')&(summary.phase=='selection')].copy()
    # A fixed rule emphasizing Session 1 RMSE, while checking consistency.
    winner=candidates.iloc[0].model
    final=get_predictions(data,len(data),[('observed_weather',0,future)],diagnostics)
    p=final[winner]['observed_weather',0]
    errors=list(calibration[winner,'weather_stress'])
    for _,g in pred[(pred.model==winner)&(pred.scenario=='weather_stress')].groupby(['fold','seed']):
        errors.append(g.actual.to_numpy()-g.forecast.to_numpy())
    if winner=='seasonal_naive': _,lo,hi=baseline(y)
    else: lo,hi=interval(p,errors)
    sub=pd.DataFrame(dict(ds=future.ds.dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy(),forecast=p,lower_95=lo,upper_95=hi))
    assert np.isfinite(sub.iloc[:,1:]).all().all() and (sub.lower_95>=0).all()
    assert (sub.lower_95<=sub.forecast).all() and (sub.forecast<=sub.upper_95).all()
    sub.to_csv(out/'selected_model_forecast.csv',index=False)
    # Save model-specific forecasts for review without replacing previous submissions.
    for name in names:
        point=final[name]['observed_weather',0]
        pd.DataFrame({'ds':sub.ds,'forecast':point}).to_csv(out/f'point_{name}.csv',index=False)
    pred.to_csv(out/'validation_predictions.csv',index=False)
    summary.to_csv(out/'model_comparison.csv',index=False)
    # Report day two separately because the final private leaderboard uses those hours.
    day_rows=[]
    for (scenario,name),g in pred[(pred.phase=='selection')&(pred.horizon>24)].groupby(['scenario','model']):
        day_rows.append(dict(scenario=scenario,model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    pd.DataFrame(day_rows).to_csv(out/'day_two_comparison.csv',index=False)
    report=dict(selected_model=winner,configs=CONFIGS,blends=BLENDS,calibration_windows=6,selection_windows=11,
                selection_rule='Lowest pooled 48-hour RMSE under illustrative weather stress; audit excluded; baseline eligible.',
                weather_stress='Same illustrative perturbation magnitudes as session1_more_models.py; seeds depend on origin, three per window. Scenarios are not independent historical windows.',
                caveats=['Expanded selection includes six additional earlier windows. Previously inspected windows and audit are not independent tests.',
                         'Weather stress is hypothetical, not calibrated to real forecast errors.',
                         'Nominal 95% intervals use earlier absolute errors; no future coverage guarantee.',
                         'No hidden targets are available; hidden-leaderboard success cannot be guaranteed.'],fit_diagnostics=diagnostics)
    (out/'run_summary.json').write_text(json.dumps(report,indent=2))
    print('\n',candidates[['model','RMSE','RMSE_ratio','MAPE','Winkler','coverage']].to_string(index=False),flush=True)
    print('\nSELECTED:',winner,flush=True)

if __name__=='__main__':run()
