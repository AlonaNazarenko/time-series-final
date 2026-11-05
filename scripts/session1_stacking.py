"""Chronological out-of-fold stacking; meta models never see their scoring-window targets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from session1_models import interval,metrics
from session1_more_models import perturb,compare_windows
from session1_robust_models import CONFIGS,make_predictor
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/stacking'
BASES=['SARIMA_21d','hour_weather_ridge_14d','weather_ridge_21d','weather_correction_28d','chronos_bolt_small_504h']
METHODS=['equal_mean','median','stack_convex','stack_convex_recent4','stack_ridge','stack_daywise','stack_shrunk']
SCENARIOS={'observed_weather':[0],'weather_stress':[1,2,3]}


def convex_weights(x,y):
    scale=max(float(np.std(y)),1.)
    a=x/scale;b=y/scale
    def loss(w):return np.mean((a@w-b)**2)
    def jac(w):return 2*a.T@(a@w-b)/len(b)
    fit=minimize(loss,np.ones(x.shape[1])/x.shape[1],jac=jac,method='SLSQP',
                 bounds=[(0,1)]*x.shape[1],constraints=[{'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones_like(w)}],
                 options={'maxiter':500,'ftol':1e-10})
    if not fit.success:raise RuntimeError(fit.message)
    assert np.min(fit.x)>-1e-8 and abs(fit.x.sum()-1)<1e-6
    return fit.x


def combine(method,past_x,past_y,future_x):
    """past arrays: [earlier_windows, scenarios, 48, models]; future: [scenarios,48,models]."""
    if method=='equal_mean':return future_x.mean(axis=-1),{'weights':dict(zip(BASES,[.2]*5))}
    if method=='median':return np.median(future_x,axis=-1),{}
    if method=='stack_convex_recent4':past_x,past_y=past_x[-4:],past_y[-4:]
    x=past_x.reshape(-1,len(BASES));y=past_y.reshape(-1)
    if method=='stack_ridge':
        model=make_pipeline(StandardScaler(),Ridge(alpha=100))
        model.fit(x,y)
        result=model.predict(future_x.reshape(-1,len(BASES))).reshape(future_x.shape[:-1])
        coef=model[-1].coef_/model[0].scale_
        intercept=model[-1].intercept_-np.dot(coef,model[0].mean_)
        return np.maximum(result,0),{'coefficients':dict(zip(BASES,coef.tolist())),'intercept':float(intercept)}
    if method=='stack_daywise':
        result=np.empty(future_x.shape[:-1]);details={}
        for day in [0,1]:
            s=slice(day*24,(day+1)*24)
            w=convex_weights(past_x[:,:,s,:].reshape(-1,len(BASES)),past_y[:,:,s].reshape(-1))
            result[:,s]=future_x[:,s,:]@w;details[f'day{day+1}']=dict(zip(BASES,w.tolist()))
        return result,details
    w=convex_weights(x,y)
    if method=='stack_shrunk':w=.5*w+.5*np.array([.5,.5,0,0,0])
    return np.maximum(future_x@w,0),{'weights':dict(zip(BASES,w.tolist()))}


def load_base_predictions(data,origins):
    previous=pd.read_csv(ROOT/'results/robust'/'validation_predictions.csv')
    hf=np.load(ROOT/'results/hf'/'point_predictions.npz')
    assert hf['origins'].tolist()==origins
    matrices={scenario:np.empty((19,len(seeds),48,len(BASES))) for scenario,seeds in SCENARIOS.items()}
    future=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds']);future.index=np.arange(len(data),len(data)+48)
    for i,origin in enumerate(origins):
        target=data.iloc[origin:origin+48] if i<18 else future
        for m,name in enumerate(BASES):
            if name.startswith('chronos'):
                for scenario in SCENARIOS:matrices[scenario][i,:,:,m]=hf[name][i]
                continue
            if i<6:
                predict,_=make_predictor(data,origin,CONFIGS[name])
                for scenario,seeds in SCENARIOS.items():
                    for j,seed in enumerate(seeds):
                        frame=target if scenario=='observed_weather' else perturb(target,origin*10+seed)
                        matrices[scenario][i,j,:,m]=predict(frame)
            elif i<18:
                for scenario,seeds in SCENARIOS.items():
                    for j,seed in enumerate(seeds):
                        g=previous[(previous.model==name)&(previous.fold==i)&(previous.scenario==scenario)&(previous.seed==seed)].sort_values('horizon')
                        assert len(g)==48
                        matrices[scenario][i,j,:,m]=g.forecast.to_numpy()
            else:
                p=pd.read_csv(ROOT/'results/robust'/f'point_{name}.csv').forecast.to_numpy()
                for scenario in SCENARIOS:matrices[scenario][i,:,:,m]=p
    assert all(np.isfinite(x).all() for x in matrices.values())
    return matrices


def run():
    OUT.mkdir(exist_ok=True)
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds']);y=data.y.to_numpy()
    origins=list(range(len(data)-18*48,len(data),48))+[len(data)]
    matrices=load_base_predictions(data,origins)
    actual=np.stack([y[o:o+48] for o in origins[:-1]])
    rows=[];weight_rows=[];final={};all_errors={}
    for scenario,seeds in SCENARIOS.items():
        x=matrices[scenario];targets=np.repeat(actual[:,None,:],len(seeds),axis=1)
        for method in METHODS:
            errors=[]
            # First three windows warm up the meta model. Windows 3-5 supply
            # genuinely forward-produced errors before selection begins at fold 6.
            for fold in range(3,19):
                point,details=combine(method,x[:fold],targets[:fold],x[fold])
                assert point.shape==(len(seeds),48) and np.isfinite(point).all()
                weight_rows.append(dict(scenario=scenario,model=method,fold=fold,
                    train_last_target=str(data.ds.iloc[origins[fold]-1]),details=details))
                if fold>=6:
                    lows=[];highs=[]
                    for j,seed in enumerate(seeds):
                        lo,hi=interval(point[j],errors);lows.append(lo);highs.append(hi)
                        if fold<18:
                            for h in range(48):
                                rows.append(dict(model=method,scenario=scenario,seed=seed,phase='selection' if fold<17 else 'last_window_audit',
                                    fold=fold,ds=data.ds.iloc[origins[fold]+h],horizon=h+1,actual=actual[fold,h],forecast=point[j,h],lower_95=lo[h],upper_95=hi[h]))
                    if fold==18:final[method,scenario]=(point[0],lows[0],highs[0])
                if fold<18:
                    errors.extend([actual[fold]-point[j] for j in range(len(seeds))])
            all_errors[method,scenario]=errors
            print(scenario,method,'complete',flush=True)
    new=pd.DataFrame(rows)
    previous=pd.read_csv(ROOT/'results/hf'/'validation_predictions.csv',parse_dates=['ds'])
    pred=pd.concat([previous,new],ignore_index=True)
    compare_windows(pred,OUT)
    scores=[]
    for (scenario,phase,name),g in pred.groupby(['scenario','phase','model']):
        scores.append(dict(scenario=scenario,phase=phase,model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    summary=pd.DataFrame(scores)
    for (scenario,phase),g in summary.groupby(['scenario','phase']):
        summary.loc[g.index,'RMSE_ratio']=g.RMSE/g[g.model=='seasonal_naive'].RMSE.iloc[0]
    summary=summary.sort_values(['phase','scenario','RMSE'])
    selection=summary[(summary.phase=='selection')&(summary.scenario=='weather_stress')]
    winner=selection.iloc[0].model
    best_ensemble=selection[selection.model.isin(METHODS)].iloc[0].model
    future=pd.read_csv(ROOT/'weather_forecast.csv')
    for method in METHODS:
        point,lo,hi=final[method,'weather_stress']
        sub=pd.DataFrame(dict(ds=future.ds,forecast=point,lower_95=lo,upper_95=hi))
        assert np.isfinite(sub.iloc[:,1:]).all().all() and (sub.lower_95>=0).all()
        assert (sub.lower_95<=sub.forecast).all() and (sub.forecast<=sub.upper_95).all()
        sub.to_csv(OUT/f'forecast_{method}.csv',index=False)
        if method==best_ensemble:sub.to_csv(OUT/'best_new_ensemble_forecast.csv',index=False)
        if method==winner:sub.to_csv(OUT/'selected_model_forecast.csv',index=False)
    if winner not in METHODS:
        prior=json.loads((ROOT/'results/hf'/'run_summary.json').read_text())['selected_model'];assert winner==prior
        pd.read_csv(ROOT/'results/hf'/'selected_model_forecast.csv').to_csv(OUT/'selected_model_forecast.csv',index=False)
    pred.to_csv(OUT/'validation_predictions.csv',index=False)
    summary.to_csv(OUT/'model_comparison.csv',index=False)
    (OUT/'weights_by_window.json').write_text(json.dumps(weight_rows,indent=2))
    report=dict(selected_model=winner,best_new_ensemble=best_ensemble,base_models=BASES,methods=METHODS,
        selection_rule='Lowest pooled weather-stress RMSE over eleven selection windows; latest audit excluded.',
        chronology='Meta-model training starts with three earlier out-of-fold windows. At fold k, fit only folds <k; score fold k; then append its errors. Selection remains folds 6-16. Audit fold17. Final refit uses all18 completed folds.',
        intervals='Stacking intervals use expanding prequential errors from fold3 onward; these differ from fixed calibration for earlier candidates, so interval comparisons include both model and calibration-policy differences.',
        caveats=['Base models/features and meta-methods were chosen after prior experiments; data reuse still creates selection optimism.',
                 'Stacking uses forward out-of-fold forecasts, not fitted values or randomly split observations.',
                 'Illustrative weather scenarios are pooled within historical windows, not counted as independent windows.',
                 'Daywise convex weights may be less stable because fewer historical errors support each fit.',
                 'No hidden targets available; a learned ensemble is not guaranteed to win.'])
    (OUT/'run_summary.json').write_text(json.dumps(report,indent=2))
    focus=METHODS+['seasonal_naive','SARIMA_21d','blend_SARIMA50_hour50']
    print(selection[selection.model.isin(focus)][['model','RMSE','RMSE_ratio','MAPE','Winkler','coverage']].to_string(index=False))
    print('Overall selected:',winner,'Best new ensemble:',best_ensemble)

if __name__=='__main__':run()
