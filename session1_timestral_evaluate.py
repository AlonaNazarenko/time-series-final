"""Prepare leakage-free Timestral inputs and compare its saved predictions."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from session1_more_models import perturb,compare_windows
from session1_models import interval,metrics
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_timestral'


def prepare():
    from session1_stacking import load_base_predictions
    OUT.mkdir(exist_ok=True)
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds'])
    future=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds'])
    columns=[c for c in future.columns if c!='ds']
    origins=list(range(len(data)-18*48,len(data),48))+[len(data)]
    context=[];known=[]
    for i,o in enumerate(origins):
        past=data.iloc[o-504:o];target=data.iloc[o:o+48] if i<18 else future
        assert len(past)==504 and len(target)==48
        variants=[target]+[perturb(target,o*10+s) if i<18 else target for s in [1,2,3]]
        context.append(past.y.to_numpy()[None,:])
        known.append(np.stack([np.concatenate([past[columns].to_numpy(),v[columns].to_numpy()],axis=0).T for v in variants]))
    # Forecast context contains only y before its origin; known rows contain weather, never y.
    context=np.stack(context).astype('float32');known=np.stack(known).astype('float32')
    assert context.shape==(19,1,504) and known.shape==(19,4,7,552)
    base=load_base_predictions(data,origins)
    equal=np.concatenate([base['observed_weather'].mean(axis=-1),base['weather_stress'].mean(axis=-1)],axis=1)
    np.savez(OUT/'inputs.npz',origins=np.array(origins),context=context,known=known,equal_mean=equal)
    print('Prepared target history and seven known weather covariates; target future excluded.',flush=True)


def evaluate():
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds']);y=data.y.to_numpy()
    future=pd.read_csv(ROOT/'weather_forecast.csv')
    raw=np.load(OUT/'point_predictions.npz');inp=np.load(OUT/'inputs.npz');origins=raw['origins']
    assert np.array_equal(origins,inp['origins'])
    points={k:raw[k] for k in raw.files if k!='origins'}
    points['blend_equal50_timestral28_weather50']=.5*inp['equal_mean']+.5*points['timestral_28m_weather_504h']
    rows=[];calibration={}
    for name,array in points.items():
        for scenario,js in [('observed_weather',[0]),('weather_stress',[1,2,3])]:
            errors=[y[o:o+48]-array[i,j] for i,o in enumerate(origins[:6]) for j in js]
            calibration[name,scenario]=errors
            for fold,o in enumerate(origins[6:-1],start=6):
                for j in js:
                    p=array[fold,j];lo,hi=interval(p,errors)
                    for h in range(48):
                        rows.append(dict(model=name,scenario=scenario,seed=j,fold=fold,phase='selection' if fold<17 else 'last_window_audit',
                            ds=data.ds.iloc[o+h],horizon=h+1,actual=y[o+h],forecast=p[h],lower_95=lo[h],upper_95=hi[h]))
    pred=pd.concat([pd.read_csv(ROOT/'results_stacking'/'validation_predictions.csv',parse_dates=['ds']),pd.DataFrame(rows)],ignore_index=True)
    compare_windows(pred,OUT)
    summary=[]
    for (scenario,phase,name),g in pred.groupby(['scenario','phase','model']):
        summary.append(dict(scenario=scenario,phase=phase,model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    summary=pd.DataFrame(summary)
    for (scenario,phase),g in summary.groupby(['scenario','phase']):
        summary.loc[g.index,'RMSE_ratio']=g.RMSE/g[g.model=='seasonal_naive'].RMSE.iloc[0]
    summary=summary.sort_values(['phase','scenario','RMSE'])
    selection=summary[(summary.phase=='selection')&(summary.scenario=='weather_stress')]
    winner=selection.iloc[0].model;best=selection[selection.model.isin(points)].iloc[0].model
    for name,array in points.items():
        errors=list(calibration[name,'weather_stress'])+[y[o:o+48]-array[i,j] for i,o in enumerate(origins[6:-1],start=6) for j in [1,2,3]]
        p=array[-1,0];lo,hi=interval(p,errors)
        sub=pd.DataFrame(dict(ds=future.ds,forecast=p,lower_95=lo,upper_95=hi))
        assert len(sub)==48 and np.isfinite(sub.iloc[:,1:]).all().all()
        assert (sub.lower_95>=0).all() and (sub.lower_95<=sub.forecast).all() and (sub.forecast<=sub.upper_95).all()
        sub.to_csv(OUT/f'forecast_{name}.csv',index=False)
        if name==best:sub.to_csv(OUT/'best_timestral_forecast.csv',index=False)
        if name==winner:sub.to_csv(OUT/'selected_model_forecast.csv',index=False)
    if winner not in points:
        assert winner==json.loads((ROOT/'results_stacking'/'run_summary.json').read_text())['selected_model']
        pd.read_csv(ROOT/'results_stacking'/'selected_model_forecast.csv').to_csv(OUT/'selected_model_forecast.csv',index=False)
    pred.to_csv(OUT/'validation_predictions.csv',index=False);summary.to_csv(OUT/'model_comparison.csv',index=False)
    report=dict(selected_model=winner,best_timestral_candidate=best,candidates=list(points),
        selection_rule='Lowest pooled weather-stress RMSE on same eleven selection windows; latest audit excluded.',
        caveats=['Weather stress assumptions are illustrative, not measured forecast-error distributions.',
        'Many model searches on reused windows can overfit model selection.','Audit is previously seen, not an untouched test.',
        'Timestral median forecasts are evaluated as point forecasts; no fine-tuning.',
        'Timestral intervals use fixed early calibration for validation; prior ensembles retain their expanding-calibration intervals.',
        'Pretraining overlap with disguised series cannot be independently verified; no original-source search performed.'])
    (OUT/'run_summary.json').write_text(json.dumps(report,indent=2))
    print(selection[selection.model.isin(list(points)+['equal_mean','seasonal_naive','SARIMA_21d'])][['model','RMSE','RMSE_ratio','MAPE','Winkler','coverage']].to_string(index=False))
    print('Overall selected:',winner,'Best Timestral candidate:',best)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    prepare() if a.prepare else evaluate()
