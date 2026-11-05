"""Evaluate locally generated HF forecasts in the original project environment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from session1_models import fit_forecast,interval,metrics
from session1_more_models import compare_windows
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results_hf'


def run():
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds']);y=data.y.to_numpy()
    future=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds'])
    saved=np.load(OUT/'point_predictions.npz')
    origins=saved['origins']
    assert origins.tolist()==list(range(len(data)-18*48,len(data),48))+[len(data)]
    points={k:saved[k] for k in saved.files if k!='origins'}
    # Fixed blends provide a direct test of whether a pretrained model complements SARIMA.
    sarima=np.stack([fit_forecast(y[:o],dict(kind='SARIMA',days=21))[0] for o in origins])
    for size in ['tiny','small']:
        points[f'blend_SARIMA50_bolt_{size}504_50']=.5*sarima+.5*points[f'chronos_bolt_{size}_504h']
    rows=[];calibration={name:[y[o:o+48]-p[i] for i,o in enumerate(origins[:6])] for name,p in points.items()}
    for name,predictions in points.items():
        for fold,origin in enumerate(origins[6:-1],start=6):
            p=predictions[fold];actual=y[origin:origin+48]
            lo,hi=interval(p,calibration[name])
            phase='selection' if fold<17 else 'last_window_audit'
            for scenario,seeds in [('observed_weather',[0]),('weather_stress',[1,2,3])]:
                # No weather inputs: all stress scenarios give the identical forecast.
                for seed in seeds:
                    for h in range(48):
                        rows.append(dict(model=name,scenario=scenario,seed=seed,phase=phase,fold=fold,ds=data.ds.iloc[origin+h],horizon=h+1,
                                         actual=actual[h],forecast=p[h],lower_95=lo[h],upper_95=hi[h]))
    hf=pd.DataFrame(rows)
    old=pd.read_csv(ROOT/'results_robust'/'validation_predictions.csv',parse_dates=['ds'])
    pred=pd.concat([old,hf],ignore_index=True)
    compare_windows(pred,OUT)
    rows=[]
    for (scenario,phase,name),g in pred.groupby(['scenario','phase','model']):
        rows.append(dict(scenario=scenario,phase=phase,model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    summary=pd.DataFrame(rows)
    for (scenario,phase),g in summary.groupby(['scenario','phase']):
        base=g[g.model=='seasonal_naive'].RMSE.iloc[0]
        summary.loc[g.index,'RMSE_ratio']=g.RMSE/base
    summary=summary.sort_values(['phase','scenario','RMSE'])
    selection=summary[(summary.phase=='selection')&(summary.scenario=='weather_stress')]
    winner=selection.iloc[0].model
    hf_winner=selection[selection.model.isin(points)].iloc[0].model
    for name in points:
        errors=list(calibration[name])+[y[o:o+48]-points[name][i] for i,o in enumerate(origins[6:-1],start=6)]
        p=points[name][-1];lo,hi=interval(p,errors)
        sub=pd.DataFrame(dict(ds=future.ds.dt.strftime('%Y-%m-%d %H:%M:%S'),forecast=p,lower_95=lo,upper_95=hi))
        assert len(sub)==48 and np.isfinite(sub.iloc[:,1:]).all().all()
        assert (sub.lower_95>=0).all() and (sub.lower_95<=sub.forecast).all() and (sub.forecast<=sub.upper_95).all()
        sub.to_csv(OUT/f'forecast_{name}.csv',index=False)
        if name==hf_winner:sub.to_csv(OUT/'best_hf_forecast.csv',index=False)
        if name==winner:sub.to_csv(OUT/'selected_model_forecast.csv',index=False)
    if winner not in points:
        previous=json.loads((ROOT/'results_robust'/'run_summary.json').read_text())['selected_model']
        assert winner==previous
        pd.read_csv(ROOT/'results_robust'/'selected_model_forecast.csv').to_csv(OUT/'selected_model_forecast.csv',index=False)
    summary.to_csv(OUT/'model_comparison.csv',index=False)
    pred.to_csv(OUT/'validation_predictions.csv',index=False)
    dayrows=[]
    for (scenario,name),g in pred[(pred.phase=='selection')&(pred.horizon>24)].groupby(['scenario','model']):
        dayrows.append(dict(scenario=scenario,model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    pd.DataFrame(dayrows).to_csv(OUT/'day_two_comparison.csv',index=False)
    report=dict(selected_model=winner,best_hf_candidate=hf_winner,hf_candidates=list(points),
        selection_rule='Lowest pooled weather-stress RMSE on same eleven selection windows, across previous and HF candidates; latest audit excluded.',
        inference='Frozen pretrained Chronos-Bolt Tiny/Small, local CPU; context lengths 168, 504, 1536 hours; native median point forecast; no fine-tuning or weather inputs.',
        intervals='Empirical absolute-error calibration from six earlier 48h windows, separately by forecast day; refreshed with all completed historical blocks for future forecast.',
        caveats=['Native model quantiles are not presented as 95% intervals.','Validation windows reused across model searches; model selection can be optimistic.',
                 'Previously inspected audit is not an untouched test.','Pretraining-data overlap cannot be independently verified; no attempt was made to identify the disguised series.',
                 'Only supplied project data are used as local inputs; pretrained weights are external model artifacts.',
                 'Zero-shot forecasting does not guarantee superiority to seasonal naive.'])
    (OUT/'run_summary.json').write_text(json.dumps(report,indent=2))
    print(selection[['model','RMSE','RMSE_ratio','MAPE','Winkler','coverage']].to_string(index=False))
    print('Overall selected:',winner,'\nBest HF candidate:',hf_winner)

if __name__=='__main__':run()
