"""Assignment-focused selection: 48h RMSE, day-two metrics, causal interval calibration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from session1_models import metrics,baseline
from session1_stacking import load_base_predictions
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results_task'
POLICIES=['absolute_day','scaled_symmetric','scaled_daynight']


def bounds(point,prior_points,prior_actual,prior_daylight,current_daylight,policy):
    residual=prior_actual-prior_points
    low=np.empty(48);high=np.empty(48)
    for day in [0,1]:
        sl=slice(day*24,(day+1)*24)
        if policy=='absolute_day':
            width=np.quantile(np.abs(residual[:,sl]),.95,method='higher')
            low[sl]=point[sl]-width;high[sl]=point[sl]+width
        elif policy=='scaled_symmetric':
            scores=np.abs(residual[:,sl])/(50+.2*prior_points[:,sl])
            q=np.quantile(scores,.95,method='higher')
            width=q*(50+.2*point[sl]);low[sl]=point[sl]-width;high[sl]=point[sl]+width
        else:
            scores=residual[:,sl]/(50+.2*prior_points[:,sl])
            for light in [0,1]:
                history=scores[prior_daylight[:,sl]==light]
                if len(history)<20:history=scores.ravel()
                qlo=np.quantile(history,.025,method='lower');qhi=np.quantile(history,.975,method='higher')
                indices=np.arange(day*24,(day+1)*24)[current_daylight[sl]==light]
                low[indices]=point[indices]+min(qlo,0)*(50+.2*point[indices])
                high[indices]=point[indices]+max(qhi,0)*(50+.2*point[indices])
    return np.maximum(low,0),np.maximum(high,point)


def run():
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds']);future=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds']);y=data.y.to_numpy()
    inp=np.load(ROOT/'results_timestral'/'inputs.npz');origins=inp['origins'];new=np.load(OUT/'point_predictions.npz')
    assert np.array_equal(origins,new['origins'])
    raw={k:new[k] for k in new.files if k!='origins'}
    old=np.load(ROOT/'results_timestral'/'point_predictions.npz')
    raw['timestral28_original']=old['timestral_28m_weather_504h']
    raw['equal_mean']=inp['equal_mean']
    bases=load_base_predictions(data,origins.tolist())
    raw['SARIMA_21d']=np.concatenate([bases['observed_weather'][:,:,:,0],bases['weather_stress'][:,:,:,0]],axis=1)
    actual=np.stack([y[o:o+48] for o in origins[:-1]])
    daylight=np.stack([data.daylight.iloc[o:o+48].to_numpy() for o in origins[:-1]]+[future.daylight.to_numpy()])
    for name in ['timestral28_original','timestral123_weather21d']:
        for mode in ['level_half','bias_half']:
            p=raw[name].copy()
            for i in range(1,19):
                for light in [0,1]:
                    pastmask=daylight[i-1]==light;mask=daylight[i]==light
                    a=actual[i-1,pastmask];b=raw[name][i-1,0,pastmask]
                    # Previous block actuals are available; never use this block's target.
                    if mode=='level_half':
                        factor=np.clip(1+.5*((a+10).sum()/(b+10).sum()-1),.5,1.5)
                        p[i][:,mask]*=factor
                    else:p[i][:,mask]=np.maximum(p[i][:,mask]+.5*np.mean(a-b),0)
            raw[name+'_'+mode]=p
    raw['timestral28_SARIMA50']=.5*raw['timestral28_original']+.5*raw['SARIMA_21d']
    raw['timestral123_SARIMA25']=.75*raw['timestral123_weather21d']+.25*raw['SARIMA_21d']
    raw['seasonal_naive']=np.stack([np.repeat(baseline(y[:o])[0][None,:],4,axis=0) for o in origins])
    records=[];forecast_files={}
    for name,array in raw.items():
        for scenario,js in [('observed_weather',[0]),('weather_stress',[1,2,3])]:
            for policy in (['official'] if name=='seasonal_naive' else POLICIES):
                for fold in range(6,19):
                    # Pool only previously completed blocks, not current/future actuals.
                    history=array[:fold,js,:].reshape(-1,48)
                    actual_history=np.repeat(actual[:fold,None,:],len(js),axis=1).reshape(-1,48)
                    light_history=np.repeat(daylight[:fold,None,:],len(js),axis=1).reshape(-1,48)
                    for j in ([0] if fold==18 else js):
                        p=array[fold,j]
                        if policy=='official':_,lo,hi=baseline(y[:origins[fold]])
                        else:lo,hi=bounds(p,history,actual_history,light_history,daylight[fold],policy)
                        if fold==18:
                            if scenario=='weather_stress':
                                sub=pd.DataFrame(dict(ds=future.ds.dt.strftime('%Y-%m-%d %H:%M:%S'),forecast=p,lower_95=lo,upper_95=hi))
                                assert np.isfinite(sub.iloc[:,1:]).all().all() and (sub.lower_95>=0).all()
                                assert (sub.lower_95<=sub.forecast).all() and (sub.forecast<=sub.upper_95).all()
                                filename=f'forecast_{name}__{policy}.csv';sub.to_csv(OUT/filename,index=False);forecast_files[name,policy]=filename
                            continue
                        for h in range(48):
                            records.append(dict(model=name,interval=policy,scenario=scenario,seed=j,fold=fold,
                                phase='selection' if fold<17 else 'last_window_audit',ds=data.ds.iloc[origins[fold]+h],horizon=h+1,
                                actual=actual[fold,h],forecast=p[h],lower_95=lo[h],upper_95=hi[h]))
        print(name,'evaluated',flush=True)
    pred=pd.DataFrame(records);rows=[]
    for (scenario,phase,name,policy),g in pred.groupby(['scenario','phase','model','interval']):
        for period,part in [('full_48h',g),('day_two',g[g.horizon>24])]:
            rows.append(dict(scenario=scenario,phase=phase,model=name,interval=policy,period=period,
                **metrics(part.actual.to_numpy(),part.forecast.to_numpy(),part.lower_95.to_numpy(),part.upper_95.to_numpy())))
    scores=pd.DataFrame(rows)
    for (scenario,phase,period),g in scores.groupby(['scenario','phase','period']):
        base=g[g.model=='seasonal_naive'].iloc[0]
        for metric in ['RMSE','MAPE','Winkler']:scores.loc[g.index,metric+'_ratio']=g[metric]/base[metric]
    scores=scores.sort_values(['scenario','phase','period','RMSE','Winkler'])
    choice=scores[(scores.scenario=='weather_stress')&(scores.phase=='selection')&(scores.period=='full_48h')]
    winner=choice.sort_values(['RMSE','Winkler']).iloc[0]
    # Point RMSE is identical for interval policies. Explicitly choose interval by Winkler.
    selected=choice[choice.model==winner.model].sort_values('Winkler').iloc[0]
    pd.read_csv(OUT/forecast_files[selected.model,selected.interval]).to_csv(OUT/'selected_model_forecast.csv',index=False)
    # One row per point model, using its best-Winkler interval within selection only.
    best=choice.sort_values('Winkler').groupby('model',as_index=False).first().sort_values('RMSE')
    winrows=[]
    for _,row in best.iterrows():
        g=pred[(pred.model==row.model)&(pred.interval==row.interval)&(pred.scenario=='weather_stress')&(pred.phase=='selection')]
        wins=0
        for fold,part in g.groupby('fold'):
            base=pred[(pred.model=='seasonal_naive')&(pred.scenario=='weather_stress')&(pred.phase=='selection')&(pred.fold==fold)]
            wins+=np.mean((part.actual-part.forecast)**2)<np.mean((base.actual-base.forecast)**2)
        winrows.append(dict(model=row.model,wins=wins,windows=11,win_rate_pct=100*wins/11))
    best=best.merge(pd.DataFrame(winrows),on='model');best.to_csv(OUT/'point_model_comparison.csv',index=False)
    pred.to_csv(OUT/'validation_predictions.csv',index=False);scores.to_csv(OUT/'all_scores.csv',index=False)
    report=dict(selected_model=selected.model,selected_interval=selected.interval,selection_rule='Session 1: lowest full-48h weather-stress RMSE, then lowest selection Winkler for that point model; audit excluded.',
        assignment={'session1':'RMSE ratio on all48 hidden hours','final':'Day-two ranks: 0.3 RMSE + 0.3 MAPE + 0.4 Winkler; class ranks cannot be known locally.'},
        adaptation='Half-strength level or bias corrections use only the immediately preceding completed48h block, split by daylight. Previous block observed-weather hindcasts diagnose model bias retrospectively.',
        interval_policies={'absolute_day':'Expanding95th percentile absolute errors, per forecast day',
            'scaled_symmetric':'Expanding95th percentile abs(error)/(50+0.2*forecast), per forecast day',
            'scaled_daynight':'Expanding signed2.5/97.5 percentiles of error/(50+0.2*forecast), split by forecast day and daylight; bounds include point.'},
        caveats=['More candidates on repeatedly used windows can overfit validation; no guarantee of beating hidden baseline.',
        'Weather stress magnitudes are assumptions, not measured forecast errors.',
        'Audit was already seen and is excluded from selection; adaptation may use actual earlier blocks only.',
        'Interval method is selected on validation Winkler; nominal95% does not guarantee future coverage.'])
    (OUT/'run_summary.json').write_text(json.dumps(report,indent=2))
    print(best[['model','interval','RMSE','RMSE_ratio','MAPE','Winkler','coverage','wins','win_rate_pct']].to_string(index=False))
    print('SELECTED:',selected.model,selected.interval,flush=True)

if __name__=='__main__':run()
