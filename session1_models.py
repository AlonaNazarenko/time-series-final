"""Run reproducible, chronological ETS/ARIMA/SARIMA comparisons on supplied data."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).resolve().parent
H = 48
CONFIGS = {
    'ETS_ANA_7d': dict(kind='ETS', days=7, trend=False),
    'ETS_ANA_21d': dict(kind='ETS', days=21, trend=False),
    'ETS_AAdA_21d': dict(kind='ETS', days=21, trend=True),
    'ARIMA_212_21d': dict(kind='ARIMA', days=21),
    'SARIMA_101_110_24_7d': dict(kind='SARIMA', days=7),
    'SARIMA_101_110_24_21d': dict(kind='SARIMA', days=21),
    'SARIMA_101_110_24_60d': dict(kind='SARIMA', days=60),
}


def baseline(history):
    point = np.tile(history[-24:], 2)
    sigma = np.std(history[24:] - history[:-24], ddof=1)
    half = 1.96 * sigma * np.sqrt(np.arange(H)//24 + 1)
    return point, np.maximum(point-half, 0), point+half


def fit_forecast(history, config):
    # Scaling is fitted separately at every origin and improves optimization.
    recent = np.asarray(history[-24*config['days']:], float)
    scale = max(float(np.std(recent)), 1.0)
    z = recent / scale
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        if config['kind'] == 'ETS':
            model = ETSModel(z, error='add', trend='add' if config['trend'] else None,
                             damped_trend=config['trend'], seasonal='add', seasonal_periods=24)
            fit = model.fit(disp=False, maxiter=1500)
        else:
            seasonal = config['kind'] == 'SARIMA'
            model = SARIMAX(z, order=(1,0,1) if seasonal else (2,1,2),
                            seasonal_order=(1,1,0,24) if seasonal else (0,0,0,0),
                            enforce_stationarity=False, enforce_invertibility=False)
            fit = model.fit(disp=False, maxiter=300)
        if not fit.mle_retvals.get('converged', True):
            if config['kind'] == 'ETS':
                fit = model.fit(start_params=fit.params, disp=False, maxiter=6000)
            else:
                fit = model.fit(start_params=fit.params, method='powell', disp=False, maxiter=500)
    if not fit.mle_retvals.get('converged', True):
        raise RuntimeError('Optimizer failed to converge')
    point = np.maximum(np.asarray(fit.forecast(H))*scale, 0)
    if point.shape != (H,) or not np.isfinite(point).all():
        raise ValueError('Invalid forecast')
    return point, sorted(set(str(w.message) for w in caught))


def metrics(actual, point, lower, upper):
    return dict(RMSE=float(np.sqrt(np.mean((actual-point)**2))),
                MAPE=float(np.mean(np.abs((actual-point)/actual))*100),
                Winkler=float(np.mean(upper-lower+40*np.maximum(lower-actual,0)+40*np.maximum(actual-upper,0))),
                coverage=float(np.mean((actual>=lower)&(actual<=upper))),
                mean_width=float(np.mean(upper-lower)))


def interval(point, errors):
    # Separate day-one/day-two scales, estimated only on earlier calibration folds.
    # This is empirical calibration, not a guaranteed 95% coverage claim under dependence.
    widths = [np.quantile(np.abs(np.concatenate([e[d*24:(d+1)*24] for e in errors])), .95,
                          method='higher') for d in range(2)]
    half = np.repeat(widths, 24)
    return np.maximum(point-half, 0), point+half


def run(surname=None):
    out = ROOT/'results'; out.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT/'train.csv', parse_dates=['ds'])
    future = pd.read_csv(ROOT/'weather_forecast.csv', parse_dates=['ds'])
    assert data.ds.is_unique and data.ds.is_monotonic_increasing
    assert data.ds.diff().dropna().eq(pd.Timedelta(hours=1)).all()
    y = data.y.to_numpy(float)
    assert np.isfinite(y).all() and (y>0).all(), 'MAPE requires positive targets here'
    assert len(future)==H and future.ds.equals(pd.Series(pd.date_range(data.ds.iloc[-1]+pd.Timedelta(hours=1),periods=H,freq='h'),name='ds'))
    errors = {name: [] for name in CONFIGS}
    rows, predictions, issues = [], [], []
    # Six calibration blocks, five model-selection blocks, then the previously seen EDA window.
    origins = list(range(len(y)-12*H, len(y), H))
    for fold, origin in enumerate(origins):
        phase = 'calibration' if fold<6 else ('selection' if fold<11 else 'last_window_audit')
        actual = y[origin:origin+H]
        forecasts = {'seasonal_naive': baseline(y[:origin])}
        for name, config in CONFIGS.items():
            try:
                point, notes = fit_forecast(y[:origin], config)
                if notes: issues.append(dict(fold=fold,model=name,notes=notes))
                if phase=='calibration': errors[name].append(actual-point)
                elif len(errors[name])==6:
                    lower, upper = interval(point, errors[name])
                    forecasts[name] = (point, lower, upper)
            except Exception as exc:
                issues.append(dict(fold=fold,model=name,error=str(exc)))
        if phase!='calibration':
            for name, (point,lower,upper) in forecasts.items():
                rows.append(dict(phase=phase,fold=fold,start=str(data.ds.iloc[origin]),model=name,
                                 **metrics(actual,point,lower,upper)))
                for h in range(H):
                    predictions.append(dict(phase=phase,fold=fold,ds=data.ds.iloc[origin+h],horizon=h+1,
                                            model=name,actual=actual[h],forecast=point[h],lower_95=lower[h],upper_95=upper[h]))
        print(f'{fold+1}/12: {phase}, origin {data.ds.iloc[origin-1]}',flush=True)
    scores = pd.DataFrame(rows)
    pred = pd.DataFrame(predictions)
    summary = []
    for name,g in pred[pred.phase=='selection'].groupby('model'):
        if len(g)!=5*H: continue
        summary.append(dict(model=name,**metrics(g.actual.to_numpy(),g.forecast.to_numpy(),g.lower_95.to_numpy(),g.upper_95.to_numpy())))
    summary = pd.DataFrame(summary).set_index('model')
    summary['RMSE_ratio'] = summary.RMSE/summary.loc['seasonal_naive','RMSE']
    selection_scores = scores[scores.phase=='selection'].pivot(index='fold',columns='model',values='RMSE')
    summary['windows_beating_baseline'] = [(selection_scores[m]<selection_scores.seasonal_naive).sum() for m in summary.index]
    summary = summary.sort_values('RMSE')
    winner = summary.index[0] # Includes baseline: do not force a worse model.
    if winner=='seasonal_naive': point,lower,upper = baseline(y)
    else:
        point,notes = fit_forecast(y,CONFIGS[winner])
        # Refresh empirical uncertainty with every completed pre-forecast validation block.
        all_errors = list(errors[winner])
        for _,g in pred[pred.model==winner].groupby('fold'):
            all_errors.append(g.actual.to_numpy()-g.forecast.to_numpy())
        lower,upper = interval(point,all_errors)
    submission = pd.DataFrame(dict(ds=future.ds.dt.strftime('%Y-%m-%d %H:%M:%S'),forecast=point,lower_95=lower,upper_95=upper))
    assert list(submission.columns)==['ds','forecast','lower_95','upper_95'] and len(submission)==H
    assert np.isfinite(submission.iloc[:,1:]).all().all()
    assert (submission.lower_95>=0).all() and (submission.lower_95<=submission.forecast).all() and (submission.forecast<=submission.upper_95).all()
    scores.to_csv(out/'validation_scores.csv',index=False)
    pred.to_csv(out/'validation_predictions.csv',index=False)
    summary.to_csv(out/'model_comparison.csv')
    submission.to_csv(out/'selected_model_forecast.csv',index=False)
    if surname:
        if not surname.strip() or any(c in surname for c in '/\\'): raise ValueError('Provide a surname, not a path')
        submission.to_csv(ROOT/f'{surname.strip().upper()}_project.csv',index=False)
    audit=scores[scores.phase=='last_window_audit'].set_index('model')
    report=dict(selected_model=winner,selection_rule='Lowest pooled RMSE on five selection windows; baseline eligible',
                calibration_windows=6,selection_windows=5,window_hours=H,
                audit_note='Last 48-hour window was already inspected during EDA and the initial model probe; it is not an untouched test set. It is excluded from model selection.',
                audit_selected=audit.loc[winner].to_dict(),audit_baseline=audit.loc['seasonal_naive'].to_dict(),
                interval_method='95th percentile absolute rolling forecast errors, separately for each forecast day; non-baseline intervals evaluated with calibration-only errors, refreshed on all past folds for final forecast.',
                configs=CONFIGS,fit_diagnostics=issues)
    (out/'run_summary.json').write_text(json.dumps(report,indent=2))
    print('\nSelection results:\n',summary.to_string(),flush=True)
    print('\nLast-window audit:\n',audit[['RMSE','MAPE','Winkler','coverage']].to_string(),flush=True)
    print('\nSelected:',winner,'\nSaved:',out/'selected_model_forecast.csv',flush=True)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--surname')
    run(parser.parse_args().surname)
