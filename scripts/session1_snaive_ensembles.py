"""Compare SNaive blends using saved forward forecasts and causal weights."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from session1_models import baseline, metrics
from session1_stacking import convex_weights
from session1_task_evaluate import bounds

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/snaive'


def run():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT / 'train.csv', parse_dates=['ds'])
    future = pd.read_csv(ROOT / 'weather_forecast.csv', parse_dates=['ds'])
    inputs = np.load(ROOT / 'results/timestral/inputs.npz')
    saved = np.load(ROOT / 'results/timestral/point_predictions.npz')
    origins = inputs['origins']
    assert np.array_equal(origins, saved['origins'])
    assert len(origins) == 19 and origins[-1] == len(data)
    assert np.all(np.diff(origins) == 48)
    actual = np.stack([data.y.iloc[o:o+48].to_numpy() for o in origins[:-1]])
    daylight = np.stack([data.daylight.iloc[o:o+48].to_numpy() for o in origins[:-1]] + [future.daylight.to_numpy()])
    naive = np.stack([np.repeat(baseline(data.y.iloc[:o].to_numpy())[0][None], 4, axis=0) for o in origins])
    raw = {'seasonal_naive': naive,
           'timestral28': saved['timestral_28m_weather_504h'],
           'equal_mean': inputs['equal_mean']}
    for base in ['timestral28', 'equal_mean']:
        for weight in [.1, .25, .5, .75]:
            raw[f'{base}_snaive{int(100*weight)}'] = (1-weight)*raw[base]+weight*naive
    members = ['seasonal_naive', 'timestral28', 'equal_mean']
    x = np.stack([raw[name] for name in members], axis=-1)
    raw['three_equal'] = x.mean(axis=-1)
    raw['three_median'] = np.median(x, axis=-1)
    weights = []
    # First three folds initialize learned weights. All subsequent fits use
    # completed folds only, separately for observed and stressed weather.
    for method in ['convex_expanding', 'convex_recent4', 'convex_daywise']:
        result = x.mean(axis=-1)
        for fold in range(3, 19):
            start = max(0, fold-4) if method == 'convex_recent4' else 0
            for scenario, seeds in [('observed_weather', [0]), ('weather_stress', [1, 2, 3])]:
                slices = [slice(0, 24), slice(24, 48)] if method == 'convex_daywise' else [slice(0, 48)]
                for sl in slices:
                    history = x[start:fold][:, seeds, sl, :].reshape(-1, 3)
                    targets = np.repeat(actual[start:fold, None, sl], len(seeds), axis=1).reshape(-1)
                    w = convex_weights(history, targets)
                    # Future uses actual forecast weather (seed 0), but weights
                    # are learned on stress scenarios for the exported forecast.
                    for seed in ([0] if fold == 18 else seeds):
                        result[fold, seed, sl] = x[fold, seed, sl] @ w
                    weights.append(dict(model=method, fold=fold, scenario=scenario,
                                        horizon_start=sl.start+1, horizon_end=sl.stop,
                                        weights=dict(zip(members, w.tolist()))))
        raw[method] = result
    records = []
    for name, array in raw.items():
        assert array.shape == (19, 4, 48) and np.isfinite(array).all()
        for scenario, seeds in [('observed_weather', [0]), ('weather_stress', [1, 2, 3])]:
            for fold in range(6, 19):
                # Learned models have genuinely forward calibration from fold 3.
                start = 3 if name.startswith('convex_') else 0
                history = array[start:fold][:, seeds, :].reshape(-1, 48)
                truth = np.repeat(actual[start:fold, None, :], len(seeds), axis=1).reshape(-1, 48)
                light = np.repeat(daylight[start:fold, None, :], len(seeds), axis=1).reshape(-1, 48)
                for seed in ([0] if fold == 18 else seeds):
                    point = array[fold, seed]
                    if name == 'seasonal_naive':
                        _, lo, hi = baseline(data.y.iloc[:origins[fold]].to_numpy())
                    else:
                        lo, hi = bounds(point, history, truth, light, daylight[fold], 'absolute_day')
                    assert np.isfinite([point, lo, hi]).all()
                    assert (lo >= 0).all() and (lo <= point).all() and (point <= hi).all()
                    if fold == 18:
                        if scenario == 'weather_stress':
                            pd.DataFrame(dict(ds=future.ds, forecast=point, lower_95=lo, upper_95=hi)).to_csv(OUT / f'forecast_{name}.csv', index=False)
                        continue
                    for h in range(48):
                        records.append(dict(model=name, scenario=scenario, seed=seed, fold=fold,
                            phase='selection' if fold < 17 else 'last_window_audit',
                            ds=data.ds.iloc[origins[fold]+h], horizon=h+1, actual=actual[fold, h],
                            forecast=point[h], lower_95=lo[h], upper_95=hi[h]))
    pred = pd.DataFrame(records)
    scores, windows = [], []
    for (scenario, phase, name), group in pred.groupby(['scenario', 'phase', 'model']):
        for period, part in [('full_48h', group), ('day_two', group[group.horizon > 24])]:
            scores.append(dict(scenario=scenario, phase=phase, model=name, period=period,
                **metrics(part.actual.to_numpy(), part.forecast.to_numpy(), part.lower_95.to_numpy(), part.upper_95.to_numpy())))
            for fold, block in part.groupby('fold'):
                windows.append(dict(scenario=scenario, phase=phase, model=name, period=period, fold=fold,
                                    RMSE=np.sqrt(np.mean((block.actual-block.forecast)**2))))
    scores = pd.DataFrame(scores)
    windows = pd.DataFrame(windows)
    keys = ['scenario', 'phase', 'period', 'fold']
    ref = windows[windows.model == 'seasonal_naive'][keys+['RMSE']].rename(columns={'RMSE': 'baseline_RMSE'})
    windows = windows.merge(ref, on=keys, validate='many_to_one')
    windows['win'] = windows.RMSE < windows.baseline_RMSE-1e-9
    wins = windows.groupby(['scenario', 'phase', 'period', 'model']).agg(wins=('win', 'sum'), windows=('fold', 'size')).reset_index()
    scores = scores.merge(wins)
    scores['win_rate_pct'] = 100*scores.wins/scores.windows
    for _, group in scores.groupby(['scenario', 'phase', 'period']):
        scores.loc[group.index, 'RMSE_ratio'] = group.RMSE/group[group.model == 'seasonal_naive'].RMSE.iloc[0]
    scores = scores.sort_values(['scenario', 'phase', 'period', 'RMSE'])
    selection = scores[(scores.scenario == 'weather_stress') & (scores.phase == 'selection') & (scores.period == 'full_48h')]
    winner = selection.iloc[0].model
    pd.read_csv(OUT / f'forecast_{winner}.csv').to_csv(OUT / 'selected_model_forecast.csv', index=False)
    pred.to_csv(OUT / 'validation_predictions.csv', index=False)
    scores.to_csv(OUT / 'model_comparison.csv', index=False)
    windows.to_csv(OUT / 'window_comparison.csv', index=False)
    (OUT / 'weights_by_window.json').write_text(json.dumps(weights, indent=2))
    (OUT / 'run_summary.json').write_text(json.dumps(dict(selected_model=winner,
        selection_rule='Lowest pooled full-48h weather-stress RMSE on folds 6–16; audit fold 17 excluded.',
        intervals='Expanding absolute errors by forecast day; official baseline intervals. Learned models calibrate from fold 3.',
        caveats=['Repeated selection-window reuse can overfit model selection.', 'Weather perturbations are hypothetical.',
                 'Latest audit has already been seen; hidden performance is unknown.']), indent=2))
    print(selection[['model', 'RMSE', 'RMSE_ratio', 'wins', 'win_rate_pct']].to_string(index=False))
    print('Selected:', winner)


if __name__ == '__main__':
    run()
