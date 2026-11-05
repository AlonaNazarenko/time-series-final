"""Extra Trees and SNaive blends on the common 18 historical windows."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from session1_more_models import predictor, perturb, compare_windows
from session1_models import baseline, interval, metrics

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'results/extra_snaive'


def run():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(ROOT/'train.csv', parse_dates=['ds'])
    future = pd.read_csv(ROOT/'weather_forecast.csv', parse_dates=['ds'])
    future.index = np.arange(len(data), len(data)+48)
    origins = np.arange(len(data)-18*48, len(data)+1, 48)
    actual = np.stack([data.y.iloc[o:o+48] for o in origins[:-1]])
    raw = {'seasonal_naive': np.stack([np.tile(baseline(data.y.iloc[:o].to_numpy())[0], (4, 1)) for o in origins])}
    for days in [21, 60, 120]:
        name = f'extra_trees_{days}d'
        cache = OUT/f'{name}_points.npz'
        if cache.exists():
            saved = np.load(cache)
            assert np.array_equal(saved['origins'], origins)
            raw[name] = saved['points']
        else:
            points = []
            for fold, origin in enumerate(origins):
                target = data.iloc[origin:origin+48] if fold < 18 else future
                predict = predictor(data, origin, dict(kind='extra', days=days, weather=True))
                variants = [target]+[perturb(target, int(origin)*10+seed) if fold < 18 else target for seed in [1, 2, 3]]
                points.append(np.stack([predict(frame) for frame in variants]))
                print(name, f'{fold+1}/19', flush=True)
            raw[name] = np.stack(points)
            np.savez_compressed(cache, origins=origins, points=raw[name])
        for n in range(1, 20):
            raw[f'{name}_snaive{n*5}'] = (1-n/20)*raw[name]+n/20*raw['seasonal_naive']
    records = []
    for name, points in raw.items():
        assert points.shape == (19, 4, 48) and np.isfinite(points).all()
        for scenario, seeds in [('observed_weather', [0]), ('weather_stress', [1, 2, 3])]:
            for fold in range(6, 19):
                errors = (actual[:fold, None, :]-points[:fold])[:, seeds].reshape(-1, 48)
                for seed in ([0] if fold == 18 else seeds):
                    point = points[fold, seed]
                    if name == 'seasonal_naive':
                        _, lo, hi = baseline(data.y.iloc[:origins[fold]].to_numpy())
                    else:
                        lo, hi = interval(point, list(errors))
                    assert np.isfinite([point, lo, hi]).all() and (lo >= 0).all()
                    assert (lo <= point).all() and (point <= hi).all()
                    if fold == 18:
                        if scenario == 'weather_stress':
                            pd.DataFrame(dict(ds=future.ds.to_numpy(), forecast=point, lower_95=lo, upper_95=hi)).to_csv(OUT/f'forecast_{name}.csv', index=False)
                        continue
                    for h in range(48):
                        records.append(dict(model=name, scenario=scenario, seed=seed, fold=fold,
                            phase='selection' if fold < 17 else 'last_window_audit',
                            ds=data.ds.iloc[origins[fold]+h], horizon=h+1, actual=actual[fold, h],
                            forecast=point[h], lower_95=lo[h], upper_95=hi[h]))
    pred = pd.DataFrame(records)
    _, consistency = compare_windows(pred, OUT)
    rows = []
    for (scenario, phase, name), g in pred.groupby(['scenario', 'phase', 'model']):
        for period, part in [('full_48h', g), ('day_two', g[g.horizon > 24])]:
            rows.append(dict(scenario=scenario, phase=phase, model=name, period=period,
                **metrics(part.actual.to_numpy(), part.forecast.to_numpy(), part.lower_95.to_numpy(), part.upper_95.to_numpy())))
    scores = pd.DataFrame(rows)
    for _, group in scores.groupby(['scenario', 'phase', 'period']):
        scores.loc[group.index, 'RMSE_ratio'] = group.RMSE/group[group.model == 'seasonal_naive'].RMSE.iloc[0]
    scores = scores.sort_values(['scenario', 'phase', 'period', 'RMSE'])
    selection = scores.query("scenario == 'weather_stress' and phase == 'selection' and period == 'full_48h'")
    winner = selection.iloc[0].model
    best_blend = selection[selection.model.str.contains('_snaive')].iloc[0].model
    pd.read_csv(OUT/f'forecast_{winner}.csv').to_csv(OUT/'selected_model_forecast.csv', index=False)
    pd.read_csv(OUT/f'forecast_{best_blend}.csv').to_csv(OUT/'best_blend_forecast.csv', index=False)
    scores.to_csv(OUT/'model_comparison.csv', index=False)
    pred.to_csv(OUT/'validation_predictions.csv', index=False)
    (OUT/'run_summary.json').write_text(json.dumps(dict(selected_model=winner, best_blend=best_blend,
        candidates=len(raw), days=[21, 60, 120], trees=120, min_samples_leaf=3, max_features=.9, random_state=42,
        selection='Lowest full-48h weather-stress RMSE over folds 6–16; audit fold17 excluded.',
        intervals='Expanding earlier errors per forecast day; official SNaive intervals.',
        caveats=['Reused validation windows; weather stress is hypothetical; audit previously seen.']), indent=2))
    print(selection.head(10)[['model', 'RMSE', 'RMSE_ratio']].to_string(index=False))
    print('SELECTED', winner, 'BEST BLEND', best_blend)


if __name__ == '__main__':
    run()
