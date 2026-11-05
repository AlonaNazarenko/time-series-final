"""Large, reproducible SNaive combination search on common saved rolling windows.

Search ranks point forecasts only. The latest audit never selects weights.
Final intervals use completed historical errors, without an independent coverage test.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from session1_models import interval

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results_snaive_search'


def load():
    frames = [pd.read_csv(ROOT / 'results_timestral/validation_predictions.csv'),
              pd.read_csv(ROOT / 'results_task/validation_predictions.csv')]
    names, arrays, futures = [], [], []
    reference = None
    for source, frame in enumerate(frames):
        for name, group in frame.groupby('model', sort=False):
            if 'interval' in group:
                group = group[group.interval == group.interval.iloc[0]]
            group = group.sort_values(['fold', 'seed', 'horizon'])
            keys = group[['fold', 'seed', 'horizon', 'ds']].reset_index(drop=True)
            if reference is None:
                reference = keys
                actual = group.actual.to_numpy().reshape(12, 4, 48)
            pd.testing.assert_frame_equal(keys, reference)
            np.testing.assert_allclose(group.actual.to_numpy().reshape(12, 4, 48), actual)
            array = group.forecast.to_numpy().reshape(12, 4, 48)
            if any(np.allclose(array, other, rtol=0, atol=1e-7) for other in arrays):
                continue
            paths = []
            if source == 1:
                paths += sorted((ROOT / 'results_task').glob(f'forecast_{name}__*.csv'))
            for folder in ['results_timestral', 'results_stacking', 'results_hf', 'results_robust']:
                paths += [ROOT / folder / f'forecast_{name}.csv', ROOT / folder / f'point_{name}.csv']
            path = next((p for p in paths if p.exists()), None)
            if path is None:
                raise FileNotFoundError(f'Future forecast missing for {name}')
            f = pd.read_csv(path)
            assert len(f) == 48
            if futures:
                assert f.ds.tolist() == future_dates
            future_dates = f.ds.tolist()
            names.append(name); arrays.append(array); futures.append(f.forecast.to_numpy())
    assert names[0] == 'seasonal_naive'
    return names, np.stack(arrays, axis=-1), actual, np.stack(futures, axis=-1), future_dates


def candidates(m):
    """Weights may differ by forecast day; every new blend contains SNaive."""
    weights, families = [], []
    def add(w, family):
        weights.append(np.broadcast_to(w, (2, m)).copy()); families.append(family)
    for j in range(m):
        w = np.zeros(m); w[j] = 1; add(w, 'single_control')
    for j in range(1, m):
        for n in range(1, 20):
            w = np.zeros(m); w[0] = n/20; w[j] = 1-n/20
            add(w, 'pair_grid_5pct')
        for d1 in range(1, 10):
            for d2 in range(1, 10):
                if d1 == d2:
                    continue
                w = np.zeros((2, m)); w[:, 0] = [d1/10, d2/10]; w[:, j] = 1-w[:, 0]
                add(w, 'pair_daywise_10pct')
    for a, b in combinations(range(1, m), 2):
        for n in range(1, 10):
            for k in range(1, 10-n):
                w = np.zeros(m); w[0] = n/10; w[a] = k/10; w[b] = 1-w[0]-w[a]
                add(w, 'triple_grid_10pct')
    rng = np.random.default_rng(20261105)
    for size in [3, 4, 5, 8]:
        for _ in range(5000):
            indices = np.r_[0, rng.choice(np.arange(1, m), size-1, replace=False)]
            parts = rng.dirichlet(np.ones(size)); parts = .95*parts; parts[0] += .05
            w = np.zeros(m); w[indices] = parts; add(w, f'random_{size}_members')
    return np.asarray(weights), np.asarray(families)


def describe(w, names):
    return json.dumps({f'day{d+1}': {names[j]: round(float(v), 8) for j, v in enumerate(w[d]) if v > 1e-10}
                       for d in range(2)}, ensure_ascii=False)


def predict(x, w):
    return np.concatenate([x[..., :24, :] @ w[0], x[..., 24:, :] @ w[1]], axis=-1)


def run():
    OUT.mkdir(exist_ok=True)
    names, x, actual, future, dates = load()
    weights, families = candidates(len(names))
    assert np.isfinite(weights).all() and (weights >= -1e-12).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1)
    print(f'{len(names)} unique base models; {len(weights):,} combinations', flush=True)
    # Quadratic loss calculates exactly the RMSE of convex combinations, without
    # storing hundreds of millions of hourly candidate predictions.
    error = x-actual[..., None]
    cov = np.empty((12, 2, 2, len(names), len(names)))
    for fold in range(12):
        for scenario, seeds in enumerate([[0], [1, 2, 3]]):
            for day in range(2):
                e = error[fold, seeds, day*24:(day+1)*24, :].reshape(-1, len(names))
                cov[fold, scenario, day] = e.T @ e/len(e)
    losses = np.empty((len(weights), 12, 2, 2))
    for start in range(0, len(weights), 1000):
        w = weights[start:start+1000]
        losses[start:start+len(w)] = np.maximum(np.einsum('cdi,fsdij,cdj->cfsd', w, cov, w, optimize=True), 0)
        if start % 20000 == 0:
            print(f'Scored {min(start+1000, len(weights)):,}/{len(weights):,}', flush=True)
    # Verify the efficient scoring against direct hourly forecast arithmetic.
    for i in [0, len(names), len(weights)-1]:
        direct = (predict(x, weights[i])-actual)**2
        np.testing.assert_allclose(losses[i, :, 1].mean(axis=-1), direct[:, 1:].mean(axis=(1, 2)), rtol=1e-10)
    selection = losses[:, :11, 1].mean(axis=(1, 2))
    rank = np.argsort(selection, kind='stable')
    table = pd.DataFrame(dict(candidate=np.arange(len(weights)), family=families,
        RMSE=np.sqrt(selection), day_two_RMSE=np.sqrt(losses[:, :11, 1, 1].mean(axis=1)),
        observed_RMSE=np.sqrt(losses[:, :11, 0].mean(axis=(1, 2))),
        wins=(losses[:, :11, 1].mean(axis=-1) < losses[0, :11, 1].mean(axis=-1)-1e-9).sum(axis=1),
        worst_window_ratio=np.sqrt((losses[:, :11, 1].mean(axis=-1)/losses[0, :11, 1].mean(axis=-1)).max(axis=1)),
        audit_RMSE=np.sqrt(losses[:, 11, 1].mean(axis=-1))))
    table['RMSE_ratio'] = table.RMSE/table.RMSE.iloc[0]
    table['weights'] = [describe(w, names) for w in weights]
    table.iloc[rank].to_csv(OUT / 'all_combinations.csv', index=False)
    table.iloc[rank[:100]].to_csv(OUT / 'top100.csv', index=False)
    table.iloc[rank].groupby('family', sort=False).head(1).to_csv(OUT / 'best_by_family.csv', index=False)
    np.savez_compressed(OUT / 'search_arrays.npz', weights=weights, losses=losses, models=np.array(names))
    # Nested chronological search: choose from this entire grid using only
    # earlier completed selection blocks, never the block being predicted.
    forward_rows = []
    for method in ['expanding', 'recent4']:
        for fold in range(1, 12):
            start = max(0, fold-4) if method == 'recent4' else 0
            chosen = int(np.argmin(losses[:, start:fold, 1].mean(axis=(1, 2))))
            forward_rows.append(dict(method=method, fold=fold+6, candidate=chosen,
                phase='selection' if fold < 11 else 'last_window_audit',
                MSE=float(losses[chosen, fold, 1].mean()),
                baseline_MSE=float(losses[0, fold, 1].mean()), weights=describe(weights[chosen], names)))
    forward = pd.DataFrame(forward_rows)
    forward.to_csv(OUT / 'chronological_grid_selection.csv', index=False)
    forward_summary = []
    for (method, phase), group in forward.groupby(['method', 'phase']):
        forward_summary.append(dict(method=method, phase=phase, windows=len(group),
            RMSE=float(np.sqrt(group.MSE.mean())), baseline_RMSE=float(np.sqrt(group.baseline_MSE.mean())),
            wins=int((group.MSE < group.baseline_MSE).sum())))
    pd.DataFrame(forward_summary).to_csv(OUT / 'chronological_summary.csv', index=False)
    # Report interval-free MAPE and day-two scores for the top 100 only.
    detail = []
    for index in rank[:100]:
        p = predict(x, weights[index])
        for phase, sl in [('selection', slice(0, 11)), ('last_window_audit', slice(11, 12))]:
            for period, hs in [('full_48h', slice(0, 48)), ('day_two', slice(24, 48))]:
                a, b = actual[sl, 1:, hs], p[sl, 1:, hs]
                detail.append(dict(candidate=int(index), phase=phase, period=period,
                    RMSE=float(np.sqrt(np.mean((a-b)**2))), MAPE=float(np.mean(np.abs((a-b)/a))*100)))
    pd.DataFrame(detail).to_csv(OUT / 'top100_detailed_metrics.csv', index=False)
    winner = int(rank[0])
    window_rows = []
    for index in [winner, 0, names.index('timestral_28m_weather_504h')]:
        for fold in range(12):
            window_rows.append(dict(candidate=int(index), fold=fold+6,
                phase='selection' if fold < 11 else 'last_window_audit',
                RMSE=float(np.sqrt(losses[index, fold, 1].mean())),
                day_two_RMSE=float(np.sqrt(losses[index, fold, 1, 1]))))
    pd.DataFrame(window_rows).to_csv(OUT / 'selected_window_comparison.csv', index=False)
    point = predict(future, weights[winner])
    residual = (actual-predict(x, weights[winner]))[:, 1:, :].reshape(-1, 48)
    lo, hi = interval(point, list(residual))
    output = pd.DataFrame(dict(ds=dates, forecast=point, lower_95=lo, upper_95=hi))
    assert len(output) == 48 and np.isfinite(output.iloc[:, 1:]).all().all()
    assert (lo >= 0).all() and (lo <= point).all() and (point <= hi).all()
    output.to_csv(OUT / 'selected_model_forecast.csv', index=False)
    report = dict(base_models=names, combinations=len(weights), selected_candidate=winner,
        selected_weights=json.loads(describe(weights[winner], names)), random_seed=20261105,
        selection='Lowest pooled weather-stress RMSE on folds 6–16; audit fold 17 excluded.',
        scope='Single controls; SNaive pair 5% grid; daywise pair 10% grid; all SNaive triples 10% simplex grid; 20,000 random 3/4/5/8-member convex blends.',
        chronology='Grid selection also evaluated forward from fold 7, using only previous completed blocks. First six calibration blocks unavailable for some bases and not used in this search.',
        intervals='Final intervals calibrated using all 12 completed blocks. No independent interval coverage evaluation; search ranks point forecasts only.',
        caveats=['Large search on repeatedly reused windows increases selection optimism.',
                 'Weather-stress draws are hypothetical, not independent historical windows.',
                 'Audit previously seen; forward grid evaluation still inherits prior model-design choices.'])
    (OUT / 'run_summary.json').write_text(json.dumps(report, indent=2))
    print(table.iloc[rank[:12]].drop(columns='weights').to_string(index=False))
    print('WINNER', describe(weights[winner], names), flush=True)
    print('CHRONOLOGICAL CHECK\n', pd.DataFrame(forward_summary).to_string(index=False))


if __name__ == '__main__':
    run()
