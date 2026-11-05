"""Final candidate: four Timestral-free models + 10% SNaive, anchored to the last observation,
with recent, half-level-scaled intervals. Writes results/sun/final_candidate.csv."""
import numpy as np, pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parent.parent; OUT = R/'results/sun'
d = pd.read_csv(R/'train.csv', parse_dates=['ds']); y = d.y.to_numpy(float); N = len(y)
w = pd.read_csv(R/'weather_forecast.csv')
z = np.load(OUT/'points.npz'); t = np.load(OUT/'train_points.npz'); origins = z['origins']
assert (t['origins'] == origins).all() and origins[-1] == N
A = np.stack([y[o:o+48] for o in origins[:-1]]); phi = z['phi'][:len(origins)]; ST = [1, 2, 3]
L = np.array([y[o-168:o].mean() for o in origins])
sn = z['snaive'][:len(origins)]
RHO = .9


def anchor(p):
    """Pull the first hours toward the last observed value; the gap decays by RHO per hour."""
    last = y[origins-1][:, None, None]; h = np.arange(1, 49)
    return np.maximum(p+(last-p[..., :1])*RHO**h, 0)


n = len(origins)
members = dict(gbm=t['log_abs_decay180_bag5'][:n], phys=z['phys_med'][:n], analog=z['analog'][:n], ridge=z['decay_ridge'][:n])
raw = .9*sum(members.values())/4+.1*sn
cand = anchor(raw)
tier = lambda p: np.where(p == 0, 0, np.where(p < .6, 1, 2))
day = np.repeat([0, 1], 24)


def interval(p, f, K=16, level=.5):
    s = max(0, f-K); E = np.abs((A[s:f, None]-p[s:f])[:, ST])/((L[s:f]/L[f])**level)[:, None, None]
    T = tier(phi[s:f]); h = np.zeros(48)
    for dd in (0, 1):
        for b in (0, 1, 2):
            here = (tier(phi[f]) == b) & (day == dd)
            if here.any():
                pick = (T == b)[:, None, :].repeat(3, 1) & (day == dd)[None, None, :]
                h[here] = np.quantile(E[pick], .95, method='higher')
    return np.maximum(p[f]-h, 0), p[f]+h


def metrics(a, p, lo, hi):
    return dict(RMSE=np.sqrt(((a-p)**2).mean()), MAPE=(np.abs(a-p)/a).mean()*100,
                Winkler=(hi-lo+40*np.maximum(lo-a, 0)+40*np.maximum(a-hi, 0)).mean(), coverage=((a >= lo) & (a <= hi)).mean())


if __name__ == '__main__':
    rows = []
    for name, p in [('raw_blend', raw), ('anchored_blend', cand)]:
        for phase, folds in [('selection', range(36, 47)), ('audit', [47])]:
            for part, sl in [('48h', slice(0, 48)), ('day2', slice(24, 48)), ('h1-6', slice(0, 6))]:
                a, pp, lo, hi = [], [], [], []
                for f in folds:
                    l, u = interval(p, f)
                    for s in ST:
                        a.append(A[f][sl]); pp.append(p[f, s][sl]); lo.append(l[s][sl]); hi.append(u[s][sl])
                rows.append(dict(model=name, phase=phase, part=part, **metrics(*[np.concatenate(v) for v in (a, pp, lo, hi)])))
    tab = pd.DataFrame(rows).round(2); tab.to_csv(OUT/'final_candidate_scores.csv', index=False); print(tab.to_string(index=False))
    lo, hi = interval(cand, 48); pt = cand[48, 0]
    f = pd.DataFrame(dict(ds=w.ds, forecast=pt, lower_95=lo[0], upper_95=hi[0]))
    assert list(f.columns) == ['ds', 'forecast', 'lower_95', 'upper_95'] and len(f) == 48 and f.notna().all().all()
    assert (f.lower_95 >= 0).all() and (f.lower_95 <= f.forecast).all() and (f.forecast <= f.upper_95).all()
    f.to_csv(OUT/'final_candidate.csv', index=False)
    print('first hours', np.round(pt[:4], 0), 'last obs', y[-1], '| mean d1 %.0f d2 %.0f | width d1 %.0f d2 %.0f' % (pt[:24].mean(), pt[24:].mean(), (hi[0]-lo[0])[:24].mean(), (hi[0]-lo[0])[24:].mean()))
