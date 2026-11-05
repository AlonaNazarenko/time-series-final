"""New Timestral-free family, all in sun-relative time. Fits only; session1_sun_evaluate.py scores.

Causality: state/scales at origin i use y[:i]; training origins end at least 48h before o.
Folds: 48 rolling 48h windows ending at the train end; the last 18 are the project's folds.
Clock-event origins start on the day the daylight flag jumps (never used for fitting choices).
"""
import os, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).resolve().parent.parent
from session1_more_models import perturb
OUT = ROOT/'results/sun'; OUT.mkdir(exist_ok=True)
warnings.filterwarnings('ignore')

data = pd.read_csv(ROOT/'train.csv', parse_dates=['ds'])
future = pd.read_csv(ROOT/'weather_forecast.csv', parse_dates=['ds'])
future.index = np.arange(len(data), len(data)+48)
N = len(data); assert N % 24 == 0
allf = pd.concat([data.drop(columns='y'), future]); T = len(allf)
y = data.y.to_numpy(float)
dl = allf.daylight.to_numpy(int)
WX = ['cloud_cover', 'temperature', 'wind_speed', 'pressure', 'humidity', 'air_density']

# Sun-relative geometry from the supplied flag only (no location is inferred or looked up).
rise = np.where((dl[1:] == 1) & (dl[:-1] == 0))[0]+1
sets = np.where((dl[1:] == 0) & (dl[:-1] == 1))[0]+1
assert rise[0] < sets[0] and len(rise) == len(sets) and (sets > rise).all()
phi, since_rise, to_set = np.zeros(T), np.full(T, -5.), np.full(T, -5.)
for r, s in zip(rise, sets):
    ts = np.arange(max(r-1, 0), min(s+1, T))
    phi[ts] = np.sin(np.pi*(ts-(r-1)+.5)/(s-r+2))      # one twilight hour either side
    since_rise[ts], to_set[ts] = ts-r, s-ts
t_all = np.arange(T)
k = np.searchsorted(sets, t_all, side='right')-1
since_set = np.where(k >= 0, t_all-sets[np.maximum(k, 0)], 14.).clip(0, 14.)
k = np.searchsorted(rise, t_all, side='left')
to_rise = np.where(k < len(rise), rise[np.minimum(k, len(rise)-1)]-t_all, 14.).clip(0, 14.)
day_len = pd.Series(dl).groupby(allf.ds.dt.date.to_numpy()).transform('sum').to_numpy(float)
doy = allf.ds.dt.dayofyear.to_numpy()
dates = allf.ds.dt.date.to_numpy()
hours = allf.ds.dt.hour.to_numpy()
day_rows = allf[allf.daylight == 1]
first = day_rows.groupby(day_rows.ds.dt.date).ds.apply(lambda s: s.dt.hour.min()).to_dict()
last = day_rows.groupby(day_rows.ds.dt.date).ds.apply(lambda s: s.dt.hour.max()).to_dict()


def clock_shift(t, ref):
    """Non-zero only when sunrise AND sunset jump together: a clock change, not the season."""
    a, b = first.get(dates[t]), last.get(dates[t])
    if a is None or ref not in first:
        return 0
    s1, s2 = a-first[ref], b-last[ref]
    return s1 if s1 == s2 else 0


def state(i):
    w = y[i-21*24:i].reshape(-1, 24); night = dl[i-21*24:i].reshape(-1, 24) == 0
    cn = max(float(w[night].mean()), 20.)
    cd = max(float(np.quantile(w.max(1), .8)), 1.2*cn, 50.)
    ln = dl[i-24:i] == 0
    return dict(cd=cd, cn=cn, night_last=y[i-24:i][ln].mean()/cn, peak_last=y[i-24:i].max()/cd,
                mean7=y[i-168:i].mean()/cd, last_y=y[i-1]/cn)


def rows(i, wx=None):
    idx = np.arange(i, i+48); fr = allf.iloc[idx] if wx is None else wx
    s = state(i); ref = dates[i-13]
    sh = np.array([clock_shift(t, ref) for t in idx])
    base = i+np.arange(48) % 24
    lag = {k: y[np.clip(base-24*k-sh, 0, i-1)] for k in (1, 2, 3)}
    S = s['cn']+phi[idx]*(s['cd']-s['cn'])
    f = dict(phi=phi[idx], since_rise=since_rise[idx], to_set=to_set[idx], since_set=since_set[idx],
             to_rise=to_rise[idx], day_len=day_len[idx], doy_s=np.sin(2*np.pi*doy[idx]/365.25),
             doy_c=np.cos(2*np.pi*doy[idx]/365.25), h=np.arange(48.))
    for c in WX:
        f[c] = fr[c].to_numpy(float)
    f['wpow'] = f['wind_speed']**3*f['air_density']
    f['wind2'] = f['wind_speed']**2
    f['cloud3'] = pd.Series(f['cloud_cover']).rolling(3, center=True, min_periods=1).mean().to_numpy()
    f['wind3h'] = pd.Series(f['wind_speed']).rolling(3, center=True, min_periods=1).mean().to_numpy()
    f['clear_sun'] = f['phi']*(1-f['cloud3'])
    for n in ['night_last', 'peak_last', 'mean7', 'last_y']:
        f[n] = np.full(48, s[n])
    f['sl1'], f['sl2'] = lag[1]/S, lag[2]/S
    return dict(x=pd.DataFrame(f), S=S, cd=np.full(48, s['cd']), cn=np.full(48, s['cn']),
                lag1=lag[1], lag3=(lag[1]+lag[2]+lag[3])/3, i=np.full(48, i))


t0 = time.time()
train_ids = np.arange(24*30, N-47, 24)
built = {i: rows(i) for i in train_ids}
print(f'built {len(built)} origins {time.time()-t0:.0f}s', flush=True)


def stack(o, days):
    ids = [i for i in train_ids if i >= o-24*days and i+48 <= o]
    out = {k: np.concatenate([built[i][k] for i in ids]) for k in ['S', 'cd', 'cn', 'i']}
    out['x'] = pd.concat([built[i]['x'] for i in ids], ignore_index=True)
    out['t'] = np.concatenate([y[i:i+48] for i in ids])
    return out


def hgb(loss, **kw):
    return HGB(loss=loss, max_iter=300, learning_rate=.04, max_leaf_nodes=15, min_samples_leaf=30,
               l2_regularization=5, early_stopping=False, random_state=0, **kw)


WF = ['wind_speed', 'wpow', 'wind3h', 'wind2', 'air_density', 'pressure', 'temperature', 'humidity',
      'night_last', 'last_y', 'h', 'doy_s', 'doy_c']
SF = ['phi', 'since_rise', 'to_set', 'day_len', 'doy_s', 'doy_c', 'cloud_cover', 'cloud3', 'clear_sun',
      'temperature', 'humidity', 'peak_last', 'mean7', 'sl1', 'sl2', 'h']
KF = dict(phi=3, cloud3=1.5, wind3h=1.5, temperature=.5, night_last=1, peak_last=.5, doy_s=.5, doy_c=.5)


def gbm_sun(tr, Vs, loss):
    m = hgb(loss).fit(tr['x'], tr['t']/tr['S'])
    return [m.predict(V['x'])*V['S'] for V in Vs]


def physics(tr, Vs, loss):
    x = tr['x']
    dark = (x.phi == 0) & (x.since_set >= 1) & (x.to_rise >= 1)
    mono = [1 if c in ('wind_speed', 'wpow', 'wind3h', 'wind2') else 0 for c in WF]
    wm = hgb(loss, monotonic_cst=mono).fit(x.loc[dark, WF], tr['t'][dark]/tr['cn'][dark])
    wind = np.maximum(wm.predict(x[WF])*tr['cn'], 0)
    sun = (x.phi > 0).to_numpy()
    sm = hgb(loss).fit(x.loc[sun, SF], (tr['t']-wind)[sun]/tr['cd'][sun])
    out = []
    for V in Vs:
        p = np.maximum(wm.predict(V['x'][WF])*V['cn'], 0)
        d = V['x'].phi.to_numpy() > 0
        if d.any():
            p[d] += np.maximum(sm.predict(V['x'].loc[d, SF]), 0)*V['cd'][d]
        out.append(p)
    return out


def analog(tr, Vs):
    c, w = list(KF), np.array(list(KF.values()))
    mu, sd = tr['x'][c].mean().to_numpy(), tr['x'][c].std().to_numpy()+1e-9
    m = KNeighborsRegressor(n_neighbors=60, weights='distance').fit((tr['x'][c].to_numpy()-mu)/sd*w, tr['t']/tr['S'])
    return [m.predict((V['x'][c].to_numpy()-mu)/sd*w)*V['S'] for V in Vs]


def ridge_design(x, lag1):
    return np.column_stack([x.phi, x.phi**2, x.phi*x.cloud3, x.clear_sun, x.phi*x.temperature, x.phi*x.day_len,
                            x.wind_speed, x.wind2, x.wpow, x.since_set.clip(0, 3), x.to_rise.clip(0, 3), lag1, x.h])


def decay_ridge(o, tr, Vs):
    keep = tr['i'] >= o-365*24
    age = (o-tr['i'][keep])/24.
    x = tr['x'][keep]; lag1 = (x.sl1*tr['S'][keep]).to_numpy()
    m = make_pipeline(StandardScaler(), Ridge(alpha=30))
    m.fit(ridge_design(x, lag1), tr['t'][keep], ridge__sample_weight=.5**(age/45))
    return [m.predict(ridge_design(V['x'], V['lag1'])) for V in Vs]


def fit_origin(o):
    tgt = data.iloc[o:o+48] if o < N else future
    wxs = [tgt]+[perturb(tgt, int(o)*10+s) if o < N else tgt for s in [1, 2, 3]]
    Vs = [rows(o, w) for w in wxs]
    tr = stack(o, 730)
    r = dict(gbm_sun=gbm_sun(tr, Vs, 'squared_error'), gbm_sun_med=gbm_sun(tr, Vs, 'absolute_error'),
             phys=physics(tr, Vs, 'squared_error'), phys_med=physics(tr, Vs, 'absolute_error'),
             analog=analog(tr, Vs), decay_ridge=decay_ridge(o, tr, Vs),
             sun_snaive=[V['lag1'] for V in Vs], sun_snaive3=[V['lag3'] for V in Vs],
             snaive=[np.tile(y[o-24:o], 2)]*4)
    return {k: np.maximum(np.stack(v), 0) for k, v in r.items()}, Vs[0]['S'], phi[o:o+48]


if __name__ == '__main__':
    origins = np.arange(N-48*48, N+1, 48)
    jumps = [pd.Timestamp(d) for d in ['2024-10-27', '2025-03-31', '2025-10-27', '2026-03-31']]
    events = np.array([allf.index[allf.ds == j+pd.Timedelta(hours=1)][0] for j in jumps])
    assert (events % 24 == 0).all()
    if os.environ.get('SMOKE'):
        for o in [N, events[0]]:
            t1 = time.time(); r, S, p = fit_origin(int(o))
            print({k: (v.shape, round(float(v.mean()), 1)) for k, v in r.items()}, f'{time.time()-t1:.1f}s', flush=True)
        sys.exit()
    res = {}
    for n, o in enumerate(np.r_[origins, events]):
        r, S, p = fit_origin(int(o))
        for k, v in r.items():
            res.setdefault(k, []).append(v)
        res.setdefault('S', []).append(S); res.setdefault('phi', []).append(p)
        print(n, f'{time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(OUT/'points.npz', origins=origins, events=events,
                        **{k: np.stack(v) for k, v in res.items()})
    print('finished', flush=True)
