"""Training variants for the sun-time boosting model, driven by the error diagnosis.

Variants are chosen on the 30 early windows (never used for selection before); the 11
selection windows and the audit only confirm the choice afterwards.
"""
import numpy as np, pandas as pd, time
from session1_sun_fit import (rows, stack, train_ids, built, hgb, perturb, data, future, N, y, OUT)

t0 = time.time()
noisy = {i: rows(i, perturb(data.iloc[i:i+48], int(i)*10+7)) for i in train_ids if i+48 <= N}
print(f'noisy rows {time.time()-t0:.0f}s', flush=True)
EPS = 5.


def stack_noisy(o, days):
    ids = [i for i in train_ids if i >= o-24*days and i+48 <= o]
    out = stack(o, days)
    extra = pd.concat([noisy[i]['x'] for i in ids], ignore_index=True)
    out['x'] = pd.concat([out['x'], extra], ignore_index=True)
    for k in ['S', 'cd', 'cn', 'i', 't']:
        out[k] = np.r_[out[k], out[k]]
    return out


def member(tr, o, Vs, log=False, half=None, d2=False, loss='absolute_error', bag=None):
    x, t, S, i = tr['x'], tr['t'], tr['S'], tr['i']
    w = np.ones(len(t))
    if half:
        w = w*.5**(((o-i)/24)/half)
    if bag is not None:
        u, inv = np.unique(i, return_inverse=True)
        w = w*np.random.default_rng(bag).multinomial(len(u), np.full(len(u), 1/len(u)))[inv]
    keep = (w > 0) & ((x.h >= 24).to_numpy() if d2 else True)
    z = np.log((t+EPS)/S) if log else t/S
    m = hgb(loss).fit(x[keep], z[keep], sample_weight=w[keep])
    if log:
        return [np.maximum(np.exp(m.predict(V['x']))*V['S']-EPS, 0) for V in Vs]
    return [np.maximum(m.predict(V['x'])*V['S'], 0) for V in Vs]


def fit_origin(o):
    tgt = data.iloc[o:o+48] if o < N else future
    wxs = [tgt]+[perturb(tgt, int(o)*10+s) if o < N else tgt for s in [1, 2, 3]]
    Vs = [rows(o, w) for w in wxs]
    tr = stack(o, 730); trn = stack_noisy(o, 730)
    r = dict(decay180=member(tr, o, Vs, half=180),
             decay90=member(tr, o, Vs, half=90),
             log_l2=member(tr, o, Vs, log=True, loss='squared_error'),
             log_abs=member(tr, o, Vs, log=True),
             log_abs_decay180=member(tr, o, Vs, log=True, half=180),
             log_abs_decay180_d2=member(tr, o, Vs, log=True, half=180, d2=True),
             log_abs_decay180_noise=member(trn, o, Vs, log=True, half=180))
    bags = [member(tr, o, Vs, log=True, half=180, bag=b) for b in range(5)]
    r['log_abs_decay180_bag5'] = list(np.mean(np.stack(bags), axis=0))
    return {k: np.stack(v) for k, v in r.items()}


if __name__ == '__main__':
    origins = np.arange(N-48*48, N+1, 48)
    res = {}
    for n, o in enumerate(origins):
        for k, v in fit_origin(int(o)).items():
            res.setdefault(k, []).append(v)
        print(n, f'{time.time()-t0:.0f}s', flush=True)
    np.savez_compressed(OUT/'train_points.npz', origins=origins, **{k: np.stack(v) for k, v in res.items()})
    print('finished', flush=True)
