"""Score Timestral-free candidates on project windows, clock events, and interval methods."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent
from session1_snaive_search import load
from session1_models import baseline
OUT = ROOT/'results/sun'
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 100)
z = np.load(OUT/'points.npz'); origins, events = z['origins'], z['events']
data = pd.read_csv(ROOT/'train.csv', parse_dates=['ds']); y = data.y.to_numpy(float); N = len(y)
BASE = ['gbm_sun', 'gbm_sun_med', 'phys', 'phys_med', 'analog', 'decay_ridge', 'sun_snaive', 'sun_snaive3', 'snaive']
n = len(origins); ne = len(events)
P = {k: z[k] for k in BASE}; S, phi = z['S'], z['phi']
assert origins[-1] == N and n == 49
names, x, actual, fut, dates = load(); actual = actual[:, 0]
A = np.stack([y[o:o+48] for o in origins[:-1]]); np.testing.assert_allclose(A[36:], actual)
AE = np.stack([y[o:o+48] for o in events])
OLD = {'chronos': 'chronos_bolt_small_504h', 'sarima': 'SARIMA_21d', 'hour_ridge60': 'hour_weather_ridge_60d'}
assert all('timestral' not in v for v in OLD.values())

def mix(**w):
    return lambda: sum(v*P[k] for k, v in w.items())
C = {k: P[k] for k in BASE}
new4 = ['gbm_sun', 'phys', 'analog', 'decay_ridge']; med4 = ['gbm_sun_med', 'phys_med', 'analog', 'decay_ridge']
C['mean_new4'] = sum(P[k] for k in new4)/4
C['mean_med4'] = sum(P[k] for k in med4)/4
C['mean_gbm_phys'] = (P['gbm_sun']+P['phys'])/2
C['mean_gbm_phys_med'] = (P['gbm_sun_med']+P['phys_med'])/2
C['mean_gbm_phys_ana_med'] = (P['gbm_sun_med']+P['phys_med']+P['analog'])/3
C['med4_snaive10'] = .9*C['mean_med4']+.1*P['sun_snaive']
C['gbmphysmed_snaive20'] = .8*C['mean_gbm_phys_med']+.2*P['sun_snaive']
ST, SEL = [1, 2, 3], list(range(36, 47)); AUD = 47
def old_blend(frac):
    """Equal mean of the new median family (frac) with Chronos+SARIMA (1-frac); selection folds only."""
    o = (x[..., names.index(OLD['chronos'])]+x[..., names.index(OLD['sarima'])])/2
    return frac*C['mean_med4'][36:48]+(1-frac)*o, frac*C['mean_med4'][48, 0]+(1-frac)*(fut[:, names.index(OLD['chronos'])]+fut[:, names.index(OLD['sarima'])])/2

def score(name, p, offset=0):
    """p indexed so that fold f of the 48-window scheme is p[f-offset]."""
    e = p[[f-offset for f in range(36, 48)]]-A[36:, None]
    i = np.arange(11)
    m = (e[i][:, ST]**2).mean(axis=(1, 2)); b = ((P['snaive'][36:47]-A[36:47, None])[:, ST]**2).mean(axis=(1, 2))
    d = e[i][:, ST][..., 24:]
    return dict(model=name, RMSE=np.sqrt(m.mean()), d2_RMSE=np.sqrt((d**2).mean()), obs_RMSE=np.sqrt((e[i, 0]**2).mean()),
                d2_MAPE=(np.abs(d)/A[36:47, None, 24:]).mean()*100, d2_MAPE_med=np.median(np.abs(d)/A[36:47, None, 24:])*100,
                wins=int((m < b-1e-9).sum()), worst=np.sqrt((m/b).max()),
                aud=np.sqrt((e[11, ST]**2).mean()), aud_d2=np.sqrt((e[11, ST][..., 24:]**2).mean()), bias_d2=d.mean())
rows = [score(k, v) for k, v in C.items()]
for f in [.5, .75]:
    p, _ = old_blend(f); rows.append(score(f'med4_{int(f*100)}_chronos_sarima', p, offset=36))
t = pd.DataFrame(rows).sort_values('RMSE').round(2); t.to_csv(OUT/'point_scores.csv', index=False)
print(t.to_string(index=False))

print('\nCLOCK EVENTS: 48h starting on the day the flag jumps; RMSE full / day two; observed weather and stress')
ev = []
for name in list(C):
    p = C[name][n:n+ne]
    r = {'model': name}
    for j, o in enumerate(events):
        lab = str(data.ds.iloc[o].date())
        r[lab] = np.sqrt(((p[j, ST]-AE[j])**2).mean())
        r[lab+'_d2'] = np.sqrt(((p[j, ST]-AE[j])[..., 24:]**2).mean())
    r['mean_ratio_vs_snaive'] = np.mean([np.sqrt(((p[j, ST]-AE[j])**2).mean())/np.sqrt(((P['snaive'][n+j, 0]-AE[j])**2).mean()) for j in range(ne)])
    ev.append(r)
ev = pd.DataFrame(ev).sort_values('mean_ratio_vs_snaive').round(2); ev.to_csv(OUT/'clock_events.csv', index=False)
print(ev.to_string(index=False))

# Intervals: at fold f calibrate only on folds < f (from fold 0 => 36+ earlier windows for selection).
def tier(ph):
    return np.where(ph == 0, 0, np.where(ph < .6, 1, 2))
def widths(p, f, method):
    """Return (lo, hi) arrays for all seeds at fold f, shape (4,48)."""
    e = (A[:f, None]-p[:f])[:, ST]                  # signed error: actual-forecast
    day = np.repeat([0, 1], 24); pf = p[f]
    if method == 'absolute_day':
        h = np.array([np.quantile(np.abs(e[..., day == d]), .95, method='higher') for d in (0, 1)])[day]
        return np.maximum(pf-h, 0), pf+h
    T, Tf = tier(phi[:f]), tier(phi[f])
    lo, hi = np.zeros(48), np.zeros(48)
    for d in (0, 1):
        for b in (0, 1, 2):
            pick = (T == b)[:, None, :].repeat(3, 1) & (day == d)[None, None, :]
            here = (Tf == b) & (day == d)
            if not here.any():
                continue
            sc = (S[:f][:, None, :].repeat(3, 1)[pick] if method.endswith('scaled') else 1.)
            v = e[pick]/sc
            f_sc = S[f][here] if method.endswith('scaled') else 1.
            if method.startswith('tier_sym'):
                q = np.quantile(np.abs(v), .95, method='higher'); lo[here] = -q*f_sc; hi[here] = q*f_sc
            else:
                lo[here] = np.quantile(v, .025)*f_sc; hi[here] = np.quantile(v, .975)*f_sc
    return np.maximum(pf+lo, 0), np.maximum(pf+hi, pf)
def wink(a, lo, hi):
    return hi-lo+40*np.maximum(lo-a, 0)+40*np.maximum(a-hi, 0)
res = []
for name in ['mean_med4', 'mean_gbm_phys_med', 'mean_gbm_phys_ana_med']:
    p = C[name][:48]
    for meth in ['absolute_day', 'tier_sym', 'tier_sym_scaled', 'tier_asym', 'tier_asym_scaled']:
        W, Cv, Wd = [], [], []
        for f in SEL:
            lo, hi = widths(p, f, meth)
            w = wink(A[f], lo[ST], hi[ST]); c = (A[f] >= lo[ST]) & (A[f] <= hi[ST])
            W.append(w.mean()); Wd.append(w[..., 24:].mean()); Cv.append(c[..., 24:].mean())
        lo, hi = widths(p, AUD, meth)
        res.append(dict(model=name, interval=meth, W48=np.mean(W), W_d2=np.mean(Wd), cov_d2=np.mean(Cv),
                        width_d2=(hi[ST]-lo[ST])[..., 24:].mean(), aud_W_d2=wink(A[AUD], lo[ST], hi[ST])[..., 24:].mean()))
print('\nINTERVALS: 11 selection windows, stress weather; compare with README Timestral day-two Winkler 1142.90, SNaive 2367.15')
r = pd.DataFrame(res).round(3); r.to_csv(OUT/'interval_scores.csv', index=False); print(r.to_string(index=False))
