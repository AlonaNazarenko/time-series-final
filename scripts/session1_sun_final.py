"""Interval recalibration for the student's own candidates. Uses only project data and saved predictions."""
import numpy as np, pandas as pd
from pathlib import Path
R=Path(__file__).resolve().parent.parent; OUT=R/'results/sun'
pd.set_option('display.width',200); pd.set_option('display.max_columns',30); pd.set_option('display.max_rows',60)
z=np.load(OUT/'points.npz'); origins=z['origins']; n=49
d=pd.read_csv(R/'train.csv',parse_dates=['ds']); y=d.y.to_numpy(float); w=pd.read_csv(R/'weather_forecast.csv')
A=np.stack([y[o:o+48] for o in origins[:-1]]); phi=z['phi'][:n]; S=z['S'][:n]
L=np.array([y[o-168:o].mean() for o in origins])          # causal recent level at each origin
ET=np.full((n,4,48),np.nan)
e=np.load(R/'results/extra_snaive/extra_trees_60d_points.npz'); assert (e['origins']==origins[30:]).all()
sn=z['snaive'][:n]
ET[30:]=.6*e['points']+.4*sn[30:]
C={'boost_phys_snaive20':(.8*(z['gbm_sun_med'][:n]+z['phys_med'][:n])/2+.2*sn,0),
   'four_new_snaive10':(.9*(z['gbm_sun_med'][:n]+z['phys_med'][:n]+z['analog'][:n]+z['decay_ridge'][:n])/4+.1*sn,0),
   'et60_snaive40':(ET,30)}
ST=[1,2,3]; SEL=range(36,47); AUD=47
tier=lambda p:np.where(p==0,0,np.where(p<.6,1,2))
day=np.repeat([0,1],24)
def iv(p,f,start,bytier,level,K=None):
    s=max(start,f-K) if K else start
    E=np.abs((A[s:f,None]-p[s:f])[:,ST])                  # (folds,3,48)
    sc=((L[s:f]/L[f])**level)[:,None,None] if level else 1.        # express old errors at today's level
    E=E/sc
    T=tier(phi[s:f]); h=np.zeros(48)
    for dd in (0,1):
        if not bytier:
            h[day==dd]=np.quantile(E[...,day==dd],.95,method='higher'); continue
        for b in (0,1,2):
            here=(tier(phi[f])==b)&(day==dd)
            if here.any():
                pick=(T==b)[:,None,:].repeat(3,1)&(day==dd)[None,None,:]
                h[here]=np.quantile(E[pick],.95,method='higher')
    return np.maximum(p[f]-h,0),p[f]+h
M={'absolute_day':(0,0,None),'tier':(1,0,None),'tier_recent16':(1,0,16),'level':(0,1,None),'tier_level':(1,1,None),'recent_half_level':(1,.5,16)}
wk=lambda a,l,u:u-l+40*np.maximum(l-a,0)+40*np.maximum(a-u,0)
rows=[]
for name,(p,st) in C.items():
    for m,(bt,lv,K) in M.items():
        W=[];Wf=[];Cv=[];Wd=[]
        for f in SEL:
            lo,hi=iv(p,f,st,bt,lv,K); r=wk(A[f],lo[ST],hi[ST])
            W.append(r[...,24:].mean()); Wf.append(r.mean()); Cv.append(((A[f]>=lo[ST])&(A[f]<=hi[ST]))[...,24:].mean()); Wd.append((hi[ST]-lo[ST])[...,24:].mean())
        lo,hi=iv(p,AUD,st,bt,lv,K); ra=wk(A[AUD],lo[ST],hi[ST])
        cv=((A[AUD]>=lo[ST])&(A[AUD]<=hi[ST]))[...,24:].mean()
        l2,h2=iv(p,48,st,bt,lv,K)
        rows.append(dict(model=name,interval=m,W_d2=np.mean(W),W_48=np.mean(Wf),cov_d2=np.mean(Cv),width_d2=np.mean(Wd),aud_W_d2=ra[...,24:].mean(),aud_cov=cv,fut_w_d1=(h2-l2)[0,:24].mean(),fut_w_d2=(h2-l2)[0,24:].mean()))
t=pd.DataFrame(rows).round(3); t.to_csv(OUT/'final_intervals.csv',index=False); print(t.to_string(index=False))
for name,(p,st) in C.items():
    sub=t[t.model==name].sort_values('W_d2').iloc[0]; bt,lv,K=M[sub.interval]
    lo,hi=iv(p,48,st,bt,lv,K); pt=p[48,0]; lo,hi=lo[0],hi[0]
    f=pd.DataFrame(dict(ds=w.ds,forecast=pt,lower_95=lo,upper_95=hi))
    assert list(f.columns)==['ds','forecast','lower_95','upper_95'] and len(f)==48 and f.notna().all().all() and (f.lower_95>=0).all() and (f.lower_95<=f.forecast).all() and (f.forecast<=f.upper_95).all()
    f.to_csv(OUT/f'candidate_{name}.csv',index=False); print(name,'->',sub.interval,'format OK')
p,st=C['four_new_snaive10']; lo,hi=iv(p,48,st,1,.5,16); pt=p[48,0]; lo,hi=lo[0],hi[0]
f=pd.DataFrame(dict(ds=w.ds,forecast=pt,lower_95=lo,upper_95=hi))
assert list(f.columns)==['ds','forecast','lower_95','upper_95'] and len(f)==48 and f.notna().all().all() and (f.lower_95>=0).all() and (f.lower_95<=f.forecast).all() and (f.forecast<=f.upper_95).all() and (f.ds==w.ds).all()
f.to_csv(OUT/'candidate_final_four_new_half_level.csv',index=False)
print('final candidate OK: mean d1 %.0f d2 %.0f width d1 %.0f d2 %.0f'%(pt[:24].mean(),pt[24:].mean(),(hi-lo)[:24].mean(),(hi-lo)[24:].mean()))
