"""Targeted Timestral checks: shorter context, clock covariate, larger model."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import json,argparse,time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi,snapshot_download
from timestral import TimestralForecaster
OUT=ROOT/'results/task'
SETTINGS={'timestral28_recent7d':('mistralai/Timestral-28M',168,False),
          'timestral28_clock21d':('mistralai/Timestral-28M',504,True),
          'timestral123_weather21d':('mistralai/Timestral-123M',504,False)}


def download():
    OUT.mkdir(exist_ok=True)
    old=json.loads((ROOT/'results/timestral'/'model_versions.json').read_text())
    file=OUT/'model_versions.json'
    if file.exists():old.update(json.loads(file.read_text()))
    versions={}
    for model in sorted(set(v[0] for v in SETTINGS.values())):
        sha=old[model]['revision'] if model in old else HfApi().model_info(model).sha
        path=snapshot_download(model,revision=sha,allow_patterns=['config.json','model.safetensors'])
        versions[model]=dict(revision=sha,local_path=str(Path(path).resolve()))
        print('Ready',model,flush=True)
    file.write_text(json.dumps(versions,indent=2))


def infer():
    torch.set_num_threads(4);torch.manual_seed(42)
    inp=np.load(ROOT/'results/timestral'/'inputs.npz')
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds']);future=pd.read_csv(ROOT/'weather_forecast.csv',parse_dates=['ds'])
    clock=[]
    for i,origin in enumerate(inp['origins']):
        target=data.iloc[origin:origin+48] if i<18 else future
        clock.append(np.r_[data.ds.iloc[origin-504:origin].dt.hour,target.ds.dt.hour][None,:])
    clock=np.stack(clock).astype('float32')
    np.savez(OUT/'clock.npz',hour=clock)
    versions=json.loads((OUT/'model_versions.json').read_text());predictions={};timings={}
    for name,(model,length,use_clock) in SETTINGS.items():
        started=time.time();forecaster=TimestralForecaster.from_pretrained(versions[model]['local_path'],local_files_only=True)
        result=[]
        for i in range(19):
            context=np.repeat(inp['context'][i:i+1,:,-length:],4,axis=0)
            known=inp['known'][i,:,:, -(length+48):]
            kwargs={'known_covariates':known}
            if use_clock:kwargs['categorical_known_covariates']=np.repeat(clock[i:i+1,:,-(length+48):],4,axis=0)
            with torch.inference_mode():f=forecaster.forecast(context,prediction_length=48,**kwargs)
            a=f.median.detach().cpu().numpy() if isinstance(f.median,torch.Tensor) else np.asarray(f.median)
            result.append(a.reshape(4,48))
            print(name,f'{i+1}/19',flush=True)
        predictions[name]=np.maximum(np.stack(result),0)
        assert predictions[name].shape==(19,4,48) and np.isfinite(predictions[name]).all()
        timings[name]=round(time.time()-started,2);del forecaster
    np.savez(OUT/'point_predictions.npz',origins=inp['origins'],**predictions)
    (OUT/'inference_metadata.json').write_text(json.dumps(dict(settings=SETTINGS,timings=timings,versions=versions),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download-only',action='store_true');a=p.parse_args()
    download() if a.download_only else infer()
