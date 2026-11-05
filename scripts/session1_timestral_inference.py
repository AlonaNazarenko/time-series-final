"""Local Timestral inference; use the .venv-hf environment."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import argparse,json,time
import numpy as np
import torch
from huggingface_hub import HfApi,snapshot_download
from timestral import TimestralForecaster
OUT=ROOT/'results/timestral'
MODELS=['mistralai/Timestral-5M','mistralai/Timestral-28M']


def download():
    OUT.mkdir(exist_ok=True)
    manifest=OUT/'model_versions.json';recorded=json.loads(manifest.read_text()) if manifest.exists() else {}
    versions={}
    for model in MODELS:
        sha=recorded[model]['revision'] if model in recorded else HfApi().model_info(model).sha
        path=snapshot_download(model,revision=sha,allow_patterns=['config.json','model.safetensors'])
        versions[model]=dict(revision=sha,local_path=str(Path(path).resolve()))
        print('Downloaded',model,sha,flush=True)
    manifest.write_text(json.dumps(versions,indent=2))


def infer():
    torch.set_num_threads(4);torch.manual_seed(42)
    inputs=np.load(OUT/'inputs.npz');versions=json.loads((OUT/'model_versions.json').read_text())
    output={};timings={}
    for model in MODELS:
        start=time.time()
        forecaster=TimestralForecaster.from_pretrained(versions[model]['local_path'],local_files_only=True)
        plain=[];weather=[]
        for i in range(len(inputs['origins'])):
            context=inputs['context'][i:i+1]
            with torch.inference_mode():
                a=forecaster.forecast(context,prediction_length=48)
                b=forecaster.forecast(np.repeat(context,4,axis=0),prediction_length=48,known_covariates=inputs['known'][i])
            def array(x):return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)
            plain.append(np.repeat(array(a.median).reshape(1,48),4,axis=0))
            weather.append(array(b.median).reshape(4,48))
            print(model,f'origin {i+1}/19',flush=True)
        name=model.split('/')[-1].lower().replace('-','_')
        output[name+'_target_504h']=np.maximum(np.stack(plain),0)
        output[name+'_weather_504h']=np.maximum(np.stack(weather),0)
        for k in [name+'_target_504h',name+'_weather_504h']:
            assert output[k].shape==(19,4,48) and np.isfinite(output[k]).all()
        timings[model]=round(time.time()-start,2)
        del forecaster
    np.savez(OUT/'point_predictions.npz',origins=inputs['origins'],**output)
    (OUT/'inference_metadata.json').write_text(json.dumps(dict(models=versions,device='cpu',context_hours=504,
        point='native median',covariates=['cloud_cover','temperature','wind_speed','pressure','humidity','daylight','air_density'],
        timings_seconds=timings,intervals='95% empirical rolling-error intervals evaluated separately; native 5th/95th quantiles are not called 95% intervals.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download-only',action='store_true');a=p.parse_args()
    download() if a.download_only else infer()
