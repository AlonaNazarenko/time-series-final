"""Run pretrained Chronos-Bolt locally in .venv-hf; evaluate with the original .venv."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME',str(ROOT/'.cache'/'huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download
from chronos import BaseChronosPipeline
MODELS=['amazon/chronos-bolt-tiny','amazon/chronos-bolt-small']
CONTEXTS=[168,504,1536]
OUT=ROOT/'results_hf'


def download():
    OUT.mkdir(exist_ok=True)
    versions={}
    manifest=OUT/'model_versions.json'
    recorded=json.loads(manifest.read_text()) if manifest.exists() else {}
    for model in MODELS:
        sha=recorded[model]['revision'] if model in recorded else HfApi().model_info(model).sha
        path=snapshot_download(model,revision=sha,allow_patterns=['config.json','model.safetensors','generation_config.json'])
        versions[model]={'revision':sha,'local_path':str(Path(path).resolve())}
        print('Downloaded',model,sha,flush=True)
    (OUT/'model_versions.json').write_text(json.dumps(versions,indent=2))


def infer():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data=pd.read_csv(ROOT/'train.csv',parse_dates=['ds'])
    y=data.y.to_numpy(dtype=np.float32)
    assert np.isfinite(y).all() and data.ds.is_unique
    origins=list(range(len(y)-18*48,len(y),48))+[len(y)]
    versions=json.loads((OUT/'model_versions.json').read_text())
    all_predictions={};timings={}
    for model in MODELS:
        started=time.time()
        pipeline=BaseChronosPipeline.from_pretrained(versions[model]['local_path'],device_map='cpu',torch_dtype=torch.float32,local_files_only=True)
        levels=list(pipeline.model.config.chronos_config['quantiles'])
        median_index=levels.index(.5)
        for context in CONTEXTS:
            name=model.split('/')[-1].replace('-','_')+f'_{context}h'
            outputs=[]
            for start in range(0,len(origins),4):
                batch=torch.tensor(np.stack([y[o-context:o] for o in origins[start:start+4]]))
                with torch.inference_mode():
                    forecasts=pipeline.predict(batch,prediction_length=48)
                outputs.append(forecasts[:,median_index,:].cpu().numpy())
            points=np.maximum(np.concatenate(outputs),0)
            assert points.shape==(len(origins),48) and np.isfinite(points).all()
            all_predictions[name]=points
            print(name,'complete',flush=True)
        timings[model]=round(time.time()-started,2)
        del pipeline
    np.savez(OUT/'point_predictions.npz',origins=np.array(origins),**all_predictions)
    (OUT/'inference_metadata.json').write_text(json.dumps(dict(device='cpu',dtype='float32',contexts=CONTEXTS,
        point_forecast='native median quantile (not claimed to be the conditional mean)',
        intervals='Evaluated using historical rolling-error calibration; native Bolt quantiles are not labeled as 95% intervals.',
        timings_seconds=timings,versions=versions),indent=2))
    print('Saved predictions for 18 historical windows and the actual future.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--download-only',action='store_true');args=parser.parse_args()
    if args.download_only:download()
    else:infer()
