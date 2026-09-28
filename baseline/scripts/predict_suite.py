"""Run fixed candidates on one shared dataset. Does NOT tune or calibrate on it."""
from __future__ import annotations
import argparse
from pathlib import Path
from riskbench.common import new_dir
from riskbench.linear import predict_rule,predict_tabular
from riskbench.llm import predict_llm
from riskbench.metrics import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--model-root',default='runs/models');p.add_argument('--llm-config',default='configs/llm.yaml')
    p.add_argument('--examples',default='runs/public_examples.json');p.add_argument('--lock')
    p.add_argument('--models',nargs='+',choices=['rule','tabular','frozen','finetune','zero','few'],
                   default=['rule','tabular','frozen','finetune','zero','few'])
    p.add_argument('--device',default='auto');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--bootstrap',type=int,default=0);p.add_argument('--cluster',choices=['week','group'],default='week')
    p.add_argument('--allow-label-transfer',action='store_true');p.add_argument('--no-evaluate',action='store_true')
    a=p.parse_args();root=new_dir(a.out);models=Path(a.model_root);paths=[]
    for name in a.models:
        path=str(root/(name+'.jsonl'));paths.append(path)
        print('Predicting:',name,flush=True)
        if name=='rule':predict_rule(a.dataset,path,name)
        elif name=='tabular':predict_tabular(str(models/name),a.dataset,path,name,a.lock)
        elif name in ('frozen','finetune'):
            from riskbench.neural import predict_neural
            predict_neural(str(models/name),a.dataset,path,name,a.batch_size,a.device,a.lock)
        else:
            predict_llm(a.llm_config,a.dataset,path,name,a.examples if name=='few' else None,a.lock)
    if not a.no_evaluate:
        report=evaluate(a.dataset,paths,str(root/'metrics.json'),allow_label_transfer=a.allow_label_transfer,
                        bootstrap=a.bootstrap,cluster=a.cluster)
        print('Evaluation complete; rows:',report['common_n'])
    else:
        print('Predictions complete. Outcome labels were not loaded.')

if __name__=='__main__':main()
