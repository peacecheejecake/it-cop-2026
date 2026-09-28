"""Offline end-to-end checks. Mock LLM + random tiny encoder, NEVER real model efficacy."""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
from riskbench.common import new_dir, write_json
from riskbench.demo import make_demo, mock_transport
from riskbench.data import split_public, internal_partition
from riskbench.views import prepare_view
from riskbench.linear import train_tabular, predict_tabular, predict_rule
from riskbench.llm import select_examples, predict_llm
from riskbench.artifacts import freeze
from riskbench.metrics import evaluate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--out', required=True); p.add_argument('--neural', action='store_true')
    a=p.parse_args(); root=new_dir(a.out)
    raw=make_demo(str(root/'raw'))
    split_public(raw['public'],str(root/'split'),'2024-03-01T00:00:00Z','2024-04-01T00:00:00Z')
    internal_partition(raw['internal'],str(root/'internal_raw'))
    for name,source in [('train',root/'split/train'),('valid',root/'split/valid'),('test',root/'split/test'),('internal',root/'internal_raw')]:
        prepare_view(str(source),str(root/'views'/name),'demo-byte',max_tokens=128,message_tokens=24)
    train=str(root/'views/train'); valid=str(root/'views/valid'); target=str(root/'views/internal')
    train_tabular(train,valid,str(root/'models/tabular'),cs=[1.0])
    if a.neural:
        import torch
        torch.set_num_threads(1)
        from riskbench.neural import train_neural
        for mode in ('frozen','finetune'):
            train_neural(train,valid,str(root/'models'/mode),'demo-random',mode,epochs=2,batch_size=8,accumulation=3,device='cpu')
    select_examples(train,str(root/'examples.json'),k=4)
    config={'base_url':'http://127.0.0.1:9999/v1','model':'MOCK-NO-LLM','model_revision_tag':'synthetic-test-only',
            'json_mode':True,'max_output_tokens':128,'max_retries':0,'retry_backoff_seconds':0,
            'internal_data':{'allow':True,'approved_origins':['http://127.0.0.1:9999']}}
    (root/'llm.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    lock=str(root/'experiment.lock.json')
    freeze([str(root/'models'),str(root/'examples.json'),str(root/'llm.yaml')],lock)
    pred=root/'predictions'; pred.mkdir(); paths=[]
    def path(name):
        value=str(pred/(name+'.jsonl')); paths.append(value); return value
    predict_rule(target,path('rule'),'rule')
    predict_tabular(str(root/'models/tabular'),target,path('tabular'),'tabular',lock)
    if a.neural:
        from riskbench.neural import predict_neural
        for mode in ('frozen','finetune'):
            predict_neural(str(root/'models'/mode),target,path(mode),mode,batch_size=8,device='cpu',lock=lock)
    for mode in ('zero','few'):
        predict_llm(str(root/'llm.yaml'),target,path(mode),'MOCK-'+mode,
                    str(root/'examples.json') if mode=='few' else None,lock,transport=mock_transport())
    report=evaluate(target,paths,str(root/'metrics.json'),allow_label_transfer=True,bootstrap=30)
    write_json(root/'SMOKE_NOTICE.json',{'synthetic':True,'real_codebert_executed':False,'real_llm_executed':False,
        'random_tiny_encoder_executed':a.neural,'paths_checked':['public split','label-blind internal inference','lock','tabular training','mock zero/few-shot','metrics'],
        'model_rows':len(report['metrics']),'n':report['common_n']})
    print('PASS: synthetic plumbing only. No real model/dataset performance claims.')

if __name__=='__main__':main()
