"""B2-S head training on identical cached CLS embeddings: CPU vs MPS, 6 epochs, seed 42 (compare with v4 CUDA history)."""
import sys, numpy as np, torch
from pathlib import Path
from diff_lab.config import load_study, resolve_local_path
from diff_lab.evidence import load_tokenizer
from diff_lab.neural import NeuralRun, build_head, encode_inputs, Encoded, _batches, seed_everything
from diff_lab.runner import load_frames
from diff_lab.policy import TrainingDatasetView, QueryView
from diff_lab.metrics import evaluate
cfg,raw,_=load_study("configs/studies/d1-learning-curve-f100.yaml")
frame,y,lin=load_frames(cfg,Path("data"),"B2-S")
tr=frame["split"]=="train"; va=frame["split"]=="valid"
train=TrainingDatasetView(split="train",frame=frame[tr].reset_index(drop=True),lineage=lin,labels=y[tr].reset_index(drop=True))
valid=QueryView(split="valid",frame=frame[va].reset_index(drop=True),lineage=lin); yv=y[va].to_numpy()
local=resolve_local_path(cfg.model.local_path,Path(".").resolve()); tok,_=load_tokenizer(local,cfg.model.revision)
run=NeuralRun(cfg,"B2-S",42,local,Path("/tmp/b2diag"),tok)
cache=Path("/private/tmp/claude-501/-Users-min-jiwon-it-cop-2026-labs-jit-zero-shot/3a46514a-2a9f-4747-b8f0-828df5960c29/scratchpad/b2cls.pt")
x_tr=run.pipe.fit(train).transform(train.frame).astype(np.float32); x_va=run.pipe.transform(valid.frame).astype(np.float32)
if cache.exists():
    cls_tr,cls_va=torch.load(cache)
else:
    enc,_=run._load_encoder(); enc=enc.to("mps").eval()
    trE=Encoded(encode_inputs(tok,train.frame["query_text"].tolist(),512),x_tr); vaE=Encoded(encode_inputs(tok,valid.frame["query_text"].tolist(),512),x_va)
    cls_tr=run._embed(enc,trE,tok.pad_token_id,torch.device("mps")).cpu(); cls_va=run._embed(enc,vaE,tok.pad_token_id,torch.device("mps")).cpu()
    torch.save((cls_tr,cls_va),cache)
ytr=train.labels.to_numpy().astype(np.float32); ft=cfg.finetune
for dev in [a for a in sys.argv[1:] if not a.startswith("--")]:
    d=torch.device(dev); seed_everything(42)
    head=build_head(768,x_tr.shape[1],ft.head_dropout,42).to(d)
    opt=torch.optim.AdamW(head.parameters(),lr=ft.head_lr,weight_decay=ft.weight_decay)
    lf=torch.nn.BCEWithLogitsLoss(reduction="sum"); gen=np.random.default_rng(42)
    ctr,cva=cls_tr.to(d),cls_va.to(d); ftr,fva=torch.tensor(x_tr,device=d),torch.tensor(x_va,device=d)
    hist=[]
    for ep in range(6):
        order=gen.permutation(len(ytr)); head.train(); w=ft.micro_batch_size*ft.gradient_accumulation_steps
        for s in range(0,len(order),w):
            widx=order[s:s+w]; opt.zero_grad(set_to_none=True)
            for idx in _batches(len(widx),ft.micro_batch_size,widx):
                it=torch.as_tensor(idx,device=d)
                loss=lf(head(ctr[it],ftr[it]),torch.tensor(ytr[idx],device=d)); (loss/len(widx)).backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()),ft.max_grad_norm); opt.step()
        head.eval()
        with torch.no_grad(): sc=torch.sigmoid(head(cva,fva)).float().cpu().numpy().astype(np.float64)
        hist.append(round(evaluate(valid.ids,yv,sc,"x")["ap"],4))
    print(dev,hist,flush=True)
print("v4 CUDA seed42:",[0.3593,0.3456,0.3469,0.3341,0.3238,0.3668]); print("d1 MPS seed42:",[0.3202,0.2933,0.2903,0.2863,0.2683,0.2727])
if "--cross" in sys.argv:
    import pandas as pd, json
    from safetensors.torch import load_file
    rd=Path("../../016-dl-v4-gitlines/codebert-diff-lab/artifacts/runs/ea2208bf5c84ba88")
    st=json.load(open(rd/"model/state.json"))
    from diff_lab.frozen import pipeline_from_state
    pipe=pipeline_from_state(st["pipeline"]); xv=pipe.transform(valid.frame).astype(np.float32)
    best=sorted((rd/"checkpoints").glob("best-*"))[-1]
    head=build_head(768,xv.shape[1],ft.head_dropout,0); head.load_state_dict(load_file(str(best/"head.safetensors"))); head.eval()
    with torch.no_grad(): sc=torch.sigmoid(head(cls_va,torch.tensor(xv))).numpy().astype(np.float64)
    ref=pd.read_parquet(best/"validation-scores.parquet")
    print("ids aligned", ref.change_id.tolist()==valid.ids, "max|diff| vs stored CUDA scores", np.abs(ref.score.to_numpy()-sc).max(), "AP", evaluate(valid.ids,yv,sc,"x")["ap"])
    print("pipeline same as local fit:", np.allclose(xv, x_va))
if "--seeds" in sys.argv:
    res=[]
    for seed in range(40,52):
        d=torch.device("cpu"); seed_everything(seed)
        head=build_head(768,x_tr.shape[1],ft.head_dropout,seed)
        opt=torch.optim.AdamW(head.parameters(),lr=ft.head_lr,weight_decay=ft.weight_decay)
        lf=torch.nn.BCEWithLogitsLoss(reduction="sum"); gen=np.random.default_rng(seed)
        ftr,fva=torch.tensor(x_tr),torch.tensor(x_va); best,stale,ep,hist=-1,0,0,[]
        while ep<ft.max_epochs and stale<ft.patience_epochs:
            ep+=1; order=gen.permutation(len(ytr)); head.train(); w=ft.micro_batch_size*ft.gradient_accumulation_steps
            for s in range(0,len(order),w):
                widx=order[s:s+w]; opt.zero_grad(set_to_none=True)
                for idx in _batches(len(widx),ft.micro_batch_size,widx):
                    it=torch.as_tensor(idx); loss=lf(head(cls_tr[it],ftr[it]),torch.tensor(ytr[idx])); (loss/len(widx)).backward()
                torch.nn.utils.clip_grad_norm_(list(head.parameters()),ft.max_grad_norm); opt.step()
            head.eval()
            with torch.no_grad(): ap=evaluate(valid.ids,yv,torch.sigmoid(head(cls_va,fva)).numpy().astype(np.float64),"x")["ap"]
            hist.append(round(ap,3))
            if ap>best: best,stale,be=ap,0,ep
            else: stale+=1
        res.append(best); print("seed",seed,"best",round(best,4),"@",be,hist,flush=True)
    print("mean",round(float(np.mean(res)),4),"sd",round(float(np.std(res,ddof=1)),4),"min",round(min(res),4),"max",round(max(res),4))
