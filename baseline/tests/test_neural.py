from pathlib import Path
import pytest

torch=pytest.importorskip('torch')
pytest.importorskip('safetensors')
from torch.utils.data import DataLoader
from riskbench.neural import DemoEncoder,RiskNet,Rows,Collator,train_epoch,mean_pool,save_network,load_network,train_neural
from riskbench.common import read_json


def _batch(n=7):
    rows=[{'input_ids':[1,10+i,20+i,2]} for i in range(n)]
    return DataLoader(Rows(rows,[i%2 for i in range(n)]),batch_size=2,shuffle=False,collate_fn=Collator(0))


def test_masked_pool_excludes_padding():
    hidden=torch.tensor([[[1.,2.],[3.,4.],[999.,999.]]]); mask=torch.tensor([[1,1,0]])
    assert torch.allclose(mean_pool(hidden,mask),torch.tensor([[2.,3.]]))


def test_frozen_encoder_stays_eval_and_unchanged():
    torch.set_num_threads(1);torch.manual_seed(42)
    net=RiskNet(DemoEncoder(),True);before={k:v.clone() for k,v in net.encoder.state_dict().items()}
    head=net.head.weight.detach().clone()
    opt=torch.optim.AdamW(net.head.parameters(),lr=.01)
    stats=train_epoch(net,_batch(),opt,torch.device('cpu'),accumulation=3)
    assert stats['optimizer_steps']==2 and stats['examples']==7
    assert not net.encoder.training
    assert all(torch.equal(before[k],v) for k,v in net.encoder.state_dict().items())
    assert not torch.equal(head,net.head.weight)


def test_finetuning_updates_encoder():
    torch.set_num_threads(1);torch.manual_seed(42)
    net=RiskNet(DemoEncoder(),False);before=net.encoder.embedding.weight.detach().clone()
    opt=torch.optim.AdamW(net.parameters(),lr=.01)
    train_epoch(net,_batch(),opt,torch.device('cpu'),accumulation=2)
    assert not torch.equal(before,net.encoder.embedding.weight)


def test_safe_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(1);net=RiskNet(DemoEncoder(),False).eval();batch=next(iter(_batch()));batch.pop('labels')
    with torch.no_grad():before=net(**batch)
    save_network(net,tmp_path,True)
    loaded=load_network(tmp_path,{'demo_encoder':True,'mode':'finetune'}).eval()
    with torch.no_grad():after=loaded(**batch)
    assert torch.allclose(before,after)


needs_mps=pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple MPS unavailable')


@needs_mps
def test_mps_step_matches_cpu():
    torch.manual_seed(42);ref=RiskNet(DemoEncoder(),False)
    nets={}
    for dev in ('cpu','mps'):
        net=RiskNet(DemoEncoder(),False);net.load_state_dict(ref.state_dict());net.to(dev)
        # SGD, not Adam: Adam rescales float-noise gradients (e.g. attention key bias, ~0 by symmetry) into lr-sized steps.
        opt=torch.optim.SGD(net.parameters(),lr=.01)
        stats=train_epoch(net,_batch(),opt,torch.device(dev),accumulation=2)
        nets[dev]=(stats,{k:v.detach().cpu() for k,v in net.state_dict().items()})
    assert nets['cpu'][0]['optimizer_steps']==nets['mps'][0]['optimizer_steps']
    assert nets['cpu'][0]['loss']==pytest.approx(nets['mps'][0]['loss'],rel=1e-4)
    for k,v in nets['cpu'][1].items():
        assert torch.allclose(v,nets['mps'][1][k],atol=1e-5),k


@needs_mps
def test_complete_neural_train_on_mps(data):
    out=data['root']/'finetune-mps'
    train_neural(data['train'],data['valid'],str(out),'demo-random','finetune',epochs=1,batch_size=8,accumulation=3,device='mps')
    assert read_json(out/'model.json')['device']=='mps'


def test_complete_neural_train_uses_public_only(data):
    torch.set_num_threads(1)
    for mode in ('frozen','finetune'):
        out=str(data['root']/mode)
        result=train_neural(data['train'],data['valid'],out,'demo-random',mode,epochs=1,batch_size=8,accumulation=3,device='cpu')
        assert result['demo_encoder'] and (Path(out)/'model.safetensors').exists()
        assert read_json(Path(out)/'model.json')['training_label_type']=='bug_inducing_commit'
