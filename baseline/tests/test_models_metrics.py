from pathlib import Path
import numpy as np
import pytest
from riskbench.common import BenchError, read_json, read_jsonl
from riskbench.artifacts import freeze, verify_lock, seal_predictions
from riskbench.linear import train_tabular,predict_tabular,predict_rule,fit_scaler,transform
from riskbench.metrics import ranking_metrics,evaluate
from riskbench.data import load_inputs


def test_constant_score_ties_are_not_optimistic():
    y=[1,1]+[0]*18
    m=ranking_metrics(y,[0.5]*20)
    assert m['top10_recall']==pytest.approx(.1)
    assert m['top10_precision']==pytest.approx(.1)
    assert m['ap']==pytest.approx(.1)


def test_perfect_ranking():
    m=ranking_metrics([1,1]+[0]*18,[1,.9]+[0]*18)
    assert m['ap']==1 and m['roc_auc']==1 and m['top10_recall']==1


def test_one_class_metrics_are_explicit():
    m=ranking_metrics([0,0],[0,1])
    assert m['ap'] is None and m['roc_auc'] is None and m['top10_recall'] is None


def test_invalid_scores():
    with pytest.raises(BenchError):ranking_metrics([0,1],[0,float('nan')])


def test_fixed_missing_indicators():
    x=np.array([[np.nan,1],[np.nan,2.]])
    scaler=fit_scaler(x); assert scaler['median'][0]==0
    z=transform(np.array([[4.,np.nan]]),scaler)
    assert z.shape==(1,4) and np.isfinite(z).all()


def test_tabular_internal_guard_lock_and_label_blindness(data):
    root=data['root']; model=str(root/'model')
    train_tabular(data['train'],data['valid'],model,cs=[1])
    with pytest.raises(BenchError,match='requires --lock'):
        predict_tabular(model,data['internal'],str(root/'fail.jsonl'),'LR')
    lock=str(root/'lock.json');freeze([model],lock)
    # Inference must succeed even with labels removed entirely.
    (Path(data['internal'])/'labels.jsonl').unlink()
    predict_tabular(model,data['internal'],str(root/'pred.jsonl'),'LR',lock)
    assert len(read_jsonl(root/'pred.jsonl'))==24
    assert all(p['status']=='ok' for p in read_jsonl(root/'pred.jsonl'))


def test_internal_training_rejected(data):
    with pytest.raises(BenchError,match='Public train'):
        train_tabular(data['internal'],data['valid'],str(data['root']/'badmodel'))


def test_modified_frozen_artifact_rejected(tmp_path):
    root=tmp_path/'model';root.mkdir();p=root/'weights.json';p.write_text('{}')
    lock=str(tmp_path/'lock.json');freeze([str(root)],lock)
    p.write_text('{"changed":1}')
    with pytest.raises(BenchError,match='integrity'):
        verify_lock(lock,[str(root)],True)


def test_new_file_after_freeze_rejected(tmp_path):
    root=tmp_path/'model';root.mkdir();(root/'weights.json').write_text('{}')
    lock=str(tmp_path/'lock.json');freeze([str(root)],lock);(root/'extra.json').write_text('{}')
    with pytest.raises(BenchError,match='added/removed'):
        verify_lock(lock,[str(root)],True)


def test_label_transfer_needs_explicit_flag(data):
    p=str(data['root']/'rule.jsonl');predict_rule(data['internal'],p)
    with pytest.raises(BenchError,match='label mismatch'):
        evaluate(data['internal'],[p],str(data['root']/'bad.json'))
    r=evaluate(data['internal'],[p],str(data['root']/'good.json'),allow_label_transfer=True)
    assert r['common_n']==24 and r['evaluation_label_type']=='operational_incident_72h'


def test_failure_coverage_and_paired_ci(data):
    root=data['root'];rows,m=load_inputs(data['test'])
    preds=[{'id':r['id'],'status':'ok','score':float(i)} for i,r in enumerate(rows)]
    meta={'inputs_sha256':m['files']['inputs.jsonl'],'training_label_type':'bug_inducing_commit'}
    p1=str(root/'p1.jsonl');p2=str(root/'p2.jsonl')
    seal_predictions(p1,preds,dict(meta,run_name='a'));seal_predictions(p2,preds,dict(meta,run_name='b'))
    r=evaluate(data['test'],[p1,p2],str(root/'ci.json'),bootstrap=20)
    assert r['bootstrap']['paired_delta_intervals']['b']['ap']['percentile95']==[0.0,0.0]
    bad=[dict(x) for x in preds];bad[0].update(status='error',score=None)
    p3=str(root/'p3.jsonl');seal_predictions(p3,bad,dict(meta,run_name='failed'))
    with pytest.raises(BenchError,match='Failed predictions'):
        evaluate(data['test'],[p1,p3],str(root/'bad.json'))
    report=evaluate(data['test'],[p1,p3],str(root/'partial.json'),allow_partial=True)
    assert report['common_n']==len(rows)-1 and report['coverage']['failed']['failed']==1
    assert 'COMMON SUCCESS' in report['population']
