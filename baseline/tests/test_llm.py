import json
from pathlib import Path
import httpx
import pytest
import yaml
from riskbench.common import BenchError,read_json,write_json
from riskbench.data import load_inputs
from riskbench.demo import mock_transport
from riskbench.artifacts import freeze
from riskbench.llm import parse_score,select_examples,endpoint_guard,messages_for,chat_request,predict_llm

@pytest.mark.parametrize('value',[0,50,100,1.25])
def test_parse_valid(value):
    assert parse_score(json.dumps({'risk_score':value}))==value/100

@pytest.mark.parametrize('text',['{"risk_score":true}','{"risk_score":101}','{"risk_score":-1}','{"risk_score":"80"}','{"risk_score":NaN}','{"risk_score":20,"explanation":"x"}','not json'])
def test_parse_invalid(text):
    with pytest.raises(BenchError):parse_score(text)


def test_target_label_not_in_prompt():
    messages=messages_for({'text':'safe code','label':1,'incident_id':'DO_NOT_SEND','label_type':'DO_NOT_SEND'},[])
    assert 'DO_NOT_SEND' not in json.dumps(messages)
    payload=json.loads(messages[-1]['content'])
    assert set(payload)=={'historical_public_examples','candidate_change'}


def test_internal_endpoint_blocked():
    c={'base_url':'https://example.com/v1','internal_data':{'allow':False,'approved_origins':[]}}
    with pytest.raises(BenchError,match='transmission blocked'):endpoint_guard(c,True)
    endpoint_guard(c,False)


def test_fixed_public_examples_and_invalid_source(data):
    one=str(data['root']/'one.json');two=str(data['root']/'two.json')
    select_examples(data['train'],one,4,42);select_examples(data['train'],two,4,42)
    assert read_json(one)==read_json(two)
    assert sum(r['observed_label'] for r in read_json(one)['examples'])==2
    with pytest.raises(BenchError,match='Public train'):
        select_examples(data['internal'],str(data['root']/'bad.json'),4,42)


def test_retry_identical_prompt_and_usage():
    seen=[]
    def handle(req):
        seen.append(req.content)
        if len(seen)==1:return httpx.Response(429,headers={'Retry-After':'0'})
        return httpx.Response(200,json={'choices':[{'message':{'content':'{"risk_score":60}'},'finish_reason':'stop'}],
                                        'usage':{'prompt_tokens':10,'completion_tokens':4}})
    cfg={'base_url':'http://127.0.0.1/v1','model':'mock','max_output_tokens':128,'max_retries':1,'retry_backoff_seconds':0}
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        r=chat_request(client,cfg,messages_for({'text':'x'},[]),None)
    assert r['score']==.6 and r['attempts']==2 and seen[0]==seen[1]


def test_failed_response_is_not_a_safe_score():
    with httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(401,text='private response not logged'))) as client:
        r=chat_request(client,{'base_url':'http://127.0.0.1/v1','model':'mock','max_output_tokens':128,'max_retries':2,'retry_backoff_seconds':0},[],None)
    assert r['status']=='error' and r['score'] is None and r['attempts']==1
    assert 'private response' not in json.dumps(r)


def test_end_to_end_mock_internal_zero_few_no_labels(data):
    root=data['root']; cfg=root/'llm.yaml'
    config={'base_url':'http://127.0.0.1:9999/v1','model':'MOCK-NO-LLM','max_retries':0,
            'internal_data':{'allow':True,'approved_origins':['http://127.0.0.1:9999']}}
    cfg.write_text(yaml.safe_dump(config)); ex=root/'examples.json'
    select_examples(data['train'],str(ex))
    lock=str(root/'lock.json');freeze([str(cfg),str(ex)],lock)
    (Path(data['internal'])/'labels.jsonl').unlink()
    for kind in ('zero','few'):
        r=predict_llm(str(cfg),data['internal'],str(root/(kind+'.jsonl')),kind,str(ex) if kind=='few' else None,
                      lock,transport=mock_transport())
        assert r['successful']==24 and r['mock_transport']


def test_resume_config_change_rejected(data):
    root=data['root']; cfg=root/'config.yaml';cfg.write_text(yaml.safe_dump({'base_url':'http://127.0.0.1/v1','model':'mock'}))
    target=str(root/'pred.jsonl');Path(target+'.partial.jsonl').touch();write_json(target+'.partial.meta.json',{'wrong':'metadata'})
    with pytest.raises(BenchError,match='Resume metadata mismatch'):
        predict_llm(str(cfg),data['test'],target,'zero',resume=True,transport=mock_transport())


def test_request_limit_blocks_before_network(data):
    cfg = data['root']/'limited.yaml'
    cfg.write_text(yaml.safe_dump({'base_url':'http://127.0.0.1/v1', 'model':'mock',
                                   'max_requests_per_run':1}))
    def forbidden(req):
        raise AssertionError('Network must not be called')
    with pytest.raises(BenchError, match='max_requests_per_run'):
        predict_llm(str(cfg), data['test'], str(data['root']/'limited.jsonl'), 'zero',
                    transport=httpx.MockTransport(forbidden))
