import csv
import hashlib
import subprocess
import zipfile
from pathlib import Path
import pytest
from riskbench.common import BenchError, read_json, read_jsonl, write_json, write_jsonl
from riskbench.data import (FEATURES, apache_build, canonical_row, download_apache, extract_commit, internal_build,
                            load_inputs, load_labeled, normalize_patch, split_public, static_features)
from riskbench.views import DemoTokenizer, prepare_view, render_view


def test_feature_parser_counts_code_triple_plus():
    diff='diff --git a/src/A.java b/src/A.java\n--- a/src/A.java\n+++ b/src/A.java\n@@ -1 +1,2 @@\n-old\n+new\n+++increment\n'
    f=static_features(diff)
    assert f['lines_added']==2 and f['lines_deleted']==1
    assert f['files_changed']==1 and f['directories_changed']==1 and f['hunks']==1
    assert set(f)==set(FEATURES)


def test_entropy_and_binary():
    diff='diff --git a/a b/a\n@@ -1 +1 @@\n-a\n+b\ndiff --git a/b b/b\n@@ -1 +1 @@\n-c\n+d\ndiff --git a/p b/p\nBinary files a/p and b/p differ\n'
    f=static_features(diff)
    assert f['change_entropy']==1.0 and f['binary_files']==1 and f['files_changed']==3


def test_patch_fingerprint_ignores_index_and_hunk_line_number():
    a='diff --git a/a b/a\nindex aaa..bbb\n@@ -1 +1 @@ f\n-a\n+b'
    b='diff --git a/a b/a\nindex ccc..ddd\n@@ -7 +7 @@ f\n-a\n+b'
    assert normalize_patch(a)==normalize_patch(b)


def test_public_role_guard(data):
    with pytest.raises(BenchError,match='Public train'):
        load_labeled(data['internal'],'train')
    with pytest.raises(BenchError,match='Public train'):
        load_labeled(data['valid'],'train')


def test_inputs_do_not_need_labels(data):
    p=Path(data['internal'])/'labels.jsonl'; p.unlink()
    rows,m=load_inputs(data['internal'])
    assert len(rows)==24
    with pytest.raises(BenchError):
        load_labeled(data['internal'])


def test_source_input_integrity(data):
    p=Path(data['test'])/'inputs.jsonl'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(BenchError,match='integrity'):
        load_inputs(data['test'])


def test_view_token_budget_and_no_labels(data):
    rows,m=load_inputs(data['train'])
    assert all(len(r['input_ids'])<=128 and 'label' not in r and 'diff' not in r for r in rows)
    assert all(r['truncated'] for r in rows)


def test_render_maintains_end_of_diff():
    out=render_view(DemoTokenizer(),'a'*100,'START'+('x'*1000)+'END',128,24)
    assert len(out['input_ids'])<=128 and 'END' in out['text'] and '[TRUNCATED]' in out['text']


def test_future_input_rejected(data):
    r=read_jsonl(data['raw']['public'])[0]
    r['input_available_at']='2099-01-01T00:00:00Z'
    with pytest.raises(BenchError,match='after prediction'):
        canonical_row(r)


def test_unresolved_not_zero(data):
    r=read_jsonl(data['raw']['public'])[0]; r['label_status']='ambiguous'; r['label']=0
    with pytest.raises(BenchError,match='Unresolved'):
        canonical_row(r)


def test_unknown_label_dates_require_explicit_retrospective(data):
    rows=read_jsonl(data['raw']['public'])
    for r in rows:r['label_available_at']=None
    path=data['root']/'unknown.jsonl'; write_jsonl(path,rows)
    with pytest.raises(BenchError,match='allow-retrospective'):
        split_public(str(path),str(data['root']/'strict'),'2024-03-01T00:00:00Z','2024-04-01T00:00:00Z')
    result=split_public(str(path),str(data['root']/'retro'),'2024-03-01T00:00:00Z','2024-04-01T00:00:00Z',True)
    assert result['retrospective_labels']


def test_late_labels_are_purged(data):
    report=read_json(data['root']/'split/split_report.json')
    assert report['exclusion_reasons']['label_not_available_at_fit_cutoff']>0


def test_cross_split_duplicates_purged(data):
    rows=read_jsonl(data['raw']['public'])
    rows[-1]['diff']=rows[0]['diff']
    path=data['root']/'cross.jsonl'; write_jsonl(path,rows)
    report=split_public(str(path),str(data['root']/'cross-split'),'2024-03-01T00:00:00Z','2024-04-01T00:00:00Z')
    rejects=read_jsonl(data['root']/'cross-split/excluded.jsonl')
    ids={r['id'] for r in rejects if r['reason']=='group_or_exact_patch_crosses_split'}
    assert rows[0]['id'] in ids and rows[-1]['id'] in ids


def test_demo_tokenizer_rejected_for_real_flag(data):
    from riskbench.data import write_partition
    rows=read_jsonl(data['raw']['public'])[:10]
    for r in rows:r['synthetic']=False
    write_partition(rows,data['root']/'real','test',True)
    with pytest.raises(BenchError,match='restricted'):
        prepare_view(str(data['root']/'real'),str(data['root']/'bad-view'),'demo-byte',128,24)


def _repo(root):
    path=root/'repos/apache/example.git'; path.mkdir(parents=True)
    subprocess.run(['git','init',str(path)],check=True,capture_output=True)
    def cmd(*args):
        return subprocess.run(['git','-C',str(path),*args],check=True,capture_output=True).stdout.decode().strip()
    cmd('config','user.name','Synthetic Tester'); cmd('config','user.email','synthetic@example.invalid')
    (path/'a.py').write_text('x = 1\n'); cmd('add','a.py'); cmd('commit','-m','initial')
    initial=cmd('rev-parse','HEAD')
    (path/'a.py').write_text('x = 2\n'); cmd('add','a.py'); cmd('commit','-m','adjust value')
    return path,initial,cmd('rev-parse','HEAD')


def test_git_root_and_parent_diff(tmp_path):
    repo,initial,sha=_repo(tmp_path)
    root=extract_commit(repo,initial); normal=extract_commit(repo,sha)
    assert '+x = 1' in root['diff'] and '-x = 1' in normal['diff'] and '+x = 2' in normal['diff']
    assert normal['message']=='adjust value'
    with pytest.raises(BenchError):extract_commit(repo,'HEAD; touch /tmp/not-allowed')


def test_apache_adapter_actual_git_local_only(tmp_path):
    repo,initial,sha=_repo(tmp_path)
    csv_path=tmp_path/'apache.csv'
    csv_path.write_text(f'commit_id,project,buggy,fix,la\n{sha},apache/example,True,False,999999\n')
    result=apache_build(str(csv_path),str(tmp_path/'repos'),str(tmp_path/'built'))
    assert result['accepted']==1
    row=read_jsonl(tmp_path/'built/records.jsonl')[0]
    assert row['features']['lines_added']==1 and row['label']==1
    assert 'fix' not in row and row['label_available_at'] is None


def test_internal_local_csv_import_and_path_escape(tmp_path):
    patch=tmp_path/'patch.diff'; patch.write_text('diff --git a/a b/a\n@@ -1 +1 @@\n-x\n+y\n')
    fields=['id','repository','commit','group_id','prediction_at','input_available_at','message','diff_path','label','label_type','label_status','label_available_at']
    row=['internal1','service/a','artifact-range','release1','2025-01-01T00:00:00Z','2024-12-31T00:00:00Z','message','patch.diff','0','operational_incident_72h','observed_negative','2025-01-08T00:00:00Z']
    with (tmp_path/'manifest.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerow(row)
        bad=list(row);bad[0]='internal2';bad[7]='../outside.diff';w.writerow(bad)
    result=internal_build(str(tmp_path/'manifest.csv'),str(tmp_path/'out'))
    assert result['accepted']==1 and result['rejected']==1


def test_archive_extracts_only_csv(tmp_path,monkeypatch):
    zpath=tmp_path/'archive.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        z.writestr('apachejit/dataset/apachejit_total.csv','commit_id,project,buggy\n')
        z.writestr('../../not-extracted.py','do not execute')
    import riskbench.data as mod
    monkeypatch.setattr(mod,'ARCHIVE_MD5',hashlib.md5(zpath.read_bytes()).hexdigest())
    download_apache(str(tmp_path/'extract'),str(zpath))
    assert (tmp_path/'extract/apachejit_total.csv').exists()
    assert not (tmp_path/'not-extracted.py').exists()


def test_archive_checksum_failure(tmp_path):
    p=tmp_path/'bad.zip';p.write_bytes(b'not a verified archive')
    with pytest.raises(BenchError,match='checksum'):
        download_apache(str(tmp_path/'out'),str(p))


def test_unquoted_directory_names_with_spaces():
    diff = ('diff --git a/first dir/a.java b/first dir/a.java\n@@ -1 +1 @@\n-a\n+b\n'
            'diff --git a/second dir/b.java b/second dir/b.java\n@@ -1 +1 @@\n-c\n+d\n')
    result = static_features(diff)
    assert result['files_changed'] == 2 and result['directories_changed'] == 2
