**현재 결과를 무효화할 직접적인 validation/test 라벨의 gradient 누출은 발견하지 못했습니다.** 다만 GPU resume 오류, CPT-dev 원문 중복, provenance 검증의 빈틈은 고쳐야 합니다. 연구 해석에서는 **B2 미수렴과 validation 최고 epoch 선택 편향**이 가장 큽니다.

파일은 수정하지 않았습니다. 사양 `SHA256SUMS` 10개 모두 일치했습니다. 리뷰 중 test 예측·개별 정답은 열지 않았으며, 쓰기를 발생시키는 pytest도 실행하지 않았습니다. 아래 GPU 오류는 정적 분석 결과입니다.

## Critical

확인된 항목 없음.

정형 전처리와 TF-IDF는 train view에서만 fit합니다(`codebert-diff-lab/src/diff_lab/features.py:36`, `codebert-diff-lab/src/diff_lab/models.py:105`). CPT 입력에서는 결함 라벨 열을 제거하고 train/role을 검사합니다(`codebert-diff-lab/src/diff_lab/runner.py:133`, `codebert-diff-lab/src/diff_lab/cpt.py:47`). **CPT-dev가 downstream supervised 학습에 들어가는 것은 사양상 허용**입니다(`codebert-diff-lab/docs/spec-v0.2/requirements.md:124`).

## High

### 1. CUDA resume에서 CPU RNG state가 GPU로 이동한다

- **근거:** downstream과 CPT 모두 `torch.load(..., map_location=device)`로 checkpoint 전체를 읽고, 그 안의 `torch_rng`를 그대로 `torch.set_rng_state()`에 전달합니다. CUDA 실행에서는 CPU RNG ByteTensor도 CUDA로 이동하므로 CPU RNG 복원에서 실패할 수 있습니다.  
  `codebert-diff-lab/src/diff_lab/neural.py:379`, `codebert-diff-lab/src/diff_lab/neural.py:384`, `codebert-diff-lab/src/diff_lab/cpt.py:359`, `codebert-diff-lab/src/diff_lab/cpt.py:362`
- **영향:** 장시간 GPU 작업의 재개 경로가 깨집니다. 현재 CPU resume 테스트로는 잡히지 않습니다(`codebert-diff-lab/tests/test_neural.py:95`).
- **수정:** checkpoint를 CPU에 로드한 뒤 모델·optimizer를 장치에 복원하거나, RNG tensor를 명시적으로 `.cpu()` 처리합니다. CUDA에서 강제 중단→resume과 연속 실행의 loss·최종 state hash를 비교해야 합니다.

### 2. 승인·snapshot·split·evidence의 부모 관계를 검증하지 않는다

- **근거:** runner는 각 파일이 자기 manifest의 hash와 맞는지만 검사합니다. split의 `parent_snapshot_manifest_sha256`, evidence의 snapshot/split 부모 hash, 현재 tokenizer hash를 서로 대조하지 않습니다. 학습 허용 판단도 snapshot의 `visibility` 문자열에 의존합니다.  
  `codebert-diff-lab/src/diff_lab/runner.py:89`, `codebert-diff-lab/src/diff_lab/runner.py:94`, `codebert-diff-lab/src/diff_lab/runner.py:106`, `codebert-diff-lab/src/diff_lab/runner.py:99`, `codebert-diff-lab/src/diff_lab/policy.py:78`
- **영향:** 서로 다른 버전의 정상 artifact를 잘못 조합해도 통과할 수 있습니다. 승인되지 않은 artifact를 manifest 편집만으로 차단해야 한다는 AT-04도 충족하지 못합니다(`codebert-diff-lab/docs/spec-v0.2/implementation-plan.md:374`). **현재 실행에서 그런 조합이 발생했다는 증거는 없습니다.**
- **수정:** 학습 전 하나의 lineage validator로 승인 source→snapshot→split→evidence→tokenizer 관계와 허용 용도를 검증합니다. valid/test ID가 train role로 재분류되지 않았는지도 canonical membership으로 확인해야 합니다.

### 3. 기존 B2 결과로 frozen encoder와 full FT의 우열을 판단하기 어렵다

- **근거:** encoder와 무작위 초기화 head가 같은 optimizer의 단일 lr을 사용합니다(`codebert-diff-lab/src/diff_lab/neural.py:193`, `codebert-diff-lab/src/diff_lab/neural.py:210`, `codebert-diff-lab/src/diff_lab/neural.py:213`). 등록 설정은 lr `1e-5`, 최대 20 epoch입니다(`codebert-diff-lab/configs/studies/public-comparison-v2.yaml:66`). B2 세 seed 모두 최고 epoch가 20이고 계속 개선 중이었습니다(`experiments/007-dl-m3-encoder/RUNLOG.md:45`, `experiments/007-dl-m3-encoder/RUNLOG.md:54`).
- **판정:** 같은 수치를 적용한 것은 사양 준수지만, **같은 최적화 적합성을 보장하지 않습니다.** 사용자 제공 민감도 AP 0.604–0.623은 B2가 특히 불리했다는 강한 증거입니다. 다만 lr과 epoch 상한을 동시에 변경했으므로 두 효과를 분리하지 못하며, 모든 encoder variant가 같은 정도로 불리했다고 단정할 수는 없습니다.
- **수정:** v2 결과는 “고정 초기 프로토콜에서의 성능”으로 보존합니다. 후속 버전에서는 `encoder_lr`와 `head_lr`를 분리하고 B2–B5에 동일한 head 설정·후보 수·선택 규칙을 적용합니다. **B2만 바꾼 값을 기존 주 행렬에 넣으면 안 됩니다.** 007 RESULTS의 “v2.1로 B2만 재실행” 선택지는 이 점을 명확히 해야 합니다(`experiments/007-dl-m3-encoder/RESULTS.md:21`, 사양 `codebert-diff-lab/docs/spec-v0.2/experiment-protocol.md:46`).

### 4. 최고 epoch를 고른 validation AP는 성능 추정치로 낙관적이다

- **근거:** 매 epoch validation AP를 계산해 최대값의 checkpoint와 예측을 저장하고, 그 예측으로 같은 validation의 최종 지표를 계산합니다.  
  `codebert-diff-lab/src/diff_lab/neural.py:252`, `codebert-diff-lab/src/diff_lab/neural.py:257`, `codebert-diff-lab/src/diff_lab/neural.py:260`, `codebert-diff-lab/src/diff_lab/runner.py:215`
- **영향:** 곡선이 흔들릴수록 우연히 높은 epoch를 고릅니다. 100 epoch 상한의 민감도와 단일 fit B1은 선택 기회가 다릅니다. 세 seed 평균도 이 편향을 제거하지 않습니다. 현재 metrics 경고는 F1만 낙관적이라고 적어 AP 선택 편향을 놓칩니다(`codebert-diff-lab/src/diff_lab/runner.py:216`).
- **수정:** 지금 숫자는 모두 **selection-validation 결과**로 표기합니다. 이미 반복해서 본 validation을 지금 나누는 것으로 독립성을 회복했다고 주장하면 안 됩니다. 최종 판단은 모든 설정·seed를 freeze한 뒤 sealed test에서 한 번에 합니다. 새 프로토콜의 개발 추정치가 필요하면 train 내부의 중복 그룹 단위 nested split/CV로 epoch·설정을 선택하고 별도 fold에서 평가합니다. 곡선 smoothing만으로 선택 편향이 해결되지는 않습니다.

## Medium

### 5. CPT-train과 CPT-dev에 완전히 같은 원문이 존재한다

- **근거:** CPT-dev를 개별 `change_id` hash 순위로 뽑고 중복 그룹은 묶지 않습니다(`codebert-diff-lab/src/diff_lab/data/splits.py:42`). CPT 실행은 ID overlap만 검사합니다(`codebert-diff-lab/src/diff_lab/cpt.py:228`).
- **실측:** 007의 split과 label-free `content_hash`를 join하면 **19개 exact-content 그룹, 39행**이 두 role에 걸칩니다. CPT-train의 **20행**이 CPT-dev와 같은 원문입니다.
- **영향:** CPT-dev loss가 독립적인 일반화 진단이 아닙니다. 현재는 dev loss로 checkpoint를 고르지 않고 전체 예산 종료 후 export하므로 downstream 모델 선택 오염으로 확대 해석할 필요는 없습니다(`codebert-diff-lab/src/diff_lab/cpt.py:295`, `codebert-diff-lab/src/diff_lab/cpt.py:304`).
- **수정:** 새 split version에서 exact/code 중복 연결 그룹 단위로 CPT-dev를 배정합니다. 기존 B4 결과는 중복된 dev 진단임을 표시하고, 기존 membership을 조용히 바꾸지 않습니다.

### 6. run ID가 실제 코드·모델 내용의 변경을 충분히 구분하지 못한다

- **근거:** run ID는 Git SHA와 `dirty` Boolean을 사용합니다. 같은 commit에서 서로 다른 두 dirty 수정은 같은 ID가 될 수 있고, SHA가 없는 번들도 허용됩니다. completed run은 내용 재검증 없이 반환합니다.  
  `codebert-diff-lab/src/diff_lab/runner.py:46`, `codebert-diff-lab/src/diff_lab/runner.py:53`, `codebert-diff-lab/src/diff_lab/runner.py:151`, `codebert-diff-lab/src/diff_lab/runner.py:154`
- **추가:** 기반 encoder는 `SOURCE.json`의 revision 문자열만 검사하고 weight 파일 checksum을 검증하지 않습니다(`codebert-diff-lab/src/diff_lab/neural.py:77`). tokenizer digest도 실제 로드될 수 있는 `tokenizer.json`을 포함하지 않습니다(`codebert-diff-lab/src/diff_lab/evidence.py:25`, `codebert-diff-lab/src/diff_lab/evidence.py:35`).
- **실제 기록:** 007의 `git.sha`는 null입니다(`experiments/007-dl-m3-encoder/results/afc2b5821f1f75eb/environment.json:9`). 번들 hash로 외부 근거는 남아 있지만 run 자체는 불완전합니다.
- **수정:** formal run에서 unknown SHA를 거부하고 dirty 실행에는 source digest를 포함합니다. 기반 weight·config·실제 tokenizer 파일을 checksum으로 pin하고 completed 결과 재사용 시 필수 artifact도 검증합니다.

### 7. checkpoint 교체가 원자적이지 않으며 crash 시 마지막 정상본을 잃을 수 있다

- **근거:** 임시 checkpoint 작성 후 기존 디렉토리를 삭제하고 `os.replace()`합니다. 삭제와 교체 사이에 죽으면 정상 checkpoint가 없습니다. best weight와 best score parquet도 별도 저장입니다.  
  `codebert-diff-lab/src/diff_lab/neural.py:361`, `codebert-diff-lab/src/diff_lab/neural.py:374`, `codebert-diff-lab/src/diff_lab/neural.py:259`, `codebert-diff-lab/src/diff_lab/cpt.py:351`
- **수정:** immutable generation 디렉토리에 checkpoint와 checksum을 쓰고, 완료 후 작은 `last`/`best` 포인터만 원자적으로 교체합니다. score·weight·trainer state를 동일 generation으로 연결하고 각 저장 경계의 강제 중단 테스트를 추가합니다.

### 8. 단일 class와 빈 cohort의 metric 정의가 사양과 다르다

- **근거:** 양성만 있는 cohort에서도 AP를 null로 만듭니다. 사양은 양성 없는 cohort의 AP를 null로 요구하며 ROC-AUC만 두 class 조건입니다. 빈 cohort는 `recall_at()`의 boundary 접근에서 실패합니다.  
  `codebert-diff-lab/src/diff_lab/metrics.py:78`, `codebert-diff-lab/src/diff_lab/metrics.py:80`, `codebert-diff-lab/src/diff_lab/metrics.py:41`
- **수정:** `AP: positives>0`, `ROC-AUC: two classes`를 분리하고 빈 cohort는 지표별 null·사유를 반환합니다. all-positive AP=1, empty cohort fixture를 추가합니다. 현재 전체 validation 결과에는 영향이 없지만 프로젝트별 집계에 영향을 줍니다.

### 9. M2–M4 run-folder·선택 ledger·비용 보고가 계약에 못 미친다

- **근거:** downstream epoch 기록은 `model/state.json`에만 들어가고 run-level `metrics.jsonl`을 생성하지 않습니다. `token-accounting.json`도 B4의 CPT에만 생성합니다. ledger는 실제 epoch 선택을 적지 않습니다.  
  `codebert-diff-lab/src/diff_lab/neural.py:254`, `codebert-diff-lab/src/diff_lab/runner.py:184`, `codebert-diff-lab/src/diff_lab/runner.py:212`, `codebert-diff-lab/src/diff_lab/runner.py:222`
- **비용 오류:** neural `predict_seconds=0`이므로 모든 예측 latency가 0ms로 저장됩니다. 이는 측정치로 해석할 수 없습니다. reserved VRAM·추론 p50/p95도 없습니다.  
  `codebert-diff-lab/src/diff_lab/runner.py:191`, `codebert-diff-lab/src/diff_lab/runner.py:203`, `codebert-diff-lab/src/diff_lab/runner.py:228`
- **수정:** 계약의 `metrics.jsonl`, downstream token/update 회계, 선택 checkpoint 사건, run summary를 생성합니다(`codebert-diff-lab/docs/spec-v0.2/implementation-plan.md:117`). 미측정 latency는 null로 두고 별도 inference benchmark로 측정합니다. M4는 현재 진행 중이므로 3-seed 비교와 resume 검증 전에는 완료로 표시하지 않습니다.

## Low

### 10. 테스트가 핵심 실패 조건을 충분히 검증하지 않는다

- **근거:** top-k tie 테스트는 선택 ID의 label 독립성을 직접 확인하지 않고 TP가 0/1인지에 그칩니다(`codebert-diff-lab/tests/test_metrics.py:29`). threshold 테스트도 실제 최대 F1 동률 사례가 아닙니다(`codebert-diff-lab/tests/test_metrics.py:40`). TrainingDatasetView는 label 길이만 확인하고 row membership·ID alignment는 확인하지 않습니다(`codebert-diff-lab/src/diff_lab/policy.py:75`).
- **수정:** tie에서 선택 ID 불변, 실제 F1 동률, label permutation, forged parent lineage, CUDA resume, 불완전 accumulation의 큰 배치 대비 gradient 일치 fixture를 추가합니다. 현재 accumulation 수식 자체는 downstream sample 수와 CPT target 수로 정확히 정규화되어 있습니다(`codebert-diff-lab/src/diff_lab/neural.py:246`, `codebert-diff-lab/src/diff_lab/cpt.py:281`).

## 권장 다음 단계

1. **진행 중인 008 결과는 그대로 보존**하고 B2 sensitivity를 exploratory 부록으로 유지합니다. 이미 별도 study/protocol로 등록한 방식은 적절합니다(`experiments/008-dl-m4-mlm-cpt/codebert-diff-lab/configs/studies/b2-head-sensitivity-v1.yaml:1`).
2. 다음 GPU 실행 전에 **CUDA resume와 lineage 검증을 먼저 수정**합니다. 기존 run에는 수정 전후 영향을 기록하고 기존 artifact를 덮어쓰지 않습니다.
3. **후속 protocol을 test 개봉 전에 등록**합니다. encoder/head lr 분리, B2–B5 공통 head 후보와 선택 예산, epoch cap·중단 규칙을 고정합니다. B3–B5도 변경된 downstream 조건으로 실행해야 합니다. CPT 조건이 그대로라면 기존 CPT export 재사용 여부와 횟수를 명시합니다.
4. test에는 사전 지정한 모든 seed를 싣고 최고 seed를 선택하지 않습니다. 기존 validation–test에 code-identical 그룹이 25개 있다는 감사 결과도 고려해야 합니다(`codebert-diff-lab/docs/m0-audit.md:52`). **원래 test cohort의 주 결과와, 사전 정의한 validation 중복 제외 sensitivity를 함께 보고**하는 것이 좋습니다.
5. 최종 보고에서 paired 예측 차이와 프로젝트 bootstrap을 계산하고, seed 분산과 분리합니다. 현재 증거로 가능한 결론은 **“초기 프로토콜의 B2는 미수렴이며, 충분히 학습한 frozen 표현이 강한 후보”**입니다. B2·B3·B4의 일반적인 우열은 freeze 후 test 비교까지 유보해야 합니다.