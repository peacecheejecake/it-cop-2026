# CodeBERT Diff Lab — 구현 계획

> v0.2 · 2026-10-01 · 구현 전 설계안  
> 요구사항·프로토콜을 구현 가능한 모듈, CLI 계약, 작업 단위와 검증 기준으로 연결한다. 이 문서의 디렉터리·명령·설정은 **앞으로 만들 코드베이스의 명세**다.

## 1. 구현 전략

**데이터 검증 → CPU B0/B1 → 동결 B2/FT B3 → MLM B4 → RMI B5 → 로컬 L0/L1 → 통합 freeze/export → 선택적 L2** 순서로 하나씩 완결한다. 초기부터 대규모 framework를 만들지 않는다. 학습 로직은 하나의 Python 패키지에 두고, 데이터 어댑터·모델·task·평가기를 명시적 registry로 연결한다.

Notebook은 진단용 보조 산출물만 허용한다. 원본 다운로드부터 실험 선택까지의 상태가 notebook cell 순서에 의존하지 않도록 한다. 처음에는 파일 기반 run registry와 JSON/Parquet artifact를 사용하며 MLflow·W&B·DB·웹 서비스는 필수 의존성에서 제외한다.

### 1.1 제안 기술 스택

| 영역 | 선택안 | 결정 규칙 |
|---|---|---|
| 패키징 | Python 3.11 제안, uv, pyproject.toml, uv.lock | M1에서 실제 호환성 검증 후 minor·lock 고정 |
| CLI | Typer | 명령은 업무 단계별, exit code·JSON output 지원 |
| 설정 | YAML + Pydantic | unknown field 거부, 검증 후 canonical JSON hash |
| 데이터 | PyArrow/Parquet + pandas | 대형 확장 시 streaming; Pandas index로 join하지 않음 |
| 정형·텍스트 모델·metric | scikit-learn, 선택 의존성 LightGBM | 버전 고정; 자체 metric은 oracle fixture와 비교 |
| 신경망 | PyTorch + Transformers | 단일 GPU, 로컬 encoder/causal-LM snapshot; 환경별 메모리 profile |
| LLM 실행 | 로컬 forward likelihood adapter + network-free mock | 외부 API 구현 제외, weights·precision·chat template pin |
| Retrieval 확장 | 고정 TF-IDF cosine index | 초기에는 별도 vector DB·학습 retriever 불필요 |
| 학습 loop | 얇은 전용 training loop | MLM/RMI 교대·token budget·resume를 같은 계약으로 관리 |
| artifact | JSON, Parquet, safetensors 중심 | legacy pickle/joblib는 신뢰·격리 정책 적용 |
| 품질 | pytest, Ruff, mypy | network-free CPU CI + 별도 CUDA smoke |
| 보고서 | Markdown + 자체 포함 HTML | remote asset/CDN·분석 서버 없음 |

원저자 legacy 실행 환경은 `reference/` 아래 별도 문서와 environment로 관리한다. 상류 저장소의 코드를 복사·재배포하기 전에 권한을 검토하고, 가능하면 revision pin과 실행 wrapper를 사용한다.

## 2. 목표 디렉터리 구조

```text
codebert-diff-lab/
  pyproject.toml
  uv.lock
  README.md
  configs/
    datasets/           # 승인 source/adapter 설정; raw 개인정보 없음
    studies/            # public-comparison-v2, cpt-scaling-v2, retrieval-v2
    experiments/        # 명시적 variant: B0-LR ... B5-S, L0-S/L1-S/L2-S
    profiles/           # cpu, cuda, internal-eval
    reference/          # upstream revision, args, deviation 정책
  src/diff_lab/
    cli/                # 명령 파싱, exit code; 학습 계산 로직 없음
    config/             # 엄격 schema, canonicalization, config hash
    contracts/          # 데이터·run·prediction·bundle schema
    policy/             # 사용 목적, source 승인, test gate, offline 검사
    data/
      adapters/         # jit_defects4j; 후속 codechangenet/apachejit/internal
      audit/            # ID, label, duplicate, temporal, coverage
      splits/           # provided, controlled, project_holdout
      transforms/       # feature pipeline, input renderer, tokenization
    models/             # structured LR/LGBM, sparse LR, frozen/FT encoder
    evidence/           # matched query spans/hash, typed structured serialization
    llm/                # local backend, candidate scorer, tokenizer boundary checks
    prompting/          # frozen templates, public demonstrations, context budget
    retrieval/          # public train index, rank/filter, immutable snapshots
    tasks/              # supervised, MLM, RMI; batch/loss 계약
    training/           # loop, precision, optimizer, schedule, callbacks
    experiments/        # run graph, seed pairing, budget, model selection
    evaluation/         # scoring, metric, paired statistics, cost
    artifacts/          # manifest, content-addressed cache, checkpoints
    reporting/          # report model, Markdown/HTML render
    export/             # inference bundle, offline verify, integrity checks
  tests/
    unit/
    contract/
    integration/
    gpu/
    fixtures/           # 작고 합성된 공개 fixture; 내부 데이터 금지
  docs/                 # 이 명세·프로토콜·ADR·upstream audit
  reference/            # 원저자 실행 명세·revision; 권한 승인 전 vendoring 금지
  scripts/              # 명시적 공개 artifact fetch·환경 검증 보조
  .github/workflows/    # CPU 품질 CI; GPU/내부 작업 자동 업로드 금지
  data/                 # gitignored, public raw/normalized/cache만
  artifacts/            # gitignored, run·checkpoint·prediction·report
```

실제 사내 데이터는 저장소 밖의 승인된 mount 경로를 사용한다. `.gitignore`는 보안 경계가 아니므로 내부 artifact directory에 외부 CI/동기화 도구가 접근하지 않게 한다.

## 3. 모듈 간 계약

| 모듈 | 입력 | 출력 | 책임 밖의 일 |
|---|---|---|---|
| DatasetAdapter | 승인 source artifact | 표준 change/edit/feature/label tables | split별 모델 선택, 학습 |
| Auditor | 표준 테이블·manifest | audit report·eligible IDs·실패 사유 | 모델 점수로 제외 대상 고르기 |
| SplitRegistry | 감사된 ID·제공 split·정책 | immutable membership manifest | raw 데이터 수동 수정 |
| FeaturePipeline | train features | fitted transform artifact | validation/internal에서 fit |
| InputRenderer | change+edits+renderer config | text/tokens·coverage metadata | labels 사용·미래 context 보강 |
| TaskBuilder | 허용된 dataset view | MLM/RMI/supervised batch | 다른 split negative pool 조회 |
| ModelFactory | model/encoder revision·config | 모델과 trainable parameter summary | 학습 데이터 탐색 |
| Trainer | 모델·task·train view·공개 검증 view | checkpoint·metrics·RNG 상태 | public test·internal labels 읽기 |
| TextFeaturePipeline | public TrainingDatasetView | fitted TF-IDF·희소 결합 schema | test/internal vocabulary fit |
| EmbeddingExtractor | frozen encoder + EvidenceView | embeddings·hash·coverage | encoder 갱신 |
| EvidenceBuilder | query change/edits + 사전 구간 규칙 | label-free EvidenceView | 모델 점수·정답으로 구간 선택 |
| DemoSelector | 승인 public train 사례·seed/k | 고정 demo manifest | query 정답 또는 internal 예시 접근 |
| Retriever | 동결 public index·label-free query | demo IDs·검색 provenance | index에 query·internal 자료 추가 |
| PromptRenderer | task/query/public demos | prompt·hash·예산 검증 | 초과 시 query 무단 절단 |
| LocalLLMScorer | 고정 LLM·prompt·label candidates | 두 log-likelihood·score·usage | 가중치 갱신·원격 호출·임의 self-score 대체 |
| Predictor | frozen model·label-free query·승인 frozen public demos/index | scores·status·latency | query label 접근·threshold fitting·보정 |
| Evaluator | frozen predictions+sealed labels | metric·coverage·오류·통계 | 모델 state 갱신 |
| Reporter | metrics·manifests·비교 정의 | Markdown/HTML·CSV | 누락 실행을 성공으로 추정 |

부작용은 파일 저장·fetch·training 단계에서만 명시한다. `fit`류 API는 path 문자열을 바로 받지 않고 검증된 `TrainingDatasetView`를 받는다. `EvaluationDatasetView`를 전달하면 policy error를 낸다. 이는 논리적 오류 방지 장치이며 악의적 운영자의 코드 변경을 막는 보안 모델은 아니다.

## 4. Artifact·캐시·실행 상태

### 4.1 Run 폴더 계약

```text
artifacts/runs/<run_id>/
  run.json                 # 상태·parent·seed·timestamps·attempt
  resolved-config.json     # 원본 YAML이 아닌 검증된 실효 설정
  environment.json         # lock/Git/GPU/driver/precision 정보
  data-lineage.json        # 실제 읽은 artifact·split IDs/hash
  metrics.jsonl            # update·epoch·task별 기록
  token-accounting.json    # 고유 입력·task별 노출·masked targets
  label-access-ledger.json # gradient/demo/index/selection 라벨 수
  evidence-manifest.json   # 같은 query 원문 hash·정형 feature profile
  prompts/manifest.json    # L계열: template/demo/index/model/scorer hashes
  usage.jsonl              # L계열: query/demo 토큰·latency·retry·상태
  checkpoints/
  predictions/validation.parquet
  reports/summary.md
  failure.json             # 실패 시 명시, 성공 파일처럼 취급하지 않음
```

상태는 `planned → running → completed / failed / cancelled`다. 체크포인트 선택·study freeze·test evaluation은 별도 사건과 artifact로 관리한다. `completed`는 성공한 저장·검증 후 원자적으로 기록한다.

### 4.2 캐시 키

token cache 키에 source content hash, split hash, renderer version, tokenizer revision/hash, special token 목록, max sequence/message length, abstraction 정책을 포함한다. feature cache에는 feature schema·fit train membership·전처리 버전을 추가한다. basename이나 max_msg_length만으로 캐시를 재사용하지 않는다.

B1 캐시는 train membership·vocabulary/IDF·analyzer·field weight hash, B2는 encoder state·pooling·eval mode·precision을 포함한다. LLM 응답 캐시는 task/query/demo IDs와 순서·원문/prompt hash·model/quantization/chat template/scorer를 포함한다. L2는 frozen index·retriever·eligible filter·tie 정책을 추가한다. 내부 query cache는 승인 로컬 경로에만 저장하고 train/index로 재사용하지 않는다.

CPT training negative pairs는 생성 seed·pool hash·sampler version·epoch/step이 식별되어야 한다. stale cache를 조용히 재사용하지 않는다. cache 삭제 없이도 key 변경으로 새 artifact가 만들어져야 한다.

### 4.3 Resume 계약

모델, optimizer, scheduler, gradient scaler, RNG(Python/NumPy/Torch CPU/CUDA), sampler cursor, task schedule 위치, token counters, optimizer update, early-stopping state를 저장한다. 초기 구현은 optimizer update 경계에서만 checkpoint한다.

중간 accumulation에서 죽은 경우 마지막 완료 update에서 재시작한다. 일부 batch를 재처리할 수 있지만 완료 update를 이중 계수하지 않는다. deterministic 테스트에서는 worker 수와 masking seed를 고정한다. loader 구조를 바꾸는 resume는 거부하거나 새 run으로 기록한다.

동일 환경의 연속 실행과 중단/재개의 loss·파라미터를 정한 허용오차로 비교한다. GPU 커널·버전 차이까지 bitwise equality를 약속하지 않는다. disk full·손상 checkpoint·hash mismatch는 명시적 실패로 처리한다.

## 5. CLI 계약

아래 명령은 **구현 목표 예시**다. `diff-lab` 실행 파일과 파일 경로는 아직 생성하지 않았다. 예시 ID는 M0/M1에서 실제 artifact로 치환한다.

```bash
# 환경·정책 확인
uv run diff-lab doctor --profile cpu
uv run diff-lab doctor --profile cuda

# 승인된 원본 import: legacy pickle은 별도 격리 importer 경유
uv run diff-lab data import --source jit-defects4j \
  --archive ./downloads/data.zip --approval ./approvals/jitd4j.json
uv run diff-lab data audit --dataset jitd4j-audit1
uv run diff-lab data split --dataset jitd4j-audit1 --config configs/studies/public-comparison-v2.yaml

# 정의 확인과 실행: 기본 run은 public train/validation까지만
uv run diff-lab study plan --config configs/studies/public-comparison-v2.yaml
uv run diff-lab experiment run --study public-comparison-v2 --models B0-LR,B0-LGBM,B1-TFIDF-S --seeds 42
uv run diff-lab experiment run --study public-comparison-v2 --models B2-S,B3-S --seeds 42
uv run diff-lab experiment run --study public-comparison-v2 --models B4-S,B5-S --seeds 42
uv run diff-lab experiment run --study public-comparison-v2 --models B0-LR,B0-LGBM,B1-TFIDF-S,B2-S,B3-S,B4-S,B5-S --seeds 42,43,44

# 동일 LLM: zero-shot / public-train few-shot. 가중치 학습이 아님
uv run diff-lab prompt validate --study public-comparison-v2
uv run diff-lab llm run --study public-comparison-v2 --models L0-S,L1-S

# 확장 전용: 사전에 승인한 공개 train corpus만 index에 사용
# uv run diff-lab retrieval build --study retrieval-v2 --source public-train

# 결과 선택·고정·평가의 분리
uv run diff-lab study freeze --study public-comparison-v2
uv run diff-lab evaluate --freeze public-comparison-v2-frozen --split public-test
uv run diff-lab report --freeze public-comparison-v2-frozen --format html,markdown,csv

# 후속 오프라인 이전: 내부에서는 학습 불가
uv run diff-lab export --freeze public-comparison-v2-frozen --profile offline-eval
uv run diff-lab bundle verify --bundle ./bundles/public-comparison-v2-frozen
uv run diff-lab predict --bundle ./bundles/public-comparison-v2-frozen \
  --dataset internal-final-v1 --profile internal-eval
```

공통 옵션은 `--dry-run`, `--json`, `--offline`, `--output-dir`로 제안한다. `--dry-run`은 명령·dependency·예상 artifact 경로만 출력하고 모델/데이터를 자동 다운로드하지 않는다. 내부 관련 환경에서 stdout JSON도 민감 ID나 원문을 포함하지 않는다.

종료코드 제안: 0 성공, 2 설정/입력 오류, 3 정책 위반, 4 무결성·데이터 오류, 5 실행 자원/학습 오류. OOM 시 token budget나 sequence length를 자동으로 바꾸지 않고 조정 가능한 값과 새 run 필요 여부를 안내한다.

## 6. 설정 예시

이 YAML은 설계 예시이며 실행 configuration이 아니다. `PIN_REQUIRED`는 formal 실행 시 허용하지 않는 값이다. schema 검증기는 alias·중복 key·암묵적인 typo를 거부한다.

```yaml
schema_version: 2
matrix_version: 2
study_id: public-comparison-v2
track: controlled
protocol_version: v2

dataset:
  snapshot_id: PIN_REQUIRED
  split_id: PIN_REQUIRED
  train_visibility: public
  feature_profile: jit14-audited-v1

model:
  base: microsoft/codebert-base
  revision: PIN_REQUIRED
  renderer: message-add-del-text-v2
  max_length: 512
  max_message_tokens: 64
  fusion_head: cls-feature-tanh-v1

experiments: [B0-LR, B0-LGBM, B1-TFIDF-S, B2-S, B3-S, B4-S, B5-S, L0-S, L1-S]
evidence_profile: matched
information_profile: structured_fused

text_baseline:
  classifier: logistic_regression
  analyzer: char
  ngram_range: [3, 5]
  lowercase: false
  min_df: 2
  max_features_per_field: 50000
  fit_role: supervised_train

llm:
  backend: local_hf
  model_id: PIN_REQUIRED
  revision: PIN_REQUIRED
  tokenizer_revision: PIN_REQUIRED
  precision_profile: PIN_REQUIRED
  chat_template_hash: PIN_REQUIRED
  task_prompt_hash: PIN_REQUIRED
  scorer: candidate_loglikelihood
  labels: ["0", "1"]
  max_context_tokens: PIN_REQUIRED
  query_truncation: forbidden_after_evidence_freeze
  train_weights: false
  examples:
    source_role: supervised_train
    visibility: public
    k: 4
    class_counts: {negative: 2, positive: 2}
    seeds: [42, 43, 44]
    index_write_from_query: false
  retrieval_enabled: false
seeds: [42, 43, 44]

cpt:
  permitted_role: cpt_train
  budget_unit: nonpadding_input_token_exposures
  budget: 10000000
  micro_batch_size: 8
  gradient_accumulation_steps: 16
  learning_rate: 0.00002
  mlm_probability: 0.15
  b5_task_schedule: alternating_1_to_1
  rmi_replacement_probability: 0.5

finetune:
  max_epochs: 20
  micro_batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 0.00001
  selection_metric: validation_ap
  patience_epochs: 5
  class_weight: null

evaluation:
  primary_metrics: [ap, recall_at_5pct, recall_at_10pct]
  budget_unit: change_count
  topk_rounding: ceil
  threshold_policy: validation_max_f1_then_highest_threshold
  calibration: none
  require_freeze_for_test: true

policy:
  allow_internal_training: false
  allow_internal_demonstrations: false
  allow_internal_index_members: false
  allow_remote_llm: false
  allow_test_selection: false
  automatic_remote_logging: false
```

금지 정책은 사용자가 YAML에서 true로 바꾸면 풀리는 옵션이 아니다. 해당 schema와 study policy가 override를 거부해야 한다. 허용 환경·데이터 source registry는 별도의 승인된 설정에서 주입한다.

## 7. Milestone와 완료 조건

| 단계 | 구현 내용 | 의존성 | 산출물·gate |
|---|---|---|---|
| M0 | upstream/data/모델 사용 범위·schema·label·feature 감사 | 없음 | source approval, 미확정 항목; LLM 모델은 명시적 미정 가능 |
| M1 | uv/CLI/config·matrix migration·contract·policy·CPU CI | M0 최소 승인 | 네트워크 없는 schema/policy fixture |
| M2 | 데이터→B0-LR/B0-LGBM/B1-TFIDF-S→평가 | M1 | 실제 공개 train/validation E2E; 공식 test 개봉 아님 |
| M3 | EvidenceView·동결 B2-S/FT B3-S·GPU profile | M2 | 같은 head/input의 동결 여부 검증·validation 비교 |
| M4 | B4-S MLM CPT·resume·encoder 이전 | M3 | 같은 downstream 조건의 B3-S/B4-S 비교 |
| M5 | B5-S MLM+RMI·예산·pool 검증 | M4 | B4-S/B5-S 비교·task별 회계 |
| M6L | 로컬 LLM·candidate scorer·L0-S/L1-S·demo freeze | M2 + M3 EvidenceView 계약 | 같은 query/model, train-only 예시, prompt/score 테스트 |
| M6 | 통합 반복·통계·전체 freeze·test/report·offline export | M2~M5, 통합 완료에는 M6L 추가 | DoD-Core 또는 DoD-Comparative를 구분해 출력 |
| M7 | L2·text-only/native·공개 데이터 확대·내부 평가 | M6 + 확장별 승인 | 별도 protocol·독립된 완료 상태 |

M6L은 M4/M5의 학습 결과를 기다릴 필요 없이 공통 계약이 준비되면 별도 개발 경로로 진행 가능하다. 이것은 작업 의존성 설계이며 실제 비동기 작업을 예약했다는 뜻이 아니다. M6의 공식 전체 비교는 모든 사전 등록 primary variant를 freeze한 후 수행한다.

원본 사용 권한 또는 LLM 자원이 불명확하면 해당 실행은 not_run/unavailable로 남긴다. CodeBERT에 제안한 24GB VRAM이 LLM에도 충분하다고 가정하지 않는다. 메모리·문맥·양자화 변경은 결과를 보기 전 profile에서 확정한다. GPU 없이도 M2는 완료 가능하다. LLM 미실행 시 통합 비교 완료라고 표시하지 않는다.

## 8. 구현 백로그

각 항목은 독립적인 작업 또는 PR로 나눌 수 있는 단위다. 상태는 모두 `planned`다. 추정 일정이 아니라 의존성과 인수 기준으로 순서를 정한다.

| Task | 작업 | 의존성 | 요구사항 | 완료 증거 |
|---|---|---|---|---|
| T00 | upstream/code/data 사용 범위·revision 감사 | 없음 | FR-02/03 | audit 문서·pin·누락 목록 |
| T01 | archive 구조·label/feature 대응 분석 | T00 | FR-03/04/05 | 알려진 양성/음성·1:1 join fixture |
| T02 | Python/uv/CLI skeleton·CPU 품질 CI | T00 | FR-01,NFR-01/09 | clean env `--help`, lint/type/test |
| T03 | config/data/run/prediction schema | T01/T02 | FR-01/03/11 | unknown field·ID/label 오류 실패 |
| T04 | source registry·용도 정책·offline gate | T03 | FR-04/13/14 | internal train/test selection 거부 |
| T05 | 격리 import·원본 hash·manifest | T01/T03/T04 | FR-02/03 | 원본 불변, 변환 재현·archive 공격 거부 |
| T06 | split·duplicate·temporal audit | T05 | FR-04 | leak fixture 검출·제외 회계 |
| T07 | feature pipeline·missingness·allowlist | T06 | FR-05 | train-only fit·열 순서·미래 특징 검증 |
| T08 | AP/ROC/F1/TopK·공통 prediction schema | T03 | FR-12 | hand-calculated oracle와 일치 |
| T09 | B0-LR train/predict·validation 선택·기본 freeze | T07/T08 | FR-06/13 | 실제 공개 데이터 CPU E2E |
| T10 | 로컬 CodeBERT·tokenizer snapshot·새 token 초기값 | T04/T06 | FR-07 | revision/hash·embedding 크기/초기값 검증 |
| T11 | 공통 EvidenceView·input renderer·coverage·token cache | T10 | FR-07 | 길이/ADD/DEL/결측·cache key 테스트 |
| T12 | fusion head·supervised step·B3 | T07/T11 | FR-08 | encoder와 head gradient·forward parity |
| T13 | training loop·precision·atomic checkpoint/resume | T12 | FR-11,NFR-06 | 연속/재개 비교·interrupt 테스트 |
| T14 | CUDA 100-step profile·환경 pin | T13 | NFR-02/03/09 | 실제 VRAM·throughput·pretrained smoke |
| T15 | reference wrapper·차이 보고서 | T00/T12/T14 | FR-08 | 원저자 옵션·loss/threshold 차이 명시 |
| T16 | MLM collator·task·fixed dev masks | T11/T13 | FR-09 | mask rate·special exclusions·NaN edge |
| T17 | token budget·task accounting·CPT encoder export | T16 | FR-09/11 | 실제 exposures·head reset·resume |
| T18 | B4→동일 B3 downstream 연결 | T17/T14 | FR-09 | B3와 공통 config diff 자동 검사 |
| T19 | RMI sampler·pool provenance·eligibility | T06/T11 | FR-10 | negative overlap/동일 메시지 거부 |
| T20 | multitask loop·B5·task별 상태 resume | T19/T17 | FR-10/11 | 1:1 task schedule·예산·부호 확인 |
| T21 | B계열 study runner·seed pairing·bounded selection | T09/T18/T20/T30/T31/T32/T40 | FR-11/13 | task graph·paired init·test 미접근 |
| T22 | paired 통계·비용·HTML/Markdown 보고서 | T08/T21 | FR-12 | seed별 원자료·bootstrap·coverage·오류 포함 |
| T23 | 최종 freeze ledger·test evaluator | T21/T22 | FR-13 | freeze 이전 거부·사전 후보만 평가 |
| T24 | offline bundle·integrity·evaluator-only profile | T13/T23 | FR-14 | 네트워크 차단 fixture와 fit 거부 |
| T25 | Core 문서·재현 명령·실제 검증 범위 정리 | T24/T30/T31/T32 | NFR-07/08/10 | DoD-Core/Ready 상태표 |
| T26 | CodeChangeNet 비라벨 adapter·dedup | T25 | FR-15/16 | 평가 프로젝트 제외·null label 유지 |
| T27 | ApacheJIT adapter·커밋 원문 복원 | T25 | FR-15/16 | clone 실패/사라진 commit·coverage 보고 |
| T28 | B2-T/B3-T·학습량·input ablation·project holdout | T25, 관련 adapter | FR-16 | 별도 protocol·고유량/노출량 구분 |
| T29 | 내부 배포 mapping·prediction/label 격리 | T24+승인 | FR-17 | 고정 집계·관찰창·반출 정책 검증 |

### v0.2 추가/변경 작업

기존 T00~T29는 기능 맥락을 보존하되 B1/B2 의미는 위 migration에 따른다. 실제 실행은 아래 의존성과 milestone의 합집합을 따른다. 모든 항목 상태는 planned다.

| Task | 작업 | 의존성 | 요구사항 | 완료 증거 |
|---|---|---|---|---|
| T30 | B0-LGBM·환경/시드/결측·tuning budget | T07/T08/T09 | FR-06 | 같은 feature·cohort CPU E2E |
| T31 | B1-TFIDF-S·sparse LR·pipeline export | T07/T08/T11의 CPU evidence 부분 | FR-18 | train-only vocab/IDF, sparse 메모리·inference parity |
| T32 | B2-S/T frozen extractor·cache·head | T12/T13 | FR-19 | encoder state 불변, B3와 head/입력 일치 |
| T33 | matched/native·T/S 비교 preflight | T11/T12 | FR-20 | 같은 source spans/hash·정형 profile 검증 |
| T34 | 로컬 LLM adapter·capability/profile·mock | T04/T10/T33 | FR-21 | 외부 API 차단·revision/precision·mock 상태 구분 |
| T35 | 두 라벨 log-likelihood scorer | T34 | FR-21 | 1/multi-token·경계·underflow·양 후보 누락 fixture |
| T36 | task prompt·structured serializer·context budget | T33/T34 | FR-22 | query 보존·template hash·injection 구획화 |
| T37 | public-train demo set·L0/L1 실행 | T35/T36 | FR-22 | 0-shot/4-shot·예시 ID/seed·label-access ledger |
| T38 | LLM 통계·비용·통합 freeze/export/report | T22/T23/T24/T37 | FR-12/13/14/24 | DoD-Comparative, 공개 demo offline 사용·내부 예시 금지 |
| T39 | L2 TF-IDF cosine·immutable index·시점 filter | T31/T37/T38 | FR-23 | k/class mix·query duplicate exclusion·offline 평가 |
| T40 | matrix v2 registry·legacy ID migration | T03 | FR-01/24 | v0.1 B1을 TF-IDF로 오해석하면 실패 |

T31 때문에 M2가 GPU를 요구하지 않도록 EvidenceView builder는 CPU와 로컬 tokenizer로 실행 가능해야 한다. 초기 CPU CI는 작은 로컬 tokenizer fixture를 쓴다. 실제 CodeBERT tokenizer를 쓰는 public run은 snapshot을 먼저 명시적으로 취득·pin한다. M3의 GPU 모델 로딩과 분리한다.

## 9. 자동 인수 테스트 목록

| Test | 상황 | 기대 결과 | 연계 |
|---|---|---|---|
| AT-01 | YAML에 오타·PIN_REQUIRED·중복 key | 모델 로드 전 실패 | FR-01 |
| AT-02 | 같은 change_id의 상충 라벨/2:1 feature join | audit 실패, 원인 ID는 승인 환경에서만 출력 | FR-03 |
| AT-03 | CPT source에 validation/test/internal ID 삽입 | gradient step 전에 정책 위반 | FR-04 |
| AT-04 | public manifest만 편집한 미승인 raw source | 승인 lineage 불일치로 학습 거부 | FR-04 |
| AT-05 | scaler fit에 validation feature 전달 | `TrainingDatasetView` 계약 위반 | FR-05 |
| AT-06 | 단일 class·동점 score·작은 N의 metric | 지정한 null/rounding/tie 규약과 일치 | FR-12 |
| AT-07 | ADD/DEL 교환·긴 메시지·빈 코드 | 입력 방향·총길이·coverage·eligibility 일관 | FR-07 |
| AT-08 | tokenizer/renderer/length 바꾸고 캐시 조회 | 새 cache key, stale cache 미사용 | FR-07/11 |
| AT-09 | B2-S/B3-S/B4-S/B5-S 초기 생성 | native tokenizer·초기 encoder·downstream head 초기값 동일 | FR-08/09/10 |
| AT-10 | B4 CPT→FT 이전 | encoder만 이전; MLM head와 optimizer는 FT에 섞이지 않음 | FR-09 |
| AT-11 | MLM의 특수 token·0 mask batch | 특수 token 제외·유효 target 확보·NaN 없음 | FR-09 |
| AT-12 | RMI pool에 자기/동일 메시지/test 후보 | 거부·skip 기록, 다른 split fallback 없음 | FR-10 |
| AT-13 | defect label을 RMI batch에 잘못 매핑 | schema/fixture 검사 실패 | FR-10 |
| AT-14 | accumulation 중 강제 종료 후 resume | 마지막 완료 update부터 재개·token 중복 계수 없음 | FR-11 |
| AT-15 | B5 resume 시 task counter 유실 | incompatible checkpoint 거부 | FR-10/11 |
| AT-16 | token budget 종료가 batch 중간에 발생 | 기록된 제한 초과는 최대 한 update 이내; 실제량 보고 | FR-09/10 |
| AT-17 | freeze 없이 public test 평가 요청 | nonzero policy exit; test labels 읽지 않음 | FR-13 |
| AT-18 | 일부 predictions 누락·NaN·중복 ID | final 비교 실패; coverage 진단은 생성 | FR-12 |
| AT-19 | 내부 데이터로 fit/calibrate/demo/index 구성 요청 | 비라벨이어도 거부; 공개 frozen 예시의 offline 참조는 별도 허용 test | FR-14/22/23 |
| AT-20 | offline bundle weight/tokenizer hash 변경 | inference 전 무결성 오류 | FR-14 |
| AT-21 | 내부 inference 중 원격 다운로드 시도 | 정책과 network sandbox에서 차단 | FR-14 |
| AT-22 | archive path traversal/symlink·미승인 pickle | 안전한 변환 경계에서 거부 | FR-02/03 |
| AT-23 | fixture tiny encoder가 성공했지만 GPU 테스트 미실행 | report에 실제 CodeBERT/GPU는 unverified | NFR-08 |
| AT-24 | 핵심 통제군의 input/head/feature 설정 불일치 | study comparison check 실패 | FR-08/11 |
| AT-25 | 내부 feature 결측·배포 매핑 누락 | 사전 정책만 적용; 임의 zero fill/새 aggregation 금지 | FR-17 |

### v0.2 추가 인수 테스트

| Test | 상황 | 기대 결과 | 연계 |
|---|---|---|---|
| AT-26 | test에만 있는 단어를 TF-IDF fit에 섞음 | train-only provenance/canary 검사 실패 | FR-18 |
| AT-27 | 큰 sparse 텍스트 feature를 dense로 강제 변환 | 구현 계약/메모리 test 실패 | FR-18 |
| AT-28 | B2 head 학습 전후 encoder state·embedding cache 비교 | encoder 불변, head 변경; stale encoder cache 거부 | FR-19 |
| AT-29 | B2/B3 query·head·pooling, L0/L1 query 내용 불일치 | matched comparison preflight 실패 | FR-20 |
| AT-30 | L0 prompt에 라벨 demo 포함 | zero-shot 계약 위반 | FR-22 |
| AT-31 | L1 demo/L2 index에 validation/test/internal·동일 patch 삽입 | provenance/duplicate 검사 실패 | FR-22/23 |
| AT-32 | 1-token/다중 token 후보·매우 작은 확률·prefix 재토큰화 | 수식 oracle와 일치; 경계 불일치 거부 | FR-21 |
| AT-33 | 한 후보 likelihood 없음 또는 자유문장 self-score만 반환 | unsupported/failed; 0.5 대체 금지 | FR-21 |
| AT-34 | 4-shot 문맥 초과 | query를 더 자르지 않고 preflight 실패 | FR-22 |
| AT-35 | internal query + 승인 public frozen demo/index | offline 추론 허용, internal label 접근/index append 금지 | FR-14/22/23 |
| AT-36 | historical index에 미래 commit/미관찰 라벨·unknown time | 해당 항목 제외; 정적 benchmark와 구분 | FR-23 |
| AT-37 | code comment에 prompt 명령·네트워크 요청 삽입 | 비신뢰 구획 유지, 도구/egress 없음; 모델 면역 주장 안 함 | FR-22 |
| AT-38 | LLM 반복 3회에 training_seed_count=3 기록 | replication schema/보고서 검증 실패 | FR-24 |
| AT-39 | schema 없이 legacy B1 실행 또는 결과 import | 명시적 migration 요구; TF-IDF 자동 매핑 거부 | FR-24 |
| AT-40 | LLM 미실행인데 DoD-Comparative 완료 선언 | 완료 상태 검증 실패; Core는 별도 판정 | FR-24 |
| AT-41 | LLM revision/quantization/demo order/index hash 변경 후 응답 cache 사용 | 새 cache key; frozen bundle 불일치 거부 | FR-11/21/23 |

CPU CI는 합성 dataset과 작은 무작위 encoder, local LLM mock으로 contract를 검증하고 인터넷 호출 없이 실행한다. 실제 공개 artifact·CodeBERT·CUDA 검증은 별도 명시적 job으로 실행하며 결과·환경·hash를 보관한다. CI green만으로 논문 실험을 재현했다고 표기하지 않는다.

## 10. 구현 단계의 의사결정 기록(ADR)

| ADR | 초기 결정 | 변경하려면 필요한 근거 |
|---|---|---|
| ADR-001 | Python CLI 단일 패키지, uv | CLI로 해결되지 않는 실제 workflow 요구 |
| ADR-002 | public-only 학습, internal evaluation-only | 기존 연구 질문의 변경이므로 별도 연구로 분리 |
| ADR-003 | reference/controlled 분리 | 분리 제거 불가; 구현 추적 방식만 개선 |
| ADR-004 | P0는 기존 공개 라벨 패키지 사용 | 목표 언어/예측 단위가 기존 자료에 없는 근거 |
| ADR-005 | 단일 CUDA GPU, 전체 파라미터 학습 | 실제 profile에서 자원 병목을 확인한 경우 |
| ADR-006 | encoder 512-token query·공통 원문 matched evidence | native 긴 문맥은 별도 protocol·coverage 보고 |
| ADR-007 | 10M token CPT부터 시작 | 공개 validation의 정해진 확장 기준 |
| ADR-008 | artifact는 파일 기반, 외부 logging off | 협업 요구·내부 보안 승인 후 선택적 연결 |
| ADR-009 | 확률적 학습 3 seeds·L1 3 demo sets, AP 선택 | 사전 등록된 더 큰 통계/예산 설계 |
| ADR-010 | no performance target in DoD | 성능 숫자가 아니라 실험 무결성으로 구현 완료 판정 |

### v0.2 추가 ADR

| ADR | 초기 결정 | 변경하려면 필요한 근거 |
|---|---|---|
| ADR-011 | 사용자 B0/B1/B2/B3/L0/L1/L2와 기존 B4/B5 통합 | matrix version·명시적 variant·legacy migration 필수 |
| ADR-012 | frozen LLM primary는 후보 log-likelihood | 다른 scoring mode는 별도 study; 내부 보정 금지 |
| ADR-013 | 공개 frozen demo/index의 내부 offline 참조 허용 | 내부 사례로 변경하려면 별도 연구, 현 study에서는 불가 |
| ADR-014 | controlled-v2는 신규 marker embedding 미사용 | reference special tokens는 보존·renderer 차이 명시 |

## 11. 위험과 대응

| 위험 | 대응 |
|---|---|
| 원저자 라이브러리가 새 환경과 호환되지 않음 | legacy reference 환경 격리, controlled port는 deviation 기록 |
| 원본 pickle 구조·라벨 의미를 잘못 이해 | 먼저 1:1 join/known-label fixture; 학습 코딩보다 M0 우선 |
| 데이터/토큰 cache가 다른 실험에 재사용됨 | 모든 의미 있는 입력·코드·정책을 key에 포함 |
| 훈련만 성공하고 평가 실험이 불공정함 | feature/head/input parity를 study preflight에서 검사 |
| 작은 validation에 반복적으로 과적합 | 후보 수·분량·selection 규칙 사전 제한; test는 freeze 후 |
| 추가 사전학습 효과가 없음 | negative result 기록; 파이프라인 완료와 성능 성공을 구분 |
| 메모리 부족·작업 중단 | micro-batch 사전 profile, accumulation, optimizer-boundary resume |
| 내부 평가가 feature/언어/코드 접근 문제로 불가 | 가능한 사전 지정 평가만 수행, 불가를 성공으로 포장하지 않음 |
| 데이터 공개 여부와 사용 허가를 혼동 | source별 approval·재배포 제외; 원본 자료를 문서 bundle에 넣지 않음 |

## 12. 첫 구현에 전달할 작업 지시

**첫 PR은 T00~T06의 최소 실행 기반과 검증 fixture를 대상으로 한다.** 원본 구조·라벨·split 확인 결과를 함께 남긴다. 그다음 T07~T09로 B0-LR E2E를 완결하고, T30/T31로 LightGBM·TF-IDF까지 확장한다. T40의 versioned registry를 초기에 넣어 B1/B2 충돌을 방지한다. 신경망 학습을 먼저 만들고 나중에 분할·평가를 끼워 넣는 순서는 사용하지 않는다.

각 PR은 관련 FR/AT ID, 변경한 artifact/schema version, 실행한 검증 명령, 실행하지 못한 검증(특히 CUDA·실제 사내 데이터), 다음 dependency를 기록한다. 이 계획의 완료 표시를 실제 검증 없이 자동으로 done으로 변경하지 않는다.

---
근거: [references.md](references.md). 이 문서의 package 구조·CLI·milestone·test·ADR는 본 프로젝트를 위한 자체 설계다.
