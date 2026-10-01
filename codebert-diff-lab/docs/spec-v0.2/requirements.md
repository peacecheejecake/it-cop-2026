# CodeBERT Diff Lab — 요구사항 명세

> 버전: 0.2 · 수정일: 2026-10-01 · 상태: 구현 전 설계안  
> 목적: 공개 코드 변경 데이터에서 정형/텍스트/encoder/CPT/고정 LLM을 재현 가능하게 비교하는 연구용 코드베이스  
> 문서 내 MUST는 필수, SHOULD는 권장이다. 제안한 기본값은 실험 전에 설정 파일로 고정한다. 구현·데이터 다운로드·GPU 학습이 완료되었다는 뜻은 아니다.

## 1. 목표와 성공의 정의

### 1.1 연구 질문

- **RQ1**: 정형 특징만 쓰는 모델에 비해 코드 표현을 결합하면 결함 위험 순위가 개선되는가?
- **RQ2**: 동일한 CodeBERT·입력·분류기 조건에서 diff MLM 추가 사전학습(CPT)이 추가 가치를 제공하는가?
- **RQ3**: MLM에 Replaced Message Identification(RMI)을 추가하는 것이 도움이 되는가?
- **RQ4**: 개선은 학습 데이터 다양성, 반복 노출량, 정형 특징, 커밋 메시지 중 무엇에서 오는가?
- **RQ5**: 공개 데이터에서 선택·고정한 모델이 새로운 프로젝트와 사내 평가 데이터로 이전되는가?

- **RQ6**: TF-IDF·동결 encoder·fine-tuned encoder·고정 LLM의 성능/비용과 라벨 사용 방식은 어떻게 다른가?
- **RQ7**: 같은 LLM에서 공개 라벨 예시와 유사 사례 검색의 추가 가치는 무엇인가?

RQ1~RQ3와 RQ6의 B계열은 핵심 구현, L0/L1 비교는 통합 비교의 후속 필수 단계다. RQ7의 L2와 RQ4는 확장, 사내 평가는 별도 승인 단계다. [통합 비교군](comparison-matrix.md)이 ID와 비교 목적의 기준 문서다.

성공은 특정 논문 점수를 넘기는 것이 아니라 **동일 조건의 비교, 데이터 경계의 검증, 재실행 가능한 결과 생성**이다. CPT 효과가 없거나 비용 대비 이득이 작다는 결과도 유효한 연구 결과로 취급한다.

### 1.2 예측 대상의 경계

| 구분 | 공개 연구의 기본 목표 | 후속 사내 평가 목표 |
|---|---|---|
| 예측 단위 | 커밋/코드 변경 | 배포 또는 사전에 정의한 내부 변경 단위 |
| 정답 | 결함을 유발한 변경 여부 | 해당 배포와 연결된 운영 장애 여부 |
| 점수 명칭 | `defect_risk_score` | `deployment_risk_score` 또는 이전 평가 점수 |
| 확률 해석 | 공개 결함 분포의 보정 여부를 별도 표시 | 사내 장애 확률이라고 자동 해석하지 않음 |
| 모델 변경 | 공개 train/validation에서 허용 | 사내 평가 데이터로 변경 금지 |

결함을 수정한 커밋과 결함을 유발한 커밋을 혼동하지 않는다. 공개 결함 라벨과 사내 SEV/장애 라벨을 하나의 학습 목표로 합치지 않는다.

## 2. 범위와 우선순위

| 단계 | 포함 범위 | 완료 기준 |
|---|---|---|
| P0: CPU 실행 기반 | 데이터 감사·정책·B0-LR/B0-LGBM/B1-TFIDF-S·평가 | 실제 공개 데이터 train/validation E2E |
| P1: encoder/CPT | B2-S/B3-S/B4-S/B5-S, reference 감사, freeze/export | DoD-Core: 7개 실행 variant의 무결한 비교 |
| P2: LLM 핵심 비교 | 로컬 고정 LLM의 L0-S/L1-S, score/prompt/demo 계약 | DoD-Comparative: 총 9개 variant 비교 |
| P3: 공개 확장 | L2, text-only ablation, CodeChangeNet/ApacheJIT, 학습량·project holdout | 확장별 protocol·한계·비용 보고 |
| P4: 내부 이전 평가 | 승인 offline adapter, 동결 public demo/index, 배포 집계 | 평가 전용·내부 학습/외부 전송 금지 검증 |

초기 범위에서 제외: 새 SZZ 라벨러, 대규모 크롤러, 웹 UI, 운영 배포 차단, 서빙 서비스 구축, 분산 학습, LoRA/QLoRA, LLM fine-tuning, 외부 LLM API 호출, CCT5/CodeReviewer 전체 재현, COBOL 적응 학습. LLM provider 경계는 로컬 backend와 CI mock만 구현한다. 데이터나 모델을 자동 구매·다운로드하지 않는다.

## 3. 실험군과 트랙

### 3.1 식별자

정식 매핑은 [comparison-matrix.md](comparison-matrix.md)와 설계용 [experiment-registry.json](experiment-registry.json)에 둔다. `matrix_version=2`와 실행 variant ID를 함께 기록하며 family ID만으로 결과를 저장하지 않는다.

| Family | 핵심 variant | 의미 |
|---|---|---|
| B0 | B0-LR, B0-LGBM | 정형 특징 분류 |
| B1 | B1-TFIDF-S | TF-IDF + 정형 특징 + LR |
| B2 | B2-S; 확장 B2-T | 동결 encoder + 학습 head |
| B3 | B3-S; 확장 B3-T | encoder 전체 fine-tuning |
| B4 | B4-S | MLM CPT → B3-S와 같은 FT |
| B5 | B5-S | MLM+RMI CPT → B3-S와 같은 FT |
| L0 | L0-S | 라벨 예시 없는 고정 LLM |
| L1 | L1-S | 공개 train 고정 예시를 쓰는 동일 LLM |
| L2 | 확장 L2-S | 공개 train에서 검색한 예시를 쓰는 동일 LLM |

B0는 LApredict와 동일 모델이라고 표기하지 않는다. v0.1 B1/B2는 각각 v0.2 B2-T/B3-T로 이관되며, 과거 결과의 ID를 덮어쓰지 않는다. T/S는 정형 특징 제공 유무이며 모델의 학습 상태와 별도 축이다.

### 3.2 실험 트랙과 정보 프로파일

**reference**는 원저자 코드·전처리·split·metric과 차이를 기록한다. revision·환경·패치·미해결 차이를 남기고 미실행은 `not_run`, 미검증은 `executed_unverified`로 표시한다.

**controlled**는 B2-S~B5-S의 초기 encoder·입력·feature·head·downstream 예산을 통일한다. `JIT-Fine-compatible` 또는 `BiCC-inspired`이며 정확한 재현이라고 주장하지 않는다. L0/L1/L2는 같은 고정 LLM과 scorer를 사용한다. reference/controlled 결과를 분리한다. 상류 누수 발견 시 차이를 기록하고 안전하게 수정한 별도 실행을 만든다.

**matched/native**는 위 트랙과 다른 축이다. 기본 matched는 고정된 같은 원문 query 구간과 승인된 정형 열을 제공한다. native는 모델별 더 긴 입력 등의 확장이다. T/S·matched/native·reference/controlled를 명시하지 않은 성능 비교는 final 상태로 출력하지 않는다.

## 4. 데이터 계약

### 4.1 원본과 표준화 데이터

초기 입력은 JIT-Fine 공개 패키지다. 해당 저장소는 코드·데이터와 재현 절차를 제공한다.[S02] 최초 단계에서는 직접 라벨 수집을 하지 않는다. 원본 파일은 불변 저장하고, 실험은 변환된 JSON/Parquet와 manifest를 통해 수행한다.

문헌의 데이터 건수는 검증용 참고값이지 하드코딩된 정답이 아니다. BiCC-BERT 본문과 초록에는 서로 다른 총건수가 있으므로 실제 패키지의 split별 행 수·유니크 커밋 수를 확정해야 한다.[S01]

### 4.2 공통 스키마

`schema_version=2`로 시작한다. 하나의 거대한 테이블 대신 아래 논리 테이블을 분리한다. 결측 timestamp를 현재 시각 등으로 만들어 채우지 않는다.

| 테이블 | 필수 키·필드 | 주요 검증 |
|---|---|---|
| `changes` | `change_id`, `dataset_id`, `project_id`, `source_record_id`, `visibility`, `message`, `language`, `content_hash`, `representation_kind` | ID 유일; 빈 메시지는 명시적으로 허용; visibility는 public/internal |
| `changes` 추가 필드 | `commit_sha`, `parent_sha`, `committed_at`, `available_at`, `time_provenance`, `source_uri` | 원본에 없는 값은 null; timestamp 미검증 여부 기록 |
| `edits` | `change_id`, `file_id`, `hunk_id`, `order`, `op`, `text` | op=add/delete/context; 패치 방향·순서 보존 |
| `features` | `change_id`, 명시된 feature 열, `feature_schema_id`, `provenance` | 1:1 join, 유한 수치 또는 허용 결측; 열 순서 고정 |
| `labels` | `change_id`, `label`, `label_kind`, `label_source`, `label_version` | defect label=1은 inducing, 0은 non-inducing, 미상은 null |
| `labels` 추가 필드 | `observed_at`, `observation_end`, `linkage_confidence` | 관찰 정보가 없으면 unknown으로 기록 |
| `splits` | `change_id`, `split`, `split_version`, `group_id` | downstream train/validation/test는 상호 배타적 |
| `predictions` | `change_id`, `run_id`, `score`, `score_semantics`, `prediction_status`, `latency_ms` | 평가 대상 ID와 정확히 대응; 실패를 누락하지 않음 |
| `evidence_views` | `change_id`, `evidence_profile`, `source_span_ids`, `query_content_hash`, `coverage` | matched 비교의 공통 원문 구간; label 없음 |
| `example_pool` | public train `change_id`, `label`, `pool_version`, `source_manifest_hash` | validation/test/internal 사례 금지; query label 접근 금지 |
| `retrieval_index` | `index_id`, `corpus_digest`, `retriever_revision`, `available_at_policy` | 승인된 public train만; 동결 후 변경 금지 |
| `prompt_runs` | `prompt_hash`, `query_hash`, `demo_ids`, `model_revision`, `scorer_profile`, `usage`, `status` | 원문은 승인 로컬 보관; 내부 내용 원격 logging 금지 |

run manifest에는 `matrix_version`, `variant_id`, `information_profile`, `adaptation_mode`, `label_access_ledger`를 추가한다. LLM prediction은 `candidate_loglikelihood_preference` 의미를 사용하며 supervised sigmoid와 의미를 합치지 않는다.

원본이 raw unified diff가 아니라 이미 추상화·필터링된 추가/삭제 행이면 `representation_kind=preprocessed_lines`로 기록한다. 누락된 context나 파일 경로를 복원했다고 주장하지 않는다. 실제 split과 샘플을 확인한 뒤 adapter가 지원하는 필드를 확정한다.

### 4.3 Artifact manifest

모든 dataset artifact는 다음을 포함해야 한다.

- 출처 URI, 검색/다운로드 시각, upstream revision(확보 가능 시), byte 크기, SHA-256, 사용·재배포 조건의 확인 상태.
- adapter·스키마 버전, 부모 artifact ID, 변환 명령과 설정, 생성 코드 Git SHA.
- split별 전체 수·양성 수·고유 ID 수·프로젝트·언어 분포; 누락·제외 사유와 건수.
- 학습 허용 용도: `cpt_train`, `supervised_train`, `selection`, `evaluation_only` 중 명시된 값.
- 학습 입력 hash, split membership hash, feature schema hash, privacy 등급.

manifest의 `visibility`를 임의 편집하는 것만으로 내부 데이터가 학습 가능해져서는 안 된다. 공개 학습은 승인된 source registry의 artifact와 그 검증된 파생물만 허용한다. 이 정책은 조직의 접근통제를 대체하지 않는다.

### 4.4 split과 데이터 사용 행렬

| 데이터 영역 | CPT gradient | 분류기·정형 모델 학습 | 모델/설정 선택 | 최종 평가 |
|---|---:|---:|---:|---:|
| public downstream train | 허용된 CPT subset만 | 허용 | train 내부 검증만 | 해당 없음 |
| public CPT-dev | 금지 | downstream train에 속하면 허용 | 비지도 CPT 진단만 | 해당 없음 |
| public validation | 금지 | 금지 | 허용 | 검증 결과로만 표시 |
| public test | 금지 | 금지 | 금지 | 고정 모델만 |
| internal final | 금지 | 금지 | 금지 | 고정 모델만 |

CPT-dev는 controlled 트랙에서 public train 내부에서 고정한 부분집합이다. supervised 학습의 validation과 혼동하지 않는다. 원저자 참조 트랙의 다른 검증 사용은 반드시 기록한다.

사내 데이터는 비라벨이라도 CPT·tokenizer/TF-IDF fit·분류기 학습·보정·few-shot 사례·RAG index 항목·앙상블 선택에 사용하지 않는다. scaler/imputer를 내부 분포에 다시 맞추는 것도 금지다. **공개 train에서 고정한 demo/index를 내부 오프라인 추론에서 참조하는 것은 허용**한다. 내부 query는 검색어로만 사용하고 index에 추가하지 않는다. 내부 결과로 후보·prompt·retriever·k를 다시 선택하지 않는다.

## 5. 기능 요구사항

| ID | 요구사항 | 우선순위 | 핵심 인수 조건 |
|---|---|---|---|
| FR-01 | CLI와 엄격한 설정 schema 제공 | P0 | 알 수 없는 필드·모델 ID·단위·금지 조합은 학습 전 실패 |
| FR-02 | 승인된 공개 데이터의 수동 import와 선택적 fetch | P0 | 외부 쓰기 없음; hash·용도·출처를 기록; 자동 다운로드는 명시적으로만 |
| FR-03 | 원본 adapter와 정규화 | P0 | ID별 join, label 의미 검증, 원본 불변, 안전한 변환 격리 |
| FR-04 | split 생성/검증·누수 감사 | P0 | test와 CPT overlap, 중복 ID, 충돌 라벨, 시간 위반을 식별 |
| FR-05 | 정형 feature pipeline | P0 | imputer/scaler/feature 선택은 public train에서만 fit |
| FR-06 | B0-LR/B0-LGBM 학습·저장·추론 | P0 | CPU에서 검증·freeze·test prediction·report까지 수행 |
| FR-07 | 공통 tokenizer와 입력 renderer | P1 | 특수 토큰 포함 512 이내; 추가/삭제 보존; 입력 손실 비율 기록 |
| FR-08 | B2-S 동결/B3-S FT fusion 모델 | P1 | B2는 encoder 불변·head만 갱신; B3/B4/B5는 동일 head와 encoder FT |
| FR-09 | MLM CPT | P1 | pad/특수 토큰 제외, label-free 입력, 동적 마스킹, 예산 카운터 |
| FR-10 | MLM+RMI CPT | P1 | train-only negative pool, RMI label 분리, task별 loss·토큰 기록 |
| FR-11 | 실험 실행·추적·resume | P1 | 설정/데이터/모델 hash 추적; 부분 파일을 완료로 오인하지 않음 |
| FR-12 | 공통 예측·평가·비교 | P0/P1 | 공통 ID 집합, AP/Recall@budget/F1/AUC·실패율·seed 통계 |
| FR-13 | freeze·공개 test 접근 정책 | P1 | freeze 전 test metric 실행 거부; 승인된 비교군 전체 목록 기록 |
| FR-14 | 오프라인 모델 export와 evaluation-only 경계 | P1/P4 | 모델·tokenizer·preprocessor·계약 포함; 내부 학습 명령 거부 |
| FR-15 | 비라벨/라벨 공개 데이터 확장 adapter | P3 | CodeChangeNet은 미라벨로 유지; ApacheJIT 원문 복원 상태 기록 |
| FR-16 | 학습량·입력·프로젝트 이전 실험 | P3 | 고유 샘플/처리량/compute를 구분하고 test로 분량을 선택하지 않음 |
| FR-17 | 사내 배포 단위 집계·평가 | P4 | 모델과 집계 규칙 사전 고정; 매핑 실패·coverage·라벨 관찰창 보고 |

### 5.0 v0.2 추가 기능 요구사항

| ID | 요구사항 | 우선순위 | 핵심 인수 조건 |
|---|---|---|---|
| FR-18 | B1 TF-IDF·희소 feature 결합 | P0 | vocabulary/IDF/feature fit은 public train만; dense 변환 없이 LR 학습 |
| FR-19 | 동결 embedding 추출·cache | P1 | encoder 전체 state/hash 불변; pooling·eval mode·cache key 고정 |
| FR-20 | 공통 EvidenceView·비교 preflight | P1/P2 | matched query 원문 hash·정형 열 일치; native 별도 study |
| FR-21 | 로컬 LLM scorer·backend capability | P2 | 양 후보 likelihood·다중 token 경계 검증; fake self-score 대체 금지 |
| FR-22 | L0/L1 prompt·public demo 관리 | P2 | 같은 모델·query; demo lineage 검증·prompt freeze·외부 전송 없음 |
| FR-23 | L2 public retrieval·frozen index | P3 | 같은 k/class mix; 중복·시간·라벨 provenance; 내부 index 갱신 금지 |
| FR-24 | 통합 study·label ledger·schema migration | P0/P2 | v0.1 B1 오해석 거부; fit/in-context/selection 라벨 사용량 분리 |

### 5.1 원본 import 보안

Python pickle 역직렬화는 임의 코드 실행 위험이 있다.[S10] 승인 hash만으로 안전성이 보장되는 것은 아니다. legacy pickle import는 네트워크·비밀정보가 없는 격리 환경에서 읽기 전용 원본을 받아 JSON/Parquet로 내보내는 별도 명령이어야 한다. 일반 학습·평가 명령은 외부 pickle을 직접 읽지 않는다. archive path traversal, symlink, 압축 폭탄에 대한 제한도 포함한다.

### 5.2 정형 특징

M0 감사에서 원본의 feature 이름·순서·단위·전처리를 확인한다. controlled 기본 후보는 NS, ND, NF, Entropy, LA, LD, LT, FIX, NDEV, AGE, NUC, EXP, REXP, SEXP이다.[S01] 실제 입력에는 승인한 allowlist만 사용한다.

역사 특징은 원칙적으로 예측 시점까지의 정보여야 한다. `FIX`가 현재 커밋의 메시지로부터 만들어졌는지, 미래의 결함 결과를 참조했는지 구분한다. 근거가 불명확한 제공 특징은 `provided_unverified`로 표시하고 역사적 실시간 검증에 적합하다고 주장하지 않는다. 미래 정보가 확인된 열은 controlled 트랙에서 제외하며 B0와 모든 S variant가 동일한 feature profile을 사용한다.

feature 열을 추가·삭제하면 `feature_schema_id`와 실험 family를 변경한다. 내부에 해당 열이 없다고 임의로 모두 0을 넣지 않는다. 공개 데이터로 미리 준비한 reduced-feature 모델을 사전에 지정하거나 해당 평가를 unavailable로 보고한다.

### 5.3 모델과 입력

기반 모델은 `microsoft/codebert-base`의 고정 revision이다. 로더가 MLM head를 신규 초기화했는지 반드시 기록한다. 다른 MLM checkpoint로 바꾸는 것은 별도 실험이다.

controlled 기본 head는 CLS 표현과 `Linear(feature_dim, hidden_size)→tanh` 결과를 결합하고 dropout과 단일 logit을 사용하도록 제안한다. JIT-Fine 공개 head의 구조를 참고하되, controlled loss는 수치 안정성을 위해 `BCEWithLogitsLoss`를 사용한다.[S03] reference 구현과의 차이를 기록한다.

B2-S는 B3-S와 같은 초기 encoder와 head에서 encoder만 고정한다. head 학습 중에도 encoder는 eval mode를 유지하고 optimizer parameter group에서 제외하며, 전후 state/hash를 검증한다. B2-T/B3-T는 별도 코드 전용 head를 공유한다.

B4/B5의 CPT head는 downstream으로 넘기지 않는다. encoder와 tokenizer만 이전하며, downstream head는 B3와 같은 규칙·같은 seed로 초기화한다. B5가 CPT에서 쓴 binary head를 그대로 결함 분류기에 재활용하지 않는다.

raw code를 trim/정규식 치환할 때 문자열·식별자 의미를 무단 변경하지 않는다. 파싱 가능한 언어 정보가 없는 상태에서 일반적인 공백 제거를 의미 보존이라고 간주하지 않는다.

### 5.4 긴 diff

MVP controlled 입력은 커밋당 하나의 최대 512-token sequence이며 메시지 상한은 64 tokens로 제안한다. 특수 토큰 비용을 총길이에 포함한다. 전처리에서 이미 소실된 코드가 있으면 관측 가능한 범위의 coverage만 보고한다.

파일/hunk 단위 계층 인코딩은 P3의 별도 family다. 라벨은 커밋 단위로 유지하고 여러 chunk에 복제한 라벨을 독립 표본처럼 세지 않는다. chunk-level loss를 별도 설계하지 않은 한 chunk를 합친 후 커밋 loss를 계산한다.

### 5.5 평가·오류 처리

기본 지표는 AP, Recall@5%, Recall@10%; 보조 지표는 ROC-AUC, validation에서 고정한 threshold의 F1·precision·recall이다. AP와 trapezoidal PR-AUC는 다른 필드로 관리한다.[S08]

모델 비교에서 테스트 집합의 결함 비율을 인위적으로 맞추지 않는다. 학습 imbalance 처리와 평가 분포를 구분한다. 모델별 예측 실패를 제거한 뒤 점수만 비교하지 않는다. 원래 eligible set에서 prediction status와 coverage를 먼저 보고한다. 핵심 controlled 평가에서 처리 실패가 남으면 점수는 진단용이며 final 비교는 fail 처리한다.

공개 결함 분류의 sigmoid 출력을 내부 장애의 교정된 확률로 표시하지 않는다. 내부 min/max나 평균을 이용해 점수를 다시 정규화하지 않는다.

### 5.6 TF-IDF·LightGBM

B1 기본은 메시지와 diff 각각의 문자 n-gram TF-IDF와 승인 정형 열을 결합한 LR이다. code punctuation·한 글자 연산자를 기본 word tokenizer가 버릴 수 있으므로 analyzer·ngram·lowercase 정책을 explicit config로 둔다.[S14] 시작 제안은 `analyzer=char, ngram_range=(3,5), lowercase=False, min_df=2, max_features=50000` per field이며 최적값이 아니다. vocabulary/IDF·희소 column 순서·feature weight를 bundle에 넣는다. 모델 선택 과정의 모든 fit은 train 내부에서 수행한다.[S13]

B0-LGBM은 같은 feature allowlist·대상 ID·결측 계약을 사용한다. categorical 취급, boosting 반복, objective, seeds·threads·정밀도·early-stopping 규칙을 기록한다.[S15] LR의 scaling을 동일하게 강제하지 않되 raw 정보가 달라지지 않아야 한다. B0-LR/B1-LR의 C 탐색과 LGBM 탐색 후보 수를 사전 등록하고 test로 추가 탐색하지 않는다.

### 5.7 LLM·prompt·retrieval

L0/L1은 v0.2 통합 완료의 필수 비교군, L2는 확장이다. 라벨 예시 없이 지시와 query만 주는 L0, 가중치 갱신 없이 예시를 주는 L1을 구분한다.[S16] 세부 계약과 log-likelihood score는 [통합 비교군 §6](comparison-matrix.md)에 따른다.

오프라인 공개 가중치 LLM 1개를 먼저 pin한다. L0/L1/L2 모델/양자화/score 방식을 통일한다. prompt와 demo/index는 독립 artifact이고, freeze에는 이들의 hash와 선택 예산도 포함한다. L1 demo와 L2 index는 train-only다. 사내 evaluation은 query 라벨이 predictor에 노출되지 않는 별도 evaluator 구조를 유지한다. 코드·메시지에 쓰인 지시문은 비신뢰 데이터이며 LLM에 도구 권한을 제공하지 않는다.

LLM이 단순 label만 반환하거나 잘못된 형식으로 응답하면 임의 연속 점수를 만들지 않는다. primary scorer에 필요한 두 후보 likelihood가 없으면 unsupported/failed를 보고한다. 평가 전체 coverage·실패·retry·비용을 누락하지 않는다. 모델별 unsupported 상태를 통합 비교 완료로 표시하지 않는다.

## 6. 비기능 요구사항

| ID | 요구사항 | 검증 방식 |
|---|---|---|
| NFR-01 | Python/uv 기반; CLI가 notebook보다 우선 | 동일 명령과 고정 lock으로 CPU CI 설치 |
| NFR-02 | 학습 표준은 Linux + 단일 CUDA GPU; macOS는 개발/CPU 검증 | backend·precision 명시; 불가능한 CUDA 요청은 조용히 CPU fallback하지 않음 |
| NFR-03 | CodeBERT 시작 후보는 24GB GPU·RAM 32~64GB; LLM 자원은 별도 profile | 가정일 뿐 보장 아님; 100-step profile 후 micro-batch·메모리 확인 |
| NFR-04 | 데이터·모델·코드·환경 lineage 보존 | resolved config, lock hash, GPU/driver, upstream/checkpoint hash 필수 |
| NFR-05 | 네트워크 호출을 명시적으로 통제 | 내부 실행 환경 egress 차단; 로컬 모델만 허용; telemetry off |
| NFR-06 | 재시작·resume·원자적 저장 | 중간 실패 주입 후 손상 파일 거부, 누적 token budget 복원 |
| NFR-07 | 결과를 외부 서버 없이 열람 가능 | JSON/CSV/Markdown·자체 포함 HTML 보고서; remote CDN 불필요 |
| NFR-08 | GPU 불필요 CI와 실제 GPU 검증을 분리 | tiny/random 모델 테스트를 실제 CodeBERT 검증이라고 표기하지 않음 |
| NFR-09 | 검증된 의존성 버전 고정 | uv.lock과 CPU/CUDA 환경 manifest를 연결; latest 범위로 학습하지 않음 |
| NFR-10 | 재현성 수준을 정직하게 표시 | 동일 환경 반복 허용오차·seed 분산 보고; OS/GPU 간 bitwise 일치 보장하지 않음 |

uv는 lock과 sync를 구분하며 PyTorch의 CPU/CUDA 배포 경로 설정을 지원한다.[S06][S07] 정확한 Python minor와 라이브러리 조합은 M0/M1에서 검증 후 고정한다. 설계상 제안은 Python 3.11이며, 원저자 legacy 환경은 별도 격리할 수 있다.

## 7. 전역 완료 기준

**DoD-Core**: B0-LR/B0-LGBM/B1-TFIDF-S/B2-S/B3-S/B4-S/B5-S의 사전 등록된 조건·반복 실행, 데이터 경계 검사, coverage·hash·성능/비용·label ledger 보고서를 생성한다. v0.1의 4군 완료를 v0.2 7군 완료로 자동 승격하지 않는다.

**DoD-Comparative**: DoD-Core + 같은 고정 로컬 LLM의 L0-S/L1-S 결과. 모든 9개 variant의 matched evidence, 라벨 사용, 실패, 모델 가용 시점·비용을 보고한다. L2는 필수가 아니며 미실행 상태를 명시한다. 각 모델의 randomness 축은 구분하고 1-seed smoke를 최종 연구라고 표기하지 않는다.

**DoD-Internal-Ready**: evaluation-only bundle과 synthetic internal fixture로 offline predict, train/보정/내부 demo/index 갱신/원격 업로드 거부를 검증한다. 공개 frozen demo/index를 이용한 오프라인 L1/L2는 허용 경로로 테스트한다. 실제 내부 평가 완료를 뜻하지 않는다.

**DoD-Internal-Evaluated**: 별도 승인된 실제 내부 데이터에서 동결 모델·prompt·index·집계·라벨 정책으로 평가하고 코드/언어/feature/배포 mapping coverage와 관찰창의 한계를 보고한다. 사용 불가능한 모델은 unavailable이며 완료로 대체하지 않는다.

## 8. 구현 전 검증해야 할 사항

| 항목 | 현재 상태 | 해결 시점 |
|---|---|---|
| 공개 archive의 hash·정확한 스키마·실제 split 행 수 | 아직 다운로드/확인하지 않음 | M0 |
| 코드·데이터·기반 모델의 사용 및 재배포 조건 | 확인 필요; 공개 노출과 재배포 허가는 다름 | M0 |
| upstream commit pin과 `only_adds` 등 옵션의 실제 영향 | 일부 소스 열람; 전체 call path 감사 필요 | M0/M3 |
| 제공된 split의 timestamp 검증 가능 여부 | 원본 확인 필요 | M0/M2 |
| 실제 GPU/드라이버/VRAM과 학습 처리량 | 사용자 환경 미확정·미측정 | M1/M3 |
| feature의 시점 적합성·라벨 부호 | 실제 원본과 코드로 검증 필요 | M0/M2 |
| 공개 테스트와 기반 모델 원래 pretraining 간 완전 중복 부재 | 입증하지 못함 | 제한사항으로 기록 |
| LLM 모델 revision·license·가용 메모리·후보 likelihood 지원 | 미확정; 실행 전 pin/profile | M0/M6L |
| 내부 코드 접근 허용 범위·배포 매핑·라벨 관찰창 | 별도 승인 필요 | M7 |

미확정 사항은 구현을 미루기 위한 질문이 아니라 초기 milestone의 검증 작업이다. 해결되지 않은 항목은 상태를 남기고 해당 성능 주장·실행만 차단한다.

---
관련 문서: [통합 비교군](comparison-matrix.md) · [실험 프로토콜](experiment-protocol.md) · [구현 계획](implementation-plan.md) · [근거와 확인 범위](references.md)
