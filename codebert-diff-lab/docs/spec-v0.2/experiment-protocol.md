# CodeBERT Diff Lab — 실험 프로토콜

> v0.2 · 2026-10-01 · 구현 전 제안  
> 요구사항: [requirements.md](requirements.md) · 구현 순서: [implementation-plan.md](implementation-plan.md)  
> 아래 CLI·설정 이름은 구현할 계약이다. 현재 실행 가능한 프로그램이나 실측 결과가 아니다.

## 1. 실험을 등록하는 단위

실험 ID는 `track / protocol_version / matrix_version / evidence_profile / dataset_snapshot / split_id / variant_id / budget_id / replicate_id`로 구성한다. 예: `controlled/v2/m2/matched/jitd4j-audit1/upstream-clean1/B4-S/t10m/train42`. LLM의 `demo42`는 training seed와 다른 축이다. v0.1 ID를 v0.2로 묵시적으로 해석하지 않는다.

실제 artifact 경로에는 timestamp가 아니라 config·input hash 기반의 고유 run ID를 함께 사용한다. 같은 설정의 반복 시도는 attempt ID로 구분하고 완료 결과를 덮어쓰지 않는다. 중단된 run은 성공한 결과에 합치지 않는다.

각 study는 실행 전에 다음을 명시한다: 연구 질문, 비교군, 사용 데이터, 분할, 전처리, 후보 설정 수, CPT 예산, seeds, 선택 지표, 주요 비교쌍, 테스트 cohort, 예외·중단 규칙. 변경 시 새 protocol version을 만든다.

## 2. 우선 수행할 공개 실험

### 2.1 Controlled-v2 초기 설정

| 항목 | 제안 기본값 | 규칙 |
|---|---|---|
| 비교군 | B0-LR, B0-LGBM, B1-TFIDF-S, B2-S, B3-S, B4-S, B5-S, L0-S, L1-S | core 7군 + LLM 2군; L2는 확장 |
| 핵심 비교쌍 | B2-S→B3-S→B4-S→B5-S; L0-S→L1-S | 정보 profile·head·예산 차이를 명시 |
| 정보 profile | matched-S; B0는 정형만 | 같은 query 원문 evidence hash; 모델별 token 수는 다를 수 있음 |
| 데이터 | JIT-Defects4J 승인 snapshot | actual count·ID·label mapping 감사 후 pin |
| downstream split | 제공 split에서 감사한 controlled variant | 원본을 수정했으면 새 이름; 원본 통계도 보존 |
| CPT source | downstream train에서 정한 CPT-train만 | 결함 라벨은 CPT loader에 전달하지 않음 |
| CPT-dev | public train 안에서 고정한 약 5% | ID hash 기반 label-free 선택; split seed 고정 |
| 기반 모델 | CodeBERT base 고정 revision; LLM 별도 고정 revision | B2~B5 동일 초기 encoder; L0~L2 동일 고정 LLM |
| encoder query 입력 | 총 512 tokens, 메시지 최대 64 | 같은 원문 구간을 B1/LLM에도 제공; LLM 예시는 별도 budget |
| 정형 특징 | 승인된 공통 feature profile | 모든 S variant와 B0는 같은 열·단위·원정보; transform 차이는 기록 |
| 초기 CPT 예산 | 10M non-padding token exposures | 과학적 최소량 아님; 작은 데이터 반복 가능 |
| B5 task 비율 | MLM:RMI = 1:1 | alternation schedule 고정; 원 논문 최적비율 재현 아님 |
| CPT optimizer | AdamW, lr=2e-5, weight_decay=0.01 | 초기 제안; reference 값과 구분 |
| CPT warmup | 계획된 optimizer update의 6% | 가변 padding을 고려해 사전 스케줄 추정·실제 차이 기록 |
| downstream optimizer | AdamW, lr=1e-5, weight_decay=0.01 | 최대 20 epochs, validation AP patience=5 epochs |
| micro-batch | GPU당 8 sequences | profile 후 일괄 고정; 메모리 부족 시 임의 자동 변경 금지 |
| accumulation | CPT 16, downstream 4 | GPU 1장 기준 effective sequence batch 128/32 |
| precision | 지원 시 BF16, 아니면 사전 지정 FP16 | 가능 여부 기록; reference FP16은 별도 환경 |
| downstream imbalance | 초기에는 자연 분포, loss weight 없음 | train-only balanced variant는 별도 sensitivity 실험 |
| stochastic 학습 seed | smoke=42; formal=[42,43,44] | split은 고정; 초기화·sampler·CPT task·mask RNG 분리 |
| LLM 변동 축 | L0 고정 scorer; L1 demo seeds=[42,43,44] | 같은 입력의 반복을 독립 학습으로 세지 않음 |
| 모델 선택 | public validation AP | 같은 AP라면 더 이른 checkpoint; test는 보지 않음 |
| threshold | 선택된 모델의 public validation F1 최대점 | 동률이면 더 높은 threshold; 비교 연산은 `score >= threshold` |
| calibration | 기본 없음 | 이후 별도 공개 calibration split/OOF를 쓸 때만 추가 |

위 값은 구현 가능성을 검증하기 위한 초기 프로토콜이다. GPU profile로 batch/precision을 확정한 뒤 formal 실행 전에 설정을 동결한다. 변경된 학습률·배치로 한 모델만 유리하게 재실행한 결과는 같은 controlled study에 섞지 않는다.

기본 supervised head 학습 batch·epoch·loss는 B2-S/B3-S/B4-S/B5-S에서 같다. B2-S의 encoder만 고정한다. B0-LR와 B1-TFIDF-S의 LR 시작 제안은 `C=1.0, class_weight=None`이며 solver·max_iter·수렴 상태를 기록한다. B1은 sparse 입력을 지원하는 solver를 명시하고 sparse matrix 전체를 dense로 바꾸지 않는다. 수렴 실패를 완료로 저장하지 않는다.

B0-LGBM은 CPU binary objective와 같은 feature allowlist를 사용한다. 시작 제안은 `num_leaves=15, learning_rate=0.05, n_estimators=300, min_child_samples=20`; 작은 공개 train에서의 최적값을 주장하지 않는다. 버전·seed·thread·determinism 설정을 pin한다. 최초 smoke는 단일 설정, formal은 미리 정한 후보 수·validation metric을 사용한다.[S15]

L0/L1의 비교 정의는 [통합 비교군 §6](comparison-matrix.md)에 따른다. 최초는 과제 설명 template 1개와 모델 1개를 사전 지정한다. 이후 prompt sweep은 별도 selection budget으로 기록한다. zero-shot이어도 validation threshold 선택은 label access다.

### 2.2 데이터·학습 seed의 구분

데이터 split과 CPT-dev membership은 모델·실험 seed와 무관하게 고정한다. 학습 seed만 42/43/44로 바꾼다. 동일 seed의 B2-S/B3-S/B4-S/B5-S는 encoder 시작점과 downstream head의 초기 상태를 맞춘다. v0.2 controlled는 신규 marker embedding 없이 native tokenizer로 시작한다. v0.1 special-token 방식은 별도 renderer version이다.

CPT seed 하나의 checkpoint를 세 번 fine-tuning했으면 `n_cpt_seeds=1, n_ft_seeds=3`이다. 이를 독립 CPT 3회라고 보고하지 않는다. formal 주 비교는 가능하면 각 seed별 CPT와 FT를 모두 수행하고 계산량을 기록한다.

## 3. 재현 트랙의 경계

BiCC-BERT 저자 script는 public train 파일을 CPT에 사용하고, validation/test 파일 경로도 인자로 전달한다.[S04] 인자가 있다는 이유만으로 테스트 누수를 단정하지 말고 실제 loader·training/evaluation 경로를 확인해야 한다.

| 감사 항목 | 확인·기록할 내용 |
|---|---|
| 코드 pin | branch 이름이 아니라 commit SHA; 변경한 파일 diff |
| 입력 renderer | 메시지 포함 방식, add/delete 순서, 추상화, 길이 제한 |
| tokenizer | 신규 token 목록·순서·embedding 초기화 |
| `only_adds` | 실제 영향 경로를 확인; 이름만으로 전체 입력을 추정하지 않음 |
| feature transform | 어떤 split에서 scaler를 fit하는지, label 연관 정보 여부 |
| CPT | task 교대 방식, loss 정규화, epoch·update·예산, validation 사용 |
| downstream | fusion head, BCE/CE, label 방향, threshold, early stopping |
| checkpoint | 실제 선택 기준, best checkpoint 복원 여부, test 호출 시점 |
| localization | line-level 정답이 commit classifier 입력에 섞이지 않는지 |

원저자 결과와 차이가 나면 `reference_deviations.md`에 환경·데이터·전처리·metric 차이를 적는다. 원저자 보고 수치, 우리의 reference 실행, controlled 실행을 세 범주로 구분한다. 수치 일치는 고정 tolerance 하나로 자동 인증하지 않고 seed 분산·실험 조건과 함께 검토한다.

## 4. 분할·누수 방지 프로토콜

### 4.1 적용 순서

1. 원본 파일의 ID·라벨·표현 형식·split을 확인하고 보존한다.
2. `change_id` 중복과 dataset 간 같은 commit을 검사한다. 같은 ID의 충돌 라벨은 격리한다.
3. 원본 representation의 엄격 content hash와 방향을 보존한 patch fingerprint를 만든다.
4. split 간 동일·유사 변경을 탐지한다. controlled에서는 오염된 train/CPT 항목을 제외하고 평가 membership은 가능한 한 유지한다.
5. test와 validation 자체의 중복/충돌은 숨기지 않는다. 새 평가 membership이 필요하면 별도 split version과 제외 보고서를 만든다.
6. split 고정 후 CPT subset과 negative pool을 만든다. 그다음 tokenization·chunking·sampling을 한다.

유사도 중복 기준은 label을 사용하지 않는 사전 규칙으로 정한다. 문장 의미를 바꿀 수 있는 공백·주석 제거를 exact duplicate 판정의 근거로 쓰지 않는다. exact/normalized/near-duplicate 결과를 별도로 저장한다.

### 4.2 프로젝트·시간 평가

`upstream_holdout`은 제공 split을 평가하며, timestamp가 없으면 시간순임을 검증했다고 표기하지 않는다. `project_holdout`은 train/validation/test 프로젝트가 겹치지 않도록 한다. 나중에 추가하는 `historical`은 모델 가용 시점, 데이터 cutoff, label observed_at, feature available_at까지 통제해야 한다.

초기 공개 실험은 upstream benchmark 재현이지 실제 과거 운영 시뮬레이션이 아니다. 후속 수정으로 붙인 라벨을 과거 학습 시점에 이미 알았던 것처럼 사용했다고 주장하지 않는다. 누적 이력 특징은 원칙적으로 `available_at <= prediction_time`을 만족해야 한다.

### 4.3 라벨과 provenance

공통 defect 라벨은 **1=결함 유발, 0=비결함, null=미라벨/미상**이다. RMI 라벨은 다른 열과 enum을 사용한다. source의 코딩을 명시적으로 변환하고 알려진 양성·음성 fixture로 검증한다.

다운스트림 입력의 `labels`와 CPT 입력의 `mlm_labels`/`rmi_target`을 이름으로도 분리한다. 미라벨 공개 변경을 0으로 채우는 코드는 금지한다. 다중 라벨 출처가 충돌하면 다수결로 조용히 정하지 말고 새 라벨 정책과 근거를 남긴다.

## 5. Tokenization과 입력 coverage

### 5.1 공통 EvidenceView

v0.2의 controlled 기본 renderer는 `message-add-del-text-v2`다. 기존 tokenizer로 인코딩되는 메시지/변경 구분 텍스트와 diff의 `+`/`-` prefix를 사용하고 **신규 special token을 추가하지 않는다**. 동결 B2가 새로 만든 무작위 marker embedding을 학습하지 못하는 조건을 피하기 위한 설계 변경이다. native tokenizer tokenization을 확인하고 `+`/`-`를 무조건 한 token으로 가정하지 않는다. 원저자 reference의 special-token 처리는 보존하고 차이를 기록한다.

원본 메시지·추가/삭제 행에서 encoder 512-token 상한에 맞는 구간을 결정하여 `EvidenceView`에 저장한다. source span/행 범위와 query content hash를 기록한다. B1/B2~B5/LLM matched 실험은 이 원문 구간을 그대로 사용한다. B0는 정형 열만 사용하고, S variant는 같은 승인 feature를 추가한다. LLM의 지시문·예시는 별도이며 query를 추가 절단하지 않는다. [통합 비교군 §4](comparison-matrix.md)를 따른다.

`encoder_init_v2` artifact에 모델 revision·native tokenizer를 고정한다. B2는 이 encoder를 고정, B3는 바로 FT, B4/B5는 여기서 CPT를 시작한다. downstream head는 같은 규칙으로 새로 초기화한다. CPT head·optimizer를 전달하지 않는다. 잘못된 tokenizer hash는 로드를 거부한다.

### 5.2 Coverage와 cache

커밋별 source의 이용 가능한 메시지/코드 토큰·행 수, 포함 범위, 추가/삭제별 보존 수, truncation, 원본에서 이미 소실된 정보 상태를 기록한다. encoder/LLM token 수는 각 tokenizer 기준으로 별도 기록하고 공통 query는 원문 hash로 비교한다. `preprocessed_lines` 패키지는 저장소 전체 대비 coverage라고 부르지 않는다.

빈 코드, 삭제만 있는 변경, 메시지만 있는 변경, 긴 메시지, 비ASCII 식별자를 fixture에 넣는다. eligibility는 label·model score와 무관하게 먼저 고정한다. 모델 실패가 있는 행을 조용히 제외하지 않는다. native 문맥/계층 처리로 확대하려면 별도 study와 evidence_profile을 만든다.

B1 vocabulary/IDF와 수치 전처리는 public train에서만 fit한다.[S13][S14] B2 embedding cache key는 encoder state hash, pooling, encoder eval mode, tokenizer/renderer/evidence hash, precision을 포함한다. B2 head 업데이트로 encoder cache를 무효화할 필요는 없지만 encoder state가 달라지면 재사용을 금지한다.

## 6. MLM과 RMI 구현 계약

### 6.1 MLM

MLM은 일부 토큰의 복원을 학습한다.[S09] controlled 기본 `mlm_probability=0.15`로 설정하고, 선택된 위치의 80%는 MASK, 10%는 허용 vocabulary의 무작위 토큰, 10%는 원래 토큰을 유지한다. 통계 테스트의 허용오차와 seed를 고정한다.

BOS/EOS/PAD 등 native special token은 MLM 대상과 random replacement 후보에서 제외한다. renderer가 삽입한 구조 prefix의 위치도 보호하되, 코드 안의 실제 `+`/`-` 연산자까지 token ID 전체로 제외하지 않는다. 구조 prefix와 코드 연산자는 offset/role mask로 구분한다. 한 샘플에 적격 토큰이 없으면 사유와 함께 제외한다. 적격 토큰은 있으나 선택된 위치가 0이면 결정적 재표본 또는 최소 1개 선택 규칙을 적용한다. loss는 예측 대상 토큰의 평균이며, 배치 전체에 target이 없어서 NaN이 되는 상황을 막는다.

validation MLM 마스크는 비교를 위해 고정 seed로 재현한다. downstream validation의 결함 정답을 MLM loss 계산에 사용하지 않는다. CPT-dev loss 감소는 진단 지표일 뿐 최종 결함 성능의 대체 지표가 아니다.

### 6.2 RMI

controlled 규약은 `rmi_target=1: 원래 대응 메시지`, `0: 다른 변경에서 교체한 메시지`다. 이 부호는 원저자 구현의 부호와 별도로 검증·기록한다. defect label과 절대로 혼용하지 않는다.

negative 후보는 같은 CPT split에 속하는 다른 change의 메시지이며, 자신·동일 메시지 hash·같은 duplicate group은 제외한다. 가능하면 같은 언어/프로젝트에서 선택하는 sampler를 확장 옵션으로 두되, 초기 기본 sampling 규칙과 pool hash를 고정한다. 동일 메시지가 아니어도 의미상 대응할 수 있다는 false-negative 한계를 기록한다.

빈 메시지나 유효 negative가 없는 샘플은 `rmi_ineligible`로 제외하고 비율을 보고한다. validation/test/internal pool로 fallback하지 않는다. 교체 확률 0.5는 초기 제안이며 실제 class 비율·seed를 기록한다. true/mismatched pair 수를 검사하는 fixture를 제공한다.

### 6.3 Task 교대와 계산 예산

B5는 공유 encoder와 MLM/RMI 전용 head를 사용하고, **한 optimizer update는 하나의 task**로 구성한다. gradient accumulation 안에서 task를 임의로 섞지 않는다. 마지막 불완전 accumulation의 loss 가중은 실제 sample/target 수에 맞춘다. MLM loss 평균과 RMI loss 평균을 단순 합계로 비교하지 않는다.

B4와 B5의 10M은 합산 non-padding input token exposures다. B5가 같은 원문을 두 task로 각각 forward했다면 두 번 계수한다. `mlm_input_tokens`, `rmi_input_tokens`, `mlm_target_tokens`, `forward_tokens`, `optimizer_updates`, `unique_changes_seen`, wall time을 별도로 기록한다.

동일 토큰 노출은 동일 FLOPs·GPU 시간·MLM 양을 의미하지 않는다. B5의 RMI 기여 해석에는 이 한계를 명시한다. P3에서 동일 MLM 노출을 유지하고 RMI를 더하는 추가 실험을 할 경우 **총 계산 예산이 증가한 실험**으로 따로 표기한다.

## 7. 학습량 실험

### 7.1 확장 조건

초기 10M CPT와 동일 downstream 학습의 validation 결과·비용을 먼저 확인한다. 50M/100M 확장은 public validation의 추세를 바탕으로 사전 선택한 프로토콜로 실행한다. test를 보고 학습량을 늘리지 않는다. 효과가 없을 때 확장을 중단하는 것은 허용된 결과다.

| 축 | 고정할 것 | 변경할 것 | 필요한 데이터 |
|---|---|---|---|
| 학습 노출량 | 같은 고유 CPT corpus, FT train, 모델 | 0/10M/50M/100M | 첫 공개 train 재사용 가능 |
| 데이터 다양성 | 총 token budget, FT train, sampling 규칙 | 외부 고유 diff 5만/10만/20만 | CodeChangeNet 등 공개 비라벨 |
| 라벨 효율 | validation/test, CPT 조건 | supervised train 25/50/100% | 같은 공개 라벨 데이터 |
| 일반 코드 대 diff | 가능한 한 corpus 언어·예산·모델 | 일반 코드 MLM / diff MLM | 별도 코드 corpus·출처 감사 |

각 subset은 중첩되게 추출하고 sampled ID 목록을 저장한다. supervised label subset은 프로젝트·양성 수를 보고하며, train 안에서만 구성한다. 비라벨 CPT subset 선택에는 결함 라벨을 사용하지 않는다.

고유 diff 수가 많아졌지만 긴 코드가 더 잘렸다면 동일한 정보량 증가가 아니다. 유효 token 수·coverage·언어/프로젝트 구성도 함께 비교한다. 단계별 큰 데이터 다운로드는 용량·출처 검토 후 명시적 명령으로 수행한다.

## 8. 평가 정의

### 8.1 공통 지표

`AP`는 `sklearn.metrics.average_precision_score(y_true, score)`의 정의를 사용하며, positive label을 1로 고정한다.[S08] ROC-AUC는 두 class가 있을 때만 계산한다. 양성이 없는 cohort의 AP/Recall은 null과 사유로 표시한다. 임의로 0을 채워 macro 평균을 왜곡하지 않는다.

F1/precision/recall은 선택된 checkpoint의 validation에서 고정한 threshold로 평가한다. threshold 선택은 모든 후보 고유 score에 따른 decision boundary에서 수행하고 동률은 높은 threshold를 택한다. reference가 0.5를 쓰면 `reference_f1_at_0_5`로 별도 표시한다.

### 8.2 Recall@budget

평가 cohort 크기를 N, budget을 q라 할 때 `K = ceil(q × N)`으로 정한다. N>0이면 K는 1~N 범위로 clamp한다. score 내림차순으로 K개를 선택한다. 동점은 사전에 정한 salt와 change_id의 hash 순으로 결정하며 라벨을 참조하지 않는다. 실제 동점 크기·유효 selection rate도 출력한다.

`Recall@q = 선택한 K개 중 양성 수 / cohort 전체 양성 수`.

기본 q는 0.05와 0.10이고, budget 단위는 **커밋 개수**다. LOC 기준 지표를 동일 이름으로 덮어쓰지 않는다. 내부 배포 평가에서는 deployment count라는 별도 budget unit을 명시한다. global cohort와 프로젝트별 macro 결과를 구분한다.

**고정 threshold와 top-q는 다르다.** top-q는 해당 평가 cohort의 점수 순위에 기반한 사후 ranking 지표이며 공개 validation의 score threshold를 그대로 적용하는 운영 정책이 아니다. 내부에서 q를 고정하여 ranking만 계산하는 것은 모델 재학습/확률 보정이 아니지만, 미래 전체 cohort를 알아야 하는 지표라 온라인 배포 통제를 입증하지는 않는다.

### 8.3 통계와 비용

초기 formal 결과는 세 seed의 개별 값·평균·표준편차와 주요 비교쌍의 차이를 출력한다. 같은 test set과 seed의 paired prediction을 사용한다. 프로젝트 단위 paired bootstrap은 P1 완료 범위로 구현하되, 양성 없는 resample 비율과 사용 가능한 replicate 수를 보고한다. time-block bootstrap은 충분한 timestamp를 확보한 P3에서 추가한다.

project bootstrap의 구간은 프로젝트를 다시 표집했을 때의 불확실성이며 3-seed 학습 분산을 대신하지 않는다. 둘을 구분해 보고한다. naive한 여러 chunk/seed 행 복제로 표본 수를 늘리지 않는다. 다수 실험의 최고점만 강조하지 않고 사전 등록한 B2-S→B3-S→B4-S→B5-S와 L0-S→L1-S 비교를 우선한다. CPT seed 분산과 demo-set 분산을 합쳐 평균내지 않는다. B↔L 비교는 같은 query의 paired prediction으로 평가하되 replicate 축과 supervised/in-context 라벨 노출 차이를 명시한다.

비용은 실제 GPU 모델·VRAM peak allocated/reserved·wall time·유효 tokens/sec·batch latency·단건 추론 p50/p95를 구분한다. warmup·CUDA synchronize·반복 횟수·preprocessing 포함 여부를 명시한다. 클라우드 금액은 사용자가 입력한 단가와 측정 시간으로 산정하며 환율·단가를 하드코딩하지 않는다.

## 9. Freeze·test·사내 평가

### 9.1 공개 평가 실행 순서

`public train → public validation 선택 → study freeze → 사전 지정된 public test 평가 → 보고서`.

freeze는 하나의 최종 모델만 의미하지 않는다. 비교 연구에서는 9개 primary variant의 사전 지정된 checkpoints·LLM snapshot·prompt·demo sets를 모두 묶어 고정한다. 확장 L2는 index·retriever도 포함한다. M2의 engineering E2E는 public validation 또는 synthetic test로 수행하고, 공식 real test는 전체 등록 cohort가 freeze된 뒤에만 연다. test 결과로 best seed·feature subset·CPT 예산을 고르지 않는다. 기술적 오류로 재평가한 경우 원인·패치·영향 범위를 남기고 protocol version을 구분한다.

### 9.2 모델 번들의 필수 내용

- inference 모델 가중치와 모델 구조 config, tokenizer 전체 파일, feature pipeline; B1은 TF-IDF vocabulary/IDF·column mapping.
- L 계열의 로컬 LLM revision/precision/chat template/scorer, frozen task prompt·public demo IDs/내용·출처. L2는 retriever/index snapshot·시간/중복 정책.
- label-access ledger, matrix version, EvidenceView 계약과 variant registry digest.
- renderer·feature schema·label 의미, eligible input 규칙, score semantics.
- public validation에서 선택한 threshold, calibration이 있으면 그 공개 provenance.
- 학습 데이터 및 split manifest digest, run IDs, 코드 SHA, dependency lock digest.
- 입력/출력 schema, checksum 목록, CPU/CUDA 지원 범위와 검증 기록.

학습 resume checkpoint와 export bundle을 구분한다. 내부 평가 번들에는 optimizer state·CPT head·불필요한 원문을 포함하지 않는다. L1/L2에 필요한 **승인된 공개** 예시 내용·라벨/index는 최소 범위에서 포함 가능하며 재배포/반입 조건을 확인한다. 내부 원문은 번들로 내보내지 않는다. Hugging Face 모델 로더의 local-only 기능을 사용하더라도 내부 환경의 egress 차단을 별도로 적용한다.[S11]

### 9.3 사내 평가 경계

내부에서는 고정 모델 inference, public frozen demo/index의 로컬 참조, 별도 evaluator의 내부 label join만 허용한다. predictor는 내부 정답 파일을 받지 않는다. internal artifact는 `evaluation_only`이며 학습·model/prompt selection·보정·threshold/ensemble tuning·demo/index 항목에 들어갈 수 없다. 내부 query는 검색어로만 사용하며 append/cache를 통해 학습용 corpus에 편입하지 않는다. 내부 모든 LLM 호출은 승인된 로컬 backend만 허용한다.

사내 라벨 정의, 배포–커밋 연결 규칙, 관찰창, 평가 기간, service stratification, 결측 처리, aggregation 규칙은 내부 결과를 보기 전에 명시한다. commit score를 배포 score로 합치는 첫 후보는 max이지만, 채택 시 변경 수가 많은 배포에 유리해지는 편향을 보고하고 내부 라벨로 다른 규칙을 고르지 않는다. 공개에서 결정했거나 사전에 등록한 규칙만 평가한다.

내부 코드 접근이 허용되지 않으면 raw diff를 외부로 전송하지 않는다. B0 등 사전 지정한 feature-only 모델만 가능한 경우 coverage와 불가 사유를 기록하고, CodeBERT를 평가했다고 표기하지 않는다. 내부 식별자 hash도 조직 정책상 민감할 수 있으므로 공개 보고서에는 승인된 집계 결과만 반출한다.

## 10. 필수 보고서 구조

1. 프로토콜·실험 트랙·상태와 연구 질문.
2. 데이터 provenance·split·label/feature 검증·제외 및 prediction coverage.
3. 재현 구현과 controlled 구현의 차이.
4. 통합 registry의 variant별 결과·주요 pair 차이·불확실성; training/demo/retrieval seed 축은 구분.
5. unique changes·실제 token exposures·task별 예산·GPU 시간·TF-IDF/index 준비 시간·LLM 입력/예시 토큰·latency·라벨 사용 ledger.
6. 입력 truncation·오류·양성 비율·프로젝트별 결과와 원인 분석.
7. 공개 결함→내부 장애 이전의 한계, 원래 base-model pretraining 오염 미확인 범위.
8. 다음 데이터 확장을 채택/보류한 근거. 결과가 불리해도 누락하지 않음.

## 11. LLM 실험 실행 절차

1. public-only source registry와 모델 license를 확인한 로컬 snapshot을 pin한다. weight 학습은 하지 않는다.
2. 동일 EvidenceView와 정형 serializer, task template, 두 label continuation/scorer를 고정한다. label-token 경계를 확인한다.
3. L0는 demo 없음. L1은 train만으로 만든 3개의 고정 4-shot set(각 2 positive/2 negative)을 전 query에 동일하게 적용한다. 실제 지표를 보며 예시를 골라 바꾸지 않는다.
4. L2 확장 시 동일 source pool·k·class mix·예시 길이 상한에서 class-conditional TF-IDF cosine 검색으로 예시 선택 방법만 바꾼다. 정적 benchmark의 index에서는 train만 쓰고, historical은 label availability까지 확인한다.
5. public validation에서 사전 규칙으로 threshold를 고정한다. prompt/model sweep을 했다면 후보 수와 접근한 validation 라벨 수를 기록한다.
6. L0/L1/L2별 가중치·prompt·score·demo/index를 freeze한다. test/internal에서는 변경하지 않는다.

모든 예측에서 query ID와 중복 patch group을 index에서 제외한다. matched-k를 채울 수 없으면 preflight에서 실패하거나 사전에 정한 unavailable 상태를 사용하며 test/internal 후보를 가져오지 않는다. 자연 class prior의 retrieval, 추가 k, 다른 LLM, different scoring mode는 새 variant/protocol이다.

## 12. LLM scorer와 실행 무결성

정식 수식·해석은 [통합 비교군 §6](comparison-matrix.md)에 둔다. 핵심 scorer는 두 label 후보의 raw conditional log-likelihood를 안정적인 logsumexp로 정규화한다. token log-probability를 얻기 위해 forward만 사용해도 되며 free generation/chain-of-thought를 필수로 하지 않는다. 모든 후보에 같은 task/query evidence를 사용한다. tokenizer의 prompt prefix가 candidate concatenation으로 달라지는 경우 경계를 재설계하거나 실행을 거부한다.

label candidate가 1개만 산출되는 top-logprobs 결과를 다른 label의 확률 0으로 해석하지 않는다. 양 후보 지원이 필요하다. quantization·chat template·scorer·precision 변경은 L0/L1/L2 모두 새 study로 묶는다. '0/1' 출력만 있으면 분류 진단을 할 수 있지만 primary AP/ranking 실험 완료로 대체하지 않는다.

기술적 retry 정책은 실행 전에 고정한다. timeout/OOM에 대해 입력을 더 줄이거나 다른 모델로 조용히 fallback하지 않는다. 실패 query는 prediction_status로 남기고 final full-coverage 비교를 차단한다. retry를 포함한 총 compute/cost를 보고한다. prompt injection fixture는 구획화·도구 미부여·네트워크 차단을 검사하며 모든 공격에 면역이라고 주장하지 않는다.

## 13. 비교 예산·재사용 규칙

통합 formal 연구의 첫 9개 primary variant를 등록한 후 개별 pipeline을 순차 구현한다. core를 먼저 validation에서 확인하고 LLM은 별도 branch로 병행 준비할 수 있다. 공식 public test를 이미 열어본 후 새로운 모델·예시·budget을 선택한다면 v0.2의 untouched 비교라고 주장할 수 없다. 사전 등록된 변경 없는 후속 실행인지, exploratory인지, 새로운 holdout이 필요한지 명시한다.

같은 CPT snapshot을 여러 FT seed에서 재사용한 횟수, 같은 L0를 반복 호출한 횟수, 같은 L1 demo set을 여러 query에 사용한 횟수는 서로 다른 의미다. 최고 seed나 prompt 하나만 선택해서 test 표에 싣지 않는다. LLM 하나의 L0/L1 결과로 encoder 계열 전체 대 LLM 계열 전체의 우열을 일반화하지 않는다.

---
출처의 정확한 범위는 [references.md](references.md)를 따른다. 이 프로토콜의 예산·seed·threshold·중단 규칙은 자체 설계이며 선행연구의 최적 설정으로 주장하지 않는다.
