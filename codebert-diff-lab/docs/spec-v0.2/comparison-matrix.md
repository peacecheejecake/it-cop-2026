# 통합 비교군과 연구 질문

> v0.2 · 2026-10-01 · 구현 전 설계안  
> 사용자가 제시한 B0~B3/L0~L2 분류에 기존 B4/B5 CPT 실험을 통합했다. 아래 값은 실험 결과가 아니라 구현할 계약이다. 코드 패키지 이름은 `codebert-diff-lab`을 유지한다.

## 1. 통합 표

여기서 **과제 라벨**은 공개 데이터의 결함 유발 여부다. MLM의 복원 정답과 RMI의 대응/비대응 표적은 별개의 자기지도 학습 신호다. 아래 표의 라벨 사용은 가중치 학습/추론 예시를 뜻하며, 공개 validation에 의한 설정 선택은 별도 기록한다.

| ID | 접근 | 과제 라벨을 사용하는 위치 | 과제 적응 시 업데이트 | 기본 우선순위 |
|---|---|---|---|---|
| **B0** | 정형 특징 + Logistic Regression / LightGBM | 분류기 학습 | LR 파라미터 / 트리 | P0 |
| **B1** | TF-IDF 텍스트 특징 + 정형 특징 + 분류기 | 분류기 학습; vocabulary/IDF는 train 텍스트로 fit | 선형 분류기 | P0 |
| **B2** | 고정 encoder 임베딩 + 분류기; 정형 결합 variant 추가 | 분류기 또는 결합 head만 학습 | encoder는 고정 | P1 |
| **B3** | BERT 규모 encoder fine-tuning + 정형 특징 | encoder와 분류기 학습 | encoder + head | P1 |
| **B4** | diff MLM 추가 사전학습 → B3와 같은 결함 fine-tuning | CPT에는 결함 라벨 없음; 이후 encoder + head 학습 | CPT와 FT 모두 encoder 갱신 | P1 |
| **B5** | diff MLM+RMI 추가 사전학습 → B3와 같은 결함 fine-tuning | CPT에는 결함 라벨 없음; 이후 encoder + head 학습 | CPT와 FT 모두 encoder 갱신 | P1 |
| **L0** | 고정된 범용/공개 가중치 LLM zero-shot | 입력에 라벨 예시 없음 | 가중치 업데이트 없음 | P2 |
| **L1** | 같은 LLM의 few-shot prompting | 공개 train의 라벨 예시를 prompt에 포함 | 가중치 업데이트 없음 | P2 |
| **L2** | 공개 과거 유사 사례 검색 + 같은 LLM | 공개 train의 라벨 사례 저장소; 검색된 예시를 prompt에 포함 | 기본 retriever·LLM 고정 | P3, 확장 |

B4/B5는 별개의 큰 생성 모델이 아니라 **B3의 사전학습 조건을 바꾼 실험군**이다. 모든 L 계열은 fine-tuning하지 않는다. LoRA/QLoRA·LLM SFT·앙상블은 이번 통합의 범위 밖이다.

## 2. 실행 ID와 번호 이관

B0~B5/L0~L2는 설명용 family이고, 실제 run에는 아래의 명시적 variant와 `matrix_version=2`가 필요하다. `-T`는 변경 텍스트(메시지+diff)만, `-S`는 같은 변경 텍스트에 정형 특징을 결합한다. B0의 입력은 정형 특징만이다. T/S는 토큰 길이나 모델 크기를 뜻하지 않는다.

| 실행 ID | 기본 구현 | 지위 |
|---|---|---|
| `B0-LR` | train-only 수치 전처리 + Logistic Regression | 필수 CPU 기준선 |
| `B0-LGBM` | 같은 feature allowlist + LightGBM | 필수 CPU 비선형 기준선 |
| `B1-TFIDF-S` | 메시지/diff TF-IDF + 정형 특징 + LR | 필수 CPU 텍스트 기준선 |
| `B2-S` | 동결 CodeBERT + B3-S와 같은 fusion head | 필수, encoder 적응 효과 비교 |
| `B3-S` | CodeBERT 전체 FT + fusion head | 필수, CPT 기준점 |
| `B4-S` | diff MLM CPT + B3-S와 같은 FT | 필수 CPT 실험 |
| `B5-S` | diff MLM+RMI CPT + B3-S와 같은 FT | 필수 CPT 실험 |
| `L0-S` | 고정 LLM, 정형 정보를 직렬화, 예시 0개 | 필수 LLM 기준선 |
| `L1-S` | L0-S와 같은 LLM, 공개 train 고정 4-shot 예시 | 필수 LLM 비교 |
| `L2-S` | 같은 LLM·4-shot, query별 공개 train 유사 사례 검색 | 확장 |
| `B2-T`, `B3-T` | 정형 특징을 뺀 동결/FT encoder 한 쌍 | 정보 기여 분석용 확장 |
| `L0-T`, `L1-T`, `L2-T` | 정형 특징을 뺀 동일 LLM 계열 | 정보 기여 분석용 확장 |

`B2-T`가 사용자가 제시한 원래의 고정 encoder+분류기 구성을 그대로 보존한다. 주 비교에는 `B2-S`를 추가하여 B3-S와 정보량·head를 맞춘다. B2-S를 B2-T의 결과처럼 쓰지 않는다. B2-T↔B3-T도 같은 코드 전용 head를 사용한다.

### v0.1 → v0.2 매핑

| v0.1 의미 | v0.2 실행 ID | 처리 |
|---|---|---|
| B0: 정형 LR | B0-LR | 의미 유지; LightGBM은 신규 variant |
| B1: 동결 encoder, 코드 전용 | B2-T | **번호 변경** |
| B2: FT encoder, 코드 전용 | B3-T | **번호 변경** |
| B3: FT encoder+정형 | B3-S | 의미 유지, variant 명시 |
| B4: MLM CPT+정형 | B4-S | 의미 유지 |
| B5: MLM+RMI CPT+정형 | B5-S | 의미 유지 |
| 해당 없음 | B1-TFIDF-S, B2-S, L0/L1/L2 | 신규 정의 |

기존 artifact의 model_id를 덮어쓰지 않는다. `legacy_model_id`, `legacy_matrix_version`, `mapped_family_id`를 갖는 명시적 migration만 허용한다. v0.1 B1을 v0.2 TF-IDF로 자동 해석하는 것은 오류다. v0.2의 입력 renderer 변경도 있으므로 번호 매핑만으로 같은 조건의 재실행이 되지는 않는다.

## 3. 주 비교쌍

| 비교 | 질문 | 맞출 조건 / 해석 한계 |
|---|---|---|
| B0-LR ↔ B0-LGBM | 정형 특징의 비선형 모델링이 도움이 되는가? | 같은 열·대상 ID·train; 모델별 맞는 전처리와 탐색 예산은 기록 |
| B0-LR ↔ B1-TFIDF-S | 가벼운 텍스트 특징의 추가 가치가 있는가? | 정형 열·LR 평가 조건; 표현 차이가 주 요인 |
| B1-TFIDF-S ↔ B2-S | 사전학습 encoder 표현이 유용한가? | 같은 query evidence/정형 열; head가 달라 엄밀한 단일요인 인과 비교는 아님 |
| **B2-S ↔ B3-S** | 결함 라벨로 encoder를 조정하는 효과는? | 같은 초기 encoder·pooling·head·입력·downstream 예산; encoder 갱신만 다름 |
| **B3-S ↔ B4-S** | diff MLM CPT의 추가 가치는? | 같은 FT·feature·입력; 추가 CPT 계산량은 별도 보고 |
| **B4-S ↔ B5-S** | 변경-메시지 관계 학습의 추가 가치는? | 같은 합산 CPT 토큰 예산; task별 MLM 노출 차이 명시 |
| **L0-S ↔ L1-S** | 공개 라벨 예시가 prompt에서 도움이 되는가? | 같은 LLM/precision/template/query; 예시 때문에 문맥·계산량은 증가 |
| **L1-S ↔ L2-S** | query별 유사 예시 선택이 도움이 되는가? | 같은 k·class mix·순서 규칙·예시별 budget·LLM; 선택 방법만 변경 |
| B3/B4/B5 ↔ L0/L1 | 작은 supervised 모델과 고정 LLM 중 목적에 맞는가? | 같은 평가 ID·query 정보·지표; 학습 이력·라벨 노출·모델 크기가 달라 단일요인 인과 비교는 아님 |

LightGBM의 전체 성능 우위를 미리 가정하지 않는다. LLM이 큰 문맥을 지원한다는 이유로 본 비교에만 더 많은 파일을 자동 제공하지 않는다.

## 4. 정보량을 맞추는 방법

**`matched`가 기본**이다. 먼저 공통 `EvidenceView`를 만들고 그 메시지·추가/삭제 코드 구간을 각 모델이 읽게 한다. CodeBERT의 총 512-token 제한에 맞춘 구간 선택을 한번 고정하고, TF-IDF와 LLM도 동일한 원문 구간을 사용한다. tokenizer가 다르므로 모든 모델에 512 tokens를 강제하는 것이 아니라 **원문 구간의 hash**를 맞춘다. LLM task 지시문과 라벨 예시는 query evidence와 별도 예산이다.

B0는 같은 대상의 정형 feature만 읽는다. S variant의 LLM은 승인된 동일 feature 이름·값·단위·결측 표기를 읽되, 숫자 표현은 deterministic하게 직렬화한다. 모델마다 전처리 알고리즘은 달라도 사용 가능한 원 정보는 같게 한다. 제공된 정형 feature가 전체 commit에서 계산되었다면 모든 S variant에 같은 feature를 제공하고 그 provenance를 명시한다.

**`native`는 확장**이다. TF-IDF의 전체 diff, LLM의 더 긴 문맥, encoder의 계층 처리 등 모델별 실제 장점을 살리는 별도 study다. matched와 native 결과는 섞지 않는다. 이미 원본 패키지에서 삭제된 정보를 다시 복원했다고 주장하지 않는다.

## 5. 라벨 사용량과 데이터 경계

각 run은 `gradient_label_count`, `demo_unique_label_count`, `index_labeled_count`, `selection_label_count`, `adaptation_mode`를 기록한다. RMI synthetic target 수는 별도 필드다. 예시 4개가 같은 train 예시 4개를 모든 query에 반복한 것인지, query마다 다른 4개인지도 분리한다. 사전학습 모델이 과거에 어떤 benchmark를 보았는지는 완전히 확인하지 못했다는 한계를 유지한다.

- **L0**: query와 과제 정의는 제공하지만 라벨 demonstration은 0개다. 공개 validation으로 모델/프롬프트/threshold를 선택했다면 selection-label access가 있었다고 기록한다. 이를 '어떤 과제 라벨도 사용하지 않은 실험'이라고 부르지 않는다.
- **L1**: 고정 4-shot(양성 2/음성 2)을 public train에서 seed로 선택하는 것을 초기 제안으로 둔다. 전 query에 같은 예시 집합을 사용한다. 예시 순서와 class prior 왜곡을 기록한다. k=4는 최적값이 아니라 출발 설정이다.
- **L2**: 같은 public train의 고정 index에서 검색한다. 초기 비교는 L1과 같은 2/2 class mix, k=4, 예시별 상한을 유지한 class-conditional TF-IDF cosine 검색으로 제안한다. 자연 top-k class mix는 별도 sensitivity 실험이다. 후보의 결함 라벨은 알려진 train 라벨이며 query의 정답은 검색에 들어가지 않는다.

validation/test/internal의 라벨·정답 사례를 demo 또는 index에 넣지 않는다. query 자신·동일 patch/duplicate group은 제외한다. historical 트랙에서는 index 항목의 available_at와 label observed_at이 query 시점 이전임을 확인한다. 시간이 없으면 정적 benchmark retrieval로만 보고한다.

사내 evaluation에서 **공개 train의 동결 demo/index를 로컬로 참조하는 것은 허용**한다. 사내 query를 로컬 검색어로 쓰는 것도 추론이다. 사내 사례·라벨·query 임베딩을 index에 추가하거나 재학습·선택에 재사용하는 것은 금지한다. 내부 내용의 외부 전송, 외부 LLM/API, 외부 logging은 금지한다.

## 6. LLM 평가 계약

### 6.1 모델과 출력

L0/L1/L2는 동일한 checkpoint revision, tokenizer/chat template, precision/quantization, scoring backend를 사용한다. 첫 비교는 로컬 공개 가중치 instruct 모델 1개로 제한하고 정확한 ID·revision·license·메모리는 M0/M6L에서 확인 후 pin한다. 최신 모델명이나 24GB 실행 가능성을 미리 확정하지 않는다. 외부 API는 v0.2 구현 범위에 넣지 않는다.

주 지표가 AP·Recall@budget이므로 primary scorer는 단순 '0/1' 생성 대신 **두 라벨 후보의 조건부 log-likelihood**를 계산한다. LLM 가중치는 업데이트하지 않는다. 후보 0,1의 점수를 각각 ell_0, ell_1이라 할 때:

`score = exp(ell_1 - logsumexp(ell_0, ell_1))`

실제 tokenizer에서 한 라벨이 여러 토큰이면 각 조건부 log-probability의 합을 사용한다. prompt/후보 경계, 공백, 라벨 순서, 종료 기호 정책을 고정하고 fixture로 검증한다. raw logits를 사용하며 temperature/top-k 처리된 generation score와 혼동하지 않는다.[S17] 이 값은 **두 언어적 답 후보 사이의 상대 선호 점수**이지 보정된 운영 장애 확률이 아니다.

자유 서술의 '위험도 80%'는 primary score로 사용하지 않는다. 후보 확률이 제공되지 않는 backend는 primary ranking 실험에서 unsupported로 표시한다. 후속 self-reported-score 실험은 별도 scoring profile이어야 한다. 임의로 0.5를 채워 실패를 숨기지 않는다.

### 6.2 예시와 보안

prompt에는 과제 정의, query evidence, 필요 시 train 예시만 포함한다. 저장소 README/주석/커밋 메시지는 지시가 아니라 비신뢰 데이터로 구획화한다. LLM에게 shell·네트워크·파일 쓰기 도구를 부여하지 않는다. prompt template과 렌더링 결과 hash를 저장하고 원문은 승인된 로컬 artifact에만 둔다.

k를 유지하기 위해 query 본문을 조용히 더 자르지 않는다. target-query budget을 먼저 예약하고, 예시·정형·task 지시문이 정한 총문맥에 맞지 않으면 계획 단계에서 실패시킨다. eligible train 예시가 부족하면 evaluation 데이터로 보충하지 않는다. 모델별 source/target/demo 토큰 수와 coverage를 따로 기록한다.

### 6.3 변동성과 비용

B 계열의 training seed, L1의 demonstration-set seed/order, L2의 검색 tie-break seed, LLM backend 반복 호출을 다른 축으로 기록한다. deterministic L0를 같은 입력으로 3번 실행했다고 독립 학습 3회로 세지 않는다. 최종 L1은 3개의 사전 지정 demo set을 보고하며, 최고 set만 고르지 않는다.

추가 학습 GPU 시간과 예측 시 token/latency/VRAM, index build/search 시간, 전체 eligible query 비용을 따로 집계한다. **gradient step 없음은 비용 없음이 아니다.** 동일 길이를 강요하는 대신 실제 사용 정보를 통제하고 비용을 함께 보고한다.

## 7. 실행 범위와 완료 상태

1. CPU 기준선: B0-LR, B0-LGBM, B1-TFIDF-S.
2. encoder 및 CPT: B2-S, B3-S, B4-S, B5-S.
3. LLM 핵심 비교: L0-S, L1-S.
4. 별도 확장: B2-T/B3-T, L2, text-only LLM, native 문맥, 공개 데이터 확대.

1~2는 `DoD-Core`, 1~3은 `DoD-Comparative`다. L2를 완료하지 않아도 1~3 완료라고 표시할 수 있지만, LLM을 실행하지 않은 상태를 통합 비교 완료라고 부르지 않는다. 모든 단계는 test를 보기 전에 study 후보·범위를 등록하고, 공식 비교는 포함할 전체 후보가 freeze된 후 수행한다. 향후 신규 LLM 모델을 기존 test 결과를 참고해 고르면 exploratory 후속 연구로 표시하거나 새 untouched holdout을 확보한다.

관련 문서: [요구사항](requirements.md) · [실험 프로토콜](experiment-protocol.md) · [구현 계획](implementation-plan.md) · [변경 이력](CHANGELOG.md)
