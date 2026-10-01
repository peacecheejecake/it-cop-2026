# 근거 자료와 확인 범위

> v0.2 수정일: 2026-10-01 · S01~S12는 v0.1(2026-09-30)의 확인 기록을 보존; S13~S17은 이번 통합에서 확인  
> 아래 자료는 요구사항의 배경과 참조 구현 감사에 사용했다. 문서의 CLI, 모듈 구조, 예산, 기본값, test, milestone는 본 프로젝트를 위한 설계 제안이다. 자료를 열람한 것과 데이터를 내려받아 실행한 것은 구분한다.

## S01 — BiCC-BERT / JIT-BiCC 논문

**Just-In-Time Software Defect Prediction via Bi-modal Change Representation Learning**, arXiv:2410.12107v1.

URL: <https://arxiv.org/html/2410.12107v1>

확인 범위: 추가 사전학습 목표 MLM/RMI, 코드·메시지 표현과 정형 특징 결합, 14개 feature, 데이터 설명. 초록은 27,391건, 본문 §4.2는 27,319건으로 기술한다. 어느 숫자가 실제 패키지에 해당하는지는 다운로드 후 행 수·유니크 ID·필터를 조사해야 한다. 본 문서는 논문 결과를 재실행하지 않았다.

## S02 — JIT-Fine 공개 재현 저장소

URL: <https://github.com/jacknichao/JIT-Fine>

확인 범위: README의 데이터 설명, data.zip 제공, 재현 절차와 설정 예시. `data.zip`의 내부 구조·정확한 hash·실제 split 건수·사용 조건까지 검증한 것은 아니다. 이름이 비슷한 일반 Defects4J 패키지와 구분한다.

## S03 — JIT-Fine fusion head 구현

URL: <https://github.com/jacknichao/JIT-Fine/blob/master/JITFine/concat/model.py>

열람한 file blob SHA: `a1fdf656e13021c92d09893b347ac891db1b00d4`.

확인 범위: CLS 표현, 정형 특징 linear projection와 tanh, concatenation·dropout·단일 logit, sigmoid와 BCELoss 사용. controlled 구현에서 BCEWithLogitsLoss를 쓰는 것은 명시적 변경이다. 이 SHA는 파일 blob의 SHA이며 repository commit pin을 대신하지 않는다.

## S04 — JIT-BiCC multitask 실행 script

URL: <https://github.com/jyz-1201/JIT-BiCC/blob/main/codes/ShellScripts/train_multitask.sh>

열람한 file blob SHA: `fe4b57ccc1d769e628ddeb06493c478e916daf1f`.

확인 범위: public train/validation/test 파일 인자, CodeBERT base 출발, CPT와 downstream 단계의 구분, 512-token 설정과 downstream `only_adds` 옵션. 인자 존재만으로 test 누수나 `only_adds`의 세부 효과를 단정하지 않으며, 실제 call path는 구현 전 감사 대상이다.

## S05 — JIT-Fine training/evaluation loop 일부

URL: <https://github.com/jacknichao/JIT-Fine/blob/master/JITFine/concat/run.py>

열람한 file blob SHA: `1e8666a6d6b611694bd1d9a528a2b4eb23f20fad`.

확인 범위: 일부 training/validation/checkpoint 코드, validation의 고정 0.5 threshold와 best-F1 선택. 파일 일부만 열람했으므로 전체 데이터 전처리·테스트 경로의 안전성을 검증한 것으로 볼 수 없다. repository commit pin은 M0에서 별도로 확정한다.

## S06 — uv lock과 sync

URL: <https://docs.astral.sh/uv/concepts/projects/sync/>

확인 범위: dependency locking과 환경 sync의 구분. 본 계획은 uv.lock과 실제 실행 환경의 hash를 기록하도록 요구한다. 프로젝트 dependency 버전을 이 문서 작성 과정에서 실제로 resolve하거나 lock하지 않았다.

## S07 — uv와 PyTorch 통합

URL: <https://docs.astral.sh/uv/guides/integration/pytorch/>

확인 범위: CPU/CUDA 환경에 맞는 PyTorch 설치 경로와 dependency 설정. 특정 CUDA·driver·PyTorch 조합의 호환성 또는 사용자 GPU에서의 성공을 보장하지 않는다.

## S08 — Average Precision 정의

URL: <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html>

확인 범위: AP 계산과 interpolation을 사용하는 PR curve 면적의 구분. Top-K 동점·rounding·cohort·null 정책은 이 프로젝트가 추가로 정한 계약이다.

## S09 — Masked language modeling

URL: <https://huggingface.co/docs/transformers/tasks/masked_language_modeling>

확인 범위: MLM workflow와 collator 사용 방식. 10M token 예산, task 비율, 특수 token 제외·0-target 처리 정책은 자체 설계이며 논문의 최적값으로 제시하지 않는다.

## S10 — Python pickle 보안 경고

URL: <https://docs.python.org/3/library/pickle.html>

확인 범위: unpickling이 안전하지 않으며 신뢰하지 않는 자료가 임의 코드를 실행할 수 있다는 경고. 격리 importer·권한·network 통제는 본 프로젝트의 요구사항이다. hash 일치만으로 악성 pickle 위험이 해소되지는 않는다.

## S11 — Transformers 모델 loading 옵션

URL: <https://huggingface.co/docs/transformers/en/model_doc/auto>

확인 범위: local_files_only와 revision 지정. 라이브러리 옵션은 전체 환경의 데이터 반출 방지 장치가 아니므로 내부 egress 차단은 별도로 요구한다.

## S12 — CodeBERT base 공개 config

URL: <https://huggingface.co/microsoft/codebert-base/raw/main/config.json>

확인 범위: RoBERTa 계열 model configuration. 실제 weights·tokenizer 다운로드, 모델 로딩·학습·GPU 검증은 하지 않았다. 예제의 모델명은 존재하는 public model ID이며 최종 snapshot revision은 구현 단계에서 pin한다.

## S13 — scikit-learn: 전처리와 데이터 누수

URL: <https://scikit-learn.org/stable/common_pitfalls.html>

확인 범위: train/test 분할 후 전처리 fit, test를 모델 선택/fit에 쓰지 않는 원칙, pipeline과 변환 상태 재사용. v0.2의 TF-IDF·scaler·imputer train-only 요구사항의 근거다. 실행 환경의 특정 패키지 버전을 이 문서에서 설치·검증한 것은 아니다.

## S14 — TfidfVectorizer

URL: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>

확인 범위: analyzer, ngram_range, lowercase, min_df/max_features, vocabulary/IDF 상태. 문서의 char 3~5 gram·5만 feature·field별 결합은 자체 초기 제안이다. CodeBERT보다 좋거나 나쁘다는 실험 결과를 확보한 것이 아니다.

## S15 — LightGBM parameters

URL: <https://lightgbm.readthedocs.io/en/stable/Parameters.html>

확인 범위: objective, tree/iteration, seed/thread·determinism·학습 parameter를 명시적으로 고정할 필요. 문서의 num_leaves·학습률·반복 수는 최적값이 아니라 출발 설정이다. 운영체제·라이브러리 버전 간 bitwise 재현을 보장하지 않는다.

## S16 — Language Models are Few-Shot Learners

URL: <https://arxiv.org/abs/2005.14165>

확인 범위: 과제별 gradient 업데이트 없이 텍스트 지시·few-shot demonstration으로 과제를 수행하는 설정. L0/L1의 라벨 사용 방식 구분의 근거로 사용한다. 이 논문이 JIT 결함 예측의 성능을 입증한다고 인용하지 않는다. 본 수정에서는 abstract만 확인했고 PDF 분석은 하지 않았다.

## S17 — Transformers generation/scoring documentation

URL: <https://huggingface.co/docs/transformers/main_classes/text_generation>

확인 범위: logits, output_scores, compute_transition_scores 등 출력 확률/점수 관련 인터페이스와 generation 설정. 본 설계의 primary scorer는 로컬 forward로 각 후보 조건부 likelihood를 계산하도록 제안하며, generation API가 모든 필요한 후보 값을 항상 반환한다고 가정하지 않는다. 정규화한 두 후보 상대 선호를 운영 장애 확률이라고 해석하지 않는 것은 본 연구의 score 계약이다.

## 검증 상태 요약

| 항목 | 상태 |
|---|---|
| 제공된 논문 HTML와 핵심 저자 저장소 설명 확인 | 완료 |
| 위에 명시한 일부 소스 파일 열람 | 완료 |
| 요구사항·프로토콜·구현 계획 작성 및 v0.2 통합 | 완료 |
| 통합 비교군·migration·계획용 registry·문서 링크/ID 정적 검사 | 문서 QA 결과 참조 |
| 자료별 전체 license 검토·재배포 승인 | 미완료, M0 작업 |
| 공개 archive 다운로드·역직렬화·통계 산출 | 미실행 |
| 실제 repository 생성·커밋·PR | 미실행 |
| pyproject/uv.lock 생성·환경 설치 | 미실행 |
| B0/TF-IDF/CodeBERT/LLM 학습·추론·GPU benchmark | 미실행 |
| 성능 수치 재현·baseline 우열 확인 | 미실행 |
| 사내 데이터 접근·추론·평가 | 미실행 |

문서 bundle에는 논문 본문, upstream 코드, 데이터, 모델 weight를 복제해 포함하지 않는다. 코드·자료의 실제 취득과 실행은 다음 구현 단계의 별도 작업이다.
