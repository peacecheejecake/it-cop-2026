# RUNLOG — 005-cross-project-jd4j

브랜치: `exp/005-cross-project-jd4j` (from `exp/003-seeds-cuda` @ `2db9f3b`, main 병합)

## 목적

ApacheJIT(Hadoop 계열 제외)으로 학습한 002/003의 모델을 **처음 보는 프로젝트**(JIT-Defects4J, 21개 Java 프로젝트, ApacheJIT과 프로젝트 겹침 0)에 **재학습 없이** 적용한다. "공개 데이터로 학습한 모델을 다른 코드베이스에 쓸 수 있는가"를 내부 적용 전에 공개 데이터로 먼저 측정한다.

## 사전 등록 (평가 전)

- 모델: 003과 같다. rule, 정형 LR(003 `runs/models/tabular`), A와 B(003 seed 42/43/44, Pod에 보관). **재학습과 재선택은 하지 않는다.**
- 대상: JIT-Defects4J를 002/003과 **같은 날짜 경계**로 나눈다.
  - `jd4j/train`(2016년 이전): ApacheJIT 학습과 같은 시기이므로 **프로젝트가 바뀐 효과만** 본다.
  - `jd4j/test`(2017년 이후): **프로젝트와 시기가 모두 바뀐 효과**를 본다.
  - 이름의 train/valid/test는 split 역할일 뿐이다. 이 실험에서는 **셋 다 평가 전용**이며 어떤 학습에도 쓰지 않는다.
- 지표: AP, ROC-AUC, Recall/Precision@10%, **AP ÷ 기저율(향상 배수)**. 기저율이 달라서 AP 절대값은 데이터셋끼리 비교하지 않는다.
- 비교 기준: 003의 ApacheJIT test("같은 프로젝트, 미래 시기").

## 데이터 (DATASETS.md §2)

- 변환: `tools/convert_jit_defects4j.py`(pickle 허용 목록 방식) → 27,319 커밋, buggy 2,332건.
- build: `build-apache --commit-col commit_hash --repo-col repo --label-col is_buggy_commit`(오프라인, 22분) → 채택 27,317개, 제외 2개(2 MB 초과 patch). → `.cache/canonical/jit-defects4j/`
- ApacheJIT과 겹침: 커밋 SHA **0개**, 정규화한 diff 내용이 같은 것 127개(0.46%, 양성 1개).
- split(`--train-before 2016-01-01 --valid-before 2017-01-01 --allow-retrospective`): 제외 154개(duplicate 150, 경계를 넘는 그룹 4).

| 파티션 | n | 양성 | 기저율 |
|---|---|---|---|
| jd4j/train (< 2016) | 23,803 | 2,199 | 0.092 |
| jd4j/valid (2016) | 1,257 | 58 | 0.046 |
| jd4j/test (≥ 2017) | 2,103 | 56 | 0.027 |

- prepare: 512 토큰 초과로 잘린 비율은 95.6% / 89.1% / 86.3%다. **view 정책(policy_hash, tokenizer, 절단 방식)이 002/003과 모두 같다.**
- 실수 기록: 처음 만든 `models` 심링크 경로가 틀려(`../../../`) prepare가 HF로 접속을 시도하다 실패했다. `../../.cache/models`로 고쳤다.

## 결과 1 — rule과 정형 LR (로컬, predict_suite, bootstrap 500)

| 대상 | 기저율 | rule AP | 정형 LR AP | 정형 LR AP ÷ 기저율 | 정형 LR ROC-AUC | 정형 LR R@10% |
|---|---|---|---|---|---|---|
| (003) ApacheJIT test | 0.229 | 0.499 | 0.576 | 2.5배 | 0.814 | 0.301 |
| jd4j < 2016 | 0.092 | 0.221 | 0.257 | 2.8배 | 0.749 | 0.327 |
| jd4j 2016 | 0.046 | 0.157 | 0.154 | 3.3배 | 0.715 | 0.345 |
| jd4j ≥ 2017 | 0.027 | 0.070 | 0.093 | 3.4배 | 0.706 | 0.339 |

- 크기 기반 feature는 처음 보는 프로젝트에서도 기저율 대비 2.8~3.4배의 향상을 유지한다. ROC-AUC는 0.81에서 0.71~0.75로 떨어진다.
- 2017년 이후는 양성이 56개뿐이라 불확실성이 크다(구간은 `runs/evaluation/local-jd4j-*/metrics.json`).

## 결과 2 — A/B (예정)
003 seed별 A/B 모델은 Runpod Pod `n2m8dtxgz6s598`(정지 상태)에 있다. 추론에는 Pod 재시작이 필요하다(약 20~30분, 약 $1). 사용자 결정을 기다리는 중이다.
