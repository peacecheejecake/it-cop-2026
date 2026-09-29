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

## 사전 등록 v2 — 새 Pod에서 다시 학습 (사용자 지시 "이전 Pod는 잊고 새 Pod에서 005 진행", 2026-09-29)

- 003의 가중치를 쓸 수 없으므로 **새 Pod에서 003 프로토콜 그대로 다시 학습**한다: fp32, batch 32×1, 상한 100 epoch, patience 2, seed 42/43/44, 002의 public view(SHA-256 확인).
- 학습 직후 seed마다 **ApacheJIT public test**(003 재현 확인과 같은 행 기준선)와 **JIT-Defects4J 세 파티션**을 모두 추론한다(`scripts/run_remote.sh`).
- 이번 Pod의 결과는 003 결과를 **대체하지 않는다**. ApacheJIT test 지표가 003과 seed 분산 범위 안에서 일치하는지를 재현성 점검으로 보고한다. 005의 주 결과는 같은 Pod·같은 모델 안에서의 "같은 프로젝트의 미래 대 처음 보는 프로젝트" 비교다.
- **저장**: 모델 가중치와 결과를 **network volume**(`/workspace`)에 둔다. Pod를 종료해도 같은 데이터센터의 다른 호스트에서 다시 붙일 수 있다(AGENTS.md 교훈).
- 로컬 002 모델(seed 42, MPS) 추론은 비용 없는 참고 대조로 병행한다.

### Runpod 설정 (v2)

| 항목 | 값 |
|---|---|
| network volume | `c8wdh0j8ek` (`jit005-vol`), 20 GB STANDARD, CA-MTL-3 → `/workspace` |
| Pod | `6y7c2qsnb5sc8b` (`jit005-cross-project`), Secure, CA-MTL-3, **$2.09/시간**, RTX PRO 6000 Blackwell SE 96 GB, 드라이버 595.91.07 |
| 이미지 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8.0+cu128, TF32 끔). 003과 같다 |
| 번들 | `.cache/bundles/jit005-61bcd6f.tar.gz` (603 MB, sha256 `3618dc5a…a6a`). 업로드 2분 24초, 6개 view 모두 manifest SHA-256 일치 |
| 데이터센터 선택 | RTX PRO 6000 재고와 network volume 지원이 모두 있는 곳 중, 같은 데이터센터에 A100도 있어 GPU를 바꿀 수 있는 CA-MTL-3 |

실행: 09:20 UTC 시작. pytest 55개 통과 → 정형 LR(09:22) → seed 42 A(09:22~).

## 결과 2 — A/B

### Pod 재시작 실패 (사용자가 005 진행을 승인한 뒤)
`pod-action start` → `400 There are not enough free GPUs on the host machine to start this pod.` 정지한 Pod는 원래 호스트에 묶여 있고, 그 호스트의 GPU가 다른 사용자에게 할당되었다. REST API로는 GPU 없이 시작할 수 없다(`update-pod`로 GPU 수를 바꿀 수 없음). **003의 seed별 A/B 가중치는 그 호스트의 persistent 디스크에 있어 지금은 접근할 수 없다.** 003 결과 회수 때 가중치를 받지 않았고, 정지를 택하면서 이 위험을 사용자에게 알리지 않았다. 교훈은 AGENTS.md에 반영했다.

### 대안: 002 모델(seed 42, MPS)로 로컬 추론
- 근거: 002 B는 003 seed 42 B와 같은 24,178행 test에서 AP 0.534 대 0.537, 점수 Spearman 0.979다(002 RESULTS). 사실상 같은 모델로 본다. 정형 LR은 002와 003의 계수가 1e-15 수준까지 같다.
- 한계: 002 A는 **5 epoch에서 멈춘 미수렴 버전**이다(003의 수렴한 A보다 ApacheJIT test에서 AP가 0.016 낮음). seed는 하나다.
- 명령: `predict_suite.py --dataset data/views/jd4j/{train,valid,test} --models rule tabular frozen finetune --model-root ../../002-no-hadoop/baseline/runs/models --device mps --bootstrap 500`
