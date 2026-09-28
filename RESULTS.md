# RESULTS — 001-baseline-smoke

baseline README를 순서대로 따라, **ApacheJIT 전체 공개 데이터**로 rule / 정형 LR / A(frozen CodeBERT) / B(fine-tuned CodeBERT)를 학습하고 공개 test에서 비교했다. 자세한 실행 기록은 `RUNLOG.md`에 있다.

## 한 줄 결론

전체 공개 test에서는 **B가 가장 좋다**(AP 0.526 vs 정형 LR 0.474, 차이 95% CI [+0.031, +0.075]). 그러나 이 우위는 대부분 **프로젝트 간 기저율 차이**에서 나오며, 특히 **ApacheJIT의 Hadoop 라벨 이상**의 영향이 크다. 이 부분을 빼면 순위가 뒤집혀 **정형 LR이 B보다 낫다**(AP 0.580 vs 0.551, 차이 CI [−0.044, −0.012]). 같은 프로젝트 안에서는 변경 크기만 보는 rule조차 B보다 평균 AP가 높다. 따라서 이 결과를 "fine-tuned CodeBERT가 코드 위험을 더 잘 판단한다"는 근거로 쓰면 안 된다.

## 설정

| 항목 | 값 |
|---|---|
| 데이터 | ApacheJIT v2(Zenodo 5907847, MD5 pin 일치), 106,674개 중 106,659개 채택(2 MB 초과 patch 15개 제외) |
| split | committer date 기준: train < 2016 ≤ valid < 2017 ≤ test, `--allow-retrospective`(**회고적 benchmark**) |
| 크기 | train 65,478(양성 29.4%) / valid 10,480(29.4%) / test 29,938(**19.4%**) |
| 입력 | CodeBERT 512 토큰(메시지 64), 앞·뒤 절단. **약 90%의 변경이 잘림** |
| 모델 | README 기본값, seed 42 한 번. A와 B는 MPS(Apple M3 Pro)에서 학습 |
| 모델 선택 | 공개 validation AP만 사용. test는 모델 선택에 쓰지 않음 |
| 제외 | C/D(LLM): 승인된 endpoint 없음 / §6 내부 데이터·freeze: 내부 export 없음 |

## 학습 (공개 validation)

| 모델 | validation AP | 비고 |
|---|---|---|
| 정형 LR | 0.588 | C = 0.1 |
| A frozen | 0.536 | 5 epoch까지 계속 상승, **미수렴** |
| B finetune | 0.721 | epoch 2 선택, epoch 4에서 조기 종료, 8.65시간 |
| project prior* | 0.559 | train의 프로젝트별 양성 비율만 사용 |

\* 해석을 돕기 위해 추가한 기준선이며, README의 비교군은 아니다.

## 공개 test (N = 29,938, 양성 5,796)

`predict_suite.py --models rule tabular frozen finetune --bootstrap 500`, week cluster 157개

| 모델 | AP [95% CI] | ROC-AUC | Recall@5% | Recall@10% | Precision@10% |
|---|---|---|---|---|---|
| rule | 0.419 [0.391, 0.448] | 0.764 | 0.140 | 0.261 | 0.504 |
| 정형 LR | 0.474 [0.443, 0.506] | 0.788 | 0.162 | 0.290 | 0.562 |
| A frozen | 0.362 [0.339, 0.389] | 0.713 | 0.127 | 0.219 | 0.425 |
| **B finetune** | **0.526** [0.496, 0.554] | **0.834** | **0.171** | **0.315** | **0.610** |
| project prior* | 0.368 | 0.708 | 0.134 | 0.245 | 0.475 |

rule 대비 paired Δ의 95% CI는 정형 LR의 AP가 [+0.044, +0.064], A가 [−0.075, −0.041], B가 [+0.087, +0.126]이다. 모든 모델이 coverage 100%이고 예측 실패는 없다.

## 민감도 분석

평가 전에 정한 것은 Hadoop 계열 제외와 거의 같은 중복 제외이고, 프로젝트별 AP는 결과를 본 뒤 추가했다(`scripts/sensitivity.py`, `paired_bootstrap_subset.py`).

### ApacheJIT의 Hadoop 라벨 이상
원본 CSV에서 `apache/hadoop`은 12,964개 **전부 음성**이다. 반면 같은 Hadoop 코드인 `apache/hadoop-hdfs`와 `hadoop-mapreduce`는 2012년 이후 약 90%가 양성이다. 2012년 이후 HDFS와 MapReduce 커밋은 실제로 `apache/hadoop` 저장소에 있다(SHA 3,100 / 3,100 확인). 즉 **라벨에 따라 project가 정해진 것으로 보인다.** 모델은 diff의 경로를 보고 이 흔적을 이용할 수 있다.

| test 범위 | n / 양성 | rule | 정형 LR | A | B | project prior |
|---|---|---|---|---|---|---|
| 전체 | 29,938 / 5,796 | 0.419 | 0.474 | 0.362 | **0.526** | 0.368 |
| Hadoop 계열 제외 | 24,178 / 5,532 | 0.499 | **0.580** | 0.408 | 0.551 | 0.332 |
| train과 거의 같은 중복 제외 | 29,689 / 5,793 | 0.420 | 0.474 | 0.362 | **0.526** | 0.370 |
| 프로젝트별 AP 평균(12개 프로젝트)† | — | 0.532 | **0.579** | 0.419 | 0.510 | — |

† 양성이 20개 이상인 프로젝트만 포함했다. 평균 기저율은 0.312다. 정형 LR이 B보다 나은 프로젝트는 12개 중 11개다(예외는 groovy).

paired week-cluster bootstrap(500회)으로 본 AP(B) − AP(정형 LR):
- 전체: **+0.052** [+0.031, +0.075]
- Hadoop 계열 제외: **−0.029** [−0.044, −0.012]

거의 같은 중복(0.8%, 거의 모두 음성)은 결과에 영향이 없다. **B의 전체 test 우위는 프로젝트를 구분하는 신호, 특히 Hadoop 라벨 이상에서 나온다.**

## 해석과 한계

- **ApacheJIT을 그대로 쓰는 비교는 프로젝트 식별 능력을 보상한다.** CodeBERT는 파일 경로로 프로젝트를 알아낼 수 있으므로, 이후 실험은 Hadoop 계열을 제외하거나 정정하고, 프로젝트별 지표 또는 층화 평가를 함께 보고해야 한다.
- test의 양성 비율(19.4%)이 train과 valid(29.4%)보다 낮다. 최근 커밋의 결함이 아직 덜 발견됐기 때문(라벨 우측 절단)일 가능성이 크다. AP를 split끼리 비교하지 않는다.
- 정형 baseline은 7개 정적 feature만 쓴다. 원 논문의 expert/history feature baseline과 다르다(README §4).
- A는 수렴하지 않았다. B는 seed 한 번만 돌렸다. seed 간 분산과 A/B의 공개 hyperparameter 탐색은 아직 하지 않았다.
- 변경의 약 90%가 512 토큰에서 잘린다. 긴 변경의 정보 손실은 이번 설정의 기본 한계다.
- 이 결과는 결함 유발 커밋 라벨에 대한 회고적 순위 평가다. 운영 장애 예측이나 내부 성능을 뜻하지 않는다. C/D와 내부 평가는 수행하지 않았다.

## 후속 실험 제안

1. **002**: Hadoop 계열 라벨을 정정하거나 제외한 공개 데이터로 같은 비교를 다시 하고, 프로젝트별 지표를 주 지표에 넣는다. 그 뒤 A/B의 epoch과 lr을 공개 validation으로 탐색한다.
2. **003**: seed 42/43/44로 분산을 보고한다(`protocol.example.yaml`).
3. C/D: 승인된 endpoint가 준비되면 같은 입력으로 비교한다.
4. baseline 개선 후보(upstream): clone timeout으로 프로젝트 전체가 빠지는 문제, `--mirror`의 PR ref, frozen 모드의 `training_seconds` 누락, `resolved_revision=null`, `.gitignore`의 `models/`가 심링크를 무시하지 못하는 문제.

## 산출물

- `results/metrics.{json,csv}`: predict_suite 원본 지표와 bootstrap 결과
- `results/public-test-sensitivity.json`, `results/paired_bootstrap_subset.txt`
- `results/model-{tabular,frozen,finetune}.json`: 학습 메타데이터(fingerprint 목록은 제외)
- 모델과 예측(git-ignored): `baseline/runs/models/`, `baseline/runs/evaluation/public-test/`
- 공유 캐시: `experiments/.cache/` (raw, git, models, canonical)
