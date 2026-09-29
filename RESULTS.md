# RESULTS — 002-no-hadoop

001과 **같은 프로토콜**(README 기본 하이퍼파라미터, 5 epoch 상한, seed 42, Apple MPS)을 쓰고, **Hadoop 계열 3개 저장소만 제외**했다. 실행 기록과 사전 등록은 `RUNLOG.md`에 있다.

## 한 줄 결론

Hadoop 라벨 이상을 데이터에서 빼고 학습·선택·평가를 모두 다시 해도 **정형 LR이 fine-tuned CodeBERT(B)보다 낫다.** AP는 0.576 대 0.534이고, 차이는 −0.042, 95% CI [−0.058, −0.024]다. 001의 사후 분석(Hadoop 제외 부분집합)이 가리킨 역전이 설계를 바꾼 실험에서도 재현되었다. 003(3 seed, CUDA)이 같은 결론을 확인했다.

## 설정과 데이터

| 항목 | 값 |
|---|---|
| 데이터 | `apachejit-full`에서 hadoop, hadoop-hdfs, hadoop-mapreduce를 제외(`tools/filter_records.py`) → 90,469개 |
| split | 2016/2017 경계, `--allow-retrospective`. train 56,870(29.3%) / valid 8,709(32.6%) / test 24,178(22.9%) |
| 모델 | 001과 같다. 학습은 `tools/train_with_progress.py`로 진행률만 추가했다 |
| test 행 | **001 test의 Hadoop 제외 부분집합과 ID 집합이 정확히 같다**(24,178건) |

## 학습 (public validation AP)

| 모델 | validation AP | 비고 |
|---|---|---|
| 정형 LR | 0.670 | C = 0.1 |
| A frozen | 0.570 | 5 epoch까지 계속 상승(미수렴, 003에서 epoch 10~22에 수렴함을 확인) |
| B finetune | 0.720 | epoch 1~4: 0.699, **0.720**, 0.708, 0.697 → epoch 2 선택, patience로 조기 종료. epoch당 약 2시간 |

## 공개 test (N = 24,178, 양성 5,532)

| 모델 | AP | ROC-AUC | Recall@10% | Precision@10% |
|---|---|---|---|---|
| rule | 0.499 | 0.784 | 0.262 | 0.600 |
| **정형 LR** | **0.576** | 0.814 | **0.301** | **0.689** |
| A frozen | 0.407 | 0.721 | 0.210 | 0.481 |
| B finetune | 0.534 | **0.816** | 0.266 | 0.608 |

AP(B) − AP(정형 LR) = **−0.042**, week-cluster paired bootstrap 95% CI [−0.058, −0.024](157개 cluster).

## 같은 행에서 001 / 002 / 003 비교 (seed 42, AP / ROC-AUC)

| 실험 | 학습 데이터 | 장치·epoch | 정형 LR | A | B |
|---|---|---|---|---|---|
| 001 | **Hadoop 포함**(train 65,478개, valid도 Hadoop 포함) | MPS, 5 epoch | 0.580 / 0.815 | 0.408 / 0.721 | **0.551** / 0.824 |
| 002 | Hadoop 제외(56,870개) | MPS, 5 epoch | 0.576 / 0.814 | 0.407 / 0.721 | 0.534 / 0.816 |
| 003 seed 42 | Hadoop 제외 | CUDA, 수렴까지 | 0.576 / 0.814 | 0.423 / 0.737 | 0.537 / 0.818 |

- **장치의 영향은 작다**: 002와 003 seed 42의 점수 Spearman 상관은 정형 LR 1.000, A 0.978, B 0.979다. B의 AP 차이 0.003은 003의 seed 간 표준편차(0.004) 안이다.
- **A는 수렴까지 학습하면 +0.016**이다(0.407 → 0.423). 그래도 가장 낮다.
- **Hadoop을 포함해 학습하면 B가 +0.017**이다(0.534 → 0.551). 003의 seed 간 표준편차의 약 4배다. 정형 LR은 +0.004다. "학습 데이터가 많을수록 B가 조금 나아진다"는 단서지만 **확정할 수 없다**: seed가 하나이고, 001은 valid에도 Hadoop이 있어 모델 선택 조건이 다르다. 후보 실험 004(train은 Hadoop 포함, valid·test는 제외, 3 seed)로 확인해야 한다.

## 누수 점검

test 중 train과 사실상 같은 변경·메시지는 246건(1.0%)이고 양성은 3건이다. 결과에 영향이 없다.

## 한계

- 이 실험의 공개 test는 001의 test 결과를 본 뒤 설계한 실험의 test다(`RUNLOG.md` 독립성 고지).
- ApacheJIT은 커밋 포함 방식이 무작위가 아니어서 기저율이 부풀려져 있을 수 있다(`RUNLOG.md` 데이터 특성, `DATASETS.md`). 결과는 이 benchmark 안의 순위 비교로만 해석한다.
- seed 하나. seed 분산은 003에서 확인했다(B AP 표준편차 0.004).
