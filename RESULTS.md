# RESULTS — 010-dl-m5-mlm-rmi (B5-S, study v3, public validation)

> 아래 값은 selection-validation 값이다. checkpoint epoch와 threshold를 같은 validation에서 골랐으므로 낙관적이다.

| variant | AP (mean ± sd, n=3) | ROC-AUC | Recall@5% | Recall@10% |
|---|---|---|---|---|
| B3-S (009) | 0.687 ± 0.008 | 0.912 | 0.483 | 0.667 |
| B4-S MLM CPT (009) | 0.674 ± 0.014 | 0.911 | 0.470 | 0.657 |
| **B5-S MLM+RMI CPT** | **0.669 ± 0.010** | 0.910 | 0.467 | 0.661 |

seed별 차이(AP):

| | 42 | 43 | 44 | 평균 |
|---|---|---|---|---|
| B5 − B4 | −0.001 | −0.011 | −0.002 | **−0.005** |
| B5 − B3 | −0.023 | −0.021 | −0.011 | **−0.018** |

- **RQ3(RMI의 추가 가치): 효과가 관찰되지 않았다.** 같은 합산 예산에서 B5는 B4와 거의 같거나 약간 낮다. 두 모델 모두 CPT가 없는 B3보다 낮다.
- 해석할 때 주의할 점:
  1. B5의 MLM 노출은 B4의 절반이다(사양상 "같은 합산 예산" 비교이므로 의도된 설계다).
  2. RMI 학습이 seed마다 불안정하다(dev accuracy 0.57~0.70).
  3. 차이가 seed 표준편차와 비슷한 크기다.
- 최종 판단은 freeze 후 public test(011)에서 한다.
