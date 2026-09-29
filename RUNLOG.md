# RUNLOG — 003-seeds-cuda

브랜치: `exp/003-seeds-cuda` (from `exp/002-no-hadoop` @ `660b981`)

## 목적

002와 같은 데이터(Hadoop 계열 제외)로 **seed 간 분산**을 보고, A(frozen)를 **수렴할 때까지** 학습한다. 학습은 Runpod CUDA GPU에서 한다(사용자 결정, 2026-09-29).

## 사전 등록 (학습 전 확정, 2026-09-29)

| 항목 | 값 | 002와의 차이 |
|---|---|---|
| 데이터 | 002의 `data/views/public/{train,valid,test}`를 그대로 사용. 같은 split, 같은 CodeBERT view(manifest의 hash로 확인) | 같음 |
| 장치 | Runpod CUDA GPU, fp32(PyTorch 기본값, TF32 끔) | **MPS → CUDA** |
| seed | **42, 43, 44** (A와 B 각각) | seed 42 한 번 → 3개 |
| epoch | **`--epochs 100`(안전 상한) + `--patience 2`**(baseline 기본값). "수렴" = validation AP가 2 epoch 연속 개선되지 않음 | 5 epoch 상한 → 사실상 없음 |
| batch | **`--batch-size 32 --accumulation 1`** | 8×4 → 32×1 |
| 그 외 | encoder lr 2e-5, head lr 1e-3, class_weight none, CodeBERT 512/64 head-tail | 같음 |
| 정형 LR | 한 번(결정적인 LBFGS이므로 seed와 무관) | 같음 |
| 평가 | seed마다 `predict_suite`(공개 test, rule/tabular/frozen/finetune, bootstrap 500). seed 간 평균 ± 표준편차, 프로젝트별 AP 평균 | seed 집계 추가 |

**batch 32×1과 8×4의 등가성**: `train_epoch`은 윈도 합계 loss를 윈도 크기로 나누고, 셔플은 batch 크기와 무관한 `randperm`(같은 generator seed)이다. 따라서 epoch마다 같은 32개 묶음으로 같은 gradient를 계산한다. padding 차이는 attention mask가 없애 준다. demo 인코더로 확인한 결과 옵티마이저 스텝 수가 같고, loss는 0.66740827 대 0.66740829였다. 파라미터 차이 4.5e-5는 Adam이 부동소수점 잡음을 키운 것이다(001의 MPS 테스트와 같은 현상). dropout 난수 흐름은 다르다.

**해석상 주의**:
- 장치(MPS → CUDA)와 epoch 상한이 002와 함께 바뀐다. 그래서 002 seed 42(MPS)와 003 seed 42(CUDA)의 차이는 장치와 epoch 상한의 효과가 섞인 값이다. B는 001과 002에서 모두 patience로 5 epoch 전에 멈췄으므로 epoch 상한의 영향은 주로 A에 나타날 것으로 예상한다.
- 002와 마찬가지로, 이 공개 test는 001의 test를 본 뒤 설계한 실험의 test다.

## 실행 기록
