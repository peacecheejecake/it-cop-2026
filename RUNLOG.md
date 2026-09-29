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

## 사전 등록 수정 (학습 전, 2026-09-29 12:55)

사용자 결정: **학습은 bf16 혼합 정밀도로 한다**(fp32 대조 실행은 하지 않음).
- 방법: `tools/train_with_progress.py --amp bf16`이 `train_epoch`를 `torch.autocast(cuda, bfloat16)`로 감싼다. 가중치, 옵티마이저 상태, BCE-with-logits loss는 fp32다(autocast가 이 연산을 fp32로 유지). GradScaler는 필요 없다. validation 채점, frozen 임베딩 추출, test 추론은 fp32다.
- fp16이 아니라 bf16인 이유: Blackwell에서 Tensor Core 속도는 같고, fp16은 GradScaler 때문에 baseline의 `train_epoch`를 고쳐야 하기 때문이다.
- **결과: 002(fp32, MPS)와의 차이에는 장치, epoch 상한, 정밀도 세 가지가 섞인다. 이것들은 분리할 수 없다.** seed 간 분산 비교(003 내부)에는 영향이 없다.
- 정밀도는 모델 디렉토리마다 `training_wrapper.json`에 기록한다.

## Runpod 설정

| 항목 | 값 |
|---|---|
| Pod | `n2m8dtxgz6s598` (`jit003-seeds-cuda`), Secure Cloud, EU-RO-1, **$2.09/시간** |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB, 드라이버 595.91.07, 호스트 CUDA 13.2 |
| 이미지 | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (Python 3.12.3, torch 2.8.0+cu128, capability 12.0) |
| 저장소 | 컨테이너 디스크 30 GB, `/workspace` 영구 디스크 30 GB |
| 접속 | direct SSH(`runpodctl ssh info`), 키는 `runpodctl doctor`로 등록 |

GPU 선택 근거: 사용자에게 fp32/bf16 처리량, 시간, 비용을 비교해 보여 준 뒤 사용자가 골랐다. 후보는 M3 Pro(실측 8 ex/s) 기준으로 환산한 추정치였다. RTX PRO 6000은 재고 HIGH였고 bf16 추정 약 1.1시간, 약 $2.3이었다.

업로드 번들: `.cache/bundles/jit003-d85890d.tar.gz` (573 MB, sha256 `01fbe490…42b84`). 번들에 포함된 `run_remote.sh`는 bf16 수정 전 버전이라, 수정된 스크립트만 따로 복사한다(아래 커밋).

## 실행 기록
