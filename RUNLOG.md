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

### 1차 실행 (bf16) — 발산해서 중지

설치 문제 두 가지를 고쳤다:
- macOS tar의 소유권 경고와 `._*` 파일 → `tar --no-same-owner --exclude="._*"`로 다시 풀었다.
- Ubuntu 24.04의 PEP 668 → `/workspace/venv`(system site-packages 사용, 이미지의 torch 재사용)를 만들었다.

view의 SHA-256은 세 split 모두 manifest와 일치했다. Pod에서 `pytest -q` 55개가 통과했다.

| 모델 (seed 42, bf16) | epoch별 train loss | validation AP |
|---|---|---|
| A frozen | 0.674 → **174.6** → 20.0 | 0.326444 (valid 기저율과 같음 = 점수가 모두 같음) |
| B finetune | 0.600 → 0.964 → 1.322 → **159.4** | 0.300 → 0.358 → 0.326 → 0.326 |

B 처리량은 약 **416 ex/s**(bf16)였다. seed 42의 predict_suite 도중 파이프라인을 중지했다.

**원인 조사**
- 같은 16개 입력의 CodeBERT 평균 임베딩: CPU fp32와 CUDA fp32의 차이는 0이다. CUDA bf16은 최대 차이 0.018, 코사인 유사도 1.0이다. 샘플 간 표준편차는 0.058(평균 |e|는 0.458)이다. → **인코더와 GPU는 정상**이다(`scripts/debug_embeddings.py`).
- 저장된 head: A의 |w| 평균 1.63, bias −1.73. logit이 수백 단위로 커져 sigmoid가 포화했다.
- 원인 분리(frozen A, 2 epoch):

  | 설정 | epoch 1 loss / val AP | epoch 2 loss / val AP |
  |---|---|---|
  | **fp32, batch 32×1 (CUDA)** | 0.566 / 0.535682 | 0.541 / 0.550982 |
  | bf16, batch 8×4 (CUDA) | 0.674 / 0.326444 | 174.6 / 0.326444 |
  | 002 fp32, batch 8×4 (MPS) | — / 0.535681 | — / 0.550982 |

  → **원인은 bf16 autocast다.** batch 32×1은 실제 데이터에서도 8×4와 같은 결과를 냈고, CUDA fp32는 MPS를 소수점 여섯 자리까지 재현했다.
- 해석: epoch 전체에 autocast를 씌우면 head(768 → 1)의 logit 계산도 bf16(유효숫자 약 3자리)으로 떨어진다. 샘플 간 차이(표준편차 0.058)가 공통 성분(0.46)보다 작아 신호가 반올림에 묻히고, 가중치가 한 방향으로 커진 것으로 보인다. bf16을 제대로 쓰려면 head를 fp32로 유지해야 한다. 이번에는 사용자 결정에 따라 하지 않았다.
- 교훈: 정밀도를 바꿀 때는 demo 인코더가 아니라 실제 데이터로 짧게 검증해야 한다. 이 bf16 제안과 검증 부족은 에이전트의 실수다.
- 1차 결과물은 Pod의 `/workspace/jit003/runs-bf16-diverged/`에 보존했다.

### 사전 등록 재수정 (학습 재시작 전, 사용자 결정)
**정밀도를 fp32로 되돌린다**(원래 사전 등록 조건). 나머지는 같다(CUDA, seed 42/43/44, epoch 상한 100, patience 2, batch 32×1). 이제 002와 다른 조건은 장치와 epoch 상한 두 가지다. 1차 실행 결과는 어떤 모델 선택에도 쓰지 않았다(test는 보지 않음).

### 2차 실행 (fp32) — 04:46 UTC 시작, 스크립트 `5612a88`

| 단계 (UTC) | 결과 |
|---|---|
| env + pytest 04:46 → 정형 LR 04:47 (11초) | — |
| seed 42 A frozen 04:47 → 04:51 | validation AP가 epoch 1~19에서 0.5357 → **0.5925**(epoch 19 선택)로 올랐고, epoch 20~21에서 개선이 없어 멈췄다. 발산 없음. 헤드 학습 약 39,000 ex/s |
| seed 42 B finetune 04:51 → | **146 ex/s**(fp32, epoch당 약 6.5분 + validation). validation AP: epoch 1 0.714, epoch 2 0.731 |

- A는 5 epoch 상한(002: 0.570)에서 **수렴하지 않은 상태였다**는 것이 확인됐다. 수렴 시점은 epoch 19다.
- 같은 seed 42인데 B의 validation AP가 002 MPS(0.699, 0.720)와 다르다. dropout 난수 흐름이 장치와 batch 구성에 따라 달라지기 때문으로 보이고, seed 간 분산 해석에 참고한다.
- 재추정(05:07 UTC 기준): seed당 약 45분이며 **07:10 UTC경 완료**될 것으로 본다. 남은 비용은 약 $4이고, 1차 bf16 실패분(약 $2)을 포함한 총비용은 약 $7이다.

seed별 학습 결과 (validation AP, 괄호는 선택된 epoch):

| seed | 시간 (UTC) | A frozen | B finetune epoch별 | B 선택 |
|---|---|---|---|---|
| 42 | 04:47–05:22 | 0.5925 (19), epoch 21에서 멈춤 | 0.714, **0.731**, 0.708, 0.709 | 0.7314 (2) |
| 43 | 05:22–06:05 | 0.5938 (22), epoch 24에서 멈춤 | 0.715, 0.721, **0.724**, 0.705, 0.679 | 0.7235 (3) |
| 44 | 06:05–06:40 | 0.5836 (10), epoch 12에서 멈춤 | 0.711, **0.727**, 0.720, 0.686 | 0.7271 (2) |

**완료 06:40:53 UTC.** 이어서 한 작업:
- 결과 회수: `tar --exclude=*.safetensors --exclude=*.npy`로 70개 파일을 받았고, 원격과 로컬의 파일 개수가 일치한다. 모델 가중치는 Pod의 `/workspace`에만 있다.
- Pod: `pod-action stop` → **EXITED**(uptime 10,854초 = 3.0시간, 약 $6.3). 후속 실험 여부가 정해지지 않아 삭제하지 않고 정지했다(`/workspace` 보존, 저장 공간 요금만 발생).
- 로컬 집계(002 venv, `data/splits` → 002의 splits 심링크): `tools/leak_check.py`, seed별 `tools/sensitivity.py`와 `tools/paired_bootstrap_subset.py`, 그리고 `scripts/aggregate_seeds.py` → `results/`.

- B는 모든 seed에서 epoch 2~3 뒤 validation AP가 떨어진다(과적합). epoch 상한을 없애도 **B는 5 epoch 안에 멈춘다**(001과 002처럼). epoch 상한 변경의 영향은 A에 한정된다.
- 06:08 UTC 재추정: **06:45 UTC경 완료**될 것으로 본다(seed당 35~43분).

### 탐색 분석: test 연도별 AP (seed 42, 사용자 질문 "왜 test에서 오히려 떨어지나"로 수행 — 사후 분석)

| 기간 | n | 기저율 | rule | 정형 LR | A | B | B − 정형 LR |
|---|---|---|---|---|---|---|---|
| valid 2016 | 8,709 | 0.326 | — | 0.670 | 0.593 | 0.731 | +0.061 |
| test 2017 | 8,417 | 0.310 | 0.620 | 0.686 | 0.543 | 0.674 | −0.012 |
| test 2018 | 6,962 | 0.272 | 0.573 | 0.638 | 0.454 | 0.585 | −0.052 |
| test 2019 | 8,799 | 0.117 | 0.328 | 0.408 | 0.232 | 0.337 | −0.070 |

- 학습 기간에서 멀어질수록 B의 우위가 사라지고 역전 폭이 커진다. 해석: **B가 시간이 지나면 낡는 패턴을 학습한다**(시간에 따른 분포 이동). 정형 feature(크기, 파일 수, 엔트로피)는 상대적으로 안정적이다.
- 한계: seed 하나이고 신뢰구간이 없다. 2019년은 기저율 11.7%로 라벨 우측 절단이 심하다. valid는 모델 선택에 쓰였으므로 B의 valid AP는 낙관적으로 편향되어 있다.
- 후속 후보(사용자에게 제안): 학습 곡선(train 25/50/100%)과 시간 경계 이동(train < 2017, test 2018~19).
