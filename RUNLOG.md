# RUNLOG — 008-dl-m4-mlm-cpt

브랜치 `exp/008-dl-m4-mlm-cpt` (main `c0376c6`에서 분기, `b2ae128`에 B2 민감도 study 등록). 코드: `codebert-diff-lab` (spec v0.2).

## 목적

1. 사양 milestone **M4: B4-S**(diff MLM CPT 10M 토큰 → B3-S와 같은 fine-tuning), public validation, seed 42/43/44.
2. **보조 민감도 분석**(사용자 요청 "B2 rerun"): `configs/studies/b2-head-sensitivity-v1.yaml`. 007의 B2-S가 20 epoch 상한에서도 계속 오르고 있었기 때문에, head lr을 1e-3으로, epoch 상한을 100으로 바꾼 설정 하나만 사전에 정해 실행한다. 나머지 설정은 동일하다. 사전 등록된 B2-S 결과(0.377)를 **대체하지 않으며**, 주 비교 행렬에도 넣지 않는다. 이 설정은 validation 곡선을 본 뒤 정했으므로 B2-S의 고정 encoder 상한을 가늠하는 진단으로만 쓴다.

## CPT 설정 (study `cpt` 섹션, 이번에 pin)

budget은 non-padding 입력 토큰(특수 토큰 포함) 10M이다. micro-batch 8 × accumulation 16 = 128 seq/update, AdamW lr 2e-5, wd 0.01, 6% linear warmup 후 linear decay, grad clip 1.0, fp32.
MLM 0.15, 80/10/10. renderer 구조 위치(메시지와 코드 사이 및 줄 사이 `\n`, 각 줄 앞 `+ `·`- `)와 특수 토큰은 대상에서 제외하고, 코드 안의 연산자는 대상에 포함한다. 사양 기본값이 아닌 자체 선택은 linear decay, grad clip 1.0, dev 평가 주기 25 update, dev mask seed `20261001`이다.

- 실제 데이터(로컬): CPT-train 15,375개 변경, 1회 순회당 4,545,793 토큰이다. 적격 토큰 비율은 0.868이고 제외된 샘플은 없다. 계획: seed 42/43/44 모두 **265 update**(약 10.03M 토큰, 약 2.2회 순회).
- CodeBERT HF checkpoint에는 MLM head가 없어 `lm_head.{dense,layer_norm,bias}`가 **새로 초기화된다**. decoder는 embedding에 묶여 있다. 초기 MLM loss는 약 16으로, 균등분포의 10.8보다 높다. 사양상 다른 MLM checkpoint를 쓰는 것은 별도 실험이므로 그대로 진행한다.
- MPS에서 실제 CodeBERT로 1 update를 점검했다. loss 15.90, dev loss 16.02 → 14.78이었다.

## Runpod

| 항목 | 값 |
|---|---|
| Pod | `1cpqzuzzupquqy` (`jit008-m4-mlm-cpt`), Secure, AP-IN-1, **$3.49/시간**(카탈로그 $2.69보다 높은 호스트에 배정됨), H100 80GB HBM3, 드라이버 580.126.09, volume 없음 |
| 환경 | uv `--frozen`, Python 3.11.13, torch 2.14.1+cu130. pytest 41 passed, 1 skipped |
| 번들 | `.cache/bundles/jit008-b2ae128.tar.gz` (495 MB, sha256 `87bccb19…5d0`). `codebert-diff-lab/CODE_SHA`=`b2ae128…`(007의 provenance 누락을 수정). 업로드 5분 7초, 원격 sha256 일치 |
| 데이터 | worktree에서 import → audit → split → evidence를 다시 실행했다. parquet 파일이 007과 모두 같다 |

## 실행 (04:20 UTC 시작, nohup, 순차)

```bash
uv run diff-lab experiment run --study configs/studies/b2-head-sensitivity-v1.yaml --models B2-S --seeds 42,43,44
uv run diff-lab experiment run --study configs/studies/public-comparison-v2.yaml --models B4-S --seeds 42,43,44
```

04:20에 시작해 05:37 UTC에 둘 다 EXIT=0으로 끝났다. 진행 줄은 stdout(`logs/*.json`)에, transformers 경고는 stderr에 남는다.

### B2 민감도 (`b2-head-sensitivity-v1`, 보조 분석)

| seed | run_id | best / 중단 epoch | AP | ROC-AUC | R@5% | R@10% |
|---|---|---|---|---|---|---|
| 42 | `fc99b6879fee7a68` | 27 / 32 | 0.6227 | 0.8851 | 0.435 | 0.623 |
| 43 | `91c51ce3ff1a8a02` | 14 / 19 | 0.6039 | 0.8850 | 0.428 | 0.602 |
| 44 | `ab8ca445ad53f265` | 19 / 24 | 0.6218 | 0.8892 | 0.426 | 0.619 |

- 평균 AP는 0.616으로 사전 등록 B2-S(0.377)보다 크게 높고, B3-S(0.527)와 B1-TFIDF-S(0.546)보다도 높다. 반면 validation AP는 epoch마다 ±0.03 정도 흔들리고 최고 epoch를 같은 validation에서 골랐기 때문에 **선택 편향이 있는 낙관적 값**이다(Codex 리뷰 High 4).
- 해석: 사전 등록 프로토콜(head와 encoder가 같은 lr 1e-5를 공유)에서 B2-S는 head가 덜 학습된 상태였다. 이 결과는 주 행렬 수치를 대체하지 않는다.

### B4-S (사전 등록 `public-comparison-v2`)

| seed | run_id | CPT dev MLM loss (0 → 265) | best / 중단 epoch | AP | 비고 |
|---|---|---|---|---|---|
| 42 | `0831e41dd9146c3f` | 16.02 → 1.619 | 3 / 8 | 0.5596 (AUC 0.8961, R@5% 0.413, R@10% 0.574) | run 폴더 전체 회수 |
| 43 | `6118bd9c4d0af1b2` | → 1.548 | 2 / 7 | 0.5450 (로그) | **회수 불완전**: metrics, 예측, state 파일 없음. best encoder 파일은 전송이 중간에 끊겨 무결성을 확인하지 못함 |
| 44 | `f077bc798500ebb2` | → 1.531 | 2 / 7 | 0.5489 (로그) | **회수 실패** |

- 로그 기준 AP 평균은 0.551 ± 0.008(n=3)이다. 같은 seed의 B3-S와 비교하면 +0.051, +0.026, −0.005로, 평균 +0.024다. B3과 마찬가지로 epoch 2~3 이후 과적합된다.
- 세 seed의 CPT는 모두 265 update, 각 10.03M 토큰을 정확히 계획대로 마쳤다(계획 sha 일치는 seed 42의 `token-accounting.json`과 `model/state.json`으로 확인). CPT artifact(`artifacts/cpt/*`의 cpt.json, metrics.jsonl)는 회수하지 못했다. seed 42의 dev 이력은 run의 `model/state.json`에 남아 있다.

## 사고: Pod 강제 종료로 결과 일부 유실

- 05:37 UTC 회수 중 SSH 연결이 끊겼고, 이후 `get-pod`가 404를 반환했다. 원인은 **Runpod 잔액 소진**이다(`clientBalance` −$0.27). 사용자나 agent가 삭제하지 않았다. 이 Pod의 비용은 04시대 $1.63, 이후 약 $1.5로 추정한다.
- seed 43/44의 run 폴더와 CPT artifact를 잃었다. seed 43/44의 AP 수치는 로그에만 남아 있으므로 run-folder 계약상 **freeze 대상이 될 수 없다.** 잔액을 충전한 뒤 seed 43/44를 다시 실행해야 한다(약 50분, $3 미만).
- 교훈(AGENTS 반영 예정): 회수가 끝나기 전에 Pod가 사라질 수 있다. 실행 직후 run별로 작은 결과 파일부터 즉시 받고, 큰 가중치는 그다음에 받는다. 장시간 실행 전에 잔액을 확인한다.

## 외부 리뷰

Codex CLI(read-only) 리뷰: `reviews/2026-10-01-codex-review.md`(프롬프트 `reviews/2026-10-01-codex-review-prompt.md`). 직접적인 라벨 누출은 없다. High 4건은 CUDA resume의 RNG device 버그, lineage 부모 관계 미검증, B2 프로토콜 불리, validation 최고 epoch 선택 편향이다.
