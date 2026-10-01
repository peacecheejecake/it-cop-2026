# RUNLOG — 007-dl-m3-encoder

브랜치 `exp/007-dl-m3-encoder` (main `a444ffa`에서 분기). 코드: `codebert-diff-lab` (spec v0.2), study `configs/studies/public-comparison-v2.yaml`.

## 목적

사양 milestone **M3: B2-S(CodeBERT 고정 + fusion head) / B3-S(전체 fine-tuning)**, public validation, seed 42/43/44. public test는 열지 않는다.

## 데이터 (로컬 재실행, `a444ffa`)

006과 같은 4개 명령(import → audit → split → evidence build)을 이 worktree에서 다시 실행했다. **모든 parquet의 SHA-256이 006과 같다.** manifest가 다른 이유는 `imported_at`, `code_git_sha`, 그리고 이에 따른 상위 manifest 해시뿐이다.

## Runpod

| 항목 | 값 |
|---|---|
| GPU 선택 | 사용자 선택 H100 SXM. RTX PRO 6000은 CUDA 12.8 기준 품절. H100이 network volume `c8wdh0j8ek`의 데이터센터(CA-MTL-3)에 없어 **volume 없이** 실행하고, 결과와 best 가중치를 로컬로 회수한다 |
| Pod | `jn0vqsv9mnbuum` (`jit007-m3-encoder`), Secure, EUR-NO-2, **$3.49/시간**(질문 시점 카탈로그 최저가 $2.69보다 높은 호스트에 배정됨), H100 80GB HBM3, 드라이버 580.126.09, 컨테이너 디스크 80 GB |
| 환경 | uv `--frozen`(main의 uv.lock), Python 3.11.13(로컬은 3.11.16), torch 2.14.1+cu130, transformers 4.57.6. `doctor`: cuda true, network_sandbox `unshare`. pytest 32 passed, 1 skipped |
| 번들 | `.cache/bundles/jit007-a444ffa.tar.gz` (495 MB, sha256 `a09dd705…b76d`): `git archive` + 데이터 + codebert-base. 업로드 4분 28초, 원격 sha256 일치 |

## T14 profile (B3 경로, train split만, 100 update = 3,200 예시, seed 42)

`uv run diff-lab experiment profile --study configs/studies/public-comparison-v2.yaml --precision <p> --updates 100 --out profile/<p>.json`

| precision | 예시/초 | non-pad 토큰/초 | peak CUDA | loss 처음 → 끝 | finite |
|---|---|---|---|---|---|
| fp32 | 113.2 | 33,458 | 4.83 GB | 0.791 → 0.340 | ✓ |
| bf16_encoder_autocast | 368.8 | 109,008 | 4.04 GB | 0.762 → 0.330 | ✓ |

- bf16(encoder만 autocast)은 3.3배 빠르고 100 update 동안 발산하지 않았다.
- **결정: fp32 유지**(study pin 변경 없음). 003에서 bf16이 epoch 단위로 발산했던 이력이 있고, 100 update는 전체 학습의 안정성을 보장하지 않는다. fp32의 예상 비용 차이는 수 달러 수준이다.
- 원본: `results/profile/*.json`.

## M3 정식 실행

```bash
uv run diff-lab experiment run --study configs/studies/public-comparison-v2.yaml --models B2-S,B3-S --seeds 42,43,44
```

02:25 UTC에 시작해 03:2x UTC에 끝났다(EXIT=0, nohup). epoch별 로그는 `logs/m3-formal-epochs.log`, 요약은 `logs/m3-formal.json`, run 파일은 `results/<run_id>/`(가중치 제외)에 있다.

| variant | seed | run_id | 선택 epoch / 중단 | AP | ROC-AUC | R@5% | R@10% | F1@thr* | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|
| B2-S | 42 | `c39303adf68c0395` | 20 / 상한 도달 | 0.3718 | 0.8146 | 0.278 | 0.435 | 0.420 | 81 s |
| B2-S | 43 | `01a8a93a163decbc` | 20 / 상한 도달 | 0.3803 | 0.8140 | 0.285 | 0.448 | 0.417 | 77 s |
| B2-S | 44 | `f5e31c08bb385290` | 20 / 상한 도달 | 0.3802 | 0.8141 | 0.283 | 0.439 | 0.415 | 77 s |
| B3-S | 42 | `afc2b5821f1f75eb` | 3 / epoch 8 조기 종료 | 0.5087 | 0.8790 | 0.356 | 0.527 | 0.493 | 20.1 min |
| B3-S | 43 | `de409cca4a05571a` | 2 / epoch 7 | 0.5193 | 0.8786 | 0.375 | 0.553 | 0.514 | 17.5 min |
| B3-S | 44 | `1203a66de57c5f65` | 2 / epoch 7 | 0.5536 | 0.8980 | 0.392 | 0.559 | 0.528 | 17.5 min |

\* threshold는 같은 validation에서 고른 값이라 F1이 낙관적이다.

- B2-S: CLS 임베딩 캐시(key `22e31335…`, 48초)를 만든 뒤 encoder state hash가 학습 전후로 같았다. encoder는 eval 모드였다. **20 epoch 상한에 걸릴 때까지 validation AP가 매 epoch 0.002~0.004씩 올랐다. 수렴하지 않았다.** pin된 lr 1e-5가 고정 encoder 위의 head에는 작다. 이 값은 사양 공통 FT 프로토콜이라 그대로 두었다. 변경 여부는 RESULTS에 판단 사항으로 남긴다.
- B3-S: epoch 2~3에서 최고점에 도달했다. 이후 train loss가 0.24에서 0.04로 떨어지는 동안 validation AP가 0.38~0.41까지 내려가는 뚜렷한 과적합을 보였다. patience 5로 epoch 7~8에 멈췄다.
- seed 간 차이는 실제로 있다(B3 AP 표준편차 0.024). GPU 결정성은 seed 42 재실행으로 확인하지 않았다.
- **provenance 누락**: Pod 번들에 `.git`이 없어서 `environment.json`의 `git.sha`가 `null`이다. 실제 코드는 번들 커밋 `a444ffa`이고, 번들 sha256으로 고정된다. 이후 번들에는 코드 SHA를 명시적으로 넣도록 고친다(main 후속 커밋).

## 회수와 정리

- 회수: `artifacts/runs`(B3 `last/state.pt` 제외), `logs`, `profile`, 1.4 GB, 4분 49초. 원격과 로컬의 sha256이 73개 파일 모두 일치한다(B2의 작은 `last/state.pt` 3개는 추가로 받았다). B3 best 가중치 3개(각 474 MB)는 `codebert-diff-lab/artifacts/runs/<id>/checkpoints/best/`에 있다(git-ignore). M6 test 평가에 쓴다.
- **Pod `jn0vqsv9mnbuum`을 삭제했다**(204). 사용 시간 02:18~03:31 UTC 약 1.2시간, 비용 약 $4.3.
