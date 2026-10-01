# RUNLOG — 006-dl-m2-cpu-baselines

브랜치 `exp/006-dl-m2-cpu-baselines` (main `d502149`에서 분기). 코드: `codebert-diff-lab` (spec v0.2).
study: `configs/studies/public-comparison-v2.yaml` (9개 primary variant 등록, LLM/CPT/FT 섹션은 PIN_REQUIRED).

## 목적

사양 milestone **M0(데이터 감사) + M2(CPU 기준선: B0-LR, B0-LGBM, B1-TFIDF-S, public validation)**. public test는 열지 않는다(freeze 전 코드가 거부).

## 환경

Python 3.11.16(uv, `--system-certs`), scikit-learn 1.9.1, pandas 2.3.3, pyarrow 21.0.0, transformers 4.57.6(토크나이저만), lightgbm 설치는 되지만 **libomp 없음 → import 실패**. CPU 전용. `logs/00-doctor.json`.

## M0 — import / audit / split / evidence (모두 `d502149`로 재실행, 이전 탐색 실행과 수치 동일 = 결정적)

```bash
uv run --system-certs diff-lab data import --source jit-defects4j --archive ../../.cache/raw/jit-defects4j/data.zip \
  --approval configs/sources/jit-defects4j.yaml --snapshot-id jitd4j-audit1
uv run --system-certs diff-lab data audit --snapshot-id jitd4j-audit1
uv run --system-certs diff-lab data split --snapshot-id jitd4j-audit1
uv run --system-certs diff-lab evidence build --study configs/studies/public-comparison-v2.yaml
```

- import: `macos-sandbox-exec-deny-network` 격리 worker, allowlist unpickler. snapshot manifest `code_git_sha=d502149`.
- 감사·split·evidence 수치와 해석: `codebert-diff-lab/docs/m0-audit.md`(main). 요약: 27,319 → upstream-clean1 train 16,184(양성 1,387) / valid 5,465(467) / test 5,480(475), train에서 eval과 같은 변경 190건 제외. 제공 split은 시간순이 아님(`upstream_holdout`). EvidenceView 27,129건 중 65%가 코드 줄 전부 포함, 중앙값 280 토큰.

## M2 — public validation (valid 5,465, 양성 467, 기저율 0.085)

`uv run --system-certs diff-lab experiment run --study … --models B0-LR,B1-TFIDF-S --seeds 42,43,44`

| variant | AP | ROC-AUC | Recall@5% | Recall@10% | F1@thr* | run_id (seed 42) |
|---|---|---|---|---|---|---|
| B0-LR | 0.3193 | 0.7920 | 0.2441 | 0.3961 | 0.369 | `2de31167bc4915ce` |
| **B1-TFIDF-S** | **0.5464** | **0.8847** | **0.3983** | **0.5653** | 0.530 | `1912ea338feda8a3` |
| B0-LGBM | 0.2106 | 0.7099 | 0.1670 | 0.2934 | 0.289 | `a2a548d20f63a287` |

\* threshold는 같은 validation에서 max-F1로 골랐으므로 F1은 낙관적이다(사양상 test에는 이 threshold를 고정 적용).

- **seed 42/43/44 결과가 bit 단위로 같다**(LR lbfgs / liblinear, LightGBM `deterministic=True`·행 샘플링 없음 모두 결정적). 사양 §8.3에 따라 이를 독립 반복 3회로 세지 않는다: `n_training_seeds_effective = 1`.
- B1 학습은 sparse 행렬 100,014열(메시지 5만 + 코드 5만 + 정형 14), liblinear 12회 반복 수렴. TF-IDF vocabulary/IDF는 train으로만 fit(테스트 AT-26).
- label-access ledger: gradient 16,184 / selection 5,465 / demo 0 / index 0.

### B1 진단 (validation 탐색, 모델 선택에 쓰지 않음)

상위 계수: 음(위험↓)은 `assert`, `Test` 등 **테스트 코드 n-gram**(결함 유발 라벨이 운영 코드 줄에서 역추적되므로 테스트만 바꾼 변경은 거의 음성), 양(위험↑)은 `if (`, `+ i`, `offset` 등 **추가된 제어 흐름**과 정형 `la`. 메시지의 `T - 1`, `- 1` 등 **이슈 번호 조각**도 상위에 있어, 같은 프로젝트·같은 시기 split(시간순 아님)에서 이슈 번호 대역을 외우는 성분이 섞였을 수 있다 → project/time holdout(P3)에서 재확인 필요.

### B0-LGBM (libomp 설치 후, 사용자 조치 2026-10-01)

첫 시도(`c10aa9e1f0d4cf9f`)는 libomp가 없어 exit 5로 실패했고 `failure.json`이 남았다(실패를 완료로 기록하지 않는 경로 확인). 설치 후 `--models B0-LGBM --seeds 42,43,44` → 위 표.

**LightGBM < LR**(AP 0.211 대 0.319). 진단(저장된 booster, 설정 변경·선택에 쓰지 않음): 반복 수별 AP

| 반복 | 50 | 100 | 200 | 300(등록 설정) |
|---|---|---|---|---|
| train | 0.420 | 0.488 | 0.587 | 0.665 |
| valid | 0.297 | 0.251 | 0.223 | 0.211 |

→ 사양의 시작 설정(300 tree, early stopping 없음)이 이 작은 불균형 데이터(16k, 양성 8.6%)에서 **과적합**한다. 결과를 보고 반복 수를 고르면 사전 등록 없는 validation 선택이 되므로 등록 설정의 결과를 그대로 보고한다. formal 단계에서는 LR의 C와 LGBM 후보(early stopping 포함)를 **미리 등록한 후보 수 안에서만** validation으로 고른다.

## 상태와 다음 단계

- M2 CPU 기준선 3종(B0-LR/B0-LGBM/B1-TFIDF-S) validation 완료. DoD-Core 미달: B2~B5 미구현.
- 다음: M3(EvidenceView 기반 B2-S/B3-S, fusion head, GPU).
