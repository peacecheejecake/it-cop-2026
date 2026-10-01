# RUNLOG — 009-dl-v3-rerun

브랜치 `exp/009-dl-v3-rerun` (main `46b1a28`에서 분기 → `9fd51de`로 fast-forward). 코드: `codebert-diff-lab`, study **`public-comparison-v3`**.

## 목적

사용자 지시("Fix everything, including the protocol change, now. Rerun all, if it is needed."): Codex 리뷰의 지적 사항을 모두 고치고 프로토콜 v3를 등록한 뒤, 구현된 모든 primary variant(B0-LR, B0-LGBM, B1-TFIDF-S, B2-S, B3-S, B4-S)를 seed 42/43/44로 다시 실행한다. B5-S(M5)와 L0/L1(M6L)은 아직 구현하지 않았다.

## v3와 v2의 차이 (`configs/studies/public-comparison-v3.yaml` 헤더 참고)

- finetune: encoder lr 1e-5, **head lr 1e-3**으로 분리. 최대 100 epoch, patience 5. B2-S부터 B5-S까지 똑같이 적용한다. head lr은 008의 B2 민감도 분석(public validation)을 참고해 정했다. validation에서 본 설정은 v2와 이 설정 둘뿐이다.
- split `upstream-clean2`: train/valid/test 구성원은 upstream-clean1과 같고(확인 완료), CPT-dev만 exact/code 중복 그룹 단위로 뽑는다(salt `cpt-dev-v2`, 809행). CPT-train 행 중 CPT-dev와 같은 evidence 텍스트를 가진 행은 20개에서 **1개**로 줄었다. 남은 1개는 evidence 절단 후에야 같아지는 경우로, split이 evidence보다 먼저 만들어지기 때문에 막을 수 없다. CPT accounting에 기록된다.
- 코드 수정(main `46b1a28`, `9fd51de`). 리뷰 항목에 대응한다:
  - resume을 CPU에 로드하도록 고쳤다(High 1). CUDA resume 테스트는 Pod에서 통과했다.
  - lineage validator를 추가했다(High 2, AT-04).
  - 선택 편향(High 4)에 대응해 지표에 `evaluation_role: selection_validation`을 표시하고, ledger에 `selection_events`를 기록한다.
  - CPT-dev를 그룹 단위로 뽑는다(Medium 5).
  - run ID에 source digest를 넣고, SHA를 알 수 없으면 실행을 거부한다. completed run은 artifact hash를 검증한 뒤에만 재사용하고, 기반 가중치는 sha256으로 고정한다(Medium 6).
  - checkpoint를 immutable generation과 원자적 pointer로 관리한다(Medium 7).
  - 지표의 null 정의를 고쳤다(Medium 8).
  - run 폴더에 `metrics.jsonl`, token accounting, `reports/summary.md`를 추가하고, 측정하지 않은 latency는 null로 기록한다(Medium 9).
  - 테스트를 보강했다(Low 10).
- **버그(실행 중 발견)**: `environment()`가 모든 run에서 torch를 import했다. torch가 설치된 venv에서는 B0-LGBM이 segfault(EXIT 139)를 냈다. torch와 LightGBM의 OpenMP 런타임이 함께 로드되었기 때문이다. 006 venv에는 torch가 없어서 드러나지 않았다. 이제 torch는 neural variant에서만 import하고, torch가 이미 로드된 프로세스에서는 B0-LGBM이 명시적 오류로 거부한다(`9fd51de`).

## 데이터 (로컬, `9fd51de`)

```bash
uv run diff-lab data import --source jit-defects4j --archive ../../.cache/raw/jit-defects4j/data.zip --approval configs/sources/jit-defects4j.yaml --snapshot-id jitd4j-audit1
uv run diff-lab data audit --snapshot-id jitd4j-audit1
uv run diff-lab data split --snapshot-id jitd4j-audit1 --split-id upstream-clean2
uv run diff-lab evidence build --study configs/studies/public-comparison-v3.yaml
```
출력은 `logs/0[1-4]-*.json`에 있다. counts: train 16,184(양성 1,387) / valid 5,465(467) / test 5,480(475), CPT-train 15,375 / CPT-dev 809.

## CPU 기준선 (로컬 macOS, 1분 19초)

`uv run diff-lab experiment run --study configs/studies/public-comparison-v3.yaml --models B0-LR,B0-LGBM,B1-TFIDF-S --seeds 42,43,44`

| variant | AP (seed 3개 동일) | run_id (seed 42) |
|---|---|---|
| B0-LR | 0.3193 | `d03f2f7dc1e6225d` |
| B0-LGBM | 0.2106 | `cabd1c09e6768333` |
| B1-TFIDF-S | 0.5464 | `8c206ab91ff2b2e9` |

v2(006)와 값이 같다. v3의 변경 사항(finetune, CPT-dev)은 이 variant들이 읽지 않으므로 예상대로다. 결정적이므로 실효 seed는 1이다.

## GPU (Runpod)

| 항목 | 값 |
|---|---|
| 잔액 | 시작 전 $19.73(사용자 충전). 예상 비용 약 $10 |
| Pod | `1qt8drujwubc98` (`jit009-v3-rerun`), Secure, AP-IN-1, $3.49/시간, H100 80GB HBM3, 드라이버 580.126.09 |
| 번들 | `.cache/bundles/jit009-9fd51de.tar.gz` (495 MB, sha256 `7ac22ae1…893e`), `CODE_SHA`=`9fd51de…`. 업로드 3분 43초, 원격 sha256 일치 |
| 환경 | uv `--frozen`, Python 3.11.13, torch 2.14.1+cu130. pytest 59 passed, 1 skipped(CPU 전용 테스트). CUDA resume 테스트(neural, CPT) 통과 |
| 회수 | 5분마다 작은 파일(가중치 제외)을 받는 루프로 받는다(008 사고에서 얻은 교훈) |

```bash
uv run diff-lab experiment run --study configs/studies/public-comparison-v3.yaml --models B2-S,B3-S,B4-S --seeds 42,43,44
```
06:08 UTC에 시작했다(nohup/setsid, `logs/v3-gpu.{out,err,done}`).

### 결과 (06:08 → 08:2x UTC, EXIT=0)

run 18개 모두 `run.json`의 `artifacts_sha256`과 로컬 파일이 일치한다. B2/B3/B4 best checkpoint 9개는 generation pointer의 hash로 검증했다. 표 전체는 `RESULTS.md`, run별 파일은 `results/<run_id>/`에 있다.

- B2-S: best epoch 27, 14, 19(각각 32, 19, 24 epoch에서 중단). 008 민감도 분석과 AP가 bit 단위로 같다. 설정이 같으므로 GPU에서도 결정적이다.
- B3-S: best epoch 3, 2, 2(8, 7, 7에서 중단). 학습 시간 약 21분. peak CUDA 4.8 GB.
- B4-S: CPT는 seed마다 265 update, 약 10.03M 토큰(2.206회 순회), target 약 1.32M, **330초**가 걸렸다. dev MLM loss는 16.0/15.1/15.4에서 1.61/1.51/1.52로 내려갔다. 제외된 샘플은 없고, dev와 같은 텍스트를 가진 CPT-train 행은 1개다. FT의 best epoch는 1, 2, 2다.
- 회수: 작은 파일은 5분 주기로 받았다. 마지막에 best generation의 safetensors 15개(B3/B4 encoder+head, B2 head)를 10분 만에 받았고 hash가 일치했다. CPT encoder export(seed당 474 MB)와 resume용 `state.pt`, 이전 generation은 받지 않았다. CPT export는 다시 만들 수 있다.
- **Pod `1qt8drujwubc98`을 삭제했다**(204). 잔액이 $19.73에서 $11.17로 줄어 **약 $8.6**을 썼다(06:03~08:3x).
