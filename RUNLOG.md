# RUNLOG — 001-baseline-smoke

브랜치: `exp/001-baseline-smoke` (from `main` @ `df538a5`)
목적: `baseline/README.md`를 순서대로 실행하며 파이프라인 배관(plumbing)을 검증. 실제 데이터/모델 성능 실험이 아니라 baseline 자체가 문서대로 동작하는지 확인하는 첫 실험.

## 환경 구축

- `uv venv --python /opt/homebrew/bin/python3` (Python 3.13.5)
  - README는 `uv venv --python 3.11`을 권장하지만, 이 샌드박스는 `python-build-standalone` 릴리스(github release asset) 다운로드에 필요한 외부 네트워크가 차단되어 있어(`invalid peer certificate: UnknownIssuer`) uv가 격리된 3.11 인터프리터를 받을 수 없었다. 시스템에 이미 설치된 Python 3.13.5(`>=3.11` 요구사항 충족)를 대신 사용.
  - **한계 기록**: 3.11에서만 검증된 동작 차이가 있을 수 있음. 정식 실험(neural 학습 등)으로 넘어가기 전에는 승인된 네트워크 환경에서 3.11로 재현하는 것을 권장.
- `uv pip install -e '.[dev]'` — 성공 (PyPI 접근은 가능, GitHub 릴리스 바이너리만 차단됨)

## 실행 결과

| 명령 | 결과 |
|---|---|
| `riskbench --help` | 정상 출력 (14개 subcommand) |
| `pytest -q` | `50 passed, 1 skipped` |
| `python scripts/smoke.py --out runs/smoke-core` | `PASS: synthetic plumbing only. No real model/dataset performance claims.` |

`runs/smoke-core/`에 `raw/`, `split/`, `views/`, `models/`, `predictions/`, `metrics.json`, `metrics.csv`, `experiment.lock.json` 등 전체 파이프라인 산출물 생성 확인.

## Neural 학습 코드 점검 — Apple Metal(MPS)

환경: Apple M3 Pro, 36 GB, macOS 27.0, torch 2.14.0, transformers 4.57.6 (`uv pip install -e '.[neural,dev]'`)

**결론: `neural.py` 학습 코드는 수정 없이 MPS에서 정상 동작한다.** `device_for("auto")`가 이미 cuda → mps → cpu 순으로 고르므로, 이 맥에서 `riskbench train-codebert`/`predict`/`predict_suite.py`는 기본값으로 MPS를 쓴다. 다만 기존 테스트와 `smoke.py`가 `device='cpu'`로 고정되어 있어 MPS 경로가 한 번도 검증되지 않았던 상태였다.

측정 (CodeBERT와 같은 RoBERTa-base 구조를 **무작위 초기화**한 모델, baseline `train_epoch` 그대로 사용 — 속도/정합성 측정용이며 성능 수치 아님):

| 항목 | 결과 |
|---|---|
| CPU vs MPS forward logit 최대 차이 | 2.7e-7 |
| CPU vs MPS gradient 상대오차 | 1.3e-6 |
| full fine-tuning 속도 (batch 8, accum 4, 길이 100–510) | CPU 394 → **MPS 112–124 ms/example (약 3.2×)** |
| 길이를 64 배수로 padding | 이득 없음 (122 ms) |
| bf16 / fp16 autocast | 이득 없음 (120 / 118 ms) → 수치가 바뀌는 AMP는 도입하지 않음 |
| microbatch마다 `.item()` 동기화 제거 | 이득 없음 (111–113 vs 113 ms) → 되돌림 |
| MPS 메모리 (길이 512 고정) | batch 8: 9.1 GiB, 16: 15.2 GiB, 32: 27.3 GiB |

대략적 소요 시간 추정: 공개 train이 N건이면 epoch당 ≈ N × 0.12초 (예: 5만 건 ≈ 1.7시간/epoch). 실제 CodeBERT 가중치·실데이터 길이 분포에서 다시 측정해야 함.

변경 사항 (검증 경로만, 학습 로직은 무변경):
- `tests/test_neural.py`: MPS 사용 가능할 때만 도는 테스트 2개 추가
  - `test_mps_step_matches_cpu` — 같은 초기 가중치로 CPU/MPS 한 epoch 후 파라미터 비교. Adam은 이론상 gradient가 0인 파라미터(attention key bias)의 float 잡음을 lr 크기 스텝으로 키워 1e-3 수준 차이를 만들므로, 비교는 SGD로 한다.
  - `test_complete_neural_train_on_mps` — `train_neural(..., device='mps')` end-to-end, artifact에 `device: mps` 기록 확인
- `scripts/smoke.py`: `--device {cpu,mps,cuda,auto}` 추가 (기본값 `cpu`로 기존 동작 유지)

| 명령 | 결과 |
|---|---|
| `pytest -q` | `57 passed` (MPS 테스트 3회 반복 안정) |
| `python scripts/smoke.py --out runs/smoke-neural-cpu --neural` | PASS |
| `python scripts/smoke.py --out runs/smoke-neural-mps --neural --device mps` | PASS, 모델·prediction meta 모두 `device: mps` |

주의: MPS 연산은 CUDA의 `cudnn.deterministic`에 해당하는 결정성 보장이 없다. 같은 seed라도 CPU/CUDA와 bitwise 동일하지 않으며, seed 간 분산 보고(README §4) 원칙은 그대로 적용된다. `PYTORCH_ENABLE_MPS_FALLBACK`은 설정하지 않는다 — 미지원 연산이 조용히 CPU로 넘어가지 않고 크게 실패하도록 둔다.

## 범위 결정 (사용자 결정, 2026-09-28)

- **C/D(LLM) 제외**: 승인된 OpenAI-compatible endpoint가 없음. endpoint가 준비되면 별도 실험으로 진행.
- **§6 내부 데이터 제외**: 이 환경에 내부 export(manifest.csv, patches/)가 없음. 따라서 `freeze`/내부 평가는 수행하지 않고 공개 test까지만 비교.
- **전체 ApacheJIT 사용**: `--limit` 샘플이 아닌 전체 106,674 커밋. 원본/저장소/CodeBERT/canonical은 이후 실험과 공유하도록 `experiments/.cache/`에 둔다(AGENTS.md §1.4).

## §2-1 fetch-apache — 회사 TLS 검사 프록시 대응

- 이 네트워크는 사내 TLS 검사 프록시가 일부 도메인(zenodo.org 등)의 TLS를 가로챈다. huggingface.co·github.com(git)은 가로채지 않거나 시스템 신뢰 저장소로 통과한다.
- 조치 1 (사용자 승인): 이 실험 venv의 `certifi/cacert.pem`에만 사내 프록시 루트 CA(macOS System 키체인에서 추출)를 추가했다.
- 조치 2 (사용자 승인): 그래도 Python 3.13의 기본 `VERIFY_X509_STRICT`가 프록시가 만든 zenodo.org 인증서(Authority Key Identifier 누락)를 거부했다. `scripts/fetch_apache_relaxed_tls.py`로 **이 프로세스에서만** strict 플래그를 끄고 riskbench의 `fetch-apache`를 그대로 호출했다. 무결성은 riskbench의 pinned MD5 검사로 보장된다.
- 결과: MD5 `528bf0ee04b15976be6bf15f8efddc65`(pin 일치), CSV SHA-256 `5097cbbb…d4e0`, 106,674행, 15개 프로젝트. → `experiments/.cache/raw/apachejit-v2/`

## §3-1 download-codebert

`riskbench download-codebert --out models/codebert-base` → `microsoft/codebert-base` revision `3b0952feddeffad0063f274080e3c23d75e7eb39`. → `experiments/.cache/models/codebert-base/`

## §2-2 build-apache — 저장소 clone 문제와 해결

첫 시도: `build-apache --limit 300 --seed 42 --allow-network` → **중단함**.
- baseline의 clone은 `git clone --mirror`를 **저장소당 1200초 제한**으로 실행한다. `--mirror`는 GitHub의 `refs/pull/*`까지 받아 크기가 커지고, 프록시 경유 약 3 MB/s에서는 hadoop, hadoop-mapreduce, kafka가 시간 초과로 `repository_clone_failed` 처리됐다. 그 결과 샘플 300개 중 42개(14%)가 **프로젝트 단위로 조용히 빠지는 편향**이 생겼다.
- spark는 `--mirror`로 1.1 GB를 넘기고도 끝나지 않았지만, `--bare`(브랜치·태그만)로는 653 MB에 완료됐다.
- **upstream 반영 후보**: clone timeout이 짧으면 프로젝트 단위 누락이 생긴다. `--mirror` 대신 heads/tags만 받거나 timeout을 설정 가능하게 해야 한다.

해결 (README §2-2의 오프라인 경로):
- 이미 받은 9개(`--mirror`)는 그대로 캐시로 옮기고, 나머지 6개(hadoop, hadoop-mapreduce, kafka, spark, zeppelin, zookeeper)는 `git clone --bare`로 시간 제한 없이 받았다. 15개 모두 exit 0, 합계 약 8.9 GB. → `experiments/.cache/git/apache/`
- 이후 `--allow-network` 없이 실행한다:
  ```bash
  caffeinate -i riskbench build-apache --csv data/raw/apachejit-v2/apachejit_total.csv \
    --repos data/git --out ../../.cache/canonical/apachejit-full
  ```
  diff 추출은 약 66 ms/커밋(camel 40개 측정)이므로 전체는 약 2시간이 예상된다.

진행률 확인: build-apache는 끝날 때까지 출력이 없으므로, `scripts/build_apache_with_progress.py`로 실행한다. 이 래퍼는 baseline 코드를 바꾸지 않고 `extract_commit`만 감싸 500개마다 진행률을 stderr에 쓴다. 확인은 `tail -f experiments/.cache/logs/build-apache-full.log`로 한다.

### 1차 결과 — hadoop 분리 저장소의 라벨 편향 누락 발견

1차 전체 build: selected 106,674 / accepted 103,559 / rejected 3,115 (1,120개/분, 약 95분). 보고서는 `.cache/logs/build_report.run1-before-alternates.json`에 보관했다.
- `patch_over_2MB_excluded` 15건은 README 정책대로다.
- `git show` exit 128(커밋 없음) 3,100건은 **전부 `apache/hadoop-hdfs`(2,304 / 2,907, 79%)와 `apache/hadoop-mapreduce`(796 / 1,321, 60%)**에서 나왔다. 나머지 13개 저장소의 누락은 0건이다.
- 누락분의 buggy 비율은 88%(2,759 / 3,100)로, 전체 26%보다 훨씬 높다. **라벨과 연관된 누락**이라 그대로 두면 양성이 편향된다.
- 원인: GitHub의 `apache/hadoop-hdfs`, `apache/hadoop-mapreduce`는 2009~2011년 분리 시기의 축소본(각 약 37 MB)이다. 해당 커밋은 다시 합쳐진 `apache/hadoop` 이력에 **같은 SHA로 존재**한다(표본 200 / 200 + 전체 3,100 / 3,100 확인).
- CSV에서 한 SHA가 둘 이상의 프로젝트로 등록된 경우는 0건이라 중복 계산 위험은 없다.

해결: 두 저장소에 git alternates(`objects/info/alternates` → `hadoop.git/objects`)를 설정했다. 데이터 복사가 없고 CSV의 저장소명과 ID가 그대로 유지되며, SHA가 같으므로 diff도 동일하다. 이후 build 출처가 하나로 유지되도록 **전체 build를 다시 실행**했다(부분 덧붙이기 대신).

### 최종 build (alternates 적용 후)

`selected 106,674 / accepted 106,659 / rejected 15`(모두 `patch_over_2MB_excluded`). 약 110분 소요. → `experiments/.cache/canonical/apachejit-full/` (2.3 GB)

## §2-3 split-public

```bash
riskbench split-public --records data/canonical/apachejit/records.jsonl --out data/splits/public \
  --train-before 2016-01-01T00:00:00Z --valid-before 2017-01-01T00:00:00Z --allow-retrospective
```
- `--allow-retrospective`: ApacheJIT CSV에는 라벨 확정 시각이 없다(`label_available_at=null`). README §2-3에 따라 **회고적 benchmark**로만 해석하며, 운영 backtest라고 부르지 않는다.
- 경계는 README와 `configs/protocol.example.yaml`의 값을 그대로 썼다(원 논문 split의 재현이 아님).
- 결과: split별 크기와 양성 수는 아래 표와 같다. 제외는 763건이다(`duplicate_patch` 759, `group_or_exact_patch_crosses_split` 4). 소요는 48초, 최대 RSS는 3.4 GB였다.

| split | n | 양성 | 양성 비율 |
|---|---|---|---|
| train (<2016) | 65,478 | 19,240 | 29.4% |
| valid (2016) | 10,480 | 3,078 | 29.4% |
| test (≥2017) | 29,938 | 5,796 | **19.4%** |

- **해석 주의**: test의 양성 비율이 낮다. 최근 커밋일수록 결함 유발이 발견될 시간이 짧았기 때문(라벨 우측 절단)일 가능성이 크다. AP처럼 기저율에 민감한 지표는 split 간에 직접 비교하지 않고, 같은 test 안에서 모델끼리만 비교한다.

## 범위 결정 — seed

`protocol.example.yaml`은 seed `[42, 43, 44]`를 요구하지만, 이번 실험은 **seed 42 한 번**만 실행한다. B 한 번이 MPS에서 최대 약 11시간이 걸리기 때문이다. seed 분산은 후속 실험으로 보고한다.

## §3-2 prepare

`riskbench prepare --dataset data/splits/public/{train,valid,test} --out data/views/public/{…} --tokenizer models/codebert-base` (3개 병렬 실행)
- 512 토큰 초과로 앞·뒤를 남기며 잘린 비율: train 60,755 / 65,478 (92.8%), valid 9,427 / 10,480 (90.0%), test 26,935 / 29,938 (90.0%).
- HF의 "sequence length > 512" 경고는 자르기 전 길이를 셀 때 나오는 것이다. 실제로 저장된 입력은 baseline이 512 이하인지 검사한다.
- 대부분의 변경이 잘린 입력으로 비교되므로, 긴 변경의 정보 손실이 이번 실험의 주된 한계다(README §3-2).

## §4 학습 (seed 42, 공개 train/valid만 사용)

### 정형 baseline
`riskbench train-tabular --public-train data/views/public/train --public-valid data/views/public/valid --out runs/models/tabular`
→ 선택된 C = 0.1, **public validation AP 0.588**

### A. frozen CodeBERT + linear head (MPS)
`riskbench train-codebert --mode frozen --model models/codebert-base … --epochs 5 --batch-size 8 --accumulation 4 --seed 42 --device mps`

| epoch | train loss | validation AP |
|---|---|---|
| 1 | 0.579 | 0.496 |
| 2 | 0.556 | 0.516 |
| 3 | 0.545 | 0.524 |
| 4 | 0.539 | 0.530 |
| 5 | 0.536 | **0.536** (선택) |

- epoch 5까지 AP가 계속 올라 **수렴하지 않았다**. README 기본값(5 epoch, head lr 1e-3)을 그대로 썼기 때문이다. 공개 validation 기준의 epoch/lr 탐색은 README가 허용하는 후속 작업이다. 이번 실험에서는 설정을 바꾸지 않는다.
- 임베딩 추출과 학습에 약 1시간이 걸렸다. artifact 크기는 494 MB로, frozen encoder 가중치가 포함되어 있다.
- **provenance 문제 (upstream 반영 후보)**:
  - `model.json`의 `training_seconds`는 36초로 기록된다. 타이머가 임베딩 캐시 추출 **이후**에 시작되기 때문이다(`neural.py` 252행). frozen 모드의 실제 비용이 기록에서 빠진다.
  - `resolved_revision`이 `null`이다. 로컬 디렉토리에서 불러오면 `config._commit_hash`가 없기 때문이다. 실제 revision은 `models/codebert-base/SOURCE.json`(`3b0952fe…`)에만 남는다. 가중치 파일 자체는 lock의 hash로 추적된다.

### B. full fine-tuning (MPS)
20:58경 시작. GPU 사용률 99%, GPU 메모리 약 9.8 GB로 앞선 벤치마크(9.1 GB)와 일치한다.

| epoch | 종료 | validation AP |
|---|---|---|
| 1 | 23:07 (2시간 9분, validation 포함) | 0.706 |
| 2 | 01:17 (2시간 10분) | 0.721 |
| 3 | 03:27 (2시간 10분) | 0.704 (하락 → 체크포인트는 epoch 2 유지) |
| 4 | 05:35 (2시간 8분) | 0.693 (2회 연속 개선 없음 → patience 2로 조기 종료) |

→ **선택: epoch 2, validation AP 0.721.** 학습 8.65시간, device `mps`. README 기본값(`--epochs 5 --patience 2`)을 그대로 썼다.

## 데이터 점검 (평가 전 계획, 결과를 보기 전에 기준을 정함)

### split 간 거의 같은 중복 — `scripts/leak_check.py`
baseline은 정확히 같은 patch만 제거한다. 그래서 공백, 파일 헤더, hunk 줄 번호를 무시하고 +/- 줄 내용만으로 비교하는 느슨한 키와, 정규화한 커밋 메시지로 추가 검사했다.

| eval split | diff 내용이 train과 같음 | 메시지가 train과 같음 | 그중 양성 |
|---|---|---|---|
| valid (10,480) | 58 (0.55%) | 20 (0.19%) | 각 1 |
| test (29,938) | 111 (0.37%) | 138 (0.46%) | 2 / 1 |

→ 합집합 249개(0.8%)이고 **거의 모두 음성**이다. B의 높은 AP를 설명할 만한 누수는 아니다. 민감도 분석용 ID 목록은 `runs/test_near_dup_ids.json`에 있다.

### 프로젝트만으로 예측 — "project prior" 기준선
train의 프로젝트별 양성 비율로만 점수를 매기면 **valid AP 0.559 / AUC 0.771, test AP 0.368 / AUC 0.708**이다. 정형 baseline의 valid AP 0.588과 거의 같다. 즉 이 benchmark에서는 "어느 프로젝트인가"만으로도 상당한 순위 성능이 나온다.

### ApacheJIT의 Hadoop 라벨 이상 (중요)
원본 CSV 기준:
- `apache/hadoop`: 2009~2019년 **12,964개 전부 buggy=False**.
- `apache/hadoop-hdfs`: 2012년 이후 약 90%가 buggy(예: 2013년 343/355). `apache/hadoop-mapreduce`도 비슷하다.
- 2012년 이후 HDFS와 MapReduce 커밋은 실제로는 통합된 `apache/hadoop` 저장소의 커밋이다(앞의 alternates 조사에서 SHA 3,100 / 3,100 확인). 결함 커밋은 hdfs/mapreduce로, 결함 없는 커밋은 hadoop으로 등록되어 있어, **데이터셋 구성 단계에서 라벨에 따라 project가 정해진 것으로 보인다.**
- 모델은 프로젝트나 diff의 파일 경로(`hadoop-hdfs-project/` 등)로 라벨을 추론할 수 있다. 이것은 코드 위험 판단이 아니라 데이터셋 구성의 흔적이다.
- 대응: 전체 test 결과와 함께, **Hadoop 계열 3개 프로젝트를 뺀 test**의 지표를 민감도 분석으로 보고한다. 이 기준은 test 결과를 보기 전에 정했다.

- A(0.536)나 정형 baseline(0.588) 대비 상승폭이 크다. 평가 후 split 간 누수 가능성(거의 같은 patch, 커밋 메시지 템플릿)을 점검한다. baseline은 완전히 같은 patch만 제거하고, 의미상 거의 같은 중복은 검사하지 않는다.

## §7 공개 test 비교

```bash
caffeinate -i python scripts/predict_suite.py --dataset data/views/public/test \
  --models rule tabular frozen finetune --model-root runs/models --device mps \
  --out runs/evaluation/public-test --bootstrap 500
```
- `--lock` 없음: 공개 test이므로 freeze하지 않았다(§6 내부 평가는 범위 밖). `--allow-label-transfer` 없음: 공개 test와 학습의 라벨이 모두 `bug_inducing_commit`이다.
- 추론 시간: A 약 27분, B 약 25분(MPS, batch 8), 이어서 bootstrap 500회. 05:36에 시작해 약 06:35에 끝났다.
- coverage는 네 모델 모두 100%이고 실패는 0이다. week cluster는 157개다.
- 결과 요약은 `RESULTS.md`, 원본 지표는 `results/metrics.{json,csv}`에 있다.

## 민감도 분석 (§8 해석)

```bash
python ../scripts/leak_check.py data/splits/public --write-exclusions runs/test_near_dup_ids.json
python ../scripts/sensitivity.py runs/evaluation/public-test data/splits/public runs/test_near_dup_ids.json runs/evaluation/public-test-sensitivity.json
python ../scripts/paired_bootstrap_subset.py runs/evaluation/public-test data/splits/public/test tabular finetune
```
- Hadoop 계열 제외와 거의 같은 중복 제외는 test 결과를 보기 전에 정했다(위 "데이터 점검"). 프로젝트별 AP는 전체 결과를 본 뒤 추가한 **사후 분석**이다.
- 핵심: AP(B) − AP(정형 LR)가 전체 test에서는 +0.052 [+0.031, +0.075]이지만, Hadoop 계열을 빼면 **−0.029 [−0.044, −0.012]로 역전**된다. 프로젝트별 AP 평균은 정형 LR 0.579 > rule 0.532 > B 0.510 > A 0.419다.
- `paired_bootstrap_subset.py`는 riskbench의 `evaluate`가 전체 집합만 bootstrap하기 때문에 따로 만든 스크립트다. cluster(ISO week), 500회, seed 42는 riskbench 기본값과 같다.
