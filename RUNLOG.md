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

## 다음 단계 (사용자 승인/외부 리소스 필요)

README 순서상 다음 단계들은 이 샌드박스에서 바로 진행할 수 없고, 각기 다른 승인/리소스가 필요해 **진행 전에 확인이 필요함**:

1. ~~`.[neural,dev]` 설치 + `scripts/smoke.py --neural`~~ — 완료 (위 MPS 점검 참고).
2. §2 `riskbench fetch-apache` / `build-apache --allow-network` — Zenodo ZIP 다운로드 + 공개 Git repository mirror clone. 외부 네트워크 허용 여부와 용량(디스크) 확인 필요.
3. §3 `riskbench download-codebert` — HuggingFace에서 CodeBERT 가중치 다운로드.
4. §4 실제 학습(A/B) — 이 맥의 MPS로 진행 가능(`--device mps` 명시 권장, `--allow-cpu-training` 불필요). 전체 공개 train 규모에 따라 수 시간~수십 시간 소요.
5. §5 LLM(C/D) — 승인된 OpenAI-compatible endpoint와 `LLM_API_KEY` 필요. Gemini/Anthropic native API는 baseline이 지원하지 않음.
6. §6 내부 데이터 연결 — 실제 내부 export(manifest.csv, patches/)가 있어야 함. 사내 정책상 원본 코드 반출 금지 시 외부 LLM API로 대체 불가(README 명시).

AGENTS.md §1.5에 따라 이 단계들은 조용히 mock으로 대체하지 않고, 실행 전에 필요한 리소스/승인을 먼저 확보한다.
