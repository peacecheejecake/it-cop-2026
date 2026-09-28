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

## 다음 단계 (사용자 승인/외부 리소스 필요)

README 순서상 다음 단계들은 이 샌드박스에서 바로 진행할 수 없고, 각기 다른 승인/리소스가 필요해 **진행 전에 확인이 필요함**:

1. `.[neural,dev]` 설치 + `scripts/smoke.py --neural` — torch/transformers 다운로드 필요 (용량 큼, PyPI 접근은 가능해 보이므로 시도 가능).
2. §2 `riskbench fetch-apache` / `build-apache --allow-network` — Zenodo ZIP 다운로드 + 공개 Git repository mirror clone. 외부 네트워크 허용 여부와 용량(디스크) 확인 필요.
3. §3 `riskbench download-codebert` — HuggingFace에서 CodeBERT 가중치 다운로드.
4. §4 실제 학습(A/B) — GPU 유무, CPU라면 `--allow-cpu-training` 명시적 승인 필요.
5. §5 LLM(C/D) — 승인된 OpenAI-compatible endpoint와 `LLM_API_KEY` 필요. Gemini/Anthropic native API는 baseline이 지원하지 않음.
6. §6 내부 데이터 연결 — 실제 내부 export(manifest.csv, patches/)가 있어야 함. 사내 정책상 원본 코드 반출 금지 시 외부 LLM API로 대체 불가(README 명시).

AGENTS.md §1.5에 따라 이 단계들은 조용히 mock으로 대체하지 않고, 실행 전에 필요한 리소스/승인을 먼저 확보한다.
