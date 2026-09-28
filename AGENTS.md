# AGENTS.md — jit-zero-shot 실험 운영 규칙

이 문서는 `baseline/`을 기반으로 `experiments/` 아래에서 진행되는 모든 실험에 적용된다.
루트/`experiments/`의 `CLAUDE.md`는 이 파일을 `@AGENTS.md`로 import하므로 모든 agent 세션에 자동 로드된다.

`baseline/`은 고정된 출발점이다. 실험을 위해 `baseline/` 자체를 직접 수정하지 않는다.
baseline 코드의 실제 버그를 고치는 경우가 아니면, 실험용 변경은 항상 해당 실험의 worktree 안에서만 한다.

## 1. 실험 기본 규칙

### 1.1 명명 규칙
- 디렉토리: `experiments/<NNN>-<short-slug>/` (예: `experiments/001-baseline-smoke`)
- 브랜치: `exp/<NNN>-<short-slug>`
- 이전 실험을 이어가는 경우가 아니면 항상 `main`에서 분기한다.

### 1.2 디렉토리 + worktree 생성 절차
```bash
git branch exp/<NNN>-<slug> main
git worktree add experiments/<NNN>-<slug> exp/<NNN>-<slug>
```
- `main`의 작업 트리 파일을 직접 건드리는 방식으로 worktree를 만들지 않는다. 항상 새 브랜치.
- worktree 생성 후 1.3의 절차로 그 안에서 환경을 새로 구축한다.
- `experiments/*/`는 루트 `.gitignore`에 등록되어 있다 — 실험 worktree의 내용은 `main`의 히스토리에 커밋되지 않는다. 각 실험은 자신의 브랜치로 독립적으로 관리되고, baseline으로 승격할 내용이 있으면 별도로 diff/PR 형태로 review한다.
- 실험이 끝나 더 이상 필요 없으면 `git worktree remove experiments/<slug>`로 정리한다(디렉토리를 그냥 `rm -rf`하지 않는다).

### 1.3 환경 구축
`experiments/<slug>/baseline/`으로 이동해 README 1절의 설치 절차를 그대로 따른다.
```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'         # neural 실험이면 '.[neural,dev]'
riskbench --help
pytest -q
python scripts/smoke.py --out runs/smoke-core   # neural이면 --neural 추가
```
- venv는 실험마다 독립적으로 만든다(공유하지 않음) — 실험 간 버전 drift를 피한다.
- `requirements-tested-core.txt`의 버전은 출발점일 뿐이며, 실제 설치 후 `uv pip freeze`로 그 실험만의 lock을 별도로 남긴다.

### 1.4 데이터/모델 캐시 재사용
- ApacheJIT ZIP, git mirror clone, CodeBERT 가중치 등은 용량이 크고 재다운로드 비용이 크다.
- 실험 간 공유 캐시를 `experiments/.cache/`(git-ignored) 아래에 두고, 각 실험의 `data/git`, `models/`에는 심볼릭 링크로 연결한다. 실험마다 수 GB짜리 repo mirror를 통째로 복제하지 않는다.
- 공유 캐시 구조 (전체 ApacheJIT 기준, 새 실험은 이것부터 링크한다):
  ```text
  experiments/.cache/
    raw/apachejit-v2/            # fetch-apache 결과 (Zenodo v2, MD5 pin 일치)
    git/apache/<repo>.git        # 15개 공개 저장소 bare/mirror clone
    models/codebert-base/        # download-codebert 결과 (SOURCE.json에 HF revision)
    canonical/apachejit-full/    # 전체 build-apache 결과 (records/rejected/build_report)
  ```
  ```bash
  # experiments/<slug>/baseline/ 에서
  mkdir -p data && ln -s ../../../.cache/raw data/raw && ln -s ../../../.cache/git data/git
  mkdir -p data/canonical && ln -s ../../../../.cache/canonical/apachejit-full data/canonical/apachejit
  ln -s ../../.cache/models models
  ```
- 공유 캐시는 읽기 전용으로 취급한다. split/view/모델/실행 결과(`data/splits`, `data/views`, `runs/`)는 프로토콜마다 달라지므로 실험 디렉토리 안에 만든다.
- `build-apache --allow-network`의 clone은 저장소당 1200초 제한이 있어, 느린 네트워크에서는 대형 저장소(hadoop 등)가 통째로 `repository_clone_failed`로 빠진다. 저장소는 캐시에 미리 `git clone --bare`로 받아 두고 build-apache는 `--allow-network` 없이 실행한다(README §2-2의 오프라인 경로).
- `hadoop-hdfs.git`, `hadoop-mapreduce.git`은 GitHub 축소본이라 ApacheJIT 커밋의 60~80%가 없다. 이 커밋들은 `hadoop.git`에 같은 SHA로 있으므로 캐시의 두 저장소에 `objects/info/alternates`로 `hadoop.git/objects`를 연결해 두었다. 이 연결을 지우면 라벨과 연관된 누락(누락분의 88%가 buggy)이 다시 생긴다.
- 캐시를 심링크로 재사용하더라도 baseline의 hash/provenance 검증(`build_report.json`, `SOURCE.json` 등)은 그대로 통과해야 한다 — 캐시된 파일을 손으로 고쳐서 검증을 우회하지 않는다.

### 1.5 실행 원칙 (README의 철학을 그대로 따른다)
- README의 명령이 실패하거나 전제 조건(GPU, 승인된 LLM endpoint, 승인된 내부 데이터 반입)이 없으면 멈추고 보고한다. mock/demo 결과(`demo-random`, `demo-byte`, `MOCK-NO-LLM`)를 실제 성능인 것처럼 대체하지 않는다.
- README의 단계 순서(설치 → 공개 데이터 구축 → 공통 입력 준비 → 학습 → LLM 평가 → 내부 데이터 연결 → freeze → 통합 비교 → 결과 해석)를 따른다. 의도적으로 순서를 바꾸거나 생략하면 실험 디렉토리에 그 이유를 남긴다.
- 공개 데이터만 학습/모델선택에 쓴다는 원칙을 어떤 코드 경로에서도 깨지 않는다. 내부 데이터는 추론·평가 전용.
- `--allow-partial`, `--allow-label-transfer`, `--allow-retrospective`, `--allow-cpu-training` 같은 플래그는 실행 로그에 왜 필요했는지 명시한다. 조용히 기본값처럼 켜지 않는다.
- 매 실행의 정확한 명령/플래그/seed를 실험 디렉토리의 `RUNLOG.md`에 남겨 재현 가능하게 한다.
- 내부 test를 보고 나서 epoch/feature/prompt/example set을 바꾸면 그 test는 더 이상 독립적인 final test가 아니다 — README 8절 규칙을 그대로 지킨다.

### 1.6 산출물 관리
- `data/`, `models/`, `runs/`는 각 worktree 안에서 git-ignore 대상이다. 큰 바이너리 산출물을 커밋하지 않는다.
- `experiment.lock.json`, `metrics.json`/`metrics.csv`, 그리고 실험 요약 `RESULTS.md`(baseline 대비 무엇이 달랐고 핵심 지표가 무엇인지)는 실험 디렉토리에 남긴다.

## 2. Subagent orchestration

- 오래 걸리거나 blocking되는 단계(repo mirror clone, CodeBERT 학습, LLM batch 평가)는 메인 스레드를 막지 않도록 백그라운드 Bash(`run_in_background`)나 forked agent로 실행한다. 짧은 간격으로 polling하지 않는다 — 완료 알림을 기다린다.
- 대화 맥락이 필요한 열린 조사(예: "모델 B의 AP가 baseline보다 낮은 이유")는 `Agent(subagent_type: "fork")`를 쓴다. 중간 로그를 메인 컨텍스트에 남기고 싶지 않을 때 특히 유용하다.
- 맥락 없이도 되는, 범위가 명확한 실행 작업(예: "train/valid/test 세 split에 대해 prepare 실행")은 새 general-purpose agent에 위임해도 된다 — 이 agent는 대화 기록이 없으므로 필요한 경로/명령/플래그를 프롬프트에 전부 포함한다.
- 데이터 의존성이 있는 단계는 병렬로 돌리지 않는다(같은 split에 대해 `build-apache`+`split-public`+`prepare`가 끝나기 전에 `train-codebert`를 시작하지 않음). 서로 독립적인 단계(예: 서로 다른 holdout repo 구성, 또는 public-valid에서의 C vs D LLM 점검)는 병렬화 가능하다.
- GPU/장시간 학습 작업을 subagent에 맡기기 전에 device 가용성과 예상 소요 시간을 먼저 확인한다. `--allow-cpu-training`처럼 명시적 승인이 필요한 플래그는 subagent가 임의로 켜지 않고 반드시 위로 보고한다.
- 장시간 단계는 진행률이 보이게 실행한다. riskbench는 build-apache는 끝날 때, train-codebert는 epoch마다만 출력하므로, baseline 코드를 바꾸지 않는 래퍼를 쓴다: `experiments/001-baseline-smoke/scripts/build_apache_with_progress.py`, `train_with_progress.py`(N 스텝마다 진행률/처리량/ETA). 래퍼 없이 시작한 학습은 `watch_training.sh <log>`로 epoch 경과 시간 기반 추정치를 본다.
- 모든 subagent는 실행한 정확한 명령, 산출물 경로, 그리고 1.5의 원칙 중 어겨야 했던 부분이 있다면 이를 보고한다 — 에러를 조용히 덮지 않는다.

## 3. Python best practices

- Python 3.11+, `src/riskbench/*`의 기존 스타일(type hints, dataclass, bare except 금지)을 따른다.
- 의존성 관리는 `uv`를 쓰고, 변경은 해당 실험의 `pyproject.toml`에 국한한다. baseline으로 역반영할 목적이 아니라면 `baseline/pyproject.toml`은 건드리지 않는다.
- 코드를 건드린 단계는 완료로 치기 전에 `pytest -q`와 `python scripts/smoke.py --out runs/smoke-core`(neural 코드를 건드렸으면 `--neural`도)를 실행한다.
- 조용한 fallback, 넓은 `except`, 하드코딩된 매직 경로를 쓰지 않는다 — baseline 자체의 철학(데이터/라벨/설정이 없으면 안전해 보이는 기본값 대신 크게 실패)과 같다.
- 실험 전용 스크립트는 `baseline/src`를 실험마다 포크하는 대신 `experiments/<slug>/scripts/`에 작은 단일 목적 스크립트로 둔다. `src/riskbench`에서 진짜 버그/개선을 발견하면 실험 브랜치에서 고치고 upstream 반영 대상으로 기록한다 — 실험마다 조용히 갈라지지 않는다.
- 코드에 "무엇을 하는지" 설명하는 주석을 달지 않는다. 자명하지 않은 WHY(숨은 제약, 특정 버그 우회 등)만 한 줄로 남긴다.

## 4. 참고 문서

- `baseline/README.md` — 실행 순서의 단일 진실 소스(source of truth)
- `baseline/VERIFICATION.md` — 실제 검증된 범위와 검증되지 않은 부분
- `baseline/docs/DATA_CONTRACT.md`, `baseline/docs/SECURITY_AND_LIMITS.md` — 내부 데이터 반입 형식과 제약
