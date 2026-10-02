# Handoff — 2026-10-02 (v4-gitlines 재측정, 사내 반입 준비)

다른 세션에서 이어서 작업하기 위한 기록이다. 끝난 항목은 지우고 갱신한다.

## 무슨 일이 있었나

- 사내 반입용 git 추출기를 검증하다가, **JIT-Defects4J 패키지의 텍스트가 라벨에 따라 다르게 만들어져 있다**는 것을 발견했다(버그 커밋은 일부 파일만 담김).
- v3 B3-S에 git 텍스트를 넣으면 test AP가 0.606에서 0.164로 떨어진다. v2·v3의 텍스트 variant 결과, 013의 EvidenceView arm, 014는 결론으로 쓸 수 없다. B0와 012는 영향이 없다.
- 근거와 다른 에이전트와의 검토 내용: `codebert-diff-lab/docs/leak-finding-2026-10-02.md`
- 재측정 계획과 사전 규칙: `codebert-diff-lab/docs/v4-correction-plan.md`
- 013·014 pod는 사용자가 중단했다. 재실행은 별도 승인 사항이다(계획 문서 §5).

## 커밋 위치

| 브랜치 | 내용 |
|---|---|
| `main` | `diff_lab.gitextract`, `label_eval`, `adapters/jit_defects4j_git`, CLI(`internal extract / label-sheet / label-eval`, `data rebuild-git`, `study export --variants/--seeds`), study `public-comparison-v4-gitlines`(9개 variant), `tools/representation_leak_check.py`, 문서 |
| `exp/015-dl-internal-pack` | 반입 패키지 스크립트(`pack/`, `scripts/build_pack.sh`, `scripts/fetch_wheels.py`, `scripts/validate_extract.py`), RUNLOG/RESULTS |
| `exp/016-dl-v4-gitlines` | v4 실행 기록(RUNLOG), pod 스크립트 |

## 지금 돌고 있는 것 (과금 중)

- **Pod `jit016-v4-gitlines`** (`2spnrcaqwyaa55`), H100 80GB, $3.49/h, 2026-10-02 08:09 UTC 생성. 접속 정보는 `runpodctl ssh info 2spnrcaqwyaa55`.
- 작업 디렉토리 `/workspace/jit016/codebert-diff-lab`. 진행 표시는 `logs/steps`(variant·seed가 끝날 때마다 한 줄).
  1. `/workspace/pod_run.sh`: B3-S seed 42/43/44. 08:38 UTC 시작. 끝나면 `done`.
  2. `/workspace/pod_run2.sh`: `done`을 기다렸다가 설정 파일을 9개 variant 등록본(`/workspace/v4.yaml`, main `7a25603`)으로 바꾸고 B2-S, B4-S, B5-S를 seed 42/43/44로 실행. 끝나면 `done2`.
- 번들: `experiments/.cache/bundles/jit016-8db0d30.tar.gz`, sha256 `5af9eab1…43aa`(pod에서 일치 확인). pod의 `CODE_SHA`는 `8db0d30`이고, 설정 파일만 `7a25603`의 것으로 바뀐다(코드는 같다).
- 잔액: 08:55 UTC 무렵 $25.08. 남은 작업 추정 5~6시간, $18~22. **여유가 $5 안팎이다.**

## 다음 순서

1. run이 끝날 때마다 작은 파일부터 회수한다(run.json, metrics, predictions, logs). 가중치(`checkpoints/best-*`)는 마지막에 받는다. 회수 위치는 `experiments/016-dl-v4-gitlines/codebert-diff-lab/artifacts/runs/`.
   - 로컬에는 이미 B0-LR(`4e86a5d1925642fa`), B0-LGBM(`85cfc20d82768ea6`), B1-TFIDF-S(`d08b71b35fbaf0c4`) run이 있다. pod의 run과 합쳐서 freeze한다.
2. `done2` 뒤에 L0-S, L1-S를 실행한다(사용자 승인됨).
   - Qwen2.5-Coder-7B-Instruct revision `c03e6d35…`를 pod의 `experiments/.cache/models/qwen2.5-coder-7b-instruct`에 받고(약 15 GB), 설정에 고정된 파일 sha256과 맞는지 확인한다. 011 RUNLOG에 같은 절차가 있다.
   - 시작 전에 잔액이 $9 이상인지 확인한다. 모자라면 사용자에게 알린다.
3. `study freeze` → `experiment test`(1회) → `study report`. 로컬 MPS에서도 된다(neural 재현 허용 오차 2e-3).
4. 결과 문서 `codebert-diff-lab/docs/results-v4-<date>.md` 작성, 계획 문서 §6의 문서 정정, AGENTS.md 실험 표 갱신.
5. 사내 반입 zip(015 worktree):
   - 반입 모델 규칙은 계획 문서 §4. `study export --variants <선택> --seeds <선택>`으로 번들을 만든다.
   - `pack/README.md`의 모델 설명·참고 수치·`<!-- FIDELITY -->` 자리를 v4 값으로 고친다. `scripts/build_pack.sh`의 번들 경로와 `PROVENANCE.json` 내용도 v4로 고친다.
   - `scripts/fetch_wheels.py`는 `nvidia-nccl-cu12==2.29.3`에서 실패했다(55개 wheel까지 받음, `pack-build/wheels`). 플랫폼 태그 문제로 보이며 아직 고치지 않았다.
   - 사내 서버는 시스템 Python이 3.9다. 패키지는 Python 3.11을 동봉하므로 상관없다. **glibc 버전(2.28 이상 필요)과 NVIDIA 드라이버 버전은 사용자에게 확인을 요청해 둔 상태다.**
   - `pack/smoke.sh`는 macOS에서만 통과했다. Linux에서는 돌려 보지 않았다.
6. 끝나면 pod를 삭제하고 RUNLOG에 비용을 적는다.

## 현재까지의 v4 수치 (validation, git 텍스트)

| variant | v3 (패키지 텍스트) | v4 |
|---|---|---|
| B0-LR | 0.319 | 0.302 |
| B0-LGBM | 0.211 | 0.240 |
| B1-TFIDF-S | 0.546 | 0.345 |
| B3-S | 0.687 | 실행 중 |
