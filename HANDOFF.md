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

## 현재 상태 (2026-10-03 00:45 KST)

- **v4 재측정 완료.** freeze `7340ee71c876c8b7`(25 run), public test 1회. 결과는 `exp/016-dl-v4-gitlines`의 `RESULTS.md`, `results/report.json`.
  - test AP: B2-S 0.264, B3-S 0.253, B1 0.247, B4-S 0.243, B5-S 0.243, B0-LR 0.224, B0-LGBM 0.182, L1 0.157, L0 0.126.
- **Pod는 삭제했다.** 켜져 있는 pod 없음. 잔액 $2.32.
- run·가중치·freeze·test 결과는 `experiments/016-dl-v4-gitlines/codebert-diff-lab/{artifacts,freeze,test-results,report}`에 있다(git-ignore, pod와 sha256 일치 확인).

## 다음 순서

1. 결과 문서 `codebert-diff-lab/docs/results-v4-2026-10-02.md` 작성, 계획 문서(`v4-correction-plan.md`) §6의 문서 정정.
2. 사내 반입 zip(015 worktree):
   - 반입 모델: 규칙에 따라 **B2-S seed 44**(run `d4987435844df076`, test AP 0.260). B0-LR 동반 반입을 권고했고 사용자 결정 대기.
   - `study export --variants B2-S[,B0-LR] --seeds 44`를 016 worktree의 artifacts/freeze로 실행해 번들을 만든다.
   - `pack/README.md`의 모델 설명·참고 수치·`<!-- FIDELITY -->` 자리, `scripts/build_pack.sh`의 번들 경로와 `PROVENANCE.json`을 v4로 고친다. `jit.sh predict`의 `--variants B3-S`도 바꾼다.
   - `scripts/fetch_wheels.py`는 `nvidia-nccl-cu12==2.29.3`에서 실패했다(55개 wheel까지 받음). 아직 고치지 않았다.
   - 사내 서버는 시스템 Python 3.9(패키지가 3.11을 동봉하므로 무관). **glibc 버전(2.28 이상 필요)과 NVIDIA 드라이버 버전은 사용자 확인 대기.** B2-S는 CPU로도 돌릴 수 있다.
   - `pack/smoke.sh`는 macOS에서만 통과했다.
3. 후속 연구 초안: `codebert-diff-lab/docs/next-studies-plan.md`(다른 모델, 더 큰 데이터셋; 미등록·미승인).
4. 로컬 커밋 중 아직 푸시하지 않은 것이 있다(`main`, `exp/016`). 이 세션에서는 push가 자동 권한 검사에 막혀 사용자가 직접 푸시했다.
