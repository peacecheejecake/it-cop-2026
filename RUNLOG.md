# RUNLOG — 016-dl-v4-gitlines

study `public-comparison-v4-gitlines`: v3 variant를 누출 없는 snapshot `jitd4j-git1`에서 다시 실행한다. 배경은 `codebert-diff-lab/docs/leak-finding-2026-10-02.md`, 계획은 `docs/v4-correction-plan.md`. 명령은 모두 `codebert-diff-lab/`에서 실행했다.

## 로컬 (2026-10-02, 코드 `8db0d30`)

1. `uv sync --offline --extra neural --extra lgbm --extra dev`
2. `data/snapshots/jitd4j-audit1`을 011 worktree에서 복사.
3. `diff-lab data rebuild-git --snapshot-id jitd4j-audit1 --mirrors ../../.cache/git/jd4j --out-snapshot-id jitd4j-git1`
   - 27,209행. 미러에 없어 빠진 커밋 110개(정상+버그: train 54+15, valid 14+6, test 4+17).
   - split: train 16,305(양성 1,375) / valid 5,445(461) / test 5,459(458).
4. `diff-lab data audit --snapshot-id jitd4j-git1`, `diff-lab data split --snapshot-id jitd4j-git1 --split-id upstream-clean2`, `diff-lab evidence build --study configs/studies/public-comparison-v4-gitlines.yaml`
   - EvidenceView 27,033행, 잘린 것 10,498, 토큰 중앙값 298.
5. `diff-lab experiment run --study … --models B0-LR,B0-LGBM,B1-TFIDF-S --seeds 42`
   - validation AP: B0-LR 0.3021(`4e86a5d1925642fa`), B0-LGBM 0.2396(`85cfc20d82768ea6`), B1-TFIDF-S 0.3447(`d08b71b35fbaf0c4`).
6. `python ../../../tools/representation_leak_check.py data/snapshots/jitd4j-audit1 data/snapshots/jitd4j-git1`
   - 낮은 coverage의 test AP: 패키지 0.324, git 0.089(기저율 0.084~0.087).

## Pod (H100. 사용자 승인: 새 H100 pod, 이후 같은 pod에서 9개 variant 계속)

- Pod `2spnrcaqwyaa55`(`jit016-v4-gitlines`), Secure, H100 80GB HBM3, $3.49/h, 08:09 UTC 생성. 생성 전 잔액 약 $29.6.
- 실수: `--ports` 없이 만들어 SSH가 열리지 않았다. `runpodctl pod update --ports 22/tcp`로 고쳤다(약 12분 지연).
- 번들 `.cache/bundles/jit016-8db0d30.tar.gz`(코드 `8db0d30` + git1 snapshot/split/evidence + CodeBERT), sha256 `5af9eab1…43aa`, pod에서 일치.
- `scripts/pod_setup.sh` → `scripts/pod_run.sh`(B3-S seed 42/43/44, 08:38 UTC 시작) → `scripts/pod_run2.sh`(B2-S, B4-S, B5-S).
- 편차: 9개 variant 등록본(main `7a25603`)은 번들 뒤에 만들었다. pod에서는 B3-S 루프가 끝난 뒤 설정 파일만 교체한다. 코드는 `8db0d30` 그대로이고, B0/B1/B3의 scoped config hash가 등록 전후로 같은 것을 확인했다.
- 013·014 pod는 08:50 UTC 무렵 사용자가 중단했다.

(결과는 run이 끝나는 대로 이어서 적는다.)
