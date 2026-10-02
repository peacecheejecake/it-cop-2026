# RUNLOG — 017-dl-d1-learning-curve

Study D1 (`codebert-diff-lab/docs/next-studies-plan.md` §3): 학습 데이터 비율 25/50/100%에서 validation AP. 탐색적, validation만 사용, test 접근 없음. 코드 `0bbf2a7`(`dataset.train_fraction`), 로컬 Mac(MPS). 명령은 `codebert-diff-lab/`에서 실행.

1. `uv sync --offline --extra neural --extra lgbm --extra dev`; `data/{snapshots,splits,evidence}/jitd4j-git1`을 016 worktree에서 복사.
2. `for p in 25 50 100; do diff-lab experiment run --study configs/studies/d1-learning-curve-f$p.yaml --models B0-LR,B1-TFIDF-S --seeds 42; done`
3. `for p in 25 50 100; do diff-lab experiment run --study configs/studies/d1-learning-curve-f$p.yaml --models B2-S --seeds 42,43,44; done` (device mps)

train 행 수: 25% 4,032 / 50% 8,143 / 100% 16,119.
