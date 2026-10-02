# RUNLOG — 018-dl-study-m

Study M stage 1 (`codebert-diff-lab/docs/next-studies-plan.md` §2, config `configs/studies/study-m-stage1.yaml`). 명령은 `codebert-diff-lab/`에서 실행. 시각은 UTC.

## 준비 (로컬, 코드 `a596057`)

1. `uv sync --offline --extra neural --extra dev`; `data/{snapshots,splits}/jitd4j-git1`을 016 worktree에서 복사.
2. `diff-lab diffllm prepare --config configs/studies/study-m-stage1.yaml --fulldiff ../../012-dl-diffllm/codebert-diff-lab/data/fulldiff/jitd4j-audit1 --tokenizer ../../.cache/models/qwen2.5-coder-7b-instruct-tokenizer --out data/diffllm-view/study-m-stage1`
   - train 16,129 / valid 5,445 / test 5,459 (v4와 같은 membership). 잘린 비율 32.5%, 내용 토큰 평균 1,012.
   - 012의 fulldiff는 커밋 SHA 기준 `git show -U3` 결과라 snapshot과 무관하다. git1에 있는 change_id만 쓴다.

## 1단계: CPT 효과 점검 (2026-10-03 19:4x–20:0x)

- Pod `6zeosj09ffxish`(`jit018-probe`), RTX 4090 24 GB, Secure, $0.74/h. `--ports 22/tcp`를 처음부터 지정.
- 번들 `.cache/bundles/jit018-a596057.tar.gz`(코드 + view + 012의 CPT adapter·cpt.json), sha256 `e6017675…aab`, pod에서 일치.
- `scripts/pod_probe.sh`: env 구축, Qwen2.5-Coder-7B-Instruct `c03e6d35`를 전체 받음(고정 파일 해시는 `_model_path`가 확인), 이어서
  `diff-lab diffllm probe-cpt --config configs/studies/study-m-stage1.yaml --view data/diffllm-view/study-m-stage1 --cpt qwen7b-apache50m --rows 512 --out results/probe-cpt.json --artifacts-dir artifacts`
- 결과 회수(`results/probe-cpt.{json,parquet}`, `logs/probe.log`) 후 pod 삭제. 잔액 $2.22 → $2.10.
