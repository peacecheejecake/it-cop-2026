# Handoff — 2026-10-03 (다른 Mac에서 이어서 작업)

다른 세션·다른 기기에서 이어서 작업하기 위한 기록이다. 끝난 항목은 지우고 갱신한다.

## 0. 작업 규칙 (이전 세션에서 사용자가 정한 것)

- **사용자에게 시각을 말할 때는 한국시간(KST)으로.** 저장소 기록(RUNLOG 등)은 UTC 그대로.
- GPU pod는 만들기 전에 GPU 종류·예상 비용을 알리고 확인받는다. 잔액이 바닥나면 Runpod가 pod를 즉시 종료한다(AGENTS.md §5).
- 푸시는 사용자가 요청할 때 `origin`, `lvu` 두 곳에 한다.

## 1. 새 Mac 준비

1. 저장소 clone 후 `git fetch --all`. 원격: `origin` (peacecheejecake/it-cop-2026), `lvu` (kb-lvu/it-cop-experiments). 저장소 루트가 곧 이 프로젝트(`jit-zero-shot`)다. 이전 Mac에서는 `~/it-cop-2026/labs/jit-zero-shot`에 있었다.
2. **git 밖의 데이터 옮기기:** 이전 Mac의 `experiments/.cache/handoff-2026-10-03/`(약 14 GB, tar 12개 + `SHA256SUMS` + `ITEMS.txt`)를 새 Mac으로 복사한다(외장 디스크, AirDrop, rsync 등). 그다음 새 Mac의 저장소 루트에서:
   ```bash
   tools/handoff_transfer.sh restore <복사한 폴더>
   ```
   - 해시를 검사한 뒤 worktree 5개(012, 015, 016, 017, 018)를 브랜치에서 만들고 데이터를 제자리에 푼다.
   - 담긴 것은 `ITEMS.txt` 참고. 이전 study(v2/v3, 011 등)의 산출물은 누출된 입력으로 만든 것이라 옮기지 않았다(Google Drive `gdrive:jit-zero-shot-weights/`에 일부 백업이 있다).
3. 각 worktree의 `codebert-diff-lab/`에서 환경 구축(사내 TLS 프록시 때문에 `--system-certs`):
   ```bash
   uv python install 3.11 --system-certs
   uv sync --system-certs --extra neural --extra lgbm --extra dev
   uv run pytest -q          # main 기준 104개 통과
   ```
   B0-LGBM은 macOS에서 `brew install libomp`가 필요하다.
4. Runpod: `runpodctl` 설치·로그인, `runpodctl doctor`로 SSH 키를 만들고 계정에 등록한다. 이전 Mac의 키(`~/.runpod/ssh/runpodctl-ssh-key`)는 옮기지 않는다.
5. 선택: `rclone`(Google Drive 백업용), Hugging Face 토큰(Llama 3.1을 쓸 경우만).
6. **Claude 메모리는 기기별이다.** 위 0절의 규칙을 새 세션에서도 지키도록 이 문서를 먼저 읽힌다.

## 2. 지금까지의 결론

- **JIT-Defects4J 패키지의 텍스트가 라벨에 따라 다르게 만들어져 있었다**(버그 커밋은 일부 파일만 담김). v2·v3의 텍스트 모델 결과는 결론으로 쓸 수 없다. 근거와 다른 에이전트와의 검토: `codebert-diff-lab/docs/leak-finding-2026-10-02.md`.
- **v4 재측정 완료** (`exp/016-dl-v4-gitlines`, freeze `7340ee71c876c8b7`, public test 1회). 누출 없는 git 텍스트 기준 test AP:
  - B2-S 0.264, B3-S 0.253, B1 0.247, B4-S 0.243, B5-S 0.243, B0-LR 0.224, B0-LGBM 0.182, L1 0.157, L0 0.126.
  - 텍스트 모델 간 차이는 모두 구간이 0을 포함한다. v3의 "full fine-tuning이 최고"는 재현되지 않았다.
- **D1 학습 곡선** (`exp/017`): B0-LR·B1은 데이터 50%→100%에서 평평하다.
  - B2-S는 **Mac에서 학습하면 H100보다 체계적으로 낮다**: 같은 입력·초기값, seed 12개에서 0.313 ± 0.007 대 0.372.
  - 원인은 미확인이다. 추론은 장치와 무관하다. **인코더 계열 학습은 반드시 CUDA(H100)에서 한다.**
- **diff CPT 점검** (`exp/018`): 012의 50M 토큰 CPT는 JD4J diff의 loss를 7% 낮췄다. 다만 버그·정상 커밋에 똑같이 작용했다(ROC-AUC 0.487).

## 3. 다음 할 일 (우선순위 순)

1. **Study M 1단계** (`exp/018-dl-study-m`, 설정 `codebert-diff-lab/configs/studies/study-m-stage1.yaml`).
   - 내용: Qwen2.5-Coder-7B R-base와 R-diff(012 CPT adapter 재사용)를 seed 42/43/44로. v4와 같은 cohort, H100.
   - 예상 약 15시간, 약 $52. **잔액이 $2.10이라 충전이 필요하다.**
   - 입력 view(`data/diffllm-view/study-m-stage1`)와 CPT adapter(`012…/artifacts/diffllm/cpt/adapter` → pod에서 `artifacts/diffllm/cpt/qwen7b-apache50m/adapter`)는 준비돼 있다. pod 번들 만드는 법은 018 RUNLOG의 probe 단계를 따른다.
   - 명령: `diff-lab diffllm arm --config … --arms R-base,R-diff --view …` → `diffllm freeze` → `diffllm test`. 그다음 v4 test 예측과 합친 report를 만든다(**report 스크립트는 아직 없음**).
   - 결과에 따라: R-diff − R-base가 확실히 양수면 CPT 말뭉치 확대 arm, 아니면 학습 목표나 LoRA 용량을 다음 변수로(`next-studies-plan.md` §2).
2. **Study M 2단계:** Qwen 14B, Gemma 4 12B(Apache-2.0, 승인 불필요), Llama 3.1 8B(라이선스 확인·토큰 필요). seed 수는 미정(권고: 선별 단계 1 seed, 상위 모델만 3 seed). Gemma 4는 고정된 transformers 4.57.6에서 로딩 확인이 먼저다.
3. **완전히 다른 test set으로 모든 모델 평가** — 사용자 답변 대기. 제안 내용:
   - ApacheJIT(Hadoop 제외)를 git에서 재구축하고, v4 freeze 모델과 Study M 모델을 재학습 없이 적용한다.
   - ApacheJIT 저장소는 다시 받아야 한다(약 20 GB).
   - **012 CPT 말뭉치 저장소 6개(activemq, cassandra, groovy, kafka, zeppelin, zookeeper)는 ApacheJIT와 겹치므로 빼거나 따로 보고한다.**
4. **사내 반입 zip** (`exp/015-dl-internal-pack`) — 사용자 답변 대기:
   - B2-S seed 44(run `d4987435844df076`)만 넣을지, B0-LR도 함께 넣을지.
   - 사내 서버의 glibc 버전(2.28 이상 필요)과 GPU 유무.
   - 할 일: `study export --variants B2-S[,B0-LR] --seeds 44`(016의 artifacts/freeze 사용), `pack/README.md`·`jit.sh`(`--variants`)·`build_pack.sh`·`PROVENANCE.json`을 v4 기준으로 수정.
   - `scripts/fetch_wheels.py`의 `nvidia-nccl-cu12` 다운로드 실패도 고쳐야 한다. GPU 없이 가면 CPU용 torch로 바꾸면 되고 zip이 작아진다.
5. 문서 정리: `codebert-diff-lab/docs/results-v4-<date>.md` 작성, `v4-correction-plan.md` §6의 정정 항목.
6. B2-S 학습 플랫폼 차이: CUDA에서 `exp/017/scripts/b2_head_platform_check.py --seeds`를 돌려 확인한다(15분). Study M pod에서 함께 하면 된다.

## 4. 브랜치·기록 위치

| 브랜치 | 내용 |
|---|---|
| `main` | 코드(gitextract, label_eval, rebuild-git, train_fraction, diffllm per-seed/probe-cpt), study 설정, 문서, `tools/` |
| `exp/015-dl-internal-pack` | 사내 반입 패키지 스크립트(`pack/`, `scripts/`) |
| `exp/016-dl-v4-gitlines` | v4 RESULTS/RUNLOG, `results/report.json`, `results/freeze.json` |
| `exp/017-dl-d1-learning-curve` | D1 RESULTS, B2 플랫폼 진단 스크립트 |
| `exp/018-dl-study-m` | Study M RUNLOG/RESULTS, CPT 점검 결과 |

계획 문서: `codebert-diff-lab/docs/next-studies-plan.md`(Study M, Study D), `v4-correction-plan.md`.

## 5. 비용·자원 상태 (2026-10-03 13:0x KST)

- Runpod 잔액 $2.10, 켜져 있는 pod 없음.
- 측정값(H100):
  - CodeBERT fine-tuning epoch 3.1분.
  - Qwen 7B 위험 분류 학습 seed당 2.0~2.6시간, 7B CPT 50M 토큰 2.9시간.
  - 한 GPU에 여러 run을 동시에 돌려도 처리량은 늘지 않았다(`next-studies-plan.md` §1a).
- pod `/workspace` 볼륨(20 GB)은 체크포인트로 금방 찬다. 산출물은 컨테이너 디스크에 두거나 볼륨을 크게 잡는다(016 RUNLOG).
