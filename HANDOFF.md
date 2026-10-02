# Handoff — 2026-10-03 (다른 Mac에서 이어서 작업: 외부 test set 평가부터)

다른 세션·다른 기기에서 이어서 작업하기 위한 기록이다. 끝난 항목은 지우고 갱신한다.

## 0. 작업 규칙 (이전 세션에서 사용자가 정한 것)

- **사용자에게 시각을 말할 때는 한국시간(KST)으로.** 저장소 기록(RUNLOG 등)은 UTC 그대로.
- GPU pod는 만들기 전에 GPU 종류·예상 비용을 알리고 확인받는다. 잔액이 바닥나면 Runpod가 pod를 즉시 종료한다(AGENTS.md §5).
- 푸시는 사용자가 요청할 때 `origin`, `lvu` 두 곳에 한다.

## 1. 새 Mac 준비

1. 저장소 clone 후 `git fetch --all`. 원격: `origin` (peacecheejecake/it-cop-2026), `lvu` (kb-lvu/it-cop-experiments). 저장소 루트가 곧 이 프로젝트(`jit-zero-shot`)다. 이전 Mac에서는 `~/it-cop-2026/labs/jit-zero-shot`에 있었다.
2. **git 밖의 데이터 옮기기 (Google Drive, `gdrive:jit-zero-shot-weights/`, 모두 로컬과 해시 대조 완료):**
   ```bash
   # 새 Mac의 저장소 루트에서 (rclone에 같은 Google 계정의 gdrive: remote가 있어야 한다)
   rclone copy gdrive:jit-zero-shot-weights/handoff-2026-10-03 ~/jit-handoff     # tar 9개 약 2.9 GB + SHA256SUMS
   tools/handoff_transfer.sh restore ~/jit-handoff                                # 해시 검사 → worktree 5개 생성 → 풀기
   rclone copy gdrive:jit-zero-shot-weights/016-dl-v4-gitlines/codebert-diff-lab/artifacts experiments/016-dl-v4-gitlines/codebert-diff-lab/artifacts   # v4 run·CPT 10.7 GB
   rclone copy gdrive:jit-zero-shot-weights/diffllm-v1-186bfade85a4b2e6/artifacts/diffllm/cpt experiments/012-dl-diffllm/codebert-diff-lab/artifacts/diffllm/cpt   # Qwen 7B CPT adapter
   ```
   - Drive의 `handoff-2026-10-03`에는 git 미러, CodeBERT, snapshot·split·evidence, v4 test 결과·report, 012 fulldiff·CPT 말뭉치, Study M 입력, 사내 패키지용 Python이 있다(`ITEMS.txt`).
   - v4 run과 CPT 인코더, 012 CPT adapter는 Drive의 기존 백업을 그대로 쓴다. 2026-10-03에 `rclone check`로 로컬과 0 differences를 확인했다.
   - rclone remote는 scope `drive.file`이라, 같은 rclone client로 올린 파일만 보인다. 새 Mac도 rclone 기본 client로 같은 계정에 연결해야 한다. rclone의 공유 client_id는 2026년 중 폐기 예정이라는 경고가 뜬다.
   - 대안: 이전 Mac의 `experiments/.cache/handoff-2026-10-03/`(tar 12개, 14 GB, 위 Drive 분량 포함)를 디스크로 직접 옮겨 `restore`해도 된다.
   - 이전 study(v2/v3, 011 등)의 산출물은 누출된 입력으로 만든 것이라 옮기지 않는다. Drive에 백업은 있다.
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

## 3. 다음 할 일 (사용자가 정한 순서)

### 3.1 [지금 할 일] 외부 test set으로 모든 모델 평가 (모델 × 언어)

사용자 지시(2026-10-03):
1. ISSTA'21 데이터가 사전학습이나 다른 학습에 쓰이지 않았다면, 이 데이터로 예측한다. 모델 × 언어별로 결과를 낸다.
2. 다음 단계는 그 결과를 보고 정한다.
3. 추가로 JavaScript test set이 있으면 좋겠다. 가능하면 COBOL도.

**학습 사용 여부 점검 결과:**
- 우리 학습(JD4J fine-tuning, CPT 말뭉치 zookeeper/zeppelin/activemq/kafka/cassandra/groovy, Qwen LoRA)에는 ISSTA'21 프로젝트(Qt, OpenStack, Eclipse Platform, JDT, Gerrit, Go)가 없다.
- 기반 모델(CodeBERT, Qwen2.5-Coder) 사전학습에 이 저장소들의 코드가 들어갔는지는 확인할 수 없다. JD4J도 같은 처지다. 결함 라벨이 사전학습에 들어갔을 가능성은 없다. 이 한계를 결과에 명시하고 진행한다.

**데이터 확보 완료 (2026-10-03, `DATASETS.md` §2a).** 새 Mac에서는 Drive에서 받는다:
```bash
for d in issta21 deepjit-zenodo-3965246 jit-js-ni2022; do rclone copy gdrive:jit-zero-shot-weights/raw/$d experiments/.cache/raw/$d; done
```
- ISSTA'21 6개 프로젝트 전부: Docker Hub 이미지 층에서 추출했다. Java = jdt, platform, gerrit.
- DeepJIT Qt·OpenStack(Zenodo).
- JavaScript 20개 프로젝트(`jacknichao/JIT-on-JavaScript-projects`).
- COBOL은 하지 않는다(사용자 결정).
- 사용자 지시: Zenodo(Qt·OpenStack)로 먼저 시작한다. Java 데이터도 원했고, ISSTA'21의 jdt·platform·gerrit으로 확보됐다.

**처리 원칙:**
- 패키지의 전처리 텍스트·지표는 쓰지 않는다(JD4J 누출 교훈). **커밋 ID와 라벨만** 가져오고, 텍스트와 jit14는 각 프로젝트 저장소를 clone해 `diff_lab.gitextract`로 다시 뽑는다.
  - 저장소 URL은 데이터의 repo 필드나 `Data_Extraction/git_base/git_extraction.py`의 `-url`/`-repo` 인자로 확정한다.
  - Platform과 OpenStack은 저장소가 여러 개일 수 있다.
- 데이터를 만든 뒤 `tools/representation_leak_check.py`를 돌린다.
- 모델은 재학습하지 않는다. v4 freeze `7340ee71c876c8b7`의 9개 variant를 그대로 적용한다:
  - `study export`(전체 또는 `--variants`) → `predict` → `internal label-eval`. label-eval은 `project`·`primary_language`별 지표를 이미 낸다.
  - B0~B5는 Mac에서 가능하다(추론은 장치 무관). L0/L1은 GPU가 필요하다.
- 구현할 것:
  - `internal extract`에 라벨된 커밋만 고르는 옵션(`extract_repo(only=...)`는 이미 있고 CLI 노출만 필요).
  - 라벨 CSV 변환(ISSTA'21 라벨 → `change_id,label`).
  - 언어 확장자: `gitextract.DEFAULT_EXTENSIONS`에 C++, Python, Go가 있다.
- 보고: 모델 × 언어(프로젝트)별 AP, ROC-AUC, Recall@10%, 양성 비율 대비 향상 배수. 프로젝트마다 양성 비율이 달라 AP 절대값보다 향상 배수와 순위를 본다.

### 3.2 그 다음 (외부 test 결과를 보고 사용자가 정함)

- **Study M 1단계** (`exp/018-dl-study-m`, `configs/studies/study-m-stage1.yaml`):
  - 내용: Qwen 7B R-base와 R-diff를 seed 3개로, v4 cohort, H100. 약 15시간, 약 $52. 충전 필요(잔액 $2.10).
  - 입력 view와 CPT adapter는 준비돼 있다. 명령: `diffllm arm` → `diffllm freeze` → `diffllm test`. v4와 합친 report 스크립트는 아직 없다.
- **Study M 2단계:** Qwen 14B, Gemma 4 12B, Llama 3.1 8B. seed 수는 미정.
- **diff CPT 말뭉치 확대:** 1단계에서 R-diff − R-base가 확실히 양수일 때만.
- **사내 반입 zip** (`exp/015`) — 결정 대기:
  - B2-S seed 44만 넣을지, B0-LR도 함께 넣을지.
  - 사내 서버의 glibc 버전과 GPU 유무.
  - `fetch_wheels.py`의 nccl 다운로드 실패 수정.
- 문서: `docs/results-v4-<date>.md`, `v4-correction-plan.md` §6 정정.
- B2-S 학습 플랫폼 차이 확인: CUDA에서 `exp/017/scripts/b2_head_platform_check.py --seeds`(15분).

## 4. 브랜치·기록 위치

| 브랜치 | 내용 |
|---|---|
| `main` | 코드(gitextract, label_eval, rebuild-git, train_fraction, diffllm per-seed/probe-cpt), study 설정, 문서, `tools/` |
| `exp/015-dl-internal-pack` | 사내 반입 패키지 스크립트(`pack/`, `scripts/`) |
| `exp/016-dl-v4-gitlines` | v4 RESULTS/RUNLOG, `results/report.json`, `results/freeze.json` |
| `exp/017-dl-d1-learning-curve` | D1 RESULTS, B2 플랫폼 진단 스크립트 |
| `exp/018-dl-study-m` | Study M RUNLOG/RESULTS, CPT 점검 결과 |

계획 문서: `codebert-diff-lab/docs/next-studies-plan.md`(Study M, Study D), `v4-correction-plan.md`.

## 5. 비용·자원 상태 (2026-10-03 14:30 KST)

- Runpod 잔액 $2.10, 켜져 있는 pod 없음.
- 측정값(H100):
  - CodeBERT fine-tuning epoch 3.1분.
  - Qwen 7B 위험 분류 학습 seed당 2.0~2.6시간, 7B CPT 50M 토큰 2.9시간.
  - 한 GPU에 여러 run을 동시에 돌려도 처리량은 늘지 않았다(`next-studies-plan.md` §1a).
- pod `/workspace` 볼륨(20 GB)은 체크포인트로 금방 찬다. 산출물은 컨테이너 디스크에 두거나 볼륨을 크게 잡는다(016 RUNLOG).
