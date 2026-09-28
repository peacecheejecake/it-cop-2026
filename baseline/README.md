# 공개 데이터 학습 · 내부 평가 전용 배포 위험 baseline

**실행 가능한 Python CLI 프로젝트**입니다. 공개 데이터에서만 학습·모델 선택을 수행하고, 내부 데이터는 모델을 고정한 뒤 추론·평가에만 사용합니다. 운영 배포를 차단하는 제품이 아니라 **연구용 출발점**입니다.

**검증 범위:** 단위·통합 테스트 및 합성 데이터 end-to-end 실행을 포함합니다. 실제 ApacheJIT 전체 수집, 원본 CodeBERT 가중치를 사용한 학습, 실서비스 LLM 호출, 사내 성능 검증은 이 패키지 제작 환경에서 수행하지 않았습니다. 자세한 결과는 `VERIFICATION.md`를 보십시오. 데이터·모델 가중치·API key는 포함하지 않습니다.

## 구현된 비교군

| 구분 | 구현 | 내부 데이터의 사용 |
|---|---|---|
| 고정 규칙 | `lines_added + lines_deleted`로 정렬 | 추론·평가만 |
| 정형 baseline | 정적 diff feature + Logistic Regression; C는 공개 validation AP로 선택 | 추론·평가만 |
| A | 동결 CodeBERT → attention-mask mean pooling → 학습된 Linear head | 추론·평가만 |
| B | 같은 CodeBERT·pooling·Linear head 전체 fine-tuning | 추론·평가만 |
| C | 고정 instruction LLM zero-shot | 추론·평가만 |
| D | C와 같은 LLM·설정 + 고정된 공개 train few-shot | 추론·평가만 |

A/B의 pooling과 head는 같습니다. A는 encoder의 gradient를 끄고 `eval()`을 유지하며, 공개 embedding을 디스크에 임시 캐시한 뒤 head를 학습합니다. B는 encoder와 head를 함께 업데이트합니다. 두 모델의 학습 가능한 부분은 모두 공개 데이터로만 학습합니다.

C/D는 **OpenAI Chat Completions-compatible HTTP endpoint**를 사용합니다. 승인된 내부 inference server 또는 호환 gateway에 연결할 수 있습니다. **Gemini/Anthropic native API adapter는 포함하지 않습니다.** provider별 모든 모델의 decoding/JSON 옵션이 동일하다고 가정하지 않습니다.

## 1. 설치와 가장 먼저 할 실행

Python 3.11 이상, Git이 필요합니다. 아래 명령은 macOS/Linux/Git Bash 기준입니다.

```bash

# uv 사용 예
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'

riskbench --help
pytest -q

# GPU·모델 다운로드·LLM 서버가 필요 없는 기본 smoke test
python scripts/smoke.py --out runs/smoke-core
```

일반 pip로도 같습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Windows PowerShell에서는 활성화를 `.venv\Scripts\Activate.ps1`로 바꾸십시오. CLI 자체는 Bash에 의존하지 않습니다.

CodeBERT 실험과 작은 Transformer 검증용 추가 패키지:

```bash
# CUDA 환경이면 먼저 승인된 PyTorch wheel/index에 맞게 torch를 설치하십시오.
uv pip install -e '.[neural,dev]'

# 임의 초기화한 작은 Transformer + 모의 LLM로 전체 경로 확인
python scripts/smoke.py --out runs/smoke-neural --neural
```

`demo-random`, `demo-byte`, `MOCK-NO-LLM`은 **배관 테스트 전용**입니다. 이 출력으로 모델 성능이나 논문의 재현 성공을 주장하면 안 됩니다. 코드가 이 토크나이저·모델을 실제 데이터에 적용하지 못하게 제한합니다.

`pyproject.toml`은 neural 경로의 Transformers를 `4.57.6`으로 지정합니다. 이 버전의 실제 HF 가중치 경로는 제작 환경에서 검증하지 못했습니다. `requirements-tested-core.txt`는 테스트에 사용한 핵심 패키지 버전 기록이며, GPU/OS별 완전한 환경 lock은 아닙니다. 실제 환경에서 설치한 뒤 `uv pip freeze` 등으로 별도 고정하십시오.

## 2. 공개 데이터 구축: ApacheJIT CSV + 원본 Git diff

첫 adapter는 ApacheJIT입니다. 원본 CSV의 `commit_id`, `project` (`owner/repo`), `buggy`를 읽고 해당 commit의 실제 Git diff와 메시지를 가져옵니다. CSV의 `fix`, 개발자 이력, 기존 코드량 값은 모델 입력으로 복사하지 않습니다.

ApacheJIT은 **결함 유발 commit** 라벨이지 운영 SEV 라벨이 아닙니다. JIT-Fine 원 논문의 완전 재현 코드도 아닙니다. JIT-Fine의 pickle importer와 라인별 결함 위치 추정은 이번 범위에서 제외했습니다.

### 2-1. 고정된 원본 버전 받기

```bash
riskbench fetch-apache --out data/raw/apachejit-v2
```

Zenodo record `5907847`의 v2 ZIP checksum을 검사한 뒤 `apachejit_total.csv`만 꺼냅니다. ZIP 내부 코드를 실행하지 않으며 전체 압축 해제를 하지 않습니다. 승인된 오프라인 사본이 있으면:

```bash
riskbench fetch-apache \
  --archive /approved/path/apachejit_dataset_replication.zip \
  --out data/raw/apachejit-v2
```

원본 파일의 라이선스, 각 repository의 코드 라이선스와 사내 반입 정책은 별도로 확인하십시오. 공개 데이터는 이 ZIP에 재배포하지 않습니다.

### 2-2. commit과 실제 diff 결합

```bash
# 승인된 외부 공개 학습 환경에서만 실행
riskbench build-apache \
  --csv data/raw/apachejit-v2/apachejit_total.csv \
  --repos data/git \
  --out data/canonical/apachejit \
  --allow-network
```

`--allow-network`는 필요한 **공개 Git repository**를 mirror clone하는 데 사용합니다. 내부 importer는 이 기능을 호출하지 않습니다.

처음 동작 확인에는 `--limit 300 --seed 42`를 붙이고 별도 `--out`을 쓰십시오. limit은 라벨을 보지 않는 ID hash sample입니다. 일부 데이터만 사용한 결과를 전체 데이터의 성능으로 보고하면 안 됩니다. repository clone 자체는 작은 샘플이어도 클 수 있습니다.

이미 받은 repository는 다음 위치에 둡니다.

```text
data/git/apache/groovy.git/
data/git/apache/camel.git/
...
```

그 뒤에는 `--allow-network` 없이 동일 명령을 실행합니다. Git은 코드를 checkout하거나 실행하지 않습니다. merge commit은 기본 제외이며, 사전 등록된 대안 실험으로 `--first-parent`를 선택할 수 있습니다. 2MB보다 큰 patch와 빈 diff 등은 제외되어 보고서에 남습니다.

CSV 컬럼명이 다른 사본에는 `--commit-col`, `--repo-col`, `--label-col`을 지정하십시오. 원본 이름을 추측해 자동으로 다른 열을 정답으로 선택하지 않습니다.

산출물:

```text
data/canonical/apachejit/
  records.jsonl
  build_report.json    # 선택/수집 성공/실패/제외 사유와 원본 SHA-256
  rejected.jsonl       # 누락을 숨기거나 score=0으로 바꾸지 않음
```

### 2-3. 공개 train/validation/test 분할

```bash
riskbench split-public \
  --records data/canonical/apachejit/records.jsonl \
  --out data/splits/public \
  --train-before 2016-01-01T00:00:00Z \
  --valid-before 2017-01-01T00:00:00Z \
  --allow-retrospective
```

위 연도는 시작용 프로토콜 예시이며 원 논문의 split 재현을 의미하지 않습니다. 이 adapter는 저자의 `author_date`가 아니라 Git **committer date**를 사용합니다.

**`--allow-retrospective`의 의미가 중요합니다.** 공개 CSV만으로는 결함 라벨의 실제 확정 시점을 알 수 없어 `label_available_at=null`로 남깁니다. 이 옵션은 그 한계를 승인한 **회고적 benchmark**입니다. 시간순으로 나눴어도 “당시 이용 가능했던 라벨만으로 실제 운영한 backtest”라고 부르면 안 됩니다. 실제 라벨 가용 시각을 확보하면 canonical data에 넣고 옵션 없이 실행하십시오. 알려진 가용 시각이 경계 이후인 라벨은 옵션을 사용해도 제외합니다.

동일 release group/정규화된 exact patch가 split 경계를 넘으면 연결된 전체 그룹을 제외하고 기록합니다. 같은 split 안의 exact patch 중복도 제거합니다. **semantic near-duplicate 탐지는 구현하지 않았습니다.** 특정 공개 repository를 통째로 외부 평가로 두려면 `--holdout-repos apache/groovy ...`를 사용합니다. 다른 공개 프로젝트의 시간순 validation도 함께 남습니다.

## 3. 공통 코드 입력 만들기

### 3-1. CodeBERT와 tokenizer 반입

```bash
# 공개 다운로드가 허용된 환경
riskbench download-codebert --out models/codebert-base
```

실제 조회된 HF commit을 `SOURCE.json`에 기록하고 그 revision으로 다운로드합니다. 엄격한 반복 재현에는 `--revision <verified_commit_sha>`를 지정하십시오. 내부에는 승인된 모델 파일과 tokenizer를 복사합니다. 내부 추론은 모델을 다운로드하지 않습니다.

### 3-2. A/B/C/D가 같은 변경 내용을 보도록 준비

```bash
riskbench prepare --dataset data/splits/public/train \
  --out data/views/public/train --tokenizer models/codebert-base
riskbench prepare --dataset data/splits/public/valid \
  --out data/views/public/valid --tokenizer models/codebert-base
riskbench prepare --dataset data/splits/public/test \
  --out data/views/public/test --tokenizer models/codebert-base
```

기본 정책은 총 **512 CodeBERT tokens**, 메시지 최대 **64 tokens**, 긴 diff는 앞/뒤를 함께 남기는 방식입니다. 실제 encode 결과가 한도를 넘지 않게 검사합니다.

A/B는 저장된 동일 `input_ids`, C/D는 그와 대응하는 동일 `text`를 사용합니다. C/D가 몰래 원본 full diff를 더 보는 비교가 아닙니다. LLM의 자체 tokenizer로 계산한 token 수까지 CodeBERT와 같다는 의미는 아닙니다. C/D few-shot의 추가 예시는 의도된 실험 변수입니다.

`manifest.json`에 tokenizer signature, 전처리 policy hash, truncation 수를 저장합니다. tokenizer/길이/절단 방식을 바꾸려면 공개 단계에서 별도 실험으로 정하고 다시 고정해야 합니다. 긴 변경의 정보 손실은 이 기본 실험의 한계이며, full-context LLM 비교는 별도 조건으로 구현·보고해야 합니다.

## 4. 학습: 공개 데이터만

### 정형 baseline

```bash
riskbench train-tabular \
  --public-train data/views/public/train \
  --public-valid data/views/public/valid \
  --out runs/models/tabular
```

7개 공통 정적 feature에 `log1p → 공개 train median 보간 → missing indicators → 공개 train scaling → Logistic Regression`을 적용합니다. validation AP로 C를 선택하고 test는 보지 않습니다. 클래스 resampling과 내부 calibration은 없습니다.

이번 7개 feature는 `lines_added`, `lines_deleted`, `files_changed`, `directories_changed`, `change_entropy`, `binary_files`, `hunks`입니다. **원 논문의 전체 expert/history feature baseline과 동일하지 않습니다.** 정보가 제한된 공통 baseline이며, 향후 동일한 시점 정의로 양쪽에서 계산 가능한 feature만 별도 버전으로 확장하십시오.

### A. 동결 CodeBERT + head

```bash
riskbench train-codebert \
  --mode frozen --model models/codebert-base \
  --public-train data/views/public/train \
  --public-valid data/views/public/valid \
  --out runs/models/frozen \
  --epochs 5 --batch-size 8 --accumulation 4 --seed 42
```

### B. 전체 CodeBERT fine-tuning

```bash
riskbench train-codebert \
  --mode finetune --model models/codebert-base \
  --public-train data/views/public/train \
  --public-valid data/views/public/valid \
  --out runs/models/finetune \
  --epochs 5 --batch-size 8 --accumulation 4 \
  --encoder-lr 2e-5 --head-lr 1e-3 --seed 42
```

validation AP가 가장 높은 checkpoint만 보관합니다. CPU full fine-tuning은 실수로 시작하지 않도록 `--allow-cpu-training`을 명시해야 합니다. `--device cpu|cuda|mps|auto`를 지원합니다. GPU 사양에 맞게 batch size를 정하고, 공개 단계에서 학습 설정을 확정하십시오.

정형 모델은 JSON 계수·scaler, neural 모델은 JSON config와 `safetensors`로 저장합니다. 내부 inference에는 원본 학습 데이터가 필요 없습니다. head만이 아니라 frozen encoder 가중치도 artifact에 포함되므로 실제 model 디렉터리는 작지 않습니다.

다른 seed는 별도 model root에 저장하고 연구 결과에 분산을 보고하십시오. A/B 각각에 대한 충분한 공개 hyperparameter 탐색은 사용자의 실험 단계이며, 이 코드가 설정 하나만으로 최적 성능을 보장하지 않습니다.

## 5. C/D: 추가 학습 없는 LLM

### 공개 few-shot 예시 고정

```bash
riskbench examples \
  --public-train data/views/public/train \
  --out runs/public_examples.json --k 4 --seed 42

cp configs/llm.example.yaml configs/llm.yaml
```

`configs/llm.yaml`에서 endpoint, 실제 model ID, immutable revision tag, decoding 옵션을 설정하십시오. API key는 YAML에 쓰지 않고 환경변수 `LLM_API_KEY`로 전달합니다.

- C와 D는 같은 YAML을 사용합니다.
- D의 예시는 공개 **train**에서만 고정 추출한 양성 2건·음성 2건입니다. 내부 예시나 내부 RAG는 없습니다.
- 관측 라벨 0/1을 0%/100% 확률로 가르치지 않습니다. 과거 label을 설명하는 예시와 현재의 ordinal score 출력을 구분합니다.
- 응답은 `{"risk_score": 0..100}`입니다. score/100은 편의상의 정규화이며 **사내 장애 확률이 아닙니다.**
- 서버가 지원하지 않는 `temperature`, `response_format`, `max_tokens` 옵션은 공개 validation에서 조정하십시오. reasoning 모델이면 출력 token 예산/`max_completion_tokens`가 특히 중요합니다. 실패했을 때 조용히 다른 모델·프롬프트로 바꾸지 않습니다.

```bash
# 먼저 공개 validation에서 API/포맷/설정을 확인
riskbench llm --config configs/llm.yaml --dataset data/views/public/valid \
  --out runs/public-valid/zero.jsonl --name zero
riskbench llm --config configs/llm.yaml --dataset data/views/public/valid \
  --examples runs/public_examples.json \
  --out runs/public-valid/few.jsonl --name few
```

한 run의 기본 요청 상한은 1,000개입니다. 전체 공개 test에 수만 건이 있다면 호출 비용·처리량을 검토하고 **내부 평가 전에** `max_requests_per_run`을 명시적으로 정하십시오. 실패 요청의 최대 재시도 횟수도 비용에 영향을 줍니다. 실제 가격을 코드에 하드코딩하지 않습니다.

응답 parsing 실패, 인증 실패, timeout은 `status=error, score=null`로 남깁니다. 이를 안전한 변경의 score 0으로 바꾸지 않습니다. 재시도는 같은 요청을 반복하며 응답을 보고 prompt를 수정하지 않습니다. 중단된 호출은 동일 입력·설정으로 `--resume`할 수 있고, 이미 기록한 성공/실패를 임의로 골라 재시도하지 않습니다. 마지막 JSONL 줄이 강제 종료로 잘렸다면 수동 점검이 필요하며 자동으로 숨기지 않습니다.

## 6. 내부 데이터 연결과 실험 잠금

### 내부 export 형식

`examples/internal_manifest.csv`와 `docs/DATA_CONTRACT.md`를 참고하여 **내부에서** 아래 형태로 내보내십시오.

```text
data/internal/export/
  manifest.csv
  patches/
    change-001.diff
    change-002.diff
```

```bash
riskbench build-internal \
  --manifest data/internal/export/manifest.csv \
  --out data/canonical/internal
riskbench internal-test \
  --records data/canonical/internal/records.jsonl \
  --out data/splits/internal
riskbench prepare \
  --dataset data/splits/internal \
  --out data/views/internal \
  --tokenizer models/codebert-base
```

내부 importer는 GitHub/API를 호출하지 않습니다. 사내 데이터의 원인 관계 판정·72시간 관찰 완료·라벨 확정은 **upstream에서 먼저 수행**해야 합니다. 코드는 사후 RCA를 읽어 원인 관계를 자동으로 보증하지 않습니다. 미확정·censored 사례를 정상 0으로 대체하지 않고 제외 건수를 기록합니다.

내부 LLM을 쓸 경우 승인된 실행 경로를 config에 명시합니다. 예를 들어 직접 운영하고 승인된 로컬 서버일 때만:

```yaml
internal_data:
  allow: true
  approved_origins: ["http://127.0.0.1:8000"]
```

이 설정은 **통신 승인 확인용 guard**이지 네트워크 sandbox/DLP가 아닙니다. 해당 서버가 외부로 전달하지 않는지, prompt/response 보관·접근 정책이 적합한지 별도로 확인해야 합니다. 원본 코드 반출이 금지된 경우 외부 LLM API를 대안으로 사용하지 마십시오.

### 공개 단계가 끝난 후 고정

```bash
riskbench freeze \
  --paths runs/models configs/llm.yaml configs/protocol.example.yaml runs/public_examples.json \
  --out experiment.lock.json
```

model/config/examples의 파일 hash와 Python 구현 hash를 저장합니다. 내부 추론 시 변경되면 중단합니다. lock과 상대 폴더 구조를 함께 내부 환경으로 반입하십시오. tokenizer도 따로 반입해야 내부 입력을 prepare할 수 있습니다. hash 검사는 무결성 장치이지, 누군가 내부 평가 결과를 보고 lock을 새로 만들지 않았다는 증명은 아닙니다.

## 7. 동일 모집단에서 전체 비교

공개 test로 먼저 실행:

```bash
python scripts/predict_suite.py \
  --dataset data/views/public/test \
  --model-root runs/models \
  --llm-config configs/llm.yaml --examples runs/public_examples.json \
  --out runs/evaluation/public-test \
  --bootstrap 500
```

내부 고정 test로 실행:

```bash
python scripts/predict_suite.py \
  --dataset data/views/internal \
  --model-root runs/models \
  --llm-config configs/llm.yaml --examples runs/public_examples.json \
  --lock experiment.lock.json \
  --out runs/evaluation/internal-test \
  --allow-label-transfer --bootstrap 500
```

`--allow-label-transfer`는 source `bug_inducing_commit`과 target `operational_incident_72h`가 다른 과제임을 명시하는 옵션입니다. 둘을 같은 정답으로 강제로 합치는 기능이 아닙니다. 내부도 같은 결함 유발 commit 라벨이라면 필요 없습니다.

처음 정형 부분만 실행하려면:

```bash
python scripts/predict_suite.py \
  --dataset data/views/public/test \
  --models rule tabular --model-root runs/models \
  --out runs/evaluation/tabular-only
```

라벨을 열지 않고 예측만 만들려면 `--no-evaluate`를 붙입니다. `inputs.jsonl`과 `labels.jsonl`은 별도 파일이며, prediction 코드는 label 파일이 없어도 동작합니다. 입력 준비/학습/평가가 각각 분리되어 있습니다.

### 단일 모델 명령

```bash
riskbench predict --kind tabular --artifact runs/models/tabular \
  --dataset data/views/internal --lock experiment.lock.json \
  --out runs/internal/tabular.jsonl --name tabular

riskbench predict --kind codebert --artifact runs/models/frozen \
  --dataset data/views/internal --lock experiment.lock.json \
  --out runs/internal/frozen.jsonl --name frozen

riskbench llm --config configs/llm.yaml --dataset data/views/internal \
  --lock experiment.lock.json --examples runs/public_examples.json \
  --out runs/internal/few.jsonl --name few
```

학습·모델 선택 함수는 public train/valid만 허용합니다. target ID/patch/group이 source train과 겹치면 중단하고, 내부 평가에는 source validation과의 겹침도 검사합니다. 단, Foundation model의 비공개 사전학습 corpus와의 중복 여부는 이 검사로 알 수 없습니다.

## 8. 결과 읽기

```text
runs/evaluation/internal-test/
  rule.jsonl
  tabular.jsonl
  frozen.jsonl
  finetune.jsonl
  zero.jsonl
  few.jsonl
  *.meta.json
  metrics.json
  metrics.csv
```

`metrics.json`에는 AP, ROC-AUC, Recall/Precision@Top5/10%, 실제 선택 건수, 양성 수, coverage, week 또는 group 단위 bootstrap, 첫 비교군 대비 paired delta 구간이 포함됩니다.

**해석 규칙:**

- AP는 `sklearn.metrics.average_precision_score`이며 사다리꼴 PR-AUC와 혼용하지 않습니다.
- 동점 경계는 무작위 tie-breaking의 기대 TP로 계산합니다. 입력 행이나 라벨 순서가 결과를 유리하게 만들지 않습니다. 따라서 TP가 소수일 수 있습니다.
- K는 `ceil(N × fraction)`입니다. 작은 test에서는 실제 검토 비율이 5/10%보다 커지므로 `review_rate`도 함께 보십시오.
- 실패 prediction이 있으면 기본 비교는 중단합니다. `riskbench evaluate --allow-partial`은 명시적으로 **모든 모델이 성공한 공통 부분집합**만 평가하고 원 모집단·실패 양성 수를 같이 기록합니다. 이것을 전체 모집단 성능으로 보고하지 마십시오.
- paired bootstrap의 기준 모델은 `--predictions`에서 첫 번째입니다. 독립 cluster가 적으면 경고를 남깁니다. week cluster는 시간 상관의 근사일 뿐, 장애의 실제 의존 구조를 완벽히 보장하지 않습니다.
- Top-K는 전체 test를 사후 정렬한 순위 평가입니다. online threshold 정책, 확률 calibration, 실제 장애 예방 효과 평가가 아닙니다.
- API `usage`, 시도 횟수, 지연, response model/fingerprint가 제공되면 prediction에 저장합니다. 서버가 주지 않는 비용·fingerprint를 추정하지 않습니다.
- 내부 테스트를 보고 epoch, feature, prompt, example set을 바꾸면 해당 test는 더 이상 독립 final test가 아닙니다.

## 9. 이번 baseline의 범위 밖

현재 구현하지 않은 것은 JIT-Fine pickle 변환, 전체 이력 feature 복원, GPU distributed/LoRA 훈련, native Gemini/Anthropic API, 자동 모델 탐색, 배포–장애 원인관계 생성, 여러 commit의 deployment-level score 집계, calibration, 실제 review/gating 정책 시뮬레이션, semantic near-duplicate 검사입니다.

내부 평가 단위가 여러 commit을 포함한 서비스 배포라면 **배포 전체의 실제 artifact-to-artifact diff**를 한 행으로 제공하거나, 미리 고정한 deployment 집계 실험을 추가하십시오. 배포 장애 라벨을 구성 commit 모두에 복사하지 마십시오.

데이터 로더는 baseline의 단순성을 위해 JSONL을 메모리에 읽습니다. 전체 corpus에 긴 diff가 많으면 충분한 메모리가 있는 공개 학습 환경에서 전처리하고, 대규모 운영에는 streaming/sharding을 추가해야 합니다. `--limit`은 동작 점검용이지 이 문제를 숨긴 성능 실험 수단이 아닙니다.

## 파일 위치

```text
src/riskbench/
  data.py       # 공개/내부 adapter, Git diff, feature, split, label 분리
  views.py      # 공통 tokenizer view와 truncation
  linear.py     # 고정 규칙 + 정형 Logistic Regression
  neural.py     # A/B, 학습·checkpoint·offline 추론
  llm.py        # C/D, 공개 예시, HTTP, 실패·resume·전송 guard
  metrics.py    # 같은 ID 모집단, 순위 지표, paired bootstrap
  artifacts.py  # 실험 lock과 prediction provenance
  cli.py        # CLI
scripts/
  smoke.py
  predict_suite.py
configs/
  llm.example.yaml
  protocol.example.yaml
tests/
docs/DATA_CONTRACT.md
docs/SECURITY_AND_LIMITS.md
docs/SOURCES.md
VERIFICATION.md
```

실제 논문 실험을 시작하기 전에는 `configs/protocol.example.yaml`을 프로젝트의 정답 정의·모집단·cutoff·seed·모델 버전으로 바꾸고 공개 검증을 완료한 뒤 잠그십시오.
