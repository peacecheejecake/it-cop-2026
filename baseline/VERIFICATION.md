# 실제 검증 결과

## 결과

**55 tests passed.** Python 3.13.5, CPU 환경에서 실행했습니다. 세부 버전은 `verification/environment.json`에 있습니다.

| 검사 | 결과 | 정확한 범위 |
|---|---|---|
| `pytest -q` | 55 passed | 단위·통합 테스트, 로그 포함 |
| `compileall src scripts tests` | 통과 | Python 문법 검사 |
| `smoke.py --neural` | 통과 | 합성 public/internal 자료의 준비·학습·고정·추론·6개 비교군 평가 |
| `predict_suite.py --models rule tabular` | 통과 | CLI helper로 합성 public test 28행의 2개 비교군 평가 |
| `build-internal` 예제 CSV | 1건 accepted, 0 rejected | 포함된 합성 patch의 로컬 import, network_used=false |
| editable 설치와 `riskbench --help` | 통과 | `pip install --no-deps --no-build-isolation -e .`로 등록 |
| YAML/TOML 파싱 | 통과 | 두 config와 pyproject 파싱 |

합성 end-to-end는 내부 역할의 24행에 대해 변경량 규칙, 학습한 Logistic Regression, frozen/finetune 작은 무작위 Transformer, 모의 LLM zero/few-shot을 비교했습니다. 실제 결함 예측 성능의 근거가 아닙니다. `verification/SMOKE_NOTICE.json`도 이를 표시합니다.

## 실제로 검사한 동작

공개 train/validation role 검사, 내부 자료의 학습·few-shot 거부, 입력의 미래 시각 거부, 라벨 시점의 unknown/late 구분, split 사이 group/exact-patch overlap 제거, 같은 tokenizer view/token budget, 레이블 파일을 제거한 내부 추론, 파일 hash 무결성 검사, 실제 로컬 Git repository의 commit/parent/root 추출, CSV에서 공급된 미래/상이한 feature 무시, archive checksum 검증을 검사했습니다.

A에서 encoder parameter가 바뀌지 않고 head가 학습되는지, B에서 encoder가 실제로 갱신되는지, 마지막 gradient accumulation 구간이 처리되는지, safe checkpoint 저장·재로딩 후 예측이 일치하는지 테스트했습니다. 이 신경망 테스트의 encoder는 **작은 무작위 PyTorch Transformer**입니다.

LLM은 httpx MockTransport로 요청 payload, 내부 endpoint guard, 공개 예시, 라벨 미포함, 동일 요청 retry, 비정상 JSON 및 인증 실패의 null score, resume 메타데이터, 사전 요청 수 상한을 검사했습니다. 실제 API 서비스는 호출하지 않았습니다.

평가에서는 cutoff 동점의 무작위 tie-breaking 기대값, 공통 평가 ID, 모델 실패의 명시적 partial 처리, 정답 정의 차이의 opt-in, paired bootstrap을 검사했습니다.

## 수행하지 못한 것

**실제 ApacheJIT 전체 다운로드·원 Git 전체 수집, 실제 microsoft/codebert-base tokenizer/weights 로딩과 fine-tuning, 실제 외부/내부 LLM endpoint 호출, 사내 데이터 평가, GPU/CUDA/MPS 검증은 하지 않았습니다.**

제작 환경에서 외부 artifact 다운로드 연결이 실패했고 Transformers가 설치되어 있지 않았습니다. 공식 자료에서 source record/checksum·schema·model API를 확인했고, adapter는 같은 형식의 합성 CSV와 실제 로컬 Git으로 검증했습니다. 따라서 원본 corpus의 모든 edge case 또는 특정 서버/모델의 지원 옵션이 검증됐다고 주장하지 않습니다.

`pyproject.toml`의 Transformers 4.57.6은 명시적 실행 후보 버전입니다. 해당 버전과 실제 HF 모델의 조합은 사용자의 공개 학습 환경에서 먼저 검증하고 고정해야 합니다. `requirements-tested-core.txt`는 관측 버전 기록이며 GPU wheel·OS별 완전 lock이 아닙니다.

## 재실행

```bash
python -m pip install -e '.[neural,dev]'
pytest -q
python scripts/smoke.py --out runs/verification-new --neural
```

동일 output 폴더를 덮어쓰지 않습니다. 새로운 경로를 사용하십시오. pytest의 neural 테스트는 torch/safetensors가 없으면 skip될 수 있으므로 사용자 환경의 passed/skipped 수를 같이 기록하십시오.

실제 연구를 시작할 때는 공개 데이터 소규모 import, 실제 tokenizer의 prepare, 실제 CodeBERT의 작은 공개 train/valid 학습, 공개 LLM format 검증 순으로 확인한 뒤 전체 공개 학습과 고정된 내부 평가를 진행하십시오.
