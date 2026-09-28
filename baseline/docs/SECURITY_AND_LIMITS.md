# 보안, 누수 방지, 실험 한계

## 내부 데이터 경계

기본 정책은 내부 데이터 학습·calibration·few-shot·RAG 금지입니다. 이를 위해 train/valid role과 public domain을 함수에서 검사하고 내부 모델 추론에는 experiment lock을 요구합니다. 고정 변경량 규칙은 학습 artifact가 없어 lock 요구의 예외입니다.

전처리 구축 담당자와 추론/평가 담당자를 분리할 수 있도록 inputs와 labels를 파일로 나눕니다. `predict_suite.py --no-evaluate`는 정답 파일을 읽지 않습니다. 다만 같은 사람이 원천 canonical data나 라벨에 접근할 수 있으면 소프트웨어만으로 blind 평가를 보증할 수 없습니다.

## 네트워크·실행

- 내부 import와 pretrained model 추론은 로컬 경로만 사용합니다. 내부 prepare는 다운로드 옵션을 거부합니다.
- LLM endpoint는 내부 데이터 기본 차단이며, allow와 정확한 origin의 사전 승인을 요구합니다. 이는 운영자 승인 표시이지 격리 장치가 아닙니다. gateway가 실제로 어디로 전달하는지와 로그·보관 정책을 확인하십시오.
- HTTP redirect와 환경변수 proxy의 암묵 사용을 끕니다. 승인 proxy가 필수라면 코드를 검토해 공개 검증 단계에서 명시적으로 추가하고 다시 잠그십시오.
- API key는 환경변수에서 읽으며 저장·로그에 넣지 않습니다. 에러 원문이나 prompt/response 전체를 prediction 로그에 저장하지 않습니다. 그러나 inputs, canonical 데이터, 공개 few-shot 파일에는 원 코드가 포함됩니다. 적절한 파일 권한·암호화·보관 정책이 필요합니다.
- 공개 Git 수집은 commit SHA와 owner/repository를 검증하고 shell 문자열 실행을 피하며 checkout/hooks/textconv/external diff 실행을 억제합니다. Git 자체 취약점까지 막는 sandbox는 아닙니다. 승인된 최신 Git 및 격리 환경을 사용하십시오.
- 원본 저장소의 코드는 실행하지 않습니다. Zenodo ZIP은 checksum 확인 뒤 CSV 한 파일만 추출합니다.
- Hugging Face는 trust_remote_code=False, 기본 offline 경로이며 모델 artifact는 safetensors를 사용합니다. 외부 모델/패키지 반입의 공급망 검증을 대체하지 않습니다.

## Prompt injection과 비밀 정보

코드 주석·메시지 안 지시문을 따르지 말라는 system 지시와 허용된 JSON 필드만 사용하며, tools/code execution 권한을 주지 않습니다. 이것이 prompt injection을 완전히 차단한다는 주장은 아닙니다. 악성 문구가 점수에 영향을 주는지 별도 robustness 평가가 필요합니다. secret scanning·익명화·비밀 제거는 구현하지 않았습니다. 원문 전송 전 조직 정책에 맞는 검사를 수행하십시오.

## 실험 lock의 의미

lock은 모델/config/examples 파일 hash와 구현 파일 hash를 검증합니다. 상대 폴더 구조를 유지해 옮기고, 반입 뒤 코드를 수정하지 마십시오. 프롬프트가 들어 있는 llm.py도 hash 검증 대상입니다. 변경이 필요하면 별도 실험 버전으로 만들고 새로운 평가 집합을 사용하십시오.

lock을 지우거나 새로 생성할 권한을 가진 사람이 내부 test에 맞춰 반복 실험하는 것을 막지는 못합니다. 실제 연구에서는 사전 등록, 평가 접근 통제, audit log가 필요합니다. LLM endpoint의 고정 model_revision_tag는 self-report이므로 immutable 모델 운영·서버 응답 메타데이터도 확인해야 합니다.

## 통계·해석

ApacheJIT 결함 라벨과 내부 운영 장애 라벨은 다릅니다. target transfer를 명시하십시오. 미래 라벨을 전부 아는 회고적 benchmark와 실제 fit 시점에 이용 가능한 라벨만 쓴 backtest를 구분합니다. 코드의 split은 알려진 날짜 누수를 차단하지만 unknown date를 복원하지 않습니다.

결함/장애 라벨은 노이즈·누락이 있을 수 있습니다. 정상 배포 전체를 포함한 평가 모집단, 양성 수, 제외 수, 관측 창과 원인 판정 정책을 보고하십시오. 작은 양성 수에서 점수 차이만 보고 승자를 단정하지 마십시오.

week/group bootstrap은 표본 의존성을 근사합니다. 여러 서비스가 공유하는 사건의 실제 의존 구조를 모두 복원하지 않습니다. top-K는 사후 순위 평가이며 실제 online gate policy나 예방된 장애 수가 아닙니다. score의 숫자는 확률 보정을 거치지 않았습니다.

수집 실패, 너무 긴 diff 제외, context truncation, LLM 실패의 partial 교집합 평가가 선택 편향을 만들 수 있습니다. 모두 coverage와 함께 보고해야 합니다. foundation model 사전학습 데이터와의 오염, semantic near-duplicate, 변경 언어별 성능은 별도 검사 대상입니다.

## 계산 자원·미구현

JSONL corpus는 메모리에 로드합니다. 전체 공개 데이터 수집에는 Git 저장 공간과 긴 diff 메모리가 필요합니다. 스트리밍/sharding, multi-GPU/distributed, mixed precision 자동 최적화, LoRA는 구현하지 않았습니다. 작은 합성 모델 smoke가 실제 CodeBERT의 GPU 메모리 요구를 검증한 것은 아닙니다.

기본 LLM adapter는 Chat Completions-compatible HTTP만 지원합니다. Gemini/Anthropic native schema, 특정 gateway 설치·배포, vendor별 비용 추정은 구현하지 않았습니다. 요청 수 상한·timeout·retry를 공개 단계에서 검증하고 과금 제한을 별도로 설정하십시오.

이 패키지는 연구용 baseline으로, production 인증·보안 심사·법적 적합성 또는 실제 장애 예측 성능을 보증하지 않습니다.
