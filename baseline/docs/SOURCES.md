# 원천 및 구현 참고

코드·문서는 이 요청을 위해 작성한 baseline이며 아래 연구의 코드 전체를 복제한 재현 패키지가 아닙니다. 공개 데이터·모델 가중치는 재배포하지 않습니다. 참고자료 확인일: 2026-09-28.

## ApacheJIT

- Dataset v2: https://zenodo.org/records/5907847
- DOI: https://doi.org/10.5281/zenodo.5907847
- ZIP: https://zenodo.org/records/5907847/files/apachejit_dataset_replication.zip?download=1
- 공식 MD5: `528bf0ee04b15976be6bf15f8efddc65`
- 논문: https://arxiv.org/abs/2203.00101

공식 record는 apachejit_total.csv의 commit ID, commit metrics, buggy 라벨과 balanced train/natural-distribution test의 차이를 설명합니다. adapter는 전체 CSV와 원 Git diff를 연결하며 원래 historical feature를 복원하지 않습니다. 공식 배포 설명과 압축 파일 목록을 확인했지만 제작 환경에서는 전체 archive download/repository cloning을 성공시키지 못했습니다. 다운로드·파싱 경로는 mock archive와 실제 로컬 Git repository로 검증했습니다.

CSV 컬럼명 commit_id/project/buggy는 후속 연구자가 공개한 데이터 설명에서도 교차 확인했습니다:
https://github.com/Ali-Sayed-Salehi/jit-dp-llm/blob/master/datasets/apachejit/README.md
이는 라벨 재정의나 후속 전처리를 가져온 것이 아니라 adapter schema 확인용입니다.

## CodeBERT / JIT-Fine

- 공식 CodeBERT model card: https://huggingface.co/microsoft/codebert-base
- 공식 CodeBERT 구현: https://github.com/microsoft/CodeBERT
- 원 논문: https://arxiv.org/abs/2002.08155
- JIT-Fine 저자 재현 코드: https://github.com/jacknichao/JIT-Fine
- Transformers RoBERTa docs: https://huggingface.co/docs/transformers/model_doc/roberta
- 지정한 Transformers 배포: https://pypi.org/project/transformers/4.57.6/

본 baseline은 같은 masked mean pooling + Linear head에서 encoder 동결 여부만 비교합니다. JIT-Fine의 전체 expert feature·결함 위치 추정·원 split·원 모델 구조를 복제한 것은 아닙니다. 실제 model weights/tokenizer를 받으면 revision을 기록하고, 공개 환경에서 먼저 실제 Hugging Face 경로를 검증해야 합니다.

## LLM HTTP

- OpenAI Chat API: https://developers.openai.com/api/reference/resources/chat

C/D는 이 계열의 Chat Completions 호환 요청/응답 형태를 사용합니다. 특정 provider/모델의 모든 파라미터 지원을 보장하지 않습니다. Native Gemini/Anthropic adapter는 아닙니다. 공개 validation에서 endpoint·모델·응답 옵션을 확인하십시오.

## 평가

- AP 정의: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
- Calibration: https://scikit-learn.org/stable/modules/calibration.html
- 연구 배경(Meta Diff Risk Score): https://arxiv.org/abs/2410.06351

본 코드는 AP, ROC-AUC, top-budget recall/precision과 paired cluster bootstrap을 구현합니다. Meta 내부 dataset, risk-aligned generative model, 실제 gating 실험의 재현이 아닙니다. 확률 calibration과 실제 예방 효과 추정은 포함하지 않습니다.
