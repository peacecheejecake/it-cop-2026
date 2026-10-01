# 변경 이력 — v0.1 → v0.2

> 2026-10-01 · 문서 개정이며 코드 구현/실험 완료가 아니다.

## 주요 변경

| 범위 | v0.1 | v0.2 |
|---|---|---|
| 연구 범위 | 정형 LR와 CodeBERT CPT 비교 중심 | 정형·TF-IDF·동결/FT encoder·CPT·고정 LLM 통합 |
| B0 | Logistic Regression | LR/LightGBM 별도 variant |
| B1 | 동결 encoder 코드 전용 | TF-IDF+정형+LR; legacy B1은 B2-T로 이관 |
| B2 | FT encoder 코드 전용 | 동결 encoder; legacy B2는 B3-T로 이관 |
| B3/B4/B5 | FT / MLM CPT / MLM+RMI CPT | 같은 의미 유지, S variant를 명시 |
| L0/L1 | 초기 범위에서 제외 | 로컬 고정 LLM의 zero/few-shot 비교 필수 |
| L2 | 미정 | 공개 train 고정 index 검색, 확장 |
| 입력 | encoder 중심 최대 512 tokens | 공통 EvidenceView matched 기본, native 확장 분리 |
| marker 처리 | [ADD]/[DEL] 신규 특수 token 제안 | controlled-v2는 native tokenizer의 diff prefix; frozen encoder의 신규 embedding 문제 회피 |
| 라벨 경계 | public train 학습, 내부 평가 전용 | 유지; public frozen demo/index의 내부 offline 참조만 허용 |
| scoring | supervised binary logit 중심 | LLM 두 라벨 후보 likelihood 상대 점수 추가 |
| traceability | seed·token·data/model hash | train/demo/index/selection label ledger·prompt/index hash 추가 |
| 완료 기준 | DoD-Public | DoD-Core(7 primary variants), DoD-Comparative(9)로 구분 |

## 번호 호환

v0.1의 `B1→v0.2 B2-T`, `B2→B3-T`, `B0→B0-LR`, `B3/B4/B5→B3-S/B4-S/B5-S`. 기존 artifact를 덮어쓰지 않고 matrix_version이 붙은 명시적 migration metadata만 생성한다. renderer가 바뀌므로 ID 매핑은 실험 조건의 동일성을 인증하지 않는다.

## 유지한 제약

공개 train 학습/예시/index, 공개 validation 선택, test·사내 평가 전용. 내부 fine-tuning·CPT·보정·prompt 튜닝·demo/index 추가·외부 전송 금지. reference/controlled, matched/native, query label/public 예시 label을 구분한다. 원본 pickle import 격리·출처 승인·누수 검사·freeze·artifact hash·실패 coverage·지표 정의·resume 계약은 유지한다.

## 산출물

requirements.md, experiment-protocol.md, implementation-plan.md, references.md를 개정했다. comparison-matrix.md와 experiment-registry.json을 추가했다. index.html은 원문 문서의 통합 열람본이며 외부 CDN 없이 동작한다. registry JSON은 설계용 목록이지 실행 가능한 라이브러리 구현이 아니다.

원본 v0.1 파일과 ZIP은 변경하지 않았다. 실제 학습 코드·의존성 lock·데이터·모델·성능 결과는 생성하지 않았다.
