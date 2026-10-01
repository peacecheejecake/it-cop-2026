# CodeBERT Diff Lab — 비교 연구 설계 v0.2

> 2026-10-01 · 구현 전 문서 패키지

사용자의 B0/B1/B2/B3/L0/L1/L2 표와 기존 MLM·MLM+RMI CPT 실험을 통합했다. 바로 읽을 문서는 **[통합 비교군](comparison-matrix.md)** 이다. **[index.html](index.html)** 은 모든 문서의 자체 포함 HTML 열람본이다.

| 문서 | 용도 |
|---|---|
| [comparison-matrix.md](comparison-matrix.md) | 전체 비교표, 번호 이관, 공정 비교, LLM 라벨/점수 계약 |
| [requirements.md](requirements.md) | 기능·비기능 요구, 데이터 경계, 완료 기준 |
| [experiment-protocol.md](experiment-protocol.md) | 데이터·입력·학습·prompt·평가·freeze 실험 조건 |
| [implementation-plan.md](implementation-plan.md) | 모듈·CLI·설정·milestone·백로그·인수 테스트 |
| [experiment-registry.json](experiment-registry.json) | 15개 계획 variant와 기본 9군, matrix version·migration |
| [CHANGELOG.md](CHANGELOG.md) | v0.1에서 바뀐 내용과 유지한 제약 |
| [references.md](references.md) | 출처와 확인/미확인 범위 |

첫 경로는 **B0-LR/B0-LGBM/B1-TFIDF-S → B2-S/B3-S → B4-S/B5-S → L0-S/L1-S**다. L2는 확장이다. public train으로만 학습·예시·index를 만들고 공개 validation에서 선택한 뒤 freeze한다. 사내는 오프라인 평가 전용이다.

이 패키지는 요구사항/계획 문서다. 예시 CLI는 아직 구현되지 않았으며, 학습 코드·모델·데이터·실측 결과를 포함하지 않는다. v0.1 원본은 변경하지 않았다.
