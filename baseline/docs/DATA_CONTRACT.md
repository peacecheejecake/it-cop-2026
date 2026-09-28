# 데이터 입력 규약과 평가 단위

## 1. 전제

공개 데이터는 `bug_inducing_commit`, 내부 평가 데이터는 사용자가 고정한 정답 정의를 사용합니다. 내부가 `operational_incident_72h`라면 명시적인 target-transfer 실험입니다. 정답·배포–장애 원인 연결을 이 코드가 자동 생성하거나 보증하지 않습니다.

한 행의 단위는 **라벨에 대응하는 한 변경 diff**입니다. commit 결함을 평가할 때는 commit 한 건입니다. 배포 장애를 평가할 때는 upstream에서 실제 배포 artifact 사이 diff와 그 배포의 정답을 연결해야 합니다. 배포의 양성 라벨을 구성 commit 전부에 복사하지 마십시오. 여러 commit 점수의 deployment aggregation과 고유 incident capture는 이번 구현 범위가 아닙니다.

## 2. 내부 CSV export

`examples/internal_manifest.csv`는 합성 샘플입니다. patch도 예시이며 실제 사내 데이터가 아닙니다. `synthetic=true`는 실제 데이터에서 false 또는 빈 값으로 바꾸십시오.

| 필드 | 요구 사항 |
|---|---|
| `id` | 연구 범위에서 고유한 변경/배포 식별자. 개인정보 대신 가명 ID 권장 |
| `repository` | 가명 repository/service 식별자. 내부 importer는 이를 URL로 사용하지 않음 |
| `commit` | 기준 commit 또는 artifact 식별자. manifest importer에서는 문자열로 보존 |
| `group_id` | 공동 release, 재시도 등 독립으로 볼 수 없는 변경 묶음. 알려진 연결이 없으면 id |
| `prediction_at` | 배포/변경 직전 예측 시점. timezone 포함 ISO 8601 |
| `input_available_at` | 포함한 입력의 가장 늦은 실제 가용 시점. prediction_at보다 늦으면 거부 |
| `message` | 해당 시점에 이미 존재했던 변경 설명. 사후 수정본·RCA·정답 문구 제외 |
| `diff_path` | manifest 폴더 내부의 로컬 unified Git diff 상대 경로. 외부 경로/symlink 탈출 거부 |
| `label` | 판정 완료된 이진 결과 0/1 |
| `label_type` | `bug_inducing_commit`, `operational_incident_72h` 등 고정한 정답 이름 |
| `label_status` | 1이면 confirmed_positive, 0이면 observed_negative |
| `label_available_at` | 실제 판정을 알 수 있게 된 시점. 알 수 없으면 빈 값이며 미확인임을 보고 |
| `synthetic` | 데모만 true. 실제 자료에는 false/빈 값 |

CSV 안의 따옴표·줄바꿈은 표준 CSV escaping을 사용하십시오. encoding은 UTF-8/UTF-8 BOM을 지원합니다. diff 파일은 UTF-8 export를 권장합니다.

### 예측 시점과 정답 시점

`input_available_at <= prediction_at`를 검사합니다. 하지만 exporter가 타임스탬프를 잘못 붙였는지, 메시지 본문에 실제 사후 정보가 포함됐는지는 이 검사로 탐지할 수 없습니다. 당시 snapshot을 확보하거나 텍스트 revision history로 재구성해야 합니다.

72시간 관찰이 끝났다는 것과 라벨이 확정됐다는 것은 다릅니다. 관련 사건의 판정 지연과 관측 공백을 upstream에서 확인한 뒤 `observed_negative`를 부여하십시오. `ambiguous`/`censored`를 0으로 대체하지 않으며 importer는 제외 사유를 남깁니다.

## 3. Canonical row

`build-apache`/`build-internal`은 원천 라벨과 입력이 함께 있는 `records.jsonl`을 생성합니다. 이 파일은 입력 구축 담당자가 관리할 중간 자료이지 blind inference 파일이 아닙니다.

```json
{
  "id": "example-change-001",
  "domain": "internal",
  "repository": "service-alias",
  "commit": "artifact-alias",
  "group_id": "release-alias",
  "prediction_at": "2026-01-01T01:00:00+00:00",
  "input_available_at": "2026-01-01T00:59:00+00:00",
  "message": "Example only",
  "diff": "diff --git a/a.txt b/a.txt\n@@ -1 +1 @@\n-old\n+new\n",
  "label": 0,
  "label_type": "operational_incident_72h",
  "label_status": "observed_negative",
  "label_available_at": "2026-01-08T00:00:00+00:00",
  "synthetic": true
}
```

추가 생성 필드: `patch_hash`, `features`, `feature_version`. 사용자 CSV에서 공급한 feature는 그대로 신뢰하지 않고 diff에서 재계산합니다.

## 4. 학습/평가 partition

각 partition은 다음 세 파일을 갖습니다.

```text
inputs.jsonl
labels.jsonl
manifest.json
```

`manifest.json`은 role (`train`, `valid`, `test`, `internal_test`), 파일 SHA-256, 크기·정답 비율·회고적 라벨 여부 등을 기록합니다. 실제 필드는 생성된 manifest를 기준으로 하십시오.

`inputs.jsonl`은 명시적 allowlist로 구성하며 label, label_type, label_status, label_available_at을 모델 입력에서 제거합니다. 원천 canonical row 전체를 LLM에 넣지 않습니다. `load_inputs()`는 label 파일을 읽거나 그 hash를 검증하지 않습니다. **label 파일을 별도 보관한 채 추론하고 나중에 평가할 수 있습니다.** 데이터 준비 작업에는 라벨이 필요하지만 blind inference에는 필요 없습니다.

`labels.jsonl`은 id로만 join하며 불일치·누락·중복은 오류입니다. 학습과 예시 선택은 public train, checkpoint/정형 hyperparameter 선택은 public valid만 허용합니다. 메타데이터가 조작되지 않는다는 운영상 전제가 필요합니다.

## 5. 공개 데이터 시간순 분할

ApacheJIT 원 CSV의 `buggy`를 정답으로 사용하며 라벨을 다시 SZZ로 계산하지 않습니다. 원본 CSV의 과거 개발자/코드 feature와 `fix`는 사용하지 않습니다. 원 Git의 committer date로 시간순 분할하므로 원 연구의 author_date 분할과 동일하지 않습니다.

공개 라벨 가용 시점이 불명확하면 null로 남고 `--allow-retrospective` 없이는 분할을 중단합니다. 이 옵션을 써도 **알려진** 라벨 시각이 해당 fit cutoff 뒤라면 학습/검증에서 제외합니다. null을 임의 시점으로 채우지 마십시오.

release group 또는 정규화 exact patch의 연결 요소가 split 경계를 넘으면 전체 연결 요소를 제외합니다. 미래 행을 과거 train으로 옮기지 않습니다. 동일 split 안 exact patch 중복은 시간순 첫 행을 유지합니다. 서로 다른 맥락의 동일 patch에 서로 다른 결과가 붙는 경우도 있을 수 있으므로, 중복 제외 내역을 보고하고 연구 목적에 맞는 별도 민감도 분석을 고려하십시오.

## 6. 공통 정형 feature v1

| Feature | 정의 |
|---|---|
| lines_added | unified diff hunk 안 `+` 코드 라인 수 |
| lines_deleted | hunk 안 `-` 코드 라인 수 |
| files_changed | `diff --git` 단위 수 |
| directories_changed | 변경 파일의 고유 parent directory 수 |
| change_entropy | 파일별 추가+삭제 라인의 Shannon entropy, base 2; churn 없으면 0 |
| binary_files | binary diff marker가 있는 파일 수 |
| hunks | `@@` hunk 수 |

Header의 `+++`/`---`와 실제 hunk의 `+++...` 코드 라인은 구분합니다. Git export는 no-renames, no-textconv, unified=3을 사용합니다. 내부에서도 같은 옵션으로 export하는 것이 권장됩니다. merge는 기본 제외, 거대한 patch(공개 Git 추출 기준 2MB 초과)는 제외 보고됩니다. rename/mode-only/binary-only 변경은 내용이 빈약하므로 별도 유형별 분석이 필요합니다.

결측은 0으로 위장하지 않습니다. 정형 모델은 공개 train median과 고정된 결측 지표, scaling만 사용합니다. scikit-learn 객체 pickle 대신 JSON으로 변환기·계수를 저장합니다.

## 7. 공통 code view

실제 CodeBERT tokenizer로 `message <= 64`, `total <= 512` token을 기본값으로 적용합니다. 짧은 전체 변경은 보존하고 긴 diff는 head-tail을 남깁니다. tokenizer vocab/backend·special token과 정책의 hash를 기록하고 내부에서 같은 정책인지 확인합니다.

A/B는 동일한 저장 input_ids를 사용합니다. C/D는 대응 text를 사용하고 raw full diff를 추가 제공하지 않습니다. 정형 feature는 전체 diff에서 계산하므로 입력 내용이 완전히 같은 것은 아니며 이는 각 모델 계열의 정보 표현 차이입니다. 모든 모델의 예측 대상 ID 집합은 같습니다.

## 8. 결과 contract

예측은 id, score, status를 필수로 갖고 오류 시 score는 null입니다. LLM 출력은 0~100의 ordinal risk score이며 100으로 나눈 값도 calibrated probability가 아닙니다. 모델 artifact의 label_type과 평가 target label_type이 다르면 명시적인 `--allow-label-transfer`가 필요합니다.

비교는 같은 prepared inputs hash와 정확히 같은 ID 집합에서 수행합니다. 실패가 있으면 기본적으로 중단합니다. 명시적 partial 비교는 모든 모델 성공 교집합과 제외된 양성 수를 보고하며, 전체 내부 모집단 성능을 대체하지 않습니다.
