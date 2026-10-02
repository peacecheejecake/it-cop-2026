# jit-b3s-internal — 사내 코드로 B3-S 모델 시험 평가

공개 데이터(JIT-Defects4J, Java 21개 프로젝트)로만 학습한 **B3-S**(CodeBERT 전체 fine-tuning + jit14 변경 지표 결합)를
사내 git 저장소의 커밋에 적용해 보고, 직접 단 라벨로 성능을 재기 위한 오프라인 패키지다.

- 모델: study `public-comparison-v3`, freeze `b6afa59f3a2a9550`, B3-S seed 43 (`733fb98bec2f83f4`).
  공개 validation AP 0.696. 공개 test AP는 seed 3개 평균 0.598이다.
- 사내 데이터로는 **학습·보정·threshold 조정을 하지 않는다.** 그런 기능도 없다. 점수는 공개 데이터 기준의 보정 안 된 위험 점수이며 확률이 아니다.
- `predict`는 실행하는 동안 네트워크를 코드 수준에서 차단한다.

## 1. 구성

| 경로 | 내용 |
|---|---|
| `install.sh` | 오프라인 설치(동봉 Python → `.venv` → wheel 설치 → 모델 번들 검증) |
| `smoke.sh` | 임시 git 저장소로 전체 흐름을 자가 점검 |
| `jit.sh` | `extract / sheet / predict / eval` 래퍼 |
| `python/` | CPython 3.11 (python-build-standalone, Linux x86_64) |
| `wheels/` | 의존성 wheel 전체(torch 2.14.1+cu126 포함)와 `codebert_diff_lab` wheel |
| `models/codebert-base/` | 기반 인코더 `microsoft/codebert-base` (revision `3b0952fe…`) |
| `bundle/bundle.tar.gz` | B3-S fine-tuned 가중치, head, 전처리 통계, frozen threshold, freeze 기록 |
| `docs/` | 추출기 충실도 검증 결과, offline 평가 설계 문서 |
| `src/` | 이 패키지를 만든 코드 커밋의 소스(감사용) |
| `PROVENANCE.json`, `SHA256SUMS` | 출처와 파일 해시 |

## 2. 요구 사항

- Linux x86_64, glibc 2.28 이상(RHEL/Rocky 8+, Ubuntu 20.04+), `git`, `sha256sum`, `unzip`
- 디스크: 압축 해제 후 약 10 GB(설치 후 `.venv` 포함)
- GPU는 선택 사항이다. torch는 CUDA 12.6 빌드이므로 NVIDIA 드라이버 525 이상이 필요하다.
  - GPU가 없거나 드라이버가 낮으면 `cpu`로 돌리면 된다. 수천 커밋이면 CPU로도 수십 분 수준이다.
- 인터넷·PyPI는 필요 없다.

## 3. 설치

```bash
unzip jit-b3s-internal.zip && cd jit-b3s-internal
./install.sh          # 해시 검사 → Python 풀기 → .venv → wheel 설치 → 번들 검증
./smoke.sh cpu        # 끝에 "smoke ok"가 나오면 정상 (GPU 확인은 ./smoke.sh cuda)
```

## 4. 사용 순서

라벨을 다는 동안 모델 점수를 보지 않는다. 점수를 보고 라벨을 달면 평가가 낙관적으로 나온다.

```bash
# (1) 커밋 추출: 라벨 없는 입력(dataset.parquet)과 메타데이터(meta.parquet)
./jit.sh extract work/extract --repo /src/svc-a --repo /src/svc-b --since 2025-01-01 --until 2025-12-31

# (2) 라벨 시트 만들기: 점수 없음. 커밋이 많으면 무작위 표본(--sample)
./jit.sh sheet work/extract work/labels.csv --sample 300

# (3) labels.csv의 label 칸 채우기 (엑셀로 열어도 된다. UTF-8 BOM)
#     1 = 결함을 유발한 커밋, 0 = 깨끗한 커밋, 빈칸 = 판단 보류(평가에서 제외)

# (4) 점수 계산 (라벨과 무관하게 언제 돌려도 되지만, 결과는 라벨링이 끝난 뒤 본다)
./jit.sh predict work/extract work/pred cuda      # GPU가 없으면 cpu

# (5) 평가
./jit.sh eval work/extract work/pred work/labels.csv work/eval
```

`jit.sh`를 쓰지 않고 `.venv/bin/diff-lab internal extract|label-sheet|label-eval`과 `.venv/bin/diff-lab predict`를 직접 불러도 된다. 옵션은 `--help`로 본다.

### extract 옵션

| 옵션 | 의미 |
|---|---|
| `--repo PATH` (반복 가능) | 로컬 clone 또는 bare 저장소. 원격 접근은 하지 않는다 |
| `--project NAME` (반복 가능) | 저장소 이름. 생략하면 디렉토리 이름. change_id는 `<project>:<commit sha>` |
| `--rev` | 기준 브랜치/ref (기본 `HEAD`). 이력 지표는 이 ref에서 도달 가능한 전체 이력으로 계산한다 |
| `--since`, `--until` | 평가 대상 커밋 기간. git의 committer date 기준 |
| `--max-commits N` | 최근 N개까지만 |
| `--extensions` | 코드 파일 확장자(예: `.java,.kt`). 기본은 주요 언어 전체 |
| `--author-key name\|email` | 개발자 식별 기준(기본 name, 공개 데이터와 같다) |

- merge 커밋, 코드 파일을 건드리지 않은 커밋, 주석·공백만 바꾼 커밋은 제외된다. 개수는 `extract-manifest.json`에 남는다.
- **모델은 Java로만 학습했다.** 다른 언어도 같은 규칙으로 추출하고 점수를 내지만, 분포 밖이다. 평가 결과는 언어별(`primary_language`)로 따로 본다.

## 5. 라벨링 기준

- `1`: 이 커밋이 나중에 수정된 결함을 들여왔다.
  - 찾는 방법: 버그 수정 커밋(이슈/장애 티켓에 연결된 것)에서 수정·삭제된 줄을 `git blame`으로 거슬러 올라가 그 줄을 마지막으로 바꾼 커밋을 찾는다(수동 SZZ).
  - 공개 학습 데이터의 라벨도 같은 방식이다.
- `0`: 관찰 기간 동안 이 커밋에서 비롯된 결함 수정이 없었다.
- 빈칸: 판단할 근거가 부족하다. 평가에서 빠진다.
- **관찰 기간을 둔다.** 최근 커밋은 결함이 아직 드러나지 않았을 수 있다. `--until`을 오늘보다 6개월 이상 이전으로 잡는 것을 권장한다.
- 표본 크기: 결함 비율이 10% 안팎이면 300개를 라벨해도 양성은 30개 정도다. AP의 95% 구간이 넓게 나오는 것이 정상이다. 결과의 `ap_bootstrap_95ci`를 함께 본다.

## 6. 결과 읽기

`work/eval/label-eval.json`과 `work/eval/scored-labeled.csv`가 생긴다.

| 지표 | 의미 |
|---|---|
| `ap` | Average Precision. 무작위 순위의 기대값은 양성 비율(`prevalence`)이다 |
| `lift_ap_over_prevalence` | AP / 양성 비율. 1보다 클수록 무작위보다 낫다 |
| `roc_auc` | 0.5가 무작위 |
| `recall_at_5pct`, `recall_at_10pct` | 점수 상위 5%/10% 커밋만 검토했을 때 잡히는 결함 커밋의 비율 |
| `at_threshold` | 공개 validation에서 정한 threshold(0.388)를 넘으면 위험으로 볼 때의 precision/recall/F1 |
| `by.project`, `by.primary_language` | 그룹별 지표(라벨 30개 미만 그룹은 개수만) |

공개 test 참고값(seed 3개 평균): AP 0.598(양성 비율 0.087, lift 약 6.9), ROC-AUC 0.909, Recall@10% 0.615.
사내 데이터에서 이보다 낮게 나오는 것 자체가 실험 결과다. 원인 분리를 위해 언어별·프로젝트별 결과를 먼저 본다.

## 7. 공개 데이터 표현과의 일치도 (추출기 검증)

모델은 JIT-Defects4J 패키지의 전처리된 표현만 봤다. 그래서 `extract`는 그 표현을 git에서 다시 만들도록 맞췄다.

- 줄: 주석 줄 제거, 연산자·구두점을 한 글자씩 분리, `_`를 공백으로 바꿈, 파일·순서 정보 없이 집합으로 다룸
- 메시지: 전체 메시지에 같은 토큰화 적용
- jit14: Kamei 변경 지표

공개 test 커밋을 캐시된 공개 저장소에서 다시 추출해 비교한 결과는 `docs/extract-fidelity-public-test.json`에 있다.

<!-- FIDELITY -->

## 8. 반출·보안 주의

- `dataset.parquet`, `meta.parquet`, `labels.csv`, `predictions.parquet`, `scored-labeled.csv`에는 사내 코드 줄, 커밋 메시지, 커밋 해시가 들어 있다. 사내 반출 정책을 따른다.
- `label-eval.json`에는 집계 지표와 프로젝트·언어 이름만 들어 있다.
- 이 패키지는 사내 데이터를 어디에도 보내지 않는다. `predict` 중 네트워크 차단은 코드 수준의 보호일 뿐이므로, 조직 차원의 egress 차단을 대신하지 않는다.

## 9. 문제 해결

| 증상 | 조치 |
|---|---|
| `sha256sum: WARNING ... did NOT match` | 반입 중 파일이 손상됐다. 다시 반입한다 |
| `cuda available: False` | `cpu`로 실행한다. 드라이버가 525 미만이거나 GPU가 보이지 않는 경우다 |
| `GLIBC_2.28 not found` | OS가 너무 오래됐다(RHEL 7 등). 더 새로운 서버에서 실행한다 |
| `extract`가 느림 | 이력이 큰 저장소다. 전체 이력의 numstat을 한 번 읽어야 해서 처음엔 시간이 걸린다. `--since`는 대상만 줄이고, 이력 계산 범위는 줄이지 않는다 |
| `labels must be 1 ..., 0 ... or empty` | label 칸에 다른 값(예: `TRUE`, `y`)이 있다 |
| `labeled change_id(s) have no prediction` | 다른 extract의 라벨 파일이다. 같은 extract 디렉토리로 predict/eval 한다 |
