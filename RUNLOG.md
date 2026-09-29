# RUNLOG — 002-no-hadoop

브랜치: `exp/002-no-hadoop` (from `main` @ `783677c`)

## 목적

001에서 fine-tuned CodeBERT(B)의 공개 test 우위가 ApacheJIT의 Hadoop 라벨 이상(`apache/hadoop`은 전부 음성, 같은 코드인 `hadoop-hdfs`/`hadoop-mapreduce`는 2012년 이후 약 90% 양성)에서 온다는 점을 발견했다. 002는 **데이터만 바꾼다.** Hadoop 계열 3개 저장소를 train/valid/test 모두에서 빼고, 나머지는 001과 똑같이 두어 비교가 어떻게 바뀌는지 본다.

## 사전 등록 (학습·평가 전에 확정, 2026-09-29)

| 항목 | 값 | 001과의 차이 |
|---|---|---|
| 원본 | `experiments/.cache/canonical/apachejit-full` (001 최종 build, 106,659개) | 같음 |
| **제외** | `apache/hadoop`, `apache/hadoop-hdfs`, `apache/hadoop-mapreduce` (저장소 단위, 라벨과 무관하게 전부) | **새로 추가** |
| split | `split-public --train-before 2016-01-01T00:00:00Z --valid-before 2017-01-01T00:00:00Z --allow-retrospective` | 같음 |
| 입력 | CodeBERT 512 / 메시지 64, head-tail | 같음 |
| 모델 | rule, 정형 LR, A frozen, B finetune. README 기본 하이퍼파라미터, seed 42 | 같음 |
| 선택 | public validation AP만 사용 | 같음 |
| 주 지표 | 공개 test의 AP와 **프로젝트별 AP 평균**(양성 20개 이상인 프로젝트) | 프로젝트별 AP를 주 지표로 올림 |
| 보조 | ROC-AUC, Recall/Precision@5·10%, week-cluster bootstrap 500회, project-prior 기준선 | project prior 추가 |

**제외 목록을 정한 근거**: 원본 CSV의 프로젝트×연도별 양성 비율(아래). Hadoop 계열만 구조적으로 극단적이다. `hadoop`은 모든 연도에서 0%, hdfs와 mapreduce는 2012년 이후 72~97%다. 다른 프로젝트는 연도별로 오르내리지만 0% 또는 90% 이상으로 고정된 패턴은 없다.

```
project                     03   04   05   06   07   08   09   10   11   12   13   14   15   16   17   18   19   total
apache/hadoop                .    .    .    .    .    .    0    0    0    0    0    0    0    0    0    0    0    0.0% n=11964
apache/hadoop-hdfs           .    .    .    .    .    .   24   44   52   94   97   91   90   86   84   59   68   76.4% n=2907
apache/hadoop-mapreduce      .    .    .    .    .    .   26   38   72   94   91   77   78   74    .    .    .   63.4% n=1321
(나머지 12개 프로젝트: 연도별 2~84%, 2018~19년에는 모든 프로젝트에서 하락 → 라벨 우측 절단)
```

**독립성에 대한 고지**: 이 제외는 001의 **공개 test 결과를 보고 나서** 설계했다. 따라서 002의 공개 test는 설계 과정과 무관한 final test가 아니다. 다만 제외 기준 자체는 train 라벨만으로도 확인할 수 있는 데이터 결함(`apache/hadoop` train 5,026개 전부 음성)이고, 모델·하이퍼파라미터는 001과 같게 고정했다. 프로젝트별 AP를 주 지표로 올린 것도 001의 사후 분석에서 나온 결정이다.

## 실행 기록

환경: 001과 같다(Python 3.13.5 uv venv, `.[neural,dev]`, `pytest -q` 55개 통과). 공유 캐시를 AGENTS.md §1.4의 방식으로 심링크했다.

### 필터 (`tools/filter_records.py`)
```bash
python ../tools/filter_records.py .cache/canonical/apachejit-full/records.jsonl .cache/canonical/apachejit-no-hadoop \
  --exclude-repos apache/hadoop apache/hadoop-hdfs apache/hadoop-mapreduce
```
- 제외: hadoop 11,962 / hadoop-hdfs 2,907 / hadoop-mapreduce 1,321, 남은 레코드 90,469개.
- 결과는 이후 실험(003 등)이 재사용하도록 공유 캐시 `experiments/.cache/canonical/apachejit-no-hadoop/`에 두었다. `filter_report.json`에 원본과 출력의 SHA-256, 원본 build_report를 기록했다.

### split-public
```bash
riskbench split-public --records data/canonical/apachejit-no-hadoop/records.jsonl --out data/splits/public \
  --train-before 2016-01-01T00:00:00Z --valid-before 2017-01-01T00:00:00Z --allow-retrospective
```

| split | n | 양성 | 비율 | 001 |
|---|---|---|---|---|
| train | 56,870 | 16,682 | 29.3% | 65,478 / 29.4% |
| valid | 8,709 | 2,843 | 32.6% | 10,480 / 29.4% |
| test | 24,178 | 5,532 | 22.9% | 29,938 / 19.4% |

- 제외 712건(`duplicate_patch` 708, `group_or_exact_patch_crosses_split` 4).
- **test 24,178건은 001 test에서 Hadoop 계열을 뺀 부분집합과 크기가 정확히 같다.** 그래서 001 모델을 이 부분집합으로 평가한 값(정형 LR 0.580, B 0.551)과 002 모델을 같은 행에서 직접 비교할 수 있다. 행 ID가 같은지는 평가 후 확인한다.

### 데이터 특성: ApacheJIT의 커밋 포함 방식 (사용자 질문으로 조사, 2026-09-29)

결함 비율(원본 전체 26.5%, 이 실험 train 29.3%)이 높은 이유를 확인했다. 캐시된 저장소에서 CSV 기간의 `git log HEAD --no-merges`와 CSV를 비교했다.
- CSV는 같은 기간 non-merge 커밋의 **일부만** 담는다. 대부분 50~61%이고 kafka 35%, zeppelin 36%, zookeeper 41%이며 **spark는 6%**(1,465 / 24,200)다.
- 빠진 커밋의 특징(프로젝트별 250개 표본):
  - camel: 빠진 커밋 중 코드 파일을 건드리는 비율 24%, 중앙값 5줄 → 주로 **비코드 커밋을 걸러낸 것**이다.
  - kafka: 81%가 코드 변경, 중앙값 40줄(포함된 쪽 108줄).
  - spark: 84%가 코드 변경, 중앙값 38줄(포함된 쪽 161줄) → 코드 변경인데도 **작은 커밋이 많이 빠졌다.**
- spark의 결함 비율 43.1%는 기간 내 전체 non-merge 커밋 대비 2.6%에 해당한다. 결함 커밋은 대부분 포함하고 결함 없는 커밋을 줄여 뽑은 것으로 **추정**되지만, 빠진 커밋의 라벨을 모르므로 확정할 수 없다.
- 의미: (1) AP, precision 같은 기저율 민감 지표의 절대값은 운영 환경으로 옮길 수 없다. (2) 포함 여부가 커밋 크기와 연관되어 있어, 크기 기반 모델(rule, 정형 LR)의 성능에 데이터셋 구성 효과가 섞일 수 있다. (3) 같은 test 안에서 모델끼리 비교하는 것은 유효하지만, "이 benchmark 안에서"로 한정해 해석한다.
- 002의 설계는 바꾸지 않는다(001과 같은 성질). 모든 코드 변경 커밋에 SZZ를 다시 적용해 라벨을 새로 만드는 것은 별도 실험 후보로 둔다.

### 이후 단계 (`scripts/run_pipeline.sh`, 08:43 시작)
prepare(3개 병렬) → tabular(CPU)와 A→B(MPS, `tools/train_with_progress.py --every 200`) → predict_suite(공개 test, bootstrap 500) → leak_check, sensitivity, paired bootstrap 순으로 실행한다. 모델과 하이퍼파라미터 명령은 001과 같다. 단계별 로그는 `baseline/runs/logs/`에 남는다.

| 단계 | 시각 | 결과 |
|---|---|---|
| prepare | 08:43 → 08:48 | 512 토큰 초과로 잘린 비율: train 52,160 / 56,870 (91.7%), valid 7,761 / 8,709 (89.1%), test 21,549 / 24,178 (89.1%) |
| 정형 LR | 08:48 시작 | C = 0.1, **validation AP 0.670** (001: 0.588, valid 기저율이 0.294에서 0.326으로 바뀌었으므로 직접 비교하지 않음) |
| A frozen | 08:48 → 09:17 (29분) | validation AP epoch 1~5: 0.536 → 0.551 → 0.559 → 0.566 → **0.570**(epoch 5 선택). 001처럼 **계속 상승해 미수렴** |
| B finetune | 09:17 시작 | 래퍼 출력이 정상이다(800 배치마다). 7.8~8.2 ex/s |
| B epoch 1 | 11:18 학습 종료(2시간 1분) + validation | validation AP **0.699** |
| B epoch 2 | 약 13:25 | validation AP **0.720** |
| B epoch 3 | 약 15:35 | validation AP 0.708 (하락) |
| B epoch 4 | 약 17:20 | validation AP 0.697 → patience 2로 조기 종료, **epoch 2 선택**(0.720). B 학습에 약 8시간 |
| predict_suite | 17:21 → 17:41 | 공개 test, bootstrap 500 |
| 분석 | 17:41 → 17:42 | leak_check, sensitivity, paired bootstrap → `RESULTS.md` |
