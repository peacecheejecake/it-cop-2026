# DATASETS — 공유 캐시(`experiments/.cache/`) 데이터 출처와 주의사항

실험들이 공유하는 공개 데이터의 출처, 무결성, 라이선스, 알려진 문제를 기록한다. 캐시 자체는 git-ignore 대상이므로, 이 문서가 출처 기록의 원본이다. 모든 데이터는 **공개 학습·모델 선택용**이며, 내부 데이터는 여기 두지 않는다(baseline README §6).

## 1. ApacheJIT v2

| 항목 | 값 |
|---|---|
| 출처 | Zenodo record 5907847, `apachejit_dataset_replication.zip` |
| 무결성 | MD5 `528bf0ee04b15976be6bf15f8efddc65`(riskbench pin과 일치), CSV SHA-256 `5097cbbb…d4e0` |
| 규모 | 106,674 커밋, 15개 Apache 프로젝트(Java), buggy 26.5% |
| 캐시 | `raw/apachejit-v2/`, `git/apache/<15 repos>.git`, `canonical/apachejit-full/`(106,659개 채택), `canonical/apachejit-no-hadoop/`(90,469개) |
| 라벨 | SZZ 계열 결함 유발 커밋. 라벨 확정 시각 없음 → 회고적 benchmark(`--allow-retrospective`) |

알려진 문제(001/002 RUNLOG):
- **Hadoop 라벨 이상**: `apache/hadoop`은 12,964개 전부 음성이고, 같은 코드인 `hadoop-hdfs`와 `hadoop-mapreduce`는 2012년 이후 약 90%가 양성이다. project가 라벨에 따라 정해진 것으로 보인다 → 평가에서는 제외한다(002~).
- `hadoop-hdfs`/`hadoop-mapreduce` GitHub 저장소는 축소본이다. 캐시에서 `objects/info/alternates`로 `hadoop.git`에 연결해 두었다.
- **커밋 포함 방식이 무작위가 아니다**: 저장소 non-merge 커밋의 6~61%만 포함한다(spark 6%). 작은 커밋과 비코드 커밋이 많이 빠져 기저율이 부풀려졌을 가능성이 있다.
- 최근 연도일수록 양성 비율이 하락한다(라벨 우측 절단).

## 2. JIT-Defects4J (JIT-Fine 패키지)

| 항목 | 값 |
|---|---|
| 출처 | GitHub `jacknichao/JIT-Fine` `data.zip`(저장소 커밋 `584799fd`, 2023-08-25) |
| 무결성 | `data.zip` SHA-256 `9e5ca1a3…2b47`, 변환 CSV SHA-256 `37e32598…fe30` |
| **라이선스** | **저장소에 라이선스가 명시되어 있지 않음**(GitHub `license: null`). 연구 분석용으로만 받았다. **사내 반입·사용 전에 정책 확인이 필요하다.** 원 프로젝트 코드는 각 Apache 저장소의 라이선스(Apache-2.0)를 따른다. |
| 규모 | **27,319 커밋, buggy 2,332건(8.5%)**, 21개 Java 프로젝트: commons-* 16개, ant-ivy, parquet-mr, opennlp, giraph, gora. 2001~2019년 |
| ApacheJIT과 겹침 | 프로젝트 기준 **0개**. 커밋 해시와 diff 내용 기준 겹침은 build 후 확인 |
| 라벨 | LLTC4J(수작업 버그 수정 라인)를 출발점으로 결함 유발 커밋·라인을 역추적해 확장했다. 역추적 오류가 있을 수 있다 |
| 원래 split | **시간순이 아니다**(train 2001~2015, test 2001~2019로 겹침) → 사용하지 않고 우리 프로토콜로 다시 나눈다 |
| 캐시 | `raw/jit-defects4j/{data.zip, jit_defects4j_labels.csv, repos.txt}`, `git/apache/<21 repos>.git` |

**pickle 안전 처리**(`tools/convert_jit_defects4j.py`):
- 배포 형식은 pickle뿐이다. 먼저 `pickletools`로 참조하는 전역 객체를 확인했다. pandas DataFrame 구성 요소, numpy 배열 복원 함수, `builtins.slice`만 있다.
- **이것만 허용하는 Unpickler**로 읽는다. 그 밖의 전역 객체는 차단하고 에러를 낸다.
- repo, commit, label, 원래 split, timestamp만 내보낸다. 작성자 이름과 이메일은 버린다.
- diff는 pickle에서 가져오지 않고, riskbench `build-apache`가 원본 git에서 다시 추출한다. 그래야 입력 형식이 ApacheJIT과 같아진다.

### 알려진 문제: 패키지의 줄 집합이 라벨에 따라 다르다 (2026-10-02 발견)

- 버그 커밋은 `added_code/removed_code`에 커밋의 일부 파일만 들어 있다(버그 커밋의 70%에서 줄의 20% 이상이 빠짐). 정상 커밋은 거의 전부 들어 있다(0.8%만 빠짐).
- 그래서 "텍스트가 커밋을 얼마나 덮는가"만으로 test AP 0.324가 나온다(기저율 0.087). 실제 git diff에는 이 신호가 없다.
- 패키지 텍스트로 학습·평가한 결과(study v2, v3의 텍스트 variant, 013의 EvidenceView arm, 014)는 이 영향을 받는다.
- 대신 쓰는 snapshot: `jitd4j-git1`(패키지의 ID·라벨·split은 그대로, 메시지·줄·jit14는 `.cache/git/jd4j`에서 다시 추출).
- 점검 도구: `tools/representation_leak_check.py <snapshot_dir>`. 상세: `codebert-diff-lab/docs/leak-finding-2026-10-02.md`.

## 3. 후보 (미확보, 외부 대화에서 제시됨 — 수치는 미검증)

| 후보 | 내용 | 용도 |
|---|---|---|
| ISSTA'21 JIT-DP 확장 benchmark(`ZZR0/ISSTA21-JIT-DP`) | Qt(C++), OpenStack(Python), Eclipse Platform/JDT, Gerrit(Java), Go. 약 31만 커밋 | **언어 간 전이 검증**. 전처리 형식은 토큰 단위라 원본 git에서 다시 만들어야 할 가능성이 크다 |
| JITLine(OpenStack, Qt) | ISSTA'21의 부분집합과 겹침 | 겹침을 확인한 뒤 사용 |
| ReDef | C/C++ 22개 프로젝트, revert 기반, 함수 단위. 라벨 구성에 LLM 관여 | 라벨 방식에 대한 강건성 보조 검증 |
| CAT 수정 라벨 | JIT-Defects4J 리팩터링 보정 라벨. 도구만 공개되어 있고 데이터 경로는 미확인 | 라벨 노이즈 검증 |
| SmartSHARK | 이슈, 커밋, 리뷰가 연결된 DB | 라벨 감사와 자체 데이터 구축 |

**데이터셋을 섞을 때 원칙**:
1. 데이터셋마다 표본 추출 방식이 달라 기저율이 다르다(ApacheJIT 26.5%, JIT-Defects4J 8.5%). 합쳐 학습하면 "어느 데이터셋 출신인가"가 지름길 신호가 될 수 있다(Hadoop과 같은 함정). 따라서 **데이터셋별로 층화해 평가**한다.
2. 커밋 해시와 diff 내용 기준으로 중복을 검사한다.
3. 결함 **수정** 패치는 결함 **유발** 양성으로 쓰지 않는다.

## 4. 언어 적용 범위 메모

내부 코드베이스는 여러 언어로 되어 있다. 확보한 두 데이터셋은 모두 Java다.
- 정형 feature(크기, 파일, 엔트로피, hunk)는 언어와 무관하다.
- CodeBERT는 사전학습 언어(Python, Java, JS, PHP, Ruby, Go)에 의존한다. 사전학습에 없는 언어에는 약할 수 있다.
- 내부 평가는 언어별로 나눠 보고한다. 언어 간 전이는 ISSTA'21(Python, Go, C++)로 사전 검증하는 것을 우선한다.
