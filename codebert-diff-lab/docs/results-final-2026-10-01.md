# CodeBERT Diff Lab — 최종 결과 (spec v0.2, M0~M6) — 2026-10-01

study `public-comparison-v3`(freeze `b6afa59f3a2a9550`)의 최종 결과다. primary variant 9개를 모두 freeze한 뒤 public test를 **한 번만** 평가했다. 실험별 상세 기록은 각 브랜치(`exp/006`~`exp/011`)의 RUNLOG/RESULTS에 있다. validation 단계만 다룬 중간 문서 `results-2026-10-01.md`는 이 문서로 대체되었다.

## 요약

- **최고 모델**은 CodeBERT 전체 fine-tuning + 정형 특징(B3-S)으로, test AP **0.598 ± 0.023**이다. TF-IDF보다 +0.158, CodeBERT를 고정한 모델보다 +0.124 높고, 두 차이 모두 프로젝트 bootstrap 95% 구간이 0을 포함하지 않는다.
- **diff MLM 추가 사전학습(B4-S, RQ2)은 추가 가치가 없었다**(−0.008, 구간이 0을 포함).
- **RMI 추가(B5-S, RQ3)도 추가 가치가 없었고, 오히려 약간 해로울 수 있다**(−0.010, 구간 [−0.020, −0.002]).
- **로컬 LLM Qwen2.5-Coder-7B를 학습 없이 점수기로만 쓰면 약하다.** zero-shot 0.125, 4-shot 0.172로, 정형 특징 LR(0.215)보다도 낮다. 다만 few-shot 예시는 효과가 있다(+0.047).
- **경과**: test를 열기 전에 프로토콜을 v2에서 v3로 바꿨다. v2처럼 encoder와 head가 lr 1e-5를 공유하면 CodeBERT 계열이 체계적으로 덜 학습된다(중간 문서와 `exp/009` 참고).

## 설정

- **데이터**: JIT-Defects4J, Java 프로젝트 21개, 커밋 27,319개.
  - split `upstream-clean2`: valid/test는 제공된 구성원을 그대로 쓴다. 평가 데이터와 동일한 train 행은 제외했고, CPT-dev는 중복 그룹 단위로 뽑았다.
  - train 16,184 / valid 5,465 / test 5,480(양성 475).
- **입력**: EvidenceView `message-add-del-text-v2`(512 토큰)와 정형 특징 14개(jit14). B1, B2~B5, L0/L1이 같은 evidence를 공유한다.
- **모델 구성**
  - **B0**: 정형 특징 LR, LGBM.
  - **B1**: 문자 n-gram TF-IDF + 정형 특징, LR.
  - **B2~B5**: CodeBERT(`3b0952fe`, 가중치 sha256 고정) + fusion head. encoder lr 1e-5, head lr 1e-3, validation AP 기준 조기 종료(patience 5).
  - **B4/B5 CPT**: non-padding 토큰 10M. B5는 같은 총예산 안에서 MLM과 RMI를 1:1로 번갈아 학습한다.
  - **L0/L1**: Qwen2.5-Coder-7B-Instruct(`c03e6d35`, Apache-2.0, bf16). 두 후보 라벨의 log-likelihood로 점수를 매긴다. L1은 public train에서 뽑은 고정 4-shot 세트 3개(양성 2 / 음성 2)를 쓴다.
- **seed**: 42/43/44. L0는 1회 실행이고, B0/B1은 결정적이다.
- **선택**: checkpoint와 threshold는 public validation으로 골랐다. test는 freeze 뒤 한 번만 썼다.

## test 결과

public test는 JIT-Defects4J `upstream-clean2` test로, **변경 5,480개, 양성 475개(기저율 0.087), 프로젝트 21개**다. primary variant 9개를 모두 freeze한 뒤 **한 번만** 평가했다. threshold는 freeze 때 고정한 validation max-F1 값이다. 수치는 `results/report/report-corrected.json`(브랜치 `exp/011`)에서 가져왔다.

| variant | test AP (평균 ± sd) | ROC-AUC | Recall@5% | Recall@10% | F1@thr | (validation AP) |
|---|---|---|---|---|---|---|
| B0-LR | 0.215 | 0.723 | 0.160 | 0.286 | 0.293 | 0.319 |
| B0-LGBM | 0.197 | 0.676 | 0.154 | 0.267 | 0.257 | 0.211 |
| B1-TFIDF-S | 0.440 | 0.852 | 0.307 | 0.491 | 0.453 | 0.546 |
| B2-S 고정 CodeBERT | 0.475 ± 0.018 | 0.851 | 0.347 | 0.528 | 0.481 | 0.616 |
| **B3-S 전체 FT** | **0.598 ± 0.023** | **0.909** | **0.425** | **0.615** | **0.580** | 0.687 |
| B4-S MLM CPT + FT | 0.590 ± 0.034 | 0.903 | 0.412 | 0.608 | 0.565 | 0.674 |
| B5-S MLM+RMI CPT + FT | 0.580 ± 0.029 | 0.902 | 0.410 | 0.590 | 0.555 | 0.669 |
| L0-S Qwen2.5-Coder-7B zero-shot | 0.125 | 0.634 | 0.074 | 0.162 | 0.207 | 0.140 |
| L1-S Qwen 4-shot | 0.172 ± 0.006 | 0.677 | 0.144 | 0.244 | 0.226 | 0.204 |

sd는 학습 seed 3개(L1은 demo seed 3개) 간 편차다. B0/B1은 결정적이고, L0는 1회 실행이다.

## 사전 등록한 쌍별 비교 (test AP, seed 대응, 프로젝트 bootstrap 2,000회)

| 비교 | 평균 차이 | seed별 | 프로젝트 bootstrap 95% 구간 | P(차이 ≤ 0) |
|---|---|---|---|---|
| B3 − B2 (fine-tuning의 효과) | **+0.124** | +0.080 / +0.158 / +0.133 | [+0.081, +0.164] | 0.000 |
| B4 − B3 (MLM CPT의 효과, RQ2) | −0.008 | −0.021 / −0.005 / +0.001 | [−0.025, +0.008] | 0.84 |
| B5 − B4 (RMI의 효과, RQ3) | −0.010 | −0.005 / −0.012 / −0.013 | [−0.020, −0.002] | 0.99 |
| B5 − B3 | −0.018 | −0.025 / −0.017 / −0.013 | [−0.039, +0.002] | 0.96 |
| L1 − L0 (few-shot의 효과) | **+0.047** | +0.053 / +0.040 / +0.046 | [+0.022, +0.081] | 0.000 |
| B1 − B0-LR (텍스트의 효과) | **+0.225** | (결정적) | [+0.164, +0.279] | 0.000 |
| B2 − B1 (고정 encoder 대 TF-IDF) | +0.035 | +0.052 / +0.016 / +0.036 | [−0.002, +0.076] | 0.03 |
| B3 − B1 | **+0.158** | +0.131 / +0.174 / +0.169 | [+0.121, +0.194] | 0.000 |
| B3 − L1 | **+0.427** | +0.393 / +0.449 / +0.438 | [+0.356, +0.485] | 0.000 |

프로젝트 bootstrap 구간은 프로젝트를 다시 뽑았을 때의 불확실성이다. seed별 편차(seed 분산)와는 다른 양이다.

## 결론 (봉인된 test, study v3)

1. **CodeBERT 전체 fine-tuning(B3-S)이 가장 좋다.** TF-IDF보다 AP가 +0.158, 고정 표현보다 +0.124 높고, 두 구간 모두 0을 포함하지 않는다. 프로젝트 21개 중 19개에서 B1과 같거나 높다.
2. **RQ2 diff MLM CPT(10M 토큰)는 추가 가치가 없다.** B4 − B3 = −0.008이고 구간이 0을 포함한다. 프로젝트별로도 B4가 B3를 앞선 곳이 21개 중 11개로, 사실상 동률이다.
3. **RQ3 RMI 추가는 추가 가치가 없고, 약간 해로울 수 있다.** B5 − B4 = −0.010, 구간 [−0.020, −0.002]. 같은 총예산이라 MLM 노출이 절반으로 줄었고, RMI 학습도 seed마다 불안정했다(dev 정확도 0.57~0.70).
4. **로컬 LLM(Qwen2.5-Coder-7B)은 학습 없이 점수기로만 쓰면 약하다.**
   - L0 0.125, L1 0.172는 기저율(0.087)보다는 높지만 정형 특징 LR(0.215)보다 낮다.
   - 4-shot 예시는 효과가 있다(+0.047, 구간이 0을 포함하지 않음).
   - 모델 1개, 템플릿 1개, 점수기 1개로 얻은 결과이므로 "encoder가 LLM보다 낫다"로 일반화하지 않는다(사양 §8).
5. **고정 encoder(B2)는 TF-IDF보다 +0.035 높지만, 구간이 0에 걸쳐 있다.** test에서의 격차는 validation보다 작다.

## validation과 test의 차이

모든 variant가 test에서 떨어진다. B3는 −0.089, B1은 −0.106이고, epoch 선택이 없는 결정적 모델 B0-LR도 −0.104 떨어진다.

- B0-LR까지 이만큼 떨어지므로, 차이의 대부분은 epoch 선택 편향보다 **validation과 test 코호트 자체의 차이**로 보인다. 원인(예: 프로젝트 구성)은 분석하지 않았다.
- 순위는 validation과 test에서 일관된다. B2−B1 격차(validation +0.070 → test +0.035)와 B4−B3 격차(−0.013 → −0.008)만 test에서 줄었다.
- 프로젝트별 AP는 보고서의 `per_project_test_ap`에 있다.

## 한계

- **데이터**: 공개 데이터셋 1개(JIT-Defects4J, Java 프로젝트 21개)만 썼다. 제공된 split은 시간순이 아니고, 라이선스는 확인되지 않았다.
  - 추가 발견: JIT-Fine 데이터의 코드 줄은 원본이 아니라 **토큰 단위로 띄어 쓴 형태**(`LOG . error (`)다. 모든 모델이 같은 입력을 받았으므로 비교의 공정성에는 영향이 없다(`exp/012` 참고).
- **프로토콜**: v3의 head lr 1e-3은 validation 결과를 본 뒤 정했다(공개). v2에서는 CodeBERT 계열이 체계적으로 덜 학습되었다.
- **CPT 규모**: 10M 토큰만 시도했고 확대는 하지 않았다(효과가 없으면 중단해도 된다고 사양이 허용한다).
- **L0/L1 재현성**: freeze 때 정확한 재현이 아니라 허용오차(bf16 배치 구성에 따른 차이)로 검증했다.
- **사내 데이터**: 아직 평가하지 않았다. 아래 "사내 평가 준비" 참고.

## 완료 기준(DoD) 상태

| DoD | 상태 | 근거 |
|---|---|---|
| DoD-Core (B0~B5) | **완료** | 모든 variant를 freeze하고 test를 1회 평가했다. evidence가 일치하고(query 내용 hash `7d8774b4…` 동일), 정형 열이 같고, 예측 실패가 없고, label ledger가 있다. `exp/011`의 `report-final.md` 참고. |
| DoD-Comparative (+L0/L1) | **완료** | 위 조건에 L0/L1 결과를 더했다. 반복 축을 구분했다: B2~B5는 학습 seed 3개, B0/B1은 결정적(실효 seed 1개), L0는 반복 축 없음, L1은 demo 세트 3개(학습 seed가 아님). |
| DoD-Internal-Ready | **완료(합성 데이터와 공개 데이터를 사내 형식으로 바꾼 fixture)** | 번들 검증/해제, 네트워크를 차단한 상태의 라벨 없는 오프라인 예측, 0으로 채우지 않음, 고정된 공개 demo만 허용(AT-19/20/21/25/30/35). 공개 test 변경을 사내 형식으로 넣었을 때 freeze된 test 점수가 재현된다(B0/B1 ≤2e-16, B2~B5 ≤2e-6). `docs/internal-offline-eval.md` 참고. |
| DoD-Internal-Evaluated | 미완료 | 승인된 사내 데이터가 필요하다(M7) |
| L2 (검색 few-shot) | 구현 완료, 실행 대기 | 확장 항목(필수 아님). v3 test를 연 뒤 등록한 확장 study `public-comparison-v3-l2ext`로, `exp/012` Pod에서 diffllm 다음에 실행한다 |

## 사내 평가 준비

- **오프라인 예측**: `diff-lab predict`는 export 번들만 사용한다. 네트워크를 차단하고, 라벨 열이 있으면 거부하며, 학습이나 보정은 하지 않는다.
- **배포 단위 평가**: `diff-lab internal evaluate`가 담당한다.
  - commit 점수를 배포 단위로 max 집계한다.
  - 라벨 관찰창, 연결 신뢰도, 평가 기간은 사전에 등록한다.
  - 큰 배포가 유리해지는 크기 편향을 함께 보고한다.
- 실행 전에 `configs/internal/internal-eval-v1.yaml`의 라벨 정의, 관찰창, 평가 기간(PIN_REQUIRED 4개 항목)을 결과를 보기 전에 고정해야 한다. 사내 데이터는 승인된 로컬 환경에서만 처리한다.

## 후속 연구 (진행 중)

- **`diffllm-v1`**: 탐색 연구로, 사양 v0.2 범위 밖이며 v3 test를 연 뒤에 시작했다. 설계는 `docs/diffllm-study-v1.md`에 있다.
- 공개 full diff(before → after)와 JD4J와 겹치지 않는 Apache 프로젝트의 diff 50M 토큰으로 Qwen2.5-Coder-7B를 추가 사전학습한다.
- {base, diff 사전학습} × {embedding + MLP, risk alignment} 2×2를 비교한다.
- 결과는 이 문서의 v3 비교와 섞지 않고 별도로 보고한다.

## 재현과 산출물

- **코드**: main(`codebert-diff-lab/`). freeze, test, report는 커밋 `e1b2b1a`에서 실행했고, 보고서 F1 수정은 `4cba508`, DoD 보고는 `358b263`이다.
- **실험 브랜치** (각 브랜치에 `results/`와 `RUNLOG.md`가 있다)
  - `exp/009-dl-v3-rerun`: B0~B4
  - `exp/010-dl-m5-mlm-rmi`: B5
  - `exp/011-dl-m6-final`: L0/L1, freeze, test, report
- **모델 가중치**: 로컬 worktree(`artifacts/`, git-ignore)에만 있다. 추론 전용 export 번들은 `experiments/011-dl-m6-final/codebert-diff-lab/export-v2/bundle.tar.gz`다(로컬, 194개 파일, L1 공개 demo 포함).
- **명령 순서**: `data import/audit/split` → `evidence build` → `experiment run`(variant별) → `study freeze` → `experiment test` → `study report` → `study export`. 자세한 내용은 `README.md`.
- **비용**: Runpod H100 SXM, 007~011 합계 약 $35.
