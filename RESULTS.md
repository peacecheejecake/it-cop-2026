# RESULTS — 007-dl-m3-encoder (spec v0.2 M3, public validation)

데이터는 JIT-Defects4J `jitd4j-audit1/upstream-clean1`이다. validation 5,465건, 양성 467건, 기저율 0.085. 모두 같은 EvidenceView(`message-add-del-text-v2`, 512 토큰)를 쓴다. public test는 아직 열지 않았다.

| variant | AP (mean ± sd, n=3) | ROC-AUC | Recall@5% | Recall@10% | 비고 |
|---|---|---|---|---|---|
| B0-LR (006) | 0.319 | 0.792 | 0.244 | 0.396 | 결정적, 실효 seed 1 |
| B0-LGBM (006) | 0.211 | 0.710 | 0.167 | 0.293 | 시작 설정 과적합 |
| B1-TFIDF-S (006) | **0.546** | 0.885 | 0.398 | 0.565 | 결정적, 실효 seed 1 |
| B2-S frozen CodeBERT + head | 0.377 ± 0.005 | 0.814 | 0.282 | 0.440 | 20 epoch 상한, 미수렴 |
| B3-S full fine-tuning | 0.527 ± 0.024 | 0.885 | 0.374 | 0.546 | epoch 2~3 선택, 이후 과적합 |

## 해석

1. **B3-S의 평균은 B1-TFIDF-S보다 낮다**(AP 0.527 대 0.546, ROC-AUC는 같다). B3가 앞선 경우는 seed 44 하나(0.554)뿐이다. 문자 n-gram TF-IDF와 정형 feature를 쓴 선형 모델이 CodeBERT 전체 fine-tuning과 비슷하거나 더 낫다. 이 비교는 validation 기준이고 B3의 epoch 선택에도 같은 validation을 썼으므로, B3 쪽이 오히려 낙관적으로 나온 값이다.
2. **CodeBERT 표현을 고정한 B2-S는 B0-LR보다 +0.058 높다.** CLS 표현에 정형 feature 이상의 정보가 있다. 다만 미수렴 상태라 이 차이는 하한이다.
3. B3-S는 1~3 epoch 안에 최고점에 도달하고 바로 과적합된다. train 16k건, 양성 1,387건에 비해 125M 파라미터 모델이 크다.

## 결정이 필요한 것 (M4/M5 전에)

- **B2-S 미수렴**: 공통 FT 프로토콜(lr 1e-5, 최대 20 epoch)을 그대로 쓰면 B2는 불리하다. 사양상 B2/B3/B4/B5는 같은 head 프로토콜을 공유하므로 바꾸려면 study 전체를 다시 pin해야 한다. 이번 M3 값을 공식 결과로 두고 미수렴을 한계로 기록할지, head 전용 lr과 epoch 상한을 study v2.1로 등록해 B2만 다시 돌릴지(GPU 몇 분)를 정해야 한다.
- bf16(encoder만)은 T14에서 3.3배 빨랐고 발산하지 않았다. M4/M5(MLM CPT)처럼 비용이 큰 단계에서는 도입을 검토할 만하다. 그 경우 B3 seed 42를 bf16으로 재실행해 fp32와 동등한지 먼저 확인한다.
