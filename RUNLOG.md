# RUNLOG — 010-dl-m5-mlm-rmi

브랜치 `exp/010-dl-m5-mlm-rmi` (main `8514ce8`에서 분기). 코드: `codebert-diff-lab`, study `public-comparison-v3`.

## 목적

사양 milestone **M5: B5-S**(diff MLM+RMI CPT를 B4와 같은 합산 10M 토큰 예산으로 수행한 뒤 v3 fine-tuning), public validation, seed 42/43/44.

## RMI 구현 (사양 §6.2/§6.3)

- `rmi_target`: 1은 원래 메시지, 0은 다른 CPT-train 변경의 메시지로 교체한 경우다. 교체 확률은 0.5다. 교체할 메시지는 균등 추출하되 자기 자신, 정규화한 메시지 hash가 같은 것, 같은 exact 중복 그룹은 거절한다(`rmi-uniform-reject-v1`). 다른 pool로의 fallback은 없다.
- rmi_ineligible: 빈 메시지 9건(0.06%), 코드 없음 0건이다.
- 입력은 교체된 메시지 + `\n` + 원래 코드 줄이다. 512 토큰을 넘으면 코드 줄을 끝에서부터 제거한다(seed당 844~879건, 약 5%).
- update 하나는 과제 하나다. MLM과 RMI를 번갈아 수행한다(MLM 133 / RMI 132 update). 예산은 두 과제의 non-padding 입력 토큰 합계로 센다(MLM 약 5.03M + RMI 약 4.98M).
- RMI head는 Linear-Tanh-Dropout-Linear 구조로, seed로 초기화한다. encoder만 export하며 MLM/RMI head는 넘기지 않는다.
- CPT-dev 진단: dev MLM loss와, 고정 seed로 만든 dev RMI 예제 809개의 loss/accuracy/AUC.

## 데이터와 환경

- worktree에서 데이터를 다시 만들었다(`upstream-clean2`). parquet은 009와 같다.
- Pod `9r0xh4efj5n1wl` (`jit010-m5-mlm-rmi`), AP-IN-1, H100 80GB, $3.49/시간.
- 번들 `jit010-8514ce8.tar.gz` (sha256 `eeb41c5b…600b`), `CODE_SHA`=`8514ce8…`.
- pytest 64 passed, 1 skipped. CUDA resume 테스트 통과.

## 실행

```bash
uv run diff-lab experiment run --study configs/studies/public-comparison-v3.yaml --models B5-S --seeds 42,43,44
```

10:05 → 11:1x UTC, EXIT=0. 로그는 `logs/m5.{out,err}`에 있다.

| seed | run_id | CPT (s) | dev MLM 0→끝 | dev RMI acc / AUC | best / 중단 epoch | AP | ROC-AUC | R@5% | R@10% |
|---|---|---|---|---|---|---|---|---|---|
| 42 | `60269328794d8df7` | 346 | 16.02 → 2.65 | 0.703 / 0.754 | 1 / 6 | 0.6574 | 0.9124 | 0.448 | 0.653 |
| 43 | `e1b8ef1f29915aaf` | 349 | 15.12 → 2.48 | 0.634 / 0.688 | 2 / 7 | 0.6746 | 0.9110 | 0.478 | 0.672 |
| 44 | `810917980633be9b` | 344 | 15.36 → 2.54 | 0.565 / 0.602 | 2 / 7 | 0.6743 | 0.9060 | 0.475 | 0.657 |

- B4(MLM 단독)는 dev MLM이 1.51~1.61까지 내려갔다. B5는 MLM 노출이 절반이라 2.48~2.65에 머문다.
- RMI는 seed마다 학습 정도가 크게 다르다(dev accuracy 0.565~0.703). 132 update로는 메시지와 변경의 대응을 안정적으로 배우지 못한다.
- 회수: 작은 파일과 best generation의 safetensors 6개를 받았고, generation pointer의 hash로 검증했다. CPT encoder export는 받지 않았다(다시 만들 수 있다).
- Pod는 B5 이후 011(L0/L1)에 계속 쓴다. 정리 기록은 011 RUNLOG에 있다.
