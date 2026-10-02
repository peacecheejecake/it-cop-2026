# RESULTS — 018-dl-study-m

## Step 1: what did the 50M-token diff CPT change? (label-free probe)

512 public-train changes (salted sha256 rank), full diff in the CPT text format, mean next-token loss of Qwen2.5-Coder-7B with and without the exp 012 CPT adapter.

| | base | base + CPT | reduction |
|---|---|---|---|
| all 512 | 0.924 | 0.859 | 0.065 (7%); 98.8% of changes improve |
| clean (464) | 0.919 | 0.855 | 0.064 |
| buggy (48) | 0.970 | 0.904 | 0.066 |

- **The CPT did change how the model reads JD4J diffs.** The loss falls by 7% on target-domain diffs it never saw, almost uniformly. So "CPT had no effect at all" is ruled out, and the corpus was not too small to move the model.
- **The change is the same for buggy and clean changes.** The ROC-AUC of "loss reduction" as a predictor of the label is 0.487, i.e. random. The base model's loss itself barely separates them (0.553).
- Reading: the CPT taught general modelling of Java/Apache diffs, not anything specific to risky changes. Whether risk SFT can still exploit the adapted representation is what R-diff − R-base measures.
- **Consequence for scaling the corpus:** more data of the same kind will likely lower this loss further. But this probe gives no sign that it would separate risky changes better. The scale-up arm should wait for stage 1:
  - if R-diff − R-base (3 seeds, v4 cohort) is positive with an interval excluding 0, a larger CPT is worth testing;
  - if it is about 0, scaling the same objective is a weak bet, and the objective (or LoRA capacity) is the better next variable.
- Cost: about $0.12 (RTX 4090, about 10 minutes).
