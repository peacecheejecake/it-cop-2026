# Plan: other models and larger datasets (after study v4)

Drafted 2026-10-02. Nothing here is registered or approved yet; each study needs its own registration document, pinned model revisions and a cost confirmation before it runs. Background: `leak-finding-2026-10-02.md`, `v4-correction-plan.md`.

## 0. What already exists

| document | covers | state |
|---|---|---|
| `diffllm-study-v1.md` (exp 012) | Qwen2.5-Coder-7B LoRA on git full diffs | done; leak-free; test AP 0.283 (R-diff), 0.274 (R-base) |
| `diffllm-study-v2.md` (exp 013) | EvidenceView prompts, then Qwen2.5-Coder-14B | stopped; its EvidenceView input was the package text, so it must be re-registered on `jitd4j-git1` |
| `cpt100m-study.md` (exp 014) | CodeBERT CPT at 100M tokens | stopped; same reason |
| `DATASETS.md` §3 | candidate datasets (ISSTA'21, JITLine, ReDef, SmartSHARK) | not acquired |

Not covered anywhere before this document: other model families (Llama, Gemma), and training on more data than JIT-Defects4J.

## 1. Rules that apply to every study below

- **Text comes from git** through `diff_lab.gitextract` or `diff_lab.fulldiff`, never from a preprocessed package.
- **`tools/representation_leak_check.py` runs on every new snapshot before any training.** A coverage AP well above the base rate stops the study.
- Selection on validation, freeze, one test evaluation; B0-LR is always run on the same snapshot as the floor.
- Internal use needs a model whose license allows it. License and revision are pinned from the model card at registration time, not from memory.

## 1a. GPU use, measured on 2026-10-02 (exp 016, H100, CodeBERT fp32, micro-batch 8)

| setup | time per unit of work | total throughput vs one run at a time |
|---|---|---|
| one run | fine-tuning epoch 3.1 min; CPT 50 updates 62 s | 1.0 |
| 3 runs on the same GPU, default time-slicing | epoch 8.5 min each | about 1.1 |
| 4 runs on the same GPU under NVIDIA MPS | CPT 50 updates 258 s each | about 0.96 |

- One run already saturates the H100, so several runs on one GPU finish no sooner in total. Lanes on one GPU are not a saving for this workload.
- What does reduce cost: a cheaper GPU per run (this model uses 8 GB), and one pod per lane when wall-clock time matters. Cost per run on a smaller GPU must be measured with a short real run first (AGENTS.md §5).
- The cost estimates below assume sequential runs on one GPU.

## 2. Study M — model family and size (LLM risk SFT)

**Question:** on leak-free input, does a larger or different LLM beat Qwen2.5-Coder-7B, and does any of them beat the CodeBERT variants of v4?

- **Data:** `jitd4j-git1`, split `upstream-clean2`.
- **Input:** stage 1 decides between the two leak-free forms with the 7B model: full diff (exp 012 form) and git EvidenceView + jit14 (the 013 design on the new snapshot). Later arms use the winner.
- **Training:** the exp 012 `R-base` recipe unchanged (LoRA r=16 on all linear layers, label-token loss, lr 1e-4, at most 2 epochs, bf16, one seed). No diff CPT: exp 012 found no established gain from it.

| arm | model (candidate; pin revision at registration) | license to verify | VRAM (bf16) | est. time on one 80–96 GB GPU |
|---|---|---|---|---|
| M-Q7 | Qwen2.5-Coder-7B-Instruct | Apache-2.0 | 16 GB | 1.2 h (reuse the 012 run if the input form is full diff) |
| M-Q14 | Qwen2.5-Coder-14B-Instruct | Apache-2.0 | 30 GB | 2–2.5 h |
| M-L8 | a Llama instruct model of about 8B | Meta community license, gated | 16 GB | 1.2 h |
| M-G | a Gemma instruct model of 9–12B, or CodeGemma 7B | Gemma terms, gated | 18–24 GB | 1.5 h |

- **Order:** stage 1 (input form, 7B) → M-Q14 → M-L8, M-G. The two gated families need a Hugging Face token with accepted terms, and a policy check before any internal use.
- **Parallelism:** memory would allow two 7–9B arms on one 80–96 GB GPU, but see section 1a: sharing one GPU did not raise throughput for CodeBERT. Measure before relying on it; otherwise use one pod per arm.
- **Cost estimate:** stage 1 about $5, the three new arms about $10–12 at $2.1–3.5/h. Total **about $15–17**.
- **Comparisons fixed in advance:** each arm − M-Q7 (same input), best arm − best v4 CodeBERT variant, best arm − B0-LR. One seed per arm, so differences are judged by the project bootstrap interval only, and stated as single-seed results.

## 3. Study D — more training data

**Question:** v4 text models sit at validation AP 0.35–0.37 with 16k training commits. Is that a data-size limit?

| stage | training data | evaluation | purpose |
|---|---|---|---|
| D1 | `jitd4j-git1` train at 25%, 50%, 100% | `jitd4j-git1` validation | learning curve; costs little and shows whether more data is likely to help at all |
| D2 | ApacheJIT without the Hadoop repositories (about 90k commits, 12 Java projects), text and jit14 from git | its own time-ordered test, and `jitd4j-git1` test without retraining | 5× the data, same language |
| D3 | D2 train + `jitd4j-git1` train | each dataset's test, reported separately | pooling; base rates differ (26.5% vs 8.5%), so results are stratified by dataset (DATASETS.md §3 rule) |
| D4 | ISSTA'21 JIT-DP (Qt, OpenStack, Eclipse, Gerrit, Go; about 310k commits), rebuilt from git | per language | cross-language transfer, which the internal trial needs (internal code is not only Java) |

- **Variants:** B0-LR, B2-S, B3-S for every stage; the best LLM arm of study M only if D1 points upward.
- **Preparation that costs no GPU:**
  - ApacheJIT: the label CSV and the 15 mirrors were deleted from the cache on 2026-10-01 and must be fetched again (about 20 GB of clones; hadoop-hdfs/mapreduce need the `alternates` link to hadoop.git, AGENTS.md §1.4).
  - A `rebuild-git`-style adapter that takes a commit list with labels (ApacheJIT CSV, ISSTA'21 lists) instead of a JIT-Fine snapshot.
  - Known ApacheJIT issues stay in force: Hadoop label anomaly (excluded), non-random commit inclusion, right-censored labels in recent years.
- **Cost estimate:** D1 about $3 (6 short runs). D2 about $10–15 (B3-S on 90k commits is about 5× a v4 run per seed). D3 about the same again. D4 not estimated until the data is rebuilt.
- **Gate:** D2 runs only if D1's curve is still rising at 100% (validation AP at 100% above 50% by more than the seed spread). D4 is decided after the internal trial shows which languages matter.

## 4. Suggested order

1. Finish v4 (freeze, test, internal package).
2. D1 learning curve (cheap, decides whether D2 is worth it).
3. Study M stage 1 and M-Q14 (Apache-2.0 models, no license question).
4. M-L8 and M-G after the license check; D2/D3 if the D1 gate passes.
5. D4 after the internal trial.

## 5. Open points for the user

- Which Llama and Gemma versions are acceptable for internal import (license terms).
- Whether exp 014 (CPT-100M) is still wanted on leak-free text; v4 will show whether 10M-token CPT does anything there.
- Budget per study; the estimates above assume sequential runs on one H100-class pod.
