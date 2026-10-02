# Plan: other models and larger datasets (after study v4)

Drafted 2026-10-02. Nothing here is registered or approved yet; each study needs its own registration document, pinned model revisions and a cost confirmation before it runs. Background: `leak-finding-2026-10-02.md`, `v4-correction-plan.md`.

## 0. What already exists

| document | covers | state |
|---|---|---|
| `diffllm-study-v1.md` (exp 012) | Qwen2.5-Coder-7B LoRA on git full diffs, with and without diff CPT | done; leak-free; test AP 0.283 (R-diff), 0.274 (R-base); one seed, package-split cohort, so not directly comparable with v4 |
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

## 2. Study M — LLM family and size, with and without diff CPT, directly comparable with v4

Revised 2026-10-03 at the user's request: (1) results must be directly comparable with study v4; (2) diff CPT stays in.

**Questions:**

- M1: does a larger or different LLM beat Qwen2.5-Coder-7B under the same recipe?
- M2: does diff CPT help, per model (R-diff − R-base)?
- M3: how does the best LLM arm compare with the v4 variants (B0-LR, B2-S, B3-S) on the same test commits?

### 2.1 What makes the comparison with v4 direct

| issue in exp 012 vs v4 | fix in Study M |
|---|---|
| cohort: 012 used the package split (train 16,184 / valid 5,465 / test 5,480); v4 uses `jitd4j-git1` (110 commits missing from the mirrors dropped) | train, select and test on **`jitd4j-git1`, split `upstream-clean2`, exactly v4's membership**. 012 runs are not reused for M3. |
| one seed | **3 seeds (42/43/44)** for every SFT arm, paired with v4 seeds as in the v4 report |
| hardware: exp 017 showed that training on a Mac and on the H100 gives systematically different B2-S results | **all training on H100 80GB SXM**, the GPU of v4 |
| separate freeze/test/report code paths | one combined report: Study M test predictions and v4 test predictions joined on change_id, seed-paired differences and the same project bootstrap (2,000 resamples) |
| test access | Study M arms are frozen on validation before their test is scored once. The cohort was already opened for v4, so Study M is labelled an extension of v4 made after the v4 test was opened. |

- **Input stays text-only** (git full diff, 012 form). v4 variants also use jit14, so the LLM arms have less information. M3 therefore reports both "best LLM arm − B2-S" and "best LLM arm − B0-LR", and states the difference in inputs.
- An LLM arm with jit14 in the prompt (the 013 design) is not included. It can be registered as a separate arm if wanted.

### 2.2 Arms

| model (checked on Hugging Face 2026-10-03) | revision | license | R-base (SFT only) | R-diff (diff CPT → SFT) |
|---|---|---|---|---|
| `Qwen/Qwen2.5-Coder-7B-Instruct` | `c03e6d35` | Apache-2.0 | 3 seeds | 3 seeds; **CPT adapter reused from exp 012** (corpus is disjoint from JD4J and independent of the split) |
| `Qwen/Qwen2.5-Coder-14B-Instruct` | `aedcc2d4` | Apache-2.0 | 3 seeds | 3 seeds; new CPT |
| `google/gemma-4-12B-it` | `707f0a3b` | Apache-2.0, ungated | 3 seeds | 3 seeds; new CPT |
| `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f2` | Llama 3.1 Community License, gated | 3 seeds | 3 seeds; new CPT |

- Recipe = exp 012 unchanged. CPT: 50M tokens of `apache-disjoint-diffs-v1`, causal-LM loss, LoRA r=64 merged into the base. SFT: LoRA r=16 on all linear layers, label-token loss, lr 1e-4, at most 2 epochs, micro-batch 1 × accumulation 16, bf16.
- **One CPT per model** (seed 42), shared by the three R-diff seeds. CPT seed variance is therefore not measured, as in v4 B4/B5, whose CPT also had one seed per run.
- The CPT corpus is tokenized per model, so "50M tokens" means 50M tokens of each model's tokenizer.
- E-arms (embeddings + MLP) are dropped: exp 012 showed risk SFT is better by +0.07 to +0.10.

### 2.3 Engineering before any paid run

- `diffllm.py` uses only `seeds[0]`; it needs per-seed arm runs, run ids that include the seed, and freeze over arms × seeds.
- New config `study-m.yaml`: snapshot `jitd4j-git1`, model registry with pinned files, arms × seeds.
- Full-diff prompts for `jitd4j-git1` (reuse 012's `fulldiff` extraction, filtered to the git1 membership; re-render per tokenizer).
- Combined v4 + Study M report script.
- `transformers==4.57.6` may not load Gemma 4 (released 2026-06). Each model gets a smoke check (a few CPT and SFT updates, a few scored rows; chat template, system role, label-token ids, LoRA target names) before its real run.

### 2.4 Time and cost (H100 SXM $3.49/h; measured in exp 012: 7B SFT 2.0–2.6 h per seed, 7B CPT 50M tokens 2.9 h)

Other sizes scaled by parameter count (an assumption; to be replaced by the smoke-check throughput).

| model | CPT | 6 SFT runs (2 arms × 3 seeds) | eval/freeze/test | total | cost |
|---|---|---|---|---|---|
| Qwen 7B | 0 (reused) | 14 h | 1 h | 15 h | $52 |
| Llama 8B | 3.2 h | 15 h | 1 h | 19 h | $66 |
| Gemma 4 12B | 4.6 h | 22 h | 1.5 h | 28 h | $98 |
| Qwen 14B | 5.4 h | 26 h | 1.5 h | 33 h | $115 |
| smoke checks | | | | 2 h | $7 |
| **all four** | | | | **about 97 h** | **about $340** |

- GPU time does not shrink by sharing one GPU (section 1a). Several H100 pods in parallel shorten wall-clock time (four pods: about 1.5 days) for the same total cost.
- **Staged option:**
  - Stage 1: Qwen 7B, both arms × 3 seeds, plus the combined report with v4. About $55, about 15 h on one pod (or 5 h on three).
  - Stage 2: the other models after Stage 1 shows whether the LLM route is worth it.
- Smaller option: 1 seed for the larger models (2 runs per model instead of 6) cuts Stage 2 to about $110. Their single-seed results would then not be paired with v4 seeds in the same way.

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
3. Study M stage 1 (Qwen 7B, R-base and R-diff × 3 seeds on `jitd4j-git1`, combined report with v4).
4. Study M stage 2 (Qwen 14B, Gemma 4 12B, Llama 3.1 8B after its license check); D2/D3 if the D1 gate passes.
5. D4 after the internal trial.

## 5. Open points for the user

- Whether Llama 3.1 8B is wanted at all (license check, gated download).
- Whether exp 014 (CPT-100M) is still wanted on leak-free text; v4 will show whether 10M-token CPT does anything there.
- Budget per study; the estimates above assume sequential runs on one H100-class pod.
