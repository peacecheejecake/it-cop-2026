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

**Question:** with the same leak-free input and training recipe, does a larger or different LLM beat Qwen2.5-Coder-7B?

- **Protocol = exp 012 `R-base`, unchanged** (`docs/diffllm-study-v1.md`): git full diff (`git show -U3`, 2,048 content tokens), no jit14, LoRA r=16 on all linear layers, label-token loss, lr 1e-4, at most 2 epochs, micro-batch 1 × accumulation 16, bf16, seed 42, selection on the fixed 2,000-change validation subset.
  - This makes the existing 012 `R-base` run (validation AP 0.322, test AP 0.274) the 7B baseline at no cost. Its text never came from the package, so it is not affected by the finding.
  - A separate input-form stage (EvidenceView + jit14, the 013 design) is dropped from this study to keep one variable. It can be added later on `jitd4j-git1`.
- **No diff CPT:** exp 012 found no established gain from it.

Candidates, checked on the Hugging Face API on 2026-10-03 (revision = current `main` head; file hashes are pinned at registration):

| arm | model | revision | license | gated | parameters | VRAM (bf16) |
|---|---|---|---|---|---|---|
| M-Q7 (exists) | `Qwen/Qwen2.5-Coder-7B-Instruct` | `c03e6d35` | Apache-2.0 | no | 7.6B | 16 GB |
| M-Q14 | `Qwen/Qwen2.5-Coder-14B-Instruct` | `aedcc2d4` | Apache-2.0 | no | 14.8B | 30 GB |
| M-G12 | `google/gemma-4-12B-it` | `707f0a3b` | Apache-2.0 | no | 12.0B | 24 GB |
| M-L8 | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f2` | Llama 3.1 Community License | yes (manual approval) | 8.0B | 16 GB |

- Gemma 4 is listed as Apache-2.0 and ungated, unlike Gemma 2/3 (Gemma terms, gated). So Qwen and Gemma 4 raise no license question for internal import; Llama needs a policy check and a Hugging Face token whose account accepted the license.
- Llama 4 exists only as 17B-expert mixture models (Scout/Maverick, about 109B parameters and up), too large for this setup. Llama 3.1 8B is the practical Llama candidate.
- M-Q14 was already pre-registered in `diffllm-study-v1.md` as the 14B arm of `R-base`.

**Engineering before any paid run (risks):**

- `google/gemma-4-12B-it` was released in 2026-06; the lock pins `transformers==4.57.6`. If it does not load, the study needs its own environment, recorded as a deviation.
- The prompt uses a system message and scores the label tokens `0`/`1`. Chat templates, the presence of a system role and the label-token ids differ per family, so each model gets a tiny end-to-end check (a few updates and a few scored rows) before the real run.
- LoRA "all linear layers" must resolve to the right module names per architecture.

**Time and cost** (from exp 012: the 7B `R-base` took 2 h on an H100; evaluation of full validation is included):

| arm | est. GPU time | at $2.09/h (RTX PRO 6000 96 GB) | at $3.49/h (H100) |
|---|---|---|---|
| per-model smoke checks | 0.5 h | $1 | $2 |
| M-L8 | 2–2.5 h | $5 | $8 |
| M-G12 | 3–3.5 h | $7 | $12 |
| M-Q14 | 3.5–4 h | $8 | $14 |
| freeze + one test evaluation of the new arms | 1 h | $2 | $3.5 |
| **total** | **10–11.5 h** | **about $23** | **about $40** |

- One GPU runs one arm at a time (section 1a). Three pods in parallel finish in about 4 h for the same total cost.
- **Comparisons fixed in advance:** each new arm − M-Q7 on test AP with the project bootstrap; best arm − v4 B2-S and − B0-LR as descriptive references (cohorts differ by 21 test commits). One seed per arm, stated as such.
- **Test access:** the diffllm-v1 test cohort was opened once for the four v1 arms. New arms are selected on validation, frozen, and scored once; the result is labelled an extension made after that test was opened.

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
3. Study M: per-model smoke checks, then M-Q14 and M-G12 (Apache-2.0, ungated).
4. M-L8 after the license check and a token; D2/D3 if the D1 gate passes.
5. D4 after the internal trial.

## 5. Open points for the user

- Whether Llama 3.1 8B is wanted at all (license check, gated download).
- Whether exp 014 (CPT-100M) is still wanted on leak-free text; v4 will show whether 10M-token CPT does anything there.
- Budget per study; the estimates above assume sequential runs on one H100-class pod.
