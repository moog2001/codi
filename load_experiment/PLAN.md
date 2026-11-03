# Plan: does CODI's latent reasoning have a working-memory limit?

## Goal

Test whether CODI's latent reasoning fails when it has to hold several
intermediate results at the same time.

## Idea

Two problems with the same number of steps can need different amounts of memory:

- **Chain:** 4×6=24 → 24−5=19 → 19×2=38. Only one result needs to be kept at a time.
- **Wide:** 4×6=24, 3×5=15 → 24−15=9. The model must keep 24 while computing 15.

**Load** is the largest number of intermediate results that must be held at
once, using the best possible order of steps. Numbers written in the question
never count. See `load.py`.

## Hypothesis

At the same number of steps, CODI's accuracy drops as load increases.
Explicit CoT's accuracy changes little, because it writes each result down
and can read it back.

**The hypothesis is wrong if** CODI's drop from load 1 to load 2 is not larger
than explicit CoT's drop (the bootstrap CI of the difference includes 0), and the
hint does not help wide problems more than chain problems.

## Why it's new

- Liang & Pan ([2602.00449](https://arxiv.org/abs/2602.00449)) found that CODI keeps only the last 1–2
  intermediate results, but they tested only chain tasks, where older results
  are never needed again. So it is unclear whether CODI *can't* keep older
  values or just *doesn't need to*. Wide problems separate the two.
- Scratchpad Thinking (NeurIPS 2025) found that CODI alternates storing and computing
  steps, which hints storage is limited, but it did not measure the limit.
- Dilgren & Wiegreffe ([2604.04902](https://arxiv.org/abs/2604.04902)) already covered "are gold tokens
  decodable?" and "are latent steps used?", which is why the earlier
  Stage 1–2 plan was dropped.

## Facts from CODI's code that shape the plan

- **Both modes from one checkpoint.** In training, the teacher sequence is
  `question + CoT + "The answer is: X"` and its loss is on by default
  (`ref_loss_factor=1.0`). So the same weights should write explicit CoT when
  given only the question. The probed checkpoint's folder name contains
  "noref", so **check this on day 2**.
- **The teacher never wrote the last step.** `include_last_cot=False`, so
  explicit mode writes every step except the final one, then the answer.
- **The GSM8K check is easy.** The `zen-E/GSM8k-Aug` test split lists each calculation:
  `<<16-3-4=9>> <<9*2=18>>`.
- **The logit lens already exists.** `probe_latent_token.py` decodes each latent step.
  In its sample output, numbers appear at even positions (0, 2, 4, 6) and
  operators at odd ones.
- **The released model attends to padding.** `test.py` passes no attention mask
  after the first call, so left-padding is visible. That is how 43.7% was measured.

## Design: 4 conditions × 2 modes, the same items

| cond | family B formula | steps | load | if the hypothesis is right |
|---|---|---|---|---|
| chain | (a−b+c)×d | 3 | 1 | both modes do well |
| wide | (a−b)×(c+d) | 3 | 2 | **latent drops**, explicit holds |
| chain_hint | a−b written in the text | 2 | 1 | small gain in both |
| wide_hint | a−b written in the text | 2 | 1 | **latent recovers**, explicit small gain |

- **Paired items:** one item is one set of numbers shown in all four conditions. Every
  item appears in all 8 cells, so the comparison is paired.
- **Family B is the main family** because chain and wide answers have similar
  sizes (medians about 100 vs 119). Family A, the pens example from the email, has
  chain answers about 3× larger, so use it only as a second family.
- **Fairness rules** (`templates.valid`):
  - all values between 2 and 200
  - no repeated values
  - results never equal a question number
  - chain and wide answers have the same digit count
- **Hint:** the sentence that produces x1 = a−b gets ", so x1 … are present". This
  removes one step in both conditions, but lowers the load only in wide problems.
- **Size:** 1,000 items × 4 conditions × 2 modes = 8,000 generations, which takes
  minutes on GPT-2 small. The main-effect CI should be about ±5 points.

## Two-week schedule

| day | work | command | checkpoint |
|---|---|---|---|
| 1 | Colab setup, download the checkpoint, reproduce | `bash scripts/test_gpt2.sh` (set `--ckpt_dir`; drop `--bf16` on a T4) | ≈ 43.7% |
| 2 | Both modes with our runner on GSM8K | `run_experiment --data …/gsm8k_load.jsonl --no_sort --batch_size 128` | latent ≈ day 1; explicit near 44% (CoT-SFT in the paper) |
| 3 | GSM8K load table | `gsm8k_parse`, then `stats --gsm8k` | Table 1 |
| 4 | Templates: read 20 problems by eye | `templates --n 1000` | wording is natural |
| 5 | Pilot: 100 items | `run_experiment --limit 400` | chain ≥ 30% in both modes |
| 6–7 | Full run | `run_experiment` on all 4,000 problems | `results/templates.csv` |
| 8 | Statistics | `stats --templates` | main effect and rescue effect with CIs |
| 9 | Error check | `error_check` | lost-value match rate vs chance |
| 10 | Logit lens (optional) | `logit_lens` | x1 and x2 readable at each position |
| 11–13 | Report and slides | | |
| 14 | Buffer | | |

The full commands are in `README.md`.

## Outputs

- **Table 1:** GSM8K accuracy by steps × load, for latent and explicit mode.
- **Fig 1 (key result):** chain vs wide accuracy for each mode, with 95% CIs.
- **Fig 2:** rescue effect for each mode.
- **Table 2:** lost-value match rate vs chance, for each mode.
- **Fig 3 (optional):** how often x1 and x2 are readable at each latent position, split by correct vs wrong.

## Fallbacks

- **Explicit mode doesn't work:** report the within-CODI results (chain vs wide, plus
  the rescue test). The rescue test is still a causal test of load. Train a separate
  CoT model only if you have spare GPU time.
- **Accuracy is near zero on templates:** use smaller numbers, copy GSM8K phrasing, or
  2-step versions.
- **No difference:** still a result. The "last 1–2 values" finding then comes from
  chain-only tasks, not from a memory limit.

## Stretch goals (only if time allows)

- Load 3, which needs about 7 steps.
- More latent steps (`--n_latent 10`): if load is the limit, extra steps shouldn't fix those errors.
- Family A as a second family.
- The Llama-1B CODI model.

## Report outline

1. **Question:** can latent reasoning hold several values at once?
2. **Background:** CODI; Liang & Pan (chain tasks only); Dilgren & Wiegreffe.
3. **Method:** the load definition, the GSM8K analysis, the paired templates, and two modes with the same weights.
4. **Results:** Table 1, Fig 1, Fig 2, Table 2, and optionally Fig 3.
5. **Limitations:** one small model, template problems, and only load 1 vs 2.
6. **Future work:** load 3, more latent steps, Coconut, and larger models.
