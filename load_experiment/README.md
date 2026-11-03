# Memory-load experiment for CODI

Does CODI's latent reasoning fail when it must hold several intermediate
results at once? See `PLAN.md` for the idea, the design and the two-week
schedule, and `PSEUDOCODE.md` for a short readable version of the code.

## Files

| file | what it does | tested here |
|---|---|---|
| `load.py` | steps → memory load (best order) | yes |
| `gsm8k_parse.py` | GSM8k-Aug test split → steps and load per problem (JSONL) | yes, on sample strings |
| `templates.py` | paired chain / wide / hint problems (JSONL) | yes |
| `codi_runner.py` | loads CODI like `test.py`; latent and explicit modes | on a tiny random GPT-2 only |
| `run_experiment.py` | runs both modes on a JSONL file and writes a CSV | no (needs the real checkpoint) |
| `stats.py` | main effect, rescue effect, GSM8K table, bootstrap CIs | yes, on simulated results |
| `error_check.py` | wrong answers vs lost-value answers vs chance | yes, on simulated results |
| `logit_lens.py` | x1 / x2 readable at each latent step (optional) | no |
| `tests/` | `pytest load_experiment/tests` | |

"Tiny random GPT-2" means the test checks the plumbing:
- `run_latent` produces exactly the same tokens as `test.py`'s loop.
- Cached explicit decoding matches step-by-step decoding.
- Checkpoints save and load correctly.

It does not check accuracy. The real numbers come from the day 1 and day 2 checks.

## Run it on Colab (GPU runtime)

Run everything from the repo root.

```bash
git clone https://github.com/moog2001/codi.git
cd codi
git checkout memory-load-experiment      # until this branch is merged
```

Colab already has PyTorch. Install the pinned versions of the other libraries:

```bash
pip install transformers==4.52.4 peft==0.15.2 accelerate==1.7.0 datasets==3.6.0 safetensors==0.5.3
```

Download the checkpoint. The folder should contain `model.safetensors` or
`pytorch_model.bin`; move the file up if it sits in a subfolder.

```bash
huggingface-cli download zen-E/CODI-gpt2 --local-dir load_experiment/ckpt/codi_gpt2
```

**Day 1:** reproduce the paper. In `scripts/test_gpt2.sh`, set
`--ckpt_dir load_experiment/ckpt/codi_gpt2`, and on a T4 remove `--bf16`. Expect about 43.7%.

```bash
SAVE_DIR=load_experiment/.tmp bash scripts/test_gpt2.sh
```

**Days 2–3:** GSM8K in both modes, plus the load table. `--no_sort --batch_size 128` copies `test.py`'s batching.

```bash
python -m load_experiment.gsm8k_parse --out load_experiment/data/gsm8k_load.jsonl
python -m load_experiment.run_experiment --data load_experiment/data/gsm8k_load.jsonl \
    --ckpt load_experiment/ckpt/codi_gpt2 --no_sort --batch_size 128 \
    --out load_experiment/results/gsm8k.csv
python -m load_experiment.stats --gsm8k load_experiment/results/gsm8k.csv
```

**Days 4–5:** generate the templates, then run the pilot on the first 100 items (400 problems).

```bash
python -m load_experiment.templates --n 1000 --out load_experiment/data/templates.jsonl
python -m load_experiment.run_experiment --data load_experiment/data/templates.jsonl \
    --ckpt load_experiment/ckpt/codi_gpt2 --limit 400 --out load_experiment/results/pilot.csv
python -m load_experiment.stats --templates load_experiment/results/pilot.csv
```

**Days 6–9:** the full run, statistics and error check.

```bash
python -m load_experiment.run_experiment --data load_experiment/data/templates.jsonl \
    --ckpt load_experiment/ckpt/codi_gpt2 --out load_experiment/results/templates.csv
python -m load_experiment.stats --templates load_experiment/results/templates.csv
python -m load_experiment.error_check --results load_experiment/results/templates.csv
```

**Day 10 (optional):** the logit lens.

```bash
python -m load_experiment.logit_lens --data load_experiment/data/templates.jsonl \
    --ckpt load_experiment/ckpt/codi_gpt2 --conds wide chain --limit 2000 \
    --out load_experiment/results/logit_lens.csv
```

## Things to check by hand

- **`[load_codi] missing keys`** should be 0. `test.py` loads with `strict=False`,
  which silently skips mismatched weights.
- **Explicit-mode outputs** should look like `<<21-8=13>> <<7+5=12>> The answer is: 156`.
  If they don't, the checkpoint may not have trained its teacher mode; see "Fallbacks" in `PLAN.md`.
- **The `output` column** in the results CSV: read about 20 rows per condition to
  confirm answers are parsed correctly.
