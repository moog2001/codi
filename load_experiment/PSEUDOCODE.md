# Pseudocode

This is the short, readable version. The real code is in the `.py` files
next to this one. Section letters match the plan.

## A. Load (`load.py`)

```python
def min_load(uses):
    """uses[i] = set of earlier steps that step i reads.
    A value is 'live' after it is computed and before its last use.
    Load = the smallest possible peak number of live values, over all valid orders."""
    users = [{w for w in range(n) if u in uses[w]} for u in range(n)]
    live(S) = number of u in S with some user not in S        # S = set of computed steps

    best(S) = 0                                              if S has every step
            = max(live(S), min(best(S ∪ {v}) for v ready))   otherwise
    return best(empty set)

# chain [∅,{0},{1}] → 1 · wide [∅,∅,{0,1}] → 2 · wide+hint [∅,{0}] → 1
```

## B. Running CODI in both modes (`codi_runner.py`, mirrors `test.py`)

```python
def run_latent(questions, n_latent=6):
    ids = tokenize(questions, left padding) + [<bot>]
    out = codi(ids, attention_mask)                 # encode the question
    h = out.hidden_states[-1][:, -1]                # last layer, last position
    for _ in range(n_latent):                       # continuous thoughts
        out = codi(inputs_embeds=prj(h), past=out.past)
        h = out.hidden_states[-1][:, -1]
    out = codi(inputs_embeds=embed(<eot>), past=out.past)
    text = greedy_decode(out)                       # "The answer is: 156"
    return last_number(text)

def run_explicit(questions):
    out = codi(tokenize(questions))                 # question only: no <bot>/<eot>
    text = greedy_decode(out)                       # "<<21-8=13>> <<7+5=12>> The answer is: 156"
    return last_number(text)

def greedy_decode(out, max_new=256):
    repeat: next = argmax(logits[:, :vocab-1]); stop at eos; feed embed(next)
```

## C. GSM8K parser (`gsm8k_parse.py`)

```python
for each test problem in zen-E/GSM8k-Aug:
    for each "<<expr=result>>" (one equation = one step):
        link each number in expr to the most recent earlier step with that result
        if that number is also in the question → flag ambiguous
    steps = number of equations; load = min_load(links)
Table 1 = accuracy grouped by (steps, load), for each mode, without ambiguous problems
```

## D. Template generator (`templates.py`)

```python
for item i in 0..999:
    frame = FRAMES[i % 5]                     # class, bakery, farm, library, bus
    sample a in 10..30; b, c, d in 2..9 until valid():
        x1 = a - b
        chain values = [x1, x1 + c, (x1 + c) * d]
        wide values  = [x1, c + d, x1 * (c + d)]
    emit four problems with the same numbers:
        chain, wide                (hint slot empty)
        chain_hint, wide_hint      (hint slot = ", so {x1} students are present")
```

## E. Experiment and statistics (`run_experiment.py`, `stats.py`)

```python
for mode in latent, explicit:
    run every problem → results.csv (item, cond, mode, pred, correct, ...)

C[item, cond, mode] = 1 if correct
main effect = (chain − wide)_latent − (chain − wide)_explicit
rescue(mode) = (wide_hint − wide) − (chain_hint − chain)
bootstrap: resample whole items 10,000 times → 95% CI
supported if main effect CI > 0 and rescue(latent) CI > 0
```

## F. Error check (`error_check.py`)

```python
for each wrong answer on a wide problem:
    candidates = { x1 replaced by a, b, c or d,  x2 alone,  the chain answer }
    match = share of wrong answers that equal a candidate
chance = the same, but using the candidates of a random other wide item
prediction: latent match ≫ chance; explicit match lower
```

## G. Logit lens (`logit_lens.py`, optional)

```python
for each latent position k = 0..6:
    top5 = lm_head(h_k).topk(5)
    x1_readable[k] = x1's token in top5 ; x2_readable[k] = x2's token in top5
plot by position, split by correct vs wrong
prediction: in wrong items, x1 fades from later positions
```
