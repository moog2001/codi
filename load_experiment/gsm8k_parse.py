"""Compute steps and memory load for every GSM8K test problem.

The test split of zen-E/GSM8k-Aug lists each calculation, for example
"<<16-3-4=9>> <<9*2=18>>". One equation counts as one step. A number in an
equation is linked to the most recent earlier step that produced it.

If that number also appears in the question, the link is unclear (the model
could have read it from the text), so the problem is flagged `ambiguous`.
Report how many problems you drop for that reason.

Usage (from the repo root):
    python -m load_experiment.gsm8k_parse --out load_experiment/data/gsm8k_load.jsonl
"""
import argparse
import json
import os
import re

from load_experiment.load import min_load, order_load

NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")
EQUATION_RE = re.compile(r"<<(.+?)=(.+?)>>")

# Number words that GSM8K questions often use instead of digits.
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
    "thousand": 1000, "dozen": 12, "half": 2, "twice": 2, "double": 2,
    "triple": 3, "quarter": 4,
}


def numbers_in(text, words=False):
    """All numbers in `text` as floats. With words=True, number words count too."""
    found = []
    for tok in NUMBER_RE.findall(text):
        tok = tok.replace(",", "").rstrip(".")
        try:
            found.append(float(tok))
        except ValueError:
            pass
    if words:
        for w in re.findall(r"[a-z]+", text.lower()):
            if w in NUMBER_WORDS:
                found.append(float(NUMBER_WORDS[w]))
    return found


def _same(x, y):
    return abs(x - y) < 1e-6


def parse_cot(cot, question=""):
    """Return dict(uses, results, ambiguous), or None if an equation can't be read."""
    q_nums = numbers_in(question, words=True)
    results, uses, ambiguous = [], [], False
    for expr, res in EQUATION_RE.findall(cot):
        res_nums = numbers_in(res)
        if not res_nums:
            return None
        dep = set()
        for x in numbers_in(expr):
            hits = [i for i, r in enumerate(results) if _same(r, x)]
            if hits:
                dep.add(hits[-1])
                if any(_same(x, q) for q in q_nums):
                    ambiguous = True
        uses.append(dep)
        results.append(res_nums[0])
    return dict(uses=uses, results=results, ambiguous=ambiguous)


def gsm8k_rows(dataset):
    """One row per problem: text, gold, steps, load (best order), ref_load (written order)."""
    rows = []
    for i, ex in enumerate(dataset):
        parsed = parse_cot(ex["cot"], ex["question"])
        if parsed is None or not parsed["uses"]:
            continue
        uses = parsed["uses"]
        rows.append(dict(
            item=i,
            text=ex["question"].strip().replace("  ", " "),  # same cleanup as test.py
            gold=float(str(ex["answer"]).replace(",", "")),
            steps=len(uses),
            load=min_load(uses),
            ref_load=order_load(list(range(len(uses))), uses),
            ambiguous=parsed["ambiguous"],
            cot=ex["cot"],
        ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="load_experiment/data/gsm8k_load.jsonl")
    ap.add_argument("--dataset", default="zen-E/GSM8k-Aug")
    args = ap.parse_args()

    from datasets import load_dataset
    rows = gsm8k_rows(load_dataset(args.dataset)["test"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_amb = sum(r["ambiguous"] for r in rows)
    print(f"wrote {len(rows)} problems to {args.out} ({n_amb} flagged ambiguous)")


if __name__ == "__main__":
    main()
