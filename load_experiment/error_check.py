"""Do wrong answers on wide problems look like the held value x1 was lost?

For a wide item the answer combines x1 (computed first, held) with x2 (newer).
Lost-value answers:
  x1 -> question number   x1 replaced by one of a, b, c, d
  newest only             the model answered with x2 alone
  treated as chain        the model computed the chain version instead

Chance level: each wrong answer is checked against the candidates of a random
other wide item (a shuffle baseline), repeated many times.

Usage (from the repo root):
    python -m load_experiment.error_check --results load_experiment/results/templates.csv
"""
import argparse

import numpy as np
import pandas as pd

from load_experiment.templates import FAMILIES


def lost_value_answers(row):
    """Dict: candidate answer -> type, for one wide item (the correct answer is excluded)."""
    fam = FAMILIES[row.get("family", "B")]
    a, b, c, d, x1, x2 = (int(row[k]) for k in ("a", "b", "c", "d", "x1", "x2"))
    cands = {}
    cands[fam["chain"](x1, c, d)[-1]] = "treated as chain"
    cands[x2] = "newest only"
    for q in (a, b, c, d):
        cands[fam["final"](q, x2)] = "x1 -> question number"
    cands.pop(int(row["gold"]), None)
    return cands


def classify(row, pred):
    if pred is None or not np.isfinite(pred) or pred != int(pred):
        return None
    return lost_value_answers(row).get(int(pred))


def match_rates(df, mode, n_shuffle=1000, seed=0):
    wide = df[(df["cond"] == "wide") & (df["mode"] == mode)].to_dict("records")
    wrong = [r for r in wide if not r["correct"]]
    if not wrong:
        return dict(mode=mode, n_wrong=0)
    types = [classify(r, r["pred"]) for r in wrong]
    rng = np.random.default_rng(seed)
    chance = np.mean([
        np.mean([classify(wide[rng.integers(len(wide))], r["pred"]) is not None for r in wrong])
        for _ in range(n_shuffle)
    ])
    out = dict(mode=mode, n_wrong=len(wrong),
               match=np.mean([t is not None for t in types]), chance=chance)
    for t in ("x1 -> question number", "newest only", "treated as chain"):
        out[t] = np.mean([x == t for x in types])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--n_shuffle", type=int, default=1000)
    args = ap.parse_args()
    df = pd.read_csv(args.results)
    table = pd.DataFrame([match_rates(df, m, args.n_shuffle) for m in sorted(set(df["mode"]))])
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
