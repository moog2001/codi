"""Main effect, rescue effect and the GSM8K table, with paired bootstrap intervals.

main effect   (chain - wide accuracy in latent mode) - (chain - wide accuracy in explicit mode)
rescue        (hint gain on wide) - (hint gain on chain), for each mode

Bootstrap resamples whole items, so each item keeps all 8 of its cells
(4 conditions x 2 modes) and the comparison stays paired.

Usage (from the repo root):
    python -m load_experiment.stats --templates load_experiment/results/templates.csv
    python -m load_experiment.stats --gsm8k load_experiment/results/gsm8k.csv
"""
import argparse

import numpy as np
import pandas as pd

CONDS = ["chain", "wide", "chain_hint", "wide_hint"]
MODES = ["latent", "explicit"]


def correctness_array(df):
    """C[item, cond, mode] in {0, 1}, keeping only items that have every cell."""
    df = df[df["cond"].isin(CONDS)]
    modes = [m for m in MODES if m in set(df["mode"])]
    fam = df["family"] if "family" in df else pd.Series("B", index=df.index)
    keys = sorted(set(zip(fam, df["item"])))
    index = {k: i for i, k in enumerate(keys)}
    C = np.full((len(keys), len(CONDS), len(modes)), np.nan)
    for f, item, cond, mode, ok in zip(fam, df["item"], df["cond"], df["mode"], df["correct"]):
        C[index[(f, item)], CONDS.index(cond), modes.index(mode)] = float(ok)
    complete = ~np.isnan(C).any(axis=(1, 2))
    return C[complete], modes


def drop(acc, m):
    return acc[0, m] - acc[1, m]                   # chain - wide


def main_effect(acc):
    return drop(acc, 0) - drop(acc, 1)             # latent drop - explicit drop


def rescue(acc, m):
    return (acc[3, m] - acc[1, m]) - (acc[2, m] - acc[0, m])


def bootstrap(C, stat, n_boot=10_000, seed=0):
    """Point estimate and 95% percentile interval, resampling items."""
    rng = np.random.default_rng(seed)
    n = len(C)
    point = stat(C.mean(0))
    boots = np.array([stat(C[rng.integers(0, n, n)].mean(0)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def _fmt(est):
    p, lo, hi = est
    return f"{100 * p:+6.1f} points  [95% CI {100 * lo:+.1f}, {100 * hi:+.1f}]"


def template_report(df, n_boot=10_000):
    C, modes = correctness_array(df)
    print(f"{len(C)} complete items\n")
    acc = C.mean(0)
    print("accuracy (%)      " + "".join(f"{m:>10}" for m in modes))
    for ci, cond in enumerate(CONDS):
        print(f"  {cond:<16}" + "".join(f"{100 * acc[ci, mi]:>10.1f}" for mi in range(len(modes))))
    print()
    summary = {}
    for mi, m in enumerate(modes):
        summary[f"drop_{m}"] = bootstrap(C, lambda a, mi=mi: drop(a, mi), n_boot)
        summary[f"rescue_{m}"] = bootstrap(C, lambda a, mi=mi: rescue(a, mi), n_boot)
        print(f"chain - wide, {m:<8}  {_fmt(summary[f'drop_{m}'])}")
        print(f"rescue,       {m:<8}  {_fmt(summary[f'rescue_{m}'])}")
    if modes == MODES:
        summary["main_effect"] = bootstrap(C, main_effect, n_boot)
        summary["rescue_diff"] = bootstrap(C, lambda a: rescue(a, 0) - rescue(a, 1), n_boot)
        print(f"\nmain effect (latent drop - explicit drop)  {_fmt(summary['main_effect'])}")
        print(f"rescue latent - rescue explicit            {_fmt(summary['rescue_diff'])}")
        supported = summary["main_effect"][1] > 0 and summary["rescue_latent"][1] > 0
        print("\nhypothesis " + ("SUPPORTED" if supported else "NOT supported") +
              " (needs main effect CI > 0 and latent rescue CI > 0)")
    return summary


def gsm8k_table(df, drop_ambiguous=True):
    """Accuracy and count by steps x load, for each mode."""
    if drop_ambiguous and "ambiguous" in df:
        n0 = df["item"].nunique()
        df = df[~df["ambiguous"].astype(bool)]
        print(f"dropped {n0 - df['item'].nunique()} ambiguous problems")
    df = df.assign(steps_bin=df["steps"].clip(upper=5).map(lambda s: "5+" if s >= 5 else str(s)),
                   load_bin=df["load"].clip(upper=3).map(lambda s: "3+" if s >= 3 else str(s)))
    tables = {}
    for mode, g in df.groupby("mode"):
        t = g.pivot_table(index="steps_bin", columns="load_bin", values="correct",
                          aggfunc=["mean", "count"])
        tables[mode] = t
        print(f"\n[{mode}] accuracy by steps (rows) x load (columns), with counts")
        print(t.round(3).to_string())
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", help="results CSV from the template run")
    ap.add_argument("--gsm8k", help="results CSV from the GSM8K run")
    ap.add_argument("--n_boot", type=int, default=10_000)
    args = ap.parse_args()
    if args.templates:
        template_report(pd.read_csv(args.templates), args.n_boot)
    if args.gsm8k:
        gsm8k_table(pd.read_csv(args.gsm8k))


if __name__ == "__main__":
    main()
