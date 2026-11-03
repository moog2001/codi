import numpy as np
import pandas as pd

from load_experiment.error_check import classify, lost_value_answers, match_rates
from load_experiment.stats import CONDS, bootstrap, correctness_array, main_effect, rescue
from load_experiment.templates import generate


def fake_results(acc, n=1000, seed=0):
    """acc[mode][cond] = accuracy; returns a results table like run_experiment writes."""
    rng = np.random.default_rng(seed)
    rows = []
    for item in range(n):
        for mode, by_cond in acc.items():
            for cond in CONDS:
                rows.append(dict(family="B", item=item, cond=cond, mode=mode,
                                 correct=bool(rng.random() < by_cond[cond])))
    return pd.DataFrame(rows)


def test_main_and_rescue_effects_are_recovered():
    df = fake_results({
        "latent":   dict(chain=0.60, wide=0.40, chain_hint=0.62, wide_hint=0.60),
        "explicit": dict(chain=0.60, wide=0.58, chain_hint=0.62, wide_hint=0.60),
    })
    C, modes = correctness_array(df)
    assert C.shape == (1000, 4, 2) and modes == ["latent", "explicit"]
    p, lo, hi = bootstrap(C, main_effect, n_boot=2000)
    assert lo > 0 and abs(p - 0.18) < 0.06
    p, lo, hi = bootstrap(C, lambda a: rescue(a, 0), n_boot=2000)
    assert lo > 0 and abs(p - 0.18) < 0.06


def test_no_effect_rarely_looks_significant():
    # A 95% interval should exclude 0 in about 5% of null datasets (100 runs gave 6%).
    same = dict(chain=0.5, wide=0.5, chain_hint=0.5, wide_hint=0.5)
    misses = 0
    for seed in range(40):
        C, _ = correctness_array(fake_results({"latent": same, "explicit": same}, n=400, seed=seed))
        _, lo, hi = bootstrap(C, main_effect, n_boot=300, seed=seed)
        misses += not (lo < 0 < hi)
    assert misses <= 6


def test_incomplete_items_are_dropped():
    df = fake_results({"latent": dict(chain=1, wide=1, chain_hint=1, wide_hint=1)}, n=5)
    df = df[~((df["item"] == 0) & (df["cond"] == "wide"))]
    C, _ = correctness_array(df)
    assert len(C) == 4


def test_lost_value_answers_and_classify():
    row = next(r for r in generate(20, seed=0) if r["cond"] == "wide")
    cands = lost_value_answers(row)
    assert row["gold"] not in cands
    assert cands.get(row["x2"]) == "newest only"
    assert classify(row, float(row["x2"])) == "newest only"
    assert classify(row, float(row["gold"])) is None
    assert classify(row, float("inf")) is None


def test_match_rate_beats_chance_when_errors_are_lost_values():
    rows = [r for r in generate(200, seed=0) if r["cond"] == "wide"]
    results = []
    for r in rows:
        results.append({**r, "mode": "latent", "pred": float(r["x2"]), "correct": False})
        results.append({**r, "mode": "explicit", "pred": float(r["gold"]) + 1, "correct": False})
    df = pd.DataFrame(results)
    lat = match_rates(df, "latent", n_shuffle=50)
    assert lat["match"] == 1.0 and lat["chance"] < 0.2
    assert match_rates(df, "explicit", n_shuffle=50)["match"] < 0.2
