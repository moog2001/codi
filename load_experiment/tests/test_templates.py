import re

import pytest

from load_experiment.templates import CONDS, FAMILIES, generate


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_items_are_consistent(family):
    rows = generate(n=300, seed=1, family=family)
    assert len(rows) == 300 * len(CONDS)
    fam = FAMILIES[family]
    for i in range(0, len(rows), len(CONDS)):
        item = {r["cond"]: r for r in rows[i:i + len(CONDS)]}
        assert set(item) == set(CONDS)
        r = item["chain"]
        a, b, c, d, x1 = r["a"], r["b"], r["c"], r["d"], r["x1"]
        assert x1 == fam["x1"](a, b)
        assert item["chain"]["gold"] == fam["chain"](x1, c, d)[-1] == item["chain_hint"]["gold"]
        assert item["wide"]["gold"] == fam["wide"](x1, c, d)[-1] == item["wide_hint"]["gold"]
        assert item["wide"]["gold"] == fam["final"](x1, item["wide"]["x2"])
        # loads and steps
        assert [item[c_]["load"] for c_ in CONDS] == [1, 2, 1, 1]
        assert [item[c_]["steps"] for c_ in CONDS] == [3, 3, 2, 2]
        # every question number is in the text; x1 appears only in hint versions
        for cond in CONDS:
            nums = [int(n) for n in re.findall(r"\d+", item[cond]["text"])]
            for v in (a, b, c, d):
                assert v in nums
            assert (x1 in nums) == cond.endswith("hint")
        # answers have the same number of digits
        assert len(str(item["chain"]["gold"])) == len(str(item["wide"]["gold"]))


def test_same_seed_same_items():
    assert generate(50, seed=3) == generate(50, seed=3)
    assert generate(50, seed=3) != generate(50, seed=4)


def test_enough_distinct_number_sets_for_family_b():
    rows = generate(n=1000, seed=0, family="B")
    sets = {(r["a"], r["b"], r["c"], r["d"]) for r in rows}
    assert len(sets) > 700
