"""Paired template problems: chain vs wide, with and without a hint.

Every item is one set of numbers shown in four conditions:

    cond         family B example          steps  load
    chain        (a-b+c)*d                 3      1
    wide         (a-b)*(c+d)               3      2
    chain_hint   a-b given in the text     2      1
    wide_hint    a-b given in the text     2      1

Family B is the main family because chain and wide answers have similar
sizes. Family A, (a*b-c)*d vs a*b-c*d, matches the email example but its
chain answers are much larger than its wide answers. Use it only as a second
family.

Usage (from the repo root):
    python -m load_experiment.templates --n 1000 --out load_experiment/data/templates.jsonl
"""
import argparse
import json
import os
import random

from load_experiment.load import min_load

CONDS = ["chain", "wide", "chain_hint", "wide_hint"]

# Step structure of each condition (see load.py). x1 is given in hint versions.
STRUCTURES = {
    "chain":      [set(), {0}, {1}],
    "wide":       [set(), set(), {0, 1}],
    "chain_hint": [set(), {0}],
    "wide_hint":  [set(), {0}],
}

# Each template has a {hint} slot right after the sentence that produces x1.
FAMILIES = {
    "B": dict(
        x1=lambda a, b: a - b,
        chain=lambda x1, c, d: [x1, x1 + c, (x1 + c) * d],
        wide=lambda x1, c, d: [x1, c + d, x1 * (c + d)],
        final=lambda x1, x2: x1 * x2,          # how the wide answer combines x1 and x2
        sample=lambda rng: (rng.randint(10, 30), rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)),
        frames=[
            dict(
                chain="A class has {a} students, but {b} are absent today{hint}. Then {c} new students "
                      "join the class. Each student in the room gets {d} pencils. How many pencils are given out?",
                wide="A class has {a} students, but {b} are absent today{hint}. Each student who is present "
                     "gets {c} red pencils and {d} blue pencils. How many pencils are given out?",
                hint=", so {x1} students are present",
            ),
            dict(
                chain="A bakery baked {a} cakes and sold {b} of them in the morning{hint}. In the afternoon it "
                      "baked {c} more cakes. Each cake needs {d} candles. How many candles does the bakery need "
                      "for the cakes it has now?",
                wide="A bakery baked {a} cakes and sold {b} of them in the morning{hint}. Each cake that is left "
                     "gets {c} strawberries and {d} cherries on top. How many pieces of fruit does the bakery use?",
                hint=", so {x1} cakes were left",
            ),
            dict(
                chain="A farmer had {a} cows and sold {b} of them{hint}. Then he bought {c} more cows. Each cow "
                      "eats {d} kilograms of hay a day. How many kilograms of hay do his cows eat a day?",
                wide="A farmer had {a} cows and sold {b} of them{hint}. Each cow he still has eats {c} kilograms "
                     "of hay and {d} kilograms of grain a day. How many kilograms of food do his cows eat a day?",
                hint=", so he had {x1} cows left",
            ),
            dict(
                chain="A library had {a} books on a shelf, and {b} were borrowed{hint}. Then {c} new books were "
                      "added to the shelf. Each book on the shelf has {d} stickers. How many stickers are on the "
                      "books on the shelf?",
                wide="A library had {a} books on a shelf, and {b} were borrowed{hint}. Each book still on the "
                     "shelf has {c} red stickers and {d} blue stickers. How many stickers are on the books on "
                     "the shelf?",
                hint=", so {x1} books were left on the shelf",
            ),
            dict(
                chain="A bus had {a} passengers. At the first stop, {b} people got off{hint}. Then {c} people got "
                      "on. Each passenger on the bus then paid {d} dollars. How many dollars did they pay in total?",
                wide="A bus had {a} passengers. At the first stop, {b} people got off{hint}. Each passenger still "
                     "on the bus paid {c} dollars for a ticket and {d} dollars for a snack. How many dollars did "
                     "they pay in total?",
                hint=", so {x1} passengers were left",
            ),
        ],
    ),
    "A": dict(
        x1=lambda a, b: a * b,
        chain=lambda x1, c, d: [x1, x1 - c, (x1 - c) * d],
        wide=lambda x1, c, d: [x1, c * d, x1 - c * d],
        final=lambda x1, x2: x1 - x2,
        sample=lambda rng: tuple(rng.randint(2, 12) for _ in range(4)),
        frames=[
            dict(
                chain="A shop has {a} boxes with {b} pens in each box{hint}. It gives away {c} pens. It sells "
                      "each remaining pen for {d} dollars. How many dollars does it make?",
                wide="A shop has {a} boxes with {b} pens in each box{hint}. It also has {c} boxes with {d} "
                     "pencils in each box. How many more pens than pencils does it have?",
                hint=", which is {x1} pens",
            ),
            dict(
                chain="A garden has {a} rows with {b} tulips in each row{hint}. Then {c} tulips are picked. "
                      "Each remaining tulip needs {d} cups of water. How many cups of water are needed?",
                wide="A garden has {a} rows with {b} tulips in each row{hint}. It also has {c} rows with {d} "
                     "roses in each row. How many more tulips than roses are there?",
                hint=", which is {x1} tulips",
            ),
        ],
    ),
}


def valid(nums, ch, wd):
    """Rules that keep chain and wide fair and the error analysis unambiguous."""
    vals = ch + wd
    return (all(2 <= v <= 200 for v in vals)
            and len(set(ch)) == 3 and len(set(wd)) == 3          # no repeated values
            and not set(nums) & set(vals)                       # results differ from question numbers
            and len(str(ch[-1])) == len(str(wd[-1])))           # answers have the same digit count


def sample_numbers(fam, rng, max_tries=100_000):
    for _ in range(max_tries):
        a, b, c, d = fam["sample"](rng)
        x1 = fam["x1"](a, b)
        ch, wd = fam["chain"](x1, c, d), fam["wide"](x1, c, d)
        if valid((a, b, c, d), ch, wd):
            return (a, b, c, d), x1, ch, wd
    raise RuntimeError("could not sample valid numbers; loosen the rules in valid()")


def make_item(i, rng, family="B"):
    fam = FAMILIES[family]
    frame_id = i % len(fam["frames"])
    frame = fam["frames"][frame_id]
    (a, b, c, d), x1, ch, wd = sample_numbers(fam, rng)
    n = dict(a=a, b=b, c=c, d=d, x1=x1)
    hint = frame["hint"].format(**n)
    texts = {
        "chain": frame["chain"].format(hint="", **n),
        "wide": frame["wide"].format(hint="", **n),
        "chain_hint": frame["chain"].format(hint=hint, **n),
        "wide_hint": frame["wide"].format(hint=hint, **n),
    }
    vals = {"chain": ch, "wide": wd, "chain_hint": ch, "wide_hint": wd}
    rows = []
    for cond in CONDS:
        rows.append(dict(
            item=i, family=family, frame=frame_id, cond=cond, text=texts[cond],
            gold=vals[cond][-1], steps=len(STRUCTURES[cond]), load=min_load(STRUCTURES[cond]),
            a=a, b=b, c=c, d=d, x1=x1, x2=vals[cond][1], values=vals[cond],
        ))
    return rows


def generate(n=1000, seed=0, family="B"):
    rng = random.Random(seed)
    return [row for i in range(n) for row in make_item(i, rng, family)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="number of items (each gives 4 problems)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--family", default="B", choices=sorted(FAMILIES))
    ap.add_argument("--out", default="load_experiment/data/templates.jsonl")
    args = ap.parse_args()

    rows = generate(args.n, args.seed, args.family)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} problems ({args.n} items x {len(CONDS)} conditions) to {args.out}")


if __name__ == "__main__":
    main()
