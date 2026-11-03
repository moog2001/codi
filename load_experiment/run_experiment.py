"""Run CODI (latent and/or explicit mode) on a JSONL file and save one CSV row per problem x mode.

Each JSONL row needs `text` and `gold`; every other field is copied to the output.

Examples (from the repo root):
    # pilot: the first 100 items x 4 conditions
    python -m load_experiment.run_experiment --data load_experiment/data/templates.jsonl \
        --ckpt ckpt/codi_gpt2 --limit 400 --out load_experiment/results/pilot.csv

    # GSM8K load analysis
    python -m load_experiment.run_experiment --data load_experiment/data/gsm8k_load.jsonl \
        --ckpt ckpt/codi_gpt2 --out load_experiment/results/gsm8k.csv
"""
import argparse
import json
import os

import pandas as pd

from load_experiment.codi_runner import load_codi, run_batched, run_explicit, run_latent


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True, help="folder with model.safetensors or pytorch_model.bin")
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--modes", nargs="+", default=["latent", "explicit"], choices=["latent", "explicit"])
    ap.add_argument("--n_latent", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N rows")
    ap.add_argument("--no_sort", action="store_true", help="batch in file order, like test.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.data)[: args.limit]
    model, tok = load_codi(args.ckpt, model_name=args.model_name, n_latent=args.n_latent)
    questions = [r["text"] for r in rows]

    out = []
    for mode in args.modes:
        print(f"[{mode}] {len(rows)} problems")
        if mode == "latent":
            preds, texts, _ = run_batched(run_latent, model, tok, questions, args.batch_size,
                                          sort_by_length=not args.no_sort,
                                          n_latent=args.n_latent, max_new=args.max_new)
        else:
            preds, texts, _ = run_batched(run_explicit, model, tok, questions, args.batch_size,
                                          sort_by_length=not args.no_sort, max_new=args.max_new)
        for r, p, t in zip(rows, preds, texts):
            out.append({**r, "mode": mode, "pred": p, "output": t,
                        "correct": abs(p - float(r["gold"])) < 1e-6})
        acc = sum(o["correct"] for o in out if o["mode"] == mode) / len(rows)
        print(f"[{mode}] accuracy {acc:.3f}")

    df = pd.DataFrame(out)
    for col in df.columns:  # store lists as JSON strings
        if df[col].map(lambda v: isinstance(v, list)).any():
            df[col] = df[col].map(json.dumps)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
