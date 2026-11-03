"""Optional: is the held value x1 still readable at each latent step?

For every latent position k = 0..6 the last-layer state goes through the LM head
(GPT-2's final layer norm is already applied to hidden_states[-1]), and we check
whether x1 and x2 are among the top-k tokens. probe_latent_token.py does the same
decoding. In its sample output, numbers appear at even positions and operators
at odd ones, so expect a zig-zag.

Usage (from the repo root):
    python -m load_experiment.logit_lens --data load_experiment/data/templates.jsonl \
        --ckpt ckpt/codi_gpt2 --conds wide chain --out load_experiment/results/logit_lens.csv
"""
import argparse
import os

import pandas as pd
import torch

from load_experiment.codi_runner import load_codi, run_batched, run_latent
from load_experiment.run_experiment import read_jsonl


def number_token_ids(tok, value):
    """First token of the number with and without a leading space."""
    v = int(value) if float(value).is_integer() else value
    ids = set()
    for s in (str(v), " " + str(v)):
        enc = tok.encode(s)
        if enc:
            ids.add(enc[0])
    return ids


@torch.no_grad()
def decodability(model, tok, rows, batch_size=64, topk=5, n_latent=6):
    preds, _, latents = run_batched(run_latent, model, tok, [r["text"] for r in rows], batch_size,
                                    keep_latents=True, n_latent=n_latent)
    p = next(model.parameters())
    records = []
    for r, pred, L in zip(rows, preds, latents):
        top = model.codi.lm_head(L.to(p.device, p.dtype)).topk(topk, dim=-1).indices.cpu()
        x1_ids, x2_ids = number_token_ids(tok, r["x1"]), number_token_ids(tok, r["x2"])
        for k in range(L.size(0)):
            t = set(top[k].tolist())
            records.append(dict(item=r["item"], cond=r["cond"], pos=k,
                                correct=abs(pred - float(r["gold"])) < 1e-6,
                                x1_readable=bool(t & x1_ids), x2_readable=bool(t & x2_ids)))
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--conds", nargs="+", default=["wide", "chain"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.data)[: args.limit] if r["cond"] in args.conds]
    model, tok = load_codi(args.ckpt)
    df = decodability(model, tok, rows, args.batch_size, args.topk)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["cond", "correct", "pos"])[["x1_readable", "x2_readable"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
