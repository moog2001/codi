"""Run CODI in two modes with the same weights.

latent    the question, then <bot>, 6 continuous thoughts, <eot>, then the answer
          (follows test.py's inference loop step by step, with greedy decoding)
explicit  only the question; the model writes its CoT as text, then the answer
          (the teacher format CODI was trained on: question + CoT + "The answer is: X")

load_codi() builds the model exactly like test.py does with scripts/test_gpt2.sh.
Run everything from the repo root so that `src.model` can be imported.
"""
import os
import re
import sys

import torch
import transformers
from peft import LoraConfig, TaskType
from safetensors.torch import load_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import CODI, DataArguments, ModelArguments, TrainingArguments  # noqa: E402


def codi_flags(ckpt_dir, model_name="gpt2", bf16=False, n_latent=6):
    """The same flags as scripts/test_gpt2.sh."""
    flags = [
        "--data_name", "gsm8k",
        "--output_dir", os.path.join(REPO_ROOT, "load_experiment", ".tmp"),
        "--model_name_or_path", model_name,
        "--seed", "11",
        "--model_max_length", "512",
        "--lora_r", "128", "--lora_alpha", "32", "--lora_init",
        "--batch_size", "128",
        "--greedy", "True",
        "--num_latent", "6",
        "--use_prj", "True",
        "--prj_dim", "768",
        "--prj_no_ln", "False",
        "--prj_dropout", "0.0",
        "--inf_latent_iterations", str(n_latent),
        "--inf_num_iterations", "1",
        "--remove_eos", "True",
        "--use_lora", "True",
        "--ckpt_dir", ckpt_dir or "",
    ]
    if bf16:
        flags.append("--bf16")
    return flags


def pick_dtype():
    # T4 GPUs (free Colab) have no bf16, and transformers refuses --bf16 there.
    # GPT-2 small is tiny, so float32 is fine.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def build_codi(model_name="gpt2", n_latent=6, dtype=None, ckpt_dir=None):
    """CODI with LoRA and the projection layer, before loading the checkpoint."""
    dtype = dtype or pick_dtype()
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, _, training_args = parser.parse_args_into_dataclasses(
        args=codi_flags(ckpt_dir, model_name, bf16=(dtype == torch.bfloat16), n_latent=n_latent))
    lora_config = LoraConfig(  # same as test.py for gpt2
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        init_lora_weights=True,
    )
    model = CODI(model_args, training_args, lora_config).to(dtype)

    tok = transformers.AutoTokenizer.from_pretrained(
        model_name, model_max_length=512, padding_side="left", use_fast=False)
    if tok.pad_token_id is None:  # same handling as test.py
        tok.add_special_tokens({"pad_token": "[PAD]"})
        tok.pad_token_id = model.pad_token_id
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.convert_tokens_to_ids("[PAD]")
    return model, tok


def load_codi(ckpt_dir, model_name="gpt2", n_latent=6, dtype=None, device=None):
    """Build CODI and load a checkpoint folder (model.safetensors or pytorch_model.bin)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = build_codi(model_name, n_latent, dtype, ckpt_dir)

    st_path = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(st_path):
        state = load_file(st_path)
    else:
        state = torch.load(os.path.join(ckpt_dir, "pytorch_model.bin"), map_location="cpu")
    # test.py uses strict=False, which silently skips keys that don't match.
    # Print what was skipped: missing LoRA or prj weights mean a broken load.
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.codi.tie_weights()
    print(f"[load_codi] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    for name in (missing + unexpected)[:10]:
        print(f"[load_codi]   {name}")
    if any("lora" in k or "prj" in k for k in missing):
        print("[load_codi] WARNING: LoRA or projection weights were not loaded")

    return model.to(device).eval(), tok


def last_number(text):
    """Same rule as test.py's extract_answer_number: the last number in the text."""
    nums = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(nums[-1]) if nums else float("inf")


def _device(model):
    return next(model.parameters()).device


@torch.no_grad()
def greedy_decode(model, tok, past, last_logits, max_new=256):
    """Greedy token loop from test.py. `last_logits` are the logits at the current last position."""
    vocab = model.codi.config.vocab_size - 1  # test.py's slice: never pick the last special token
    embed = model.get_embd(model.codi, model.model_name)
    batch = last_logits.size(0)
    tokens = [[] for _ in range(batch)]
    done = [False] * batch
    for _ in range(max_new):
        nxt = last_logits[:, :vocab].argmax(-1)
        for b in range(batch):
            if not done[b]:
                tokens[b].append(nxt[b].item())
                done[b] = nxt[b].item() == tok.eos_token_id
        if all(done):
            break
        out = model.codi(inputs_embeds=embed(nxt).unsqueeze(1), past_key_values=past, use_cache=True)
        past, last_logits = out.past_key_values, out.logits[:, -1]
    return [tok.decode(t, skip_special_tokens=True) for t in tokens]


@torch.no_grad()
def run_latent(model, tok, questions, n_latent=6, keep_latents=False, max_new=256):
    """Latent mode. Returns preds, texts, and the latent states (B, n_latent+1, H) if asked.

    Like test.py, no attention mask is passed after the first call, so left-padding
    tokens are visible to the latent steps. That is how the paper's 43.7% was measured.
    """
    device = _device(model)
    enc = tok(questions, return_tensors="pt", padding="longest").to(device)
    batch = enc.input_ids.size(0)
    bot = torch.full((batch, 1), model.bot_id, dtype=torch.long, device=device)
    ids = torch.cat([enc.input_ids, bot], dim=1)
    mask = torch.cat([enc.attention_mask, torch.ones_like(bot)], dim=1)

    out = model.codi(input_ids=ids, attention_mask=mask, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    h = out.hidden_states[-1][:, -1:, :]          # last layer, last position (the <bot> token)
    states = [h]
    for _ in range(n_latent):                     # the continuous thought loop
        z = model.prj(h) if model.use_prj else h
        out = model.codi(inputs_embeds=z, past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        h = out.hidden_states[-1][:, -1:, :]
        states.append(h)

    eot = model.get_embd(model.codi, model.model_name)(
        torch.tensor([model.eot_id], dtype=torch.long, device=device))
    eot = eot.unsqueeze(0).expand(batch, -1, -1)
    out = model.codi(inputs_embeds=eot, past_key_values=past, use_cache=True)
    texts = greedy_decode(model, tok, out.past_key_values, out.logits[:, -1], max_new)
    preds = [last_number(t) for t in texts]
    latents = torch.cat(states, dim=1).float().cpu() if keep_latents else None
    return preds, texts, latents


@torch.no_grad()
def run_explicit(model, tok, questions, max_new=256):
    """Explicit mode: the same weights write CoT text, e.g. '<<21-8=13>> <<7+5=12>> The answer is: 156'.

    The teacher was trained without the last CoT step (include_last_cot=False),
    so it writes every step except the final one, then the answer.
    """
    device = _device(model)
    enc = tok(questions, return_tensors="pt", padding="longest").to(device)
    out = model.codi(input_ids=enc.input_ids, attention_mask=enc.attention_mask, use_cache=True)
    texts = greedy_decode(model, tok, out.past_key_values, out.logits[:, -1], max_new)
    return [last_number(t) for t in texts], texts, None


def run_batched(fn, model, tok, questions, batch_size=64, sort_by_length=True, verbose=True, **kw):
    """Run `fn` (run_latent or run_explicit) over all questions; results keep the input order.

    sort_by_length groups similar lengths together, which keeps padding small.
    Use sort_by_length=False and batch_size=128 to copy test.py's batching exactly.
    """
    n = len(questions)
    order = list(range(n))
    if sort_by_length:
        order.sort(key=lambda i: len(tok(questions[i]).input_ids))
    preds, texts, latents = [None] * n, [None] * n, [None] * n
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        p, t, lat = fn(model, tok, [questions[i] for i in idx], **kw)
        for j, i in enumerate(idx):
            preds[i], texts[i] = p[j], t[j]
            if lat is not None:
                latents[i] = lat[j]
        if verbose:
            print(f"  {min(start + batch_size, n)}/{n}", flush=True)
    return preds, texts, latents
