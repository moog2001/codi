"""Plumbing test on a tiny random GPT-2 (no download, CPU only).

Checks that run_latent gives exactly the same tokens as test.py's own loop,
that run_explicit's cached decoding matches step-by-step decoding, and that
saving and loading a checkpoint works. The model is random, so the outputs are
nonsense; only equality between the two code paths matters.
"""
import json
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from load_experiment import codi_runner as cr  # noqa: E402

MAX_NEW = 12
QUESTIONS = ["A class has 25 students, but 7 are absent today.",
             "Tom has 4 boxes of 6 pens. How many pens?",
             "2 plus 3?"]


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    name = str(root / "tiny-gpt2")          # "gpt2" must be in the name for CODI.get_embd
    os.makedirs(name)
    vocab = {ch: i for i, ch in enumerate(_bytes_to_unicode().values())}
    vocab["<|endoftext|>"] = len(vocab)
    with open(os.path.join(name, "vocab.json"), "w") as f:
        json.dump(vocab, f)
    with open(os.path.join(name, "merges.txt"), "w") as f:
        f.write("#version: 0.2\n")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=len(vocab), n_positions=512, n_embd=32, n_layer=2, n_head=2,
                                  bos_token_id=256, eos_token_id=256)
    transformers.GPT2LMHeadModel(cfg).save_pretrained(name)

    # make a "checkpoint": a CODI with random prj and nonzero LoRA weights
    model, _ = cr.build_codi(name, dtype=torch.float32)
    with torch.no_grad():
        for n_, p in model.named_parameters():
            if "lora_B" in n_:
                p.normal_(0, 0.02)
    ckpt = str(root / "ckpt")
    os.makedirs(ckpt)
    torch.save(model.state_dict(), os.path.join(ckpt, "pytorch_model.bin"))
    loaded, tok = cr.load_codi(ckpt, model_name=name, dtype=torch.float32, device="cpu")
    return model.eval(), loaded, tok


@torch.no_grad()
def reference_test_py(model, tok, questions, n_latent=6, max_new=MAX_NEW):
    """test.py's greedy path, copied as closely as possible."""
    batch = tok(questions, return_tensors="pt", padding="longest")
    bot_tensor = torch.tensor([model.bot_id], dtype=torch.long).expand(batch["input_ids"].size(0), 1)
    input_ids = torch.cat((batch["input_ids"], bot_tensor), dim=1)
    attention_mask = torch.cat((batch["attention_mask"], torch.ones_like(bot_tensor)), dim=1)
    batch_size = input_ids.size(0)
    outputs = model.codi(input_ids=input_ids, use_cache=True, output_hidden_states=True,
                         past_key_values=None, attention_mask=attention_mask)
    past_key_values = outputs.past_key_values
    latent_embd = model.prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
    for _ in range(n_latent):
        outputs = model.codi(inputs_embeds=latent_embd, use_cache=True, output_hidden_states=True,
                             past_key_values=past_key_values)
        past_key_values = outputs.past_key_values
        latent_embd = model.prj(outputs.hidden_states[-1][:, -1, :].unsqueeze(1))
    eot_emb = model.get_embd(model.codi, model.model_name)(
        torch.tensor([model.eot_id], dtype=torch.long)).unsqueeze(0)
    output = eot_emb.expand(batch_size, -1, -1)
    finished = torch.zeros(batch_size, dtype=torch.bool)
    pred_tokens = [[] for _ in range(batch_size)]
    for _ in range(max_new):
        out = model.codi(inputs_embeds=output, output_hidden_states=False, attention_mask=None,
                         use_cache=True, output_attentions=False, past_key_values=past_key_values)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :model.codi.config.vocab_size - 1]
        next_token_ids = torch.argmax(logits, dim=-1).squeeze(-1)
        for b in range(batch_size):
            if not finished[b]:
                pred_tokens[b].append(next_token_ids[b].item())
                if next_token_ids[b] == tok.eos_token_id:
                    finished[b] = True
        if finished.all():
            break
        output = model.get_embd(model.codi, model.model_name)(next_token_ids).unsqueeze(1)
    return [tok.decode(t, skip_special_tokens=True) for t in pred_tokens]


def test_checkpoint_round_trip(tiny):
    model, loaded, _ = tiny
    a, b = model.state_dict(), loaded.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_run_latent_matches_test_py(tiny):
    _, model, tok = tiny
    ref = reference_test_py(model, tok, QUESTIONS)
    _, texts, latents = cr.run_latent(model, tok, QUESTIONS, keep_latents=True, max_new=MAX_NEW)
    assert texts == ref
    assert latents.shape == (len(QUESTIONS), 7, model.codi.config.n_embd)


def test_run_explicit_matches_step_by_step_decoding(tiny):
    _, model, tok = tiny
    vocab = model.codi.config.vocab_size - 1
    for q in QUESTIONS:
        ids = tok([q], return_tensors="pt").input_ids
        toks = []
        for _ in range(MAX_NEW):
            nxt = model.codi(input_ids=ids).logits[:, -1, :vocab].argmax(-1)
            toks.append(nxt.item())
            if nxt.item() == tok.eos_token_id:
                break
            ids = torch.cat([ids, nxt[:, None]], dim=1)
        _, texts, _ = cr.run_explicit(model, tok, [q], max_new=MAX_NEW)
        assert texts[0] == tok.decode(toks, skip_special_tokens=True)


def test_run_batched_keeps_input_order(tiny):
    _, model, tok = tiny
    one_by_one = [cr.run_latent(model, tok, [q], max_new=MAX_NEW)[1][0] for q in QUESTIONS]
    _, texts, _ = cr.run_batched(cr.run_latent, model, tok, QUESTIONS, batch_size=1,
                                 verbose=False, max_new=MAX_NEW)
    assert texts == one_by_one


def test_last_number():
    assert cr.last_number("<<4*6=24>> The answer is: 1,200") == 1200.0
    assert cr.last_number("no digits") == float("inf")
