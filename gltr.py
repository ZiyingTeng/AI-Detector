"""GLTR-style token colouring (F's part).

For each token, find its rank in GPT-2's prediction at that position:
top-10 green, top-100 yellow, top-1000 red, rarer purple.
Falls back to random ranks if torch/transformers are not installed.
"""
import html
import random

import streamlit as st

REF_MODEL = "gpt2"  # keep the same as B's reference model
MAX_LEN = 512

COLORS = {
    "top10": "rgba(76, 175, 80, 0.35)",
    "top100": "rgba(255, 193, 7, 0.40)",
    "top1000": "rgba(244, 67, 54, 0.35)",
    "rest": "rgba(156, 39, 176, 0.30)",
}
LABELS = {"top10": "top-10", "top100": "top-100", "top1000": "top-1000", "rest": "rarer"}


def bucket(rank):
    if rank is None:
        return None
    if rank <= 10:
        return "top10"
    if rank <= 100:
        return "top100"
    if rank <= 1000:
        return "top1000"
    return "rest"


@st.cache_resource(show_spinner="Loading GPT-2…")
def _load(ref: str):
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained(ref)
    lm = GPT2LMHeadModel.from_pretrained(ref).eval()
    return tok, lm


def token_ranks(text: str):
    """Return (list of (token, rank), is_mock). The first token has no rank."""
    try:
        import torch

        tok, lm = _load(REF_MODEL)
    except Exception:
        rnd = random.Random(text)
        words = text.split(" ")
        ranks = [None] + [rnd.choice([1, 2, 3, 5, 8, 30, 70, 400, 3000]) for _ in words[1:]]
        return [((" " if i else "") + w, r) for i, (w, r) in enumerate(zip(words, ranks))], True

    ids = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).input_ids
    with torch.no_grad():
        logp = torch.log_softmax(lm(ids).logits[0, :-1], -1)
    target = ids[0, 1:]
    tok_logp = logp[torch.arange(len(target)), target]
    ranks = ((logp > tok_logp[:, None]).sum(-1) + 1).tolist()
    tokens = [tok.decode([i]) for i in ids[0]]
    return [(tokens[0], None)] + list(zip(tokens[1:], ranks)), False


def render(pairs) -> str:
    spans = []
    for token, rank in pairs:
        b = bucket(rank)
        t = html.escape(token).replace("\n", "<br>")
        if b:
            spans.append(f'<span style="background:{COLORS[b]};border-radius:3px" title="rank {rank}">{t}</span>')
        else:
            spans.append(t)
    return '<div style="line-height:2;font-size:1.02rem">' + "".join(spans) + "</div>"


def legend() -> str:
    items = "".join(
        f'<span style="margin-right:14px"><span style="display:inline-block;width:12px;height:12px;'
        f'border-radius:2px;background:{COLORS[k]};vertical-align:middle;margin-right:4px"></span>{LABELS[k]}</span>'
        for k in COLORS
    )
    return f'<div style="font-size:0.85rem;opacity:0.8">{items}</div>'


def fractions(pairs) -> dict:
    counts = {k: 0 for k in COLORS}
    n = 0
    for _, r in pairs:
        b = bucket(r)
        if b:
            counts[b] += 1
            n += 1
    return {LABELS[k]: (v / n if n else 0) for k, v in counts.items()}
