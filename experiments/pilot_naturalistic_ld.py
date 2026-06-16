#!/usr/bin/env python3
"""Workstream B3 gate: does z-dominance survive the naturalistic prompt frame?

Builds the height dense (x,z) grid in the toy-implicit frame vs the naturalistic frame (identical
sampled context), extracts LD = logit(tall) - logit(short) on gemma-2-9b, and compares
corr(mean_LD, z) vs corr(mean_LD, x) per frame (paper Table 2 methodology).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "circuit" / "scripts"))
from data_gen import prompts as P            # noqa: E402
from _token_utils import tokens_of_word      # noqa: E402

MODEL = "google/gemma-2-9b"
PAIR = P.PAIRS_BY_NAME["height"]
N_X, N_Z, N_SEEDS, K = 12, 12, 3, 15


def x_grid(pair, n):
    lo, hi = float(min(pair.target_values)), float(max(pair.target_values))
    span = hi - lo
    return np.linspace(max(0.5, lo - 0.1 * span), hi + 0.1 * span, n).round(1)


def build_rows():
    xs, zs = x_grid(PAIR, N_X), np.linspace(-3, 3, N_Z).round(2)
    lo, hi = PAIR.target_values[0] * 0.4, PAIR.target_values[-1] * 2.5
    rows = []
    for x in xs:
        for z in zs:
            mu = P.derive_mu(PAIR, float(x), float(z))
            if not (lo <= mu <= hi):
                continue
            for s in range(N_SEEDS):
                toy = P.make_implicit_prompt(PAIR, float(x), mu, s, k=K)
                nat = P.make_naturalistic_prompt(PAIR, float(x), mu, s, k=K, style="primed")
                nat_neutral = P.make_naturalistic_prompt(PAIR, float(x), mu, s, k=K, style="neutral")
                rows.append({"x": float(x), "z": float(z),
                             "toy": toy, "nat": nat, "nat_neutral": nat_neutral})
    return rows


@torch.inference_mode()
def extract_ld(model, tok, prompts, high_id, low_id, bs=8, device="cuda"):
    out = np.zeros(len(prompts))
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True).to(device)
        lg = model(**enc).logits[:, -1, :].float()
        out[i:i + bs] = (lg[:, high_id] - lg[:, low_id]).cpu().numpy()
    return out


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def cell_means(rows, key):
    cells = {}
    for r in rows:
        cells.setdefault((r["x"], r["z"]), []).append(r[key])
    xs = np.array([k[0] for k in cells])
    zs = np.array([k[1] for k in cells])
    ld = np.array([np.mean(v) for v in cells.values()])
    return xs, zs, ld


def main():
    rows = build_rows()
    print(f"{len(rows)} prompts  ({len(rows)//N_SEEDS} cells x {N_SEEDS} seeds), k={K}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"": "cuda"}
    ).eval()
    high_id = tokens_of_word(tok, PAIR.high_word)[0]
    low_id = tokens_of_word(tok, PAIR.low_word)[0]
    print(f"high='{PAIR.high_word}'({high_id}) low='{PAIR.low_word}'({low_id})", flush=True)
    print(f"{'frame':16s} {'corr(LD,z)':>12s} {'corr(LD,x)':>12s}")
    for key, label in [("toy", "toy implicit"), ("nat", "naturalistic-primed"),
                       ("nat_neutral", "naturalistic-neutral")]:
        L = extract_ld(model, tok, [r[key] for r in rows], high_id, low_id)
        for i, r in enumerate(rows):
            r[key + "_ld"] = L[i]
        xs, zs, mld = cell_means(rows, key + "_ld")
        print(f"{label:16s} {corr(mld, zs):>+12.3f} {corr(mld, xs):>+12.3f}", flush=True)


if __name__ == "__main__":
    main()
