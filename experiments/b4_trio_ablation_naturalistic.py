#!/usr/bin/env python3
"""B4: does the DLA trio (L23H14/L26H4/L31H3) still collapse corr(LD,z) on naturalistic prompts?

Reuses the paper's resample-ablation mechanism (p2u_n_sweep_xfeat.run_sweep): capture each head's
pre-o_proj output, then at the target position replace the trio heads' output with that of a random
donor prompt of different (x,z). Compares baseline vs trio-ablated corr(LD,z) for the toy implicit
frame and the naturalistic-neutral frame on gemma-2-9b (height dense grid).
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "circuit" / "scripts"))
from data_gen import prompts as P                                   # noqa: E402
from p2u_n_sweep_xfeat import run_sweep, get_decoder_layers         # noqa: E402

MODEL = "google/gemma-2-9b"
PAIR = P.PAIRS_BY_NAME["height"]
TRIO = [(23, 14), (26, 4), (31, 3)]   # DLA-ranked trio on gemma-2-9b
NS = [1, 2, 3]
N_X, N_Z, N_SEEDS, K = 12, 12, 3, 15


def x_grid(pair, n):
    lo, hi = float(min(pair.target_values)), float(max(pair.target_values))
    span = hi - lo
    return np.linspace(max(0.5, lo - 0.1 * span), hi + 0.1 * span, n).round(1)


def write_prompts(style, path):
    xs, zs = x_grid(PAIR, N_X), np.linspace(-3, 3, N_Z).round(2)
    lo, hi = PAIR.target_values[0] * 0.4, PAIR.target_values[-1] * 2.5
    n = 0
    with open(path, "w") as f:
        for x in xs:
            for z in zs:
                mu = P.derive_mu(PAIR, float(x), float(z))
                if not (lo <= mu <= hi):
                    continue
                for s in range(N_SEEDS):
                    p = (P.make_implicit_prompt(PAIR, float(x), mu, s, k=K) if style == "toy"
                         else P.make_naturalistic_prompt(PAIR, float(x), mu, s, k=K, style="neutral"))
                    f.write(json.dumps({"prompt": p, "x": float(x), "z": float(z)}) + "\n")
                    n += 1
    return n


def main():
    print(f"loading {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="eager", device_map="auto").eval()
    layers = get_decoder_layers(model)
    n_layers = len(layers)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    vocab = model.config.vocab_size
    high_id = tok.encode(" " + PAIR.high_word, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + PAIR.low_word, add_special_tokens=False)[-1]
    print(f"trio={TRIO}  high='{PAIR.high_word}'({high_id}) low='{PAIR.low_word}'({low_id})  "
          f"n_layers={n_layers} n_heads={n_heads} head_dim={head_dim}", flush=True)

    out = {}
    for style in ("toy", "neutral"):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as tf:
            path = Path(tf.name)
        n = write_prompts(style, path)
        print(f"\n=== {style} (n={n}) ===", flush=True)
        out[style] = run_sweep(path, "z", TRIO, NS, model, tok, layers,
                               n_layers, n_heads, head_dim, vocab, high_id, low_id)
        path.unlink()

    print("\n================ B4 SUMMARY (trio = L23H14, L26H4, L31H3) ================")
    print(f"{'frame':22s} {'base r(LD,z)':>12s} {'trio r(LD,z)':>12s} {'Δr(trio)':>10s} {'base r(LD,x)':>12s}")
    for style, label in [("toy", "toy implicit"), ("neutral", "naturalistic-neutral")]:
        r = out[style]
        trio = r["runs"][-1]  # N=3
        print(f"{label:22s} {r['baseline_r']:>+12.3f} {trio['r_LD_label']:>+12.3f} "
              f"{trio['delta_r']:>+10.3f} {r['baseline_r_x']:>+12.3f}")


if __name__ == "__main__":
    main()
