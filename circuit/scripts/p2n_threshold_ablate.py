"""Phase 2N — joint multi-head ablation gated by Phase 2M cosine threshold.

Reads `results/p2m_alignment_<short>.json` to get the (n_layer × n_head) cosine
grid (each cell = cos(Δ_ablate(h, L'), Δ_manifold) at the readout SAE).
Sweeps thresholds and signed modes, jointly zeros all selected heads, and
measures δr(LD, z) and δ⟨LD⟩ vs the actual baseline.

Modes per threshold t:
  - positive: cells with cos > +t
  - negative: cells with cos < −t
  - both:     cells with |cos| > t  (= positive ∪ negative)
  - random:   same n_cells as `both`, picked uniformly (control)

Output:
  results/p2n_threshold_ablate_<short>.json
  figures/p2n_threshold_ablate_<short>.png

Compute (2B, height k=15, n=990 prompts, 5 thresholds × 4 modes ≈ 20 forwards × 990
prompts × 14ms ≈ 5 min). Cheap.

Usage:
  python p2n_threshold_ablate.py --short gemma2-2b
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent


def get_decoder_layers(model):
    for path in [("model", "layers"), ("model", "model", "layers")]:
        m = model
        ok = True
        for attr in path:
            if hasattr(m, attr):
                m = getattr(m, attr)
            else:
                ok = False; break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", required=True,
                    choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--feature", default="height")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.10, 0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="cap n stimuli (default all)")
    ap.add_argument("--seed", type=int, default=42,
                    help="rng for random control")
    ap.add_argument("--n-random-trials", type=int, default=3,
                    help="repeat the random control n times")
    args = ap.parse_args()

    # 1. Load cos_grid
    cos_path = REPO / "results" / f"p2m_alignment_{args.short}.json"
    if not cos_path.exists():
        raise SystemExit(f"missing {cos_path} — run p2m_sae_alignment_grid first")
    cos_data = json.loads(cos_path.read_text())
    cos_grid = np.array(cos_data["cos_grid"])
    layer_idxs = cos_data["layer_idxs"]
    head_idxs = cos_data["head_idxs"]
    print(f"loaded cos_grid {cos_grid.shape} from {cos_path.name}")
    print(f"  layers: {layer_idxs}")
    print(f"  heads:  {head_idxs}")
    print(f"  cos range: {cos_grid.min():+.3f} .. {cos_grid.max():+.3f}")

    model_id = cos_data["model"]
    feature = cos_data["feature"]
    k = cos_data["k"]
    if feature != args.feature or k != args.k:
        print(f"WARN: cos_grid was for {feature} k={k}, but you asked for "
              f"{args.feature} k={args.k}. Continuing with the cos_grid.")
    feature, k = cos_data["feature"], cos_data["k"]

    # 2. Load stimuli
    stim_path = REPO / "data" / "p2_shot_sweep" / f"{feature}_k{k}.jsonl"
    rows = [json.loads(l) for l in stim_path.open()]
    if args.n_prompts:
        rows = rows[:args.n_prompts]
    n = len(rows)
    print(f"\nloaded {n} prompts from {stim_path.name}")

    # 3. Load model
    print(f"\nloading {model_id}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN")).eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    layers = get_decoder_layers(model)
    n_heads = layers[0].self_attn.config.num_attention_heads
    head_dim = layers[0].self_attn.o_proj.in_features // n_heads
    high_w = rows[0].get("high_word", "tall")
    low_w = rows[0].get("low_word", "short")
    high_id = tok.encode(" " + high_w, add_special_tokens=False)[-1]
    low_id = tok.encode(" " + low_w, add_special_tokens=False)[-1]

    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                       dtype=np.float32)

    # 4. Forward fn with multi-head zero ablations
    def fwd_LD(layer_to_heads: dict[int, set[int]]):
        """layer_to_heads: {layer_idx: set of head indices to zero at last token}.
        Returns LD array shape (n,)."""
        handles = []
        for L_abl, head_set in layer_to_heads.items():
            if not head_set:
                continue
            head_set_local = set(head_set)
            def make_hook(hs):
                def hook(module, args_):
                    x = args_[0]
                    xm = x.clone()
                    for h in hs:
                        xm[:, -1, h * head_dim:(h + 1) * head_dim] = 0
                    return (xm,) + args_[1:]
                return hook
            h_abl = layers[L_abl].self_attn.o_proj.register_forward_pre_hook(
                make_hook(head_set_local))
            handles.append(h_abl)

        ld = np.zeros(n, dtype=np.float32)
        try:
            with torch.inference_mode():
                for i, row in enumerate(rows):
                    inp = tok(row["prompt"], return_tensors="pt").to(model.device)
                    out = model(**inp, use_cache=False)
                    logits = out.logits[0, -1].float()
                    ld[i] = float(logits[high_id] - logits[low_id])
        finally:
            for h in handles:
                h.remove()
        return ld

    # 5. Baseline (no ablation) — sanity check
    print(f"\n=== baseline ===")
    t1 = time.time()
    ld_base = fwd_LD({})
    base_r = float(np.corrcoef(ld_base, z_arr)[0, 1])
    print(f"  baseline r(LD,z) = {base_r:+.3f}  <LD>={ld_base.mean():+.2f}  "
          f"({time.time()-t1:.0f}s)")

    # 6. Threshold sweep
    rng = np.random.RandomState(args.seed)
    results = {
        "model": model_id, "short": args.short,
        "feature": feature, "k": k, "n_prompts": int(n),
        "baseline_r_LD_z": base_r,
        "baseline_LD_mean": float(ld_base.mean()),
        "n_total_cells": int(cos_grid.size),
        "thresholds": [],
    }

    def cells_for(mask: np.ndarray) -> dict[int, set[int]]:
        out: dict = {}
        for li in range(mask.shape[0]):
            for hi in range(mask.shape[1]):
                if mask[li, hi]:
                    out.setdefault(layer_idxs[li], set()).add(head_idxs[hi])
        return out

    for t in args.thresholds:
        pos_mask = cos_grid > t
        neg_mask = cos_grid < -t
        both_mask = pos_mask | neg_mask
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        n_both = int(both_mask.sum())
        print(f"\n=== t={t:.2f}  pos={n_pos}, neg={n_neg}, both={n_both} ===")
        if n_both == 0:
            continue

        # positive only
        ld_pos = fwd_LD(cells_for(pos_mask))
        r_pos = float(np.corrcoef(ld_pos, z_arr)[0, 1])
        # negative only
        ld_neg = fwd_LD(cells_for(neg_mask))
        r_neg = float(np.corrcoef(ld_neg, z_arr)[0, 1])
        # both
        ld_both = fwd_LD(cells_for(both_mask))
        r_both = float(np.corrcoef(ld_both, z_arr)[0, 1])
        # random control: same count as both, repeated
        rand_rs: list[float] = []
        rand_lds: list[float] = []
        all_idx = np.arange(cos_grid.size)
        for trial in range(args.n_random_trials):
            chosen = rng.choice(all_idx, size=n_both, replace=False)
            rand_mask = np.zeros_like(cos_grid, dtype=bool)
            for c in chosen:
                li, hi = np.unravel_index(c, cos_grid.shape)
                rand_mask[li, hi] = True
            ld_r = fwd_LD(cells_for(rand_mask))
            r_r = float(np.corrcoef(ld_r, z_arr)[0, 1])
            rand_rs.append(r_r)
            rand_lds.append(float(ld_r.mean()))

        rand_r_mean = float(np.mean(rand_rs))
        rand_r_std = float(np.std(rand_rs))

        print(f"  positive ({n_pos:>3d} cells): r={r_pos:+.3f}  Δr={r_pos-base_r:+.3f}  <LD>={ld_pos.mean():+.2f}")
        print(f"  negative ({n_neg:>3d} cells): r={r_neg:+.3f}  Δr={r_neg-base_r:+.3f}  <LD>={ld_neg.mean():+.2f}")
        print(f"  both     ({n_both:>3d} cells): r={r_both:+.3f}  Δr={r_both-base_r:+.3f}  <LD>={ld_both.mean():+.2f}")
        print(f"  random×{args.n_random_trials} ({n_both:>3d} cells): "
              f"r={rand_r_mean:+.3f}±{rand_r_std:.3f}  Δr={rand_r_mean-base_r:+.3f}")

        results["thresholds"].append({
            "threshold": t,
            "n_positive_cells": n_pos, "r_positive": r_pos,
            "n_negative_cells": n_neg, "r_negative": r_neg,
            "n_both_cells": n_both, "r_both": r_both,
            "ld_mean_positive": float(ld_pos.mean()),
            "ld_mean_negative": float(ld_neg.mean()),
            "ld_mean_both": float(ld_both.mean()),
            "rand_r_mean": rand_r_mean, "rand_r_std": rand_r_std,
            "rand_r_trials": rand_rs,
            "rand_ld_means": rand_lds,
        })

    out_json = REPO / "results" / f"p2n_threshold_ablate_{args.short}.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_json}")

    # 7. Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ts = [t["threshold"] for t in results["thresholds"]]

    ax = axes[0]
    ax.axhline(base_r, color="black", lw=1.2, ls="--", label=f"baseline r={base_r:+.3f}")
    ax.plot(ts, [t["r_positive"] for t in results["thresholds"]], "-o",
             color="C0", label="positive cos > t (z-encoders)")
    ax.plot(ts, [t["r_negative"] for t in results["thresholds"]], "-s",
             color="C3", label="negative cos < −t (anti-z)")
    ax.plot(ts, [t["r_both"] for t in results["thresholds"]], "-^",
             color="C2", lw=2, label="both signs (|cos| > t)")
    rand_ms = [t["rand_r_mean"] for t in results["thresholds"]]
    rand_ss = [t["rand_r_std"] for t in results["thresholds"]]
    ax.errorbar(ts, rand_ms, yerr=rand_ss, fmt="-x", color="gray",
                 alpha=0.7, label="random same-count control")
    ax.set_xlabel("|cos| threshold")
    ax.set_ylabel("r(LD, z) under joint ablation")
    ax.set_title(f"{args.short} {feature} k={k} — joint ablation by cosine threshold\n"
                  f"baseline r={base_r:+.3f}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(ts, [t["n_positive_cells"] for t in results["thresholds"]], "-o",
             color="C0", label="positive cells")
    ax.plot(ts, [t["n_negative_cells"] for t in results["thresholds"]], "-s",
             color="C3", label="negative cells")
    ax.plot(ts, [t["n_both_cells"] for t in results["thresholds"]], "-^",
             color="C2", lw=2, label="both")
    ax.axhline(cos_grid.size, color="black", lw=0.5, ls=":",
                label=f"{cos_grid.size} total cells")
    ax.set_xlabel("|cos| threshold")
    ax.set_ylabel("n cells selected")
    ax.set_title("cell counts per threshold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2n_threshold_ablate_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
