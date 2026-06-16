"""Phase 2B — per-head decomposition: which heads' OUTPUT carries z?

For each (layer, head, k):
  - r²(z_eff) of a ridge probe fit on head_outs[N, L, h, head_dim]
  - DLA-onto-z: project head_outs[N, L, h] through the W_O slice for that
    head, dot into a "primal_z" direction at that layer, average.

This complements attention-pattern analysis: a head can have unimpressive
attention mass on context-value tokens but still write a strong z-signal
into the residual via QK aggregation.

primal_z is built per-layer from the same trial set:
  primal_z[L] = mean(residual[L] | z_eff > +1) − mean(residual[L] | z_eff < -1)
  (computed on cell_seed=0 prompts; held-out r² uses cell_seed != 0)

We use z_eff (sample-mean-based z) as the regression target since that's
what the model actually has access to.

Usage:
  python scripts/analyze_p2b_per_head_z.py --model gemma2-2b --pair height
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results" / "p2_attn"
FIG_DIR = REPO / "figures"


def cv_r2(X, y, k=5, seed=0):
    """5-fold CV R² of a ridge regression."""
    if y.std() < 1e-9 or X.shape[0] < k * 2:
        return 0.0
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=np.float64)
    for tr, te in kf.split(X):
        m = RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0)).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12))


def primal_at_layer(residuals, z_eff, mask_train):
    """primal_z[L] = mean(h | z_eff>+1) − mean(h | z_eff<-1) on train rows."""
    n_layers = residuals.shape[1]
    primals = np.zeros((n_layers, residuals.shape[2]), dtype=np.float64)
    for L in range(n_layers):
        h = residuals[:, L, :].astype(np.float64)
        m_high = mask_train & (z_eff > +1.0)
        m_low = mask_train & (z_eff < -1.0)
        if m_high.sum() < 3 or m_low.sum() < 3:
            continue
        primals[L] = h[m_high].mean(0) - h[m_low].mean(0)
    return primals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--pair", required=True)
    p.add_argument("--k", nargs="+", type=int, default=[1, 4, 15])
    p.add_argument("--n-prompts", type=int, default=600,
                   help="cap per (k) for CV speed")
    args = p.parse_args()

    in_dir = RESULTS / args.model

    # Load each k's residuals + head_outs.
    per_k = {}
    for k in args.k:
        path = in_dir / f"{args.pair}_k{k}.npz"
        if not path.exists():
            print(f"[skip] {path} missing")
            continue
        print(f"[load] {path}")
        d = np.load(path, allow_pickle=True)
        z_eff = d['z_eff']
        # subset for CV speed
        n_total = z_eff.shape[0]
        rng = np.random.default_rng(0)
        idx = rng.choice(n_total, size=min(args.n_prompts, n_total), replace=False)
        per_k[k] = {
            'residuals': d['residuals'][idx].astype(np.float64),  # (n, L, d)
            'head_outs': d['head_outs'][idx].astype(np.float64),  # (n, L, H, hd)
            'z_eff': z_eff[idx].astype(np.float64),
            'cell_seed': d['cell_seed'][idx],
            'ld': d['ld'][idx].astype(np.float64),
        }

    if not per_k:
        return

    # --- per-head r²(z_eff) on head_outs ---
    sample = next(iter(per_k.values()))
    n_layers = sample['head_outs'].shape[1]
    n_heads = sample['head_outs'].shape[2]
    r2_table: dict[int, np.ndarray] = {}
    r2_resid_table: dict[int, np.ndarray] = {}
    primals_per_k: dict[int, np.ndarray] = {}
    dla_table: dict[int, np.ndarray] = {}

    for k, data in per_k.items():
        print(f"\n[p2b-z] computing per-head r²(z_eff) for k={k}...")
        z = data['z_eff']
        head_outs = data['head_outs']  # (n, L, H, hd)
        residuals = data['residuals']  # (n, L, d)

        # train/test split: cell_seed=0 train, others test (matches Phase 1 convention).
        cs = data['cell_seed']
        train_mask = cs == 0
        # if no cell_seed=0 (some k=0 case), fallback to first half.
        if train_mask.sum() < 50:
            train_mask = np.zeros(len(z), dtype=bool)
            train_mask[:len(z)//2] = True
        primals = primal_at_layer(residuals, z, train_mask)
        primals_per_k[k] = primals

        r2 = np.zeros((n_layers, n_heads), dtype=np.float64)
        dla = np.zeros((n_layers, n_heads), dtype=np.float64)
        # r²(z) of full residual at each layer, for context.
        r2_resid = np.zeros(n_layers, dtype=np.float64)
        for L in range(n_layers):
            X_resid = residuals[:, L, :]
            r2_resid[L] = cv_r2(X_resid, z)
            for H in range(n_heads):
                X_head = head_outs[:, L, H, :]   # (n, hd)
                r2[L, H] = cv_r2(X_head, z)
                # DLA: project head out through W_O slice. We don't have W_O
                # cached here. Approximation: dot product between head_outs and
                # the portion of primals[L] in the "head writes here" slice.
                # In a standard transformer:
                #   delta_h_layer = sum_h W_O[:, h*hd:(h+1)*hd] @ head_outs[h]
                # so the "write contribution to primals[L]" of head h is
                #   primals[L] @ W_O[:, h*hd:(h+1)*hd] @ head_outs[h]
                # Without W_O, we use the per-head output directly projected
                # onto the per-head SLICE of primals (treats W_O as identity).
                # This is an approximation; results should be checked with
                # actual W_O later.
                hd = head_outs.shape[-1]
                pr_slice = primals[L][H * hd : (H + 1) * hd]
                # only meaningful if the "primal direction" has support there
                if np.linalg.norm(pr_slice) > 1e-9 and head_outs[:, L, H].std() > 1e-9:
                    dla[L, H] = float(np.mean(head_outs[:, L, H] @ pr_slice))
        r2_table[k] = r2
        dla_table[k] = dla
        r2_resid_table[k] = r2_resid

    # --- print top heads by r²(z_eff) at each k ---
    for k in sorted(r2_table.keys()):
        r2 = r2_table[k]
        flat = np.argsort(-r2.flatten())[:12]
        print(f"\n=== {args.model} {args.pair} k={k}: top 12 (L, H) by r²(z_eff) ===")
        print(f"{'rank':>4} {'L':>3} {'H':>2} {'r²(z)':>8}")
        for rank, idx in enumerate(flat):
            L = int(idx // n_heads); H = int(idx % n_heads)
            print(f"{rank+1:>4} {L:>3} {H:>2} {r2[L, H]:>8.3f}")

    # --- residual-stream r²(z) per layer per k (the Phase 1 "ΔR² peaks at L1" check) ---
    print(f"\n=== Residual-stream r²(z_eff) per layer per k ({args.model} {args.pair}) ===")
    print(f"{'L':>3}", end=" ")
    for k in sorted(r2_resid_table.keys()):
        print(f"{'k='+str(k):>10}", end=" ")
    print()
    for L in range(n_layers):
        print(f"{L:>3}", end=" ")
        for k in sorted(r2_resid_table.keys()):
            print(f"{r2_resid_table[k][L]:>10.3f}", end=" ")
        print()

    # --- figure: r²(z) heatmap per k ---
    ks = sorted(r2_table.keys())
    fig, axes = plt.subplots(2, len(ks), figsize=(3.6 * len(ks), 7), squeeze=False)
    for ki, k in enumerate(ks):
        ax = axes[0, ki]
        im = ax.imshow(r2_table[k], aspect='auto', cmap='magma',
                       vmin=0, vmax=max(0.05, r2_table[k].max()))
        ax.set_title(f"per-head r²(z_eff)  |  k={k}")
        ax.set_ylabel("layer")
        ax.set_xlabel("head")
        fig.colorbar(im, ax=ax, fraction=0.04)
        ax2 = axes[1, ki]
        ax2.plot(r2_resid_table[k], 'o-', markersize=3)
        ax2.set_title(f"residual r²(z_eff) per layer | k={k}")
        ax2.set_xlabel("layer")
        ax2.set_ylim(-0.05, 1.0)
        ax2.grid(alpha=0.3)
    fig.suptitle(f"P2B z-encoding: head-output and residual probes ({args.model} | {args.pair})", y=1.0)
    fig.tight_layout()
    out_fig = FIG_DIR / f"p2b_per_head_r2z_{args.model}_{args.pair}.png"
    fig.savefig(out_fig, dpi=140, bbox_inches="tight")
    print(f"\n  -> {out_fig}")

    # --- figure: ΔR²(z_eff) per layer per k (the "where is z first encoded" check) ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in ks:
        r2 = r2_resid_table[k]
        # ΔR² = R²[L] − R²[L-1], with R²[-1] = 0
        d = np.diff(np.concatenate([[0.0], r2]))
        ax.plot(range(len(d)), d, "o-", markersize=4, label=f"k={k}")
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("layer")
    ax.set_ylabel("ΔR²(z_eff) at this layer")
    ax.set_title(f"Where is z newly encoded? ({args.model} | {args.pair})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_fig2 = FIG_DIR / f"p2b_increment_r2_{args.model}_{args.pair}.png"
    fig.savefig(out_fig2, dpi=140, bbox_inches="tight")
    print(f"  -> {out_fig2}")

    # --- save JSON ---
    out_json = REPO / "results" / f"p2b_per_head_r2_{args.model}_{args.pair}.json"
    with out_json.open("w") as f:
        json.dump({
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "r2_per_head": {str(k): v.tolist() for k, v in r2_table.items()},
            "r2_residual_per_layer": {str(k): v.tolist() for k, v in r2_resid_table.items()},
            "dla_per_head": {str(k): v.tolist() for k, v in dla_table.items()},
        }, f)
    print(f"  -> {out_json}")


if __name__ == "__main__":
    main()
