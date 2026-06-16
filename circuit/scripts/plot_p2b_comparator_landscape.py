"""Phase 2B — clean summary figure: per-head comparator score across L0..L3.

For each (model, layer, head, k), the comparator score is
  r(Δattn, z_eff) = pearson(attn_mass[tgt] − attn_mass[last_ctx], z_eff)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = REPO / "figures"


def load(model: str, pair: str = "height"):
    p = REPO / "results" / f"p2b_l0l1_focused_{model}_{pair}.json"
    with p.open() as f:
        return json.load(f)


def main():
    pair = "height"
    models = ["gemma2-2b", "gemma2-9b"]
    ks = [1, 4, 15]
    layers = [0, 1, 2, 3]

    fig, axes = plt.subplots(len(models), len(layers),
                              figsize=(3.6 * len(layers), 3.0 * len(models)),
                              squeeze=False)

    for mi, model in enumerate(models):
        data = load(model, pair)
        first_layer_heads = len(next(iter(data.values()))['bos'])
        for li, L in enumerate(layers):
            ax = axes[mi, li]
            heads_avail = None
            for k, color in zip(ks, ["C0", "C1", "C2"]):
                key = f"L{L}_k{k}"
                if key not in data:
                    continue
                r = np.asarray(data[key]['r_attn_diff_z'])
                heads_avail = np.arange(len(r))
                ax.plot(heads_avail, r, "o-", label=f"k={k}", color=color, markersize=5)
            ax.axhline(0, color='k', linewidth=0.4, alpha=0.4)
            ax.axhline(0.4, color='red', linewidth=0.4, linestyle="--", alpha=0.4)
            ax.set_ylim(-0.4, 0.9)
            ax.set_title(f"{model.replace('gemma2-', '').upper()}  |  L{L}")
            if li == 0:
                ax.set_ylabel("r(Δattn, z_eff)")
            if mi == len(models) - 1:
                ax.set_xlabel("head")
            if heads_avail is not None and len(heads_avail) > 8:
                ax.set_xticks(heads_avail[::2])
            if mi == 0 and li == len(layers) - 1:
                ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Phase 2B — comparator score per head (red dashed = 0.4 threshold)", y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "p2b_comparator_landscape.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
