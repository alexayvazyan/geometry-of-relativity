"""Phase 2R refined — re-analyze captured attention to answer:

(A) row-normalized overlay EXCLUDING BOS and tail — reveals secondary
    attention patterns hidden under the BOS attention sink.
(B) attention conditional on numerical content — for each head, does its
    attention to a context number correlate with:
    - the number's raw value (max/min selector?)
    - distance from the target value (comparator?)
    - distance from the context mean (outlier detector?)
    - position rank (recency / primacy?)

Loads the previously captured p2r_attn_gemma2-2b.{npz,json}; no GPU needed.

Output:
  figures/p2r_attn_overlay_no_bos_gemma2-2b.png
  figures/p2r_attn_value_correlation_gemma2-2b.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
SHORT = "gemma2-2b"

CELLS: list[tuple[int, int, str]] = [
    (16, 4, "trio (hub)"),
    (17, 7, "trio (cooperator)"),
    (14, 2, "trio (backup)"),
    (15, 7, "H7-column"),
    (14, 7, "H7-column"),
    (12, 6, "cos·σ candidate"),
    (12, 4, "cos·σ candidate"),
    (13, 2, "cos·σ candidate"),
    (11, 2, "cos·σ candidate"),
    (13, 6, "cos·σ candidate"),
    (16, 2, "high-||Δa|| control"),
    (20, 2, "high-||Δa|| control"),
]
GROUP_COLORS = {
    "trio (hub)": "#d62728",
    "trio (cooperator)": "#ff7f0e",
    "trio (backup)": "#bcbd22",
    "H7-column": "#9467bd",
    "cos·σ candidate": "#1f77b4",
    "high-||Δa|| control": "#7f7f7f",
}


def find_number_runs(cats: list[str]) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) positions for each context-number run."""
    runs = []
    in_run = False
    start = 0
    for i, c in enumerate(cats):
        if c == "context_num":
            if not in_run:
                start = i
                in_run = True
        else:
            if in_run:
                runs.append((start, i))
                in_run = False
    if in_run:
        runs.append((start, len(cats)))
    return runs


def main() -> None:
    npz = np.load(REPO / "results" / f"p2r_attn_{SHORT}.npz")
    js = json.loads((REPO / "results" / f"p2r_attn_{SHORT}.json").read_text())

    # Reload original stimuli to get context_values per prompt
    stim_path = REPO / "data" / "p2_shot_sweep" / "height_k15.jsonl"
    rows_all = [json.loads(l) for l in stim_path.open()]

    n_prompts = len(js["prompts"])
    n_layers = 26
    n_heads = 8
    print(f"loaded {n_prompts} prompts, {len(CELLS)} cells")

    # Per-prompt analysis: pull attn[L,h] for each (L,h) in CELLS,
    # group consecutive context_num tokens into N runs, sum attn within each run,
    # match to the n-th context value from the stimulus.
    per_head_corrs = {(L, h): {"value": [], "dist_target": [],
                                "dist_mean": [], "rank": []}
                       for L, h, _ in CELLS}
    per_head_pos_attn = {(L, h): [] for L, h, _ in CELLS}  # list of (n_runs,) arrays

    for pi, p in enumerate(js["prompts"]):
        prompt_idx = p["prompt_idx"]
        row = rows_all[prompt_idx]
        ctx_values = row["context_values"]  # list of 15 numbers
        target_x = row["x"]
        mu = float(row.get("mu_eff", row.get("mu", 0)))
        attn = npz[f"attn_p{pi}"]  # (n_layers, n_heads, T) fp16

        cats = p["cats"]
        runs = find_number_runs(cats)
        # Pair runs to ctx_values: the runs in prompt order are Person 1..15
        # Skip the LAST number run if it corresponds to the target (target_num)
        # — we filter on cats since target is labeled differently.
        # Actually find_number_runs() only includes context_num runs, so target is
        # excluded by construction.
        if len(runs) != len(ctx_values):
            # Some prompts may have fewer runs (tokenization edge case). Trim.
            n_use = min(len(runs), len(ctx_values))
            runs = runs[:n_use]
            ctx_v = ctx_values[:n_use]
        else:
            ctx_v = ctx_values
        if len(runs) < 5:
            continue
        ctx_v = np.array(ctx_v, dtype=np.float32)
        ranks = np.arange(1, len(runs) + 1, dtype=np.float32)
        dist_target = np.abs(ctx_v - target_x)
        dist_mean = np.abs(ctx_v - ctx_v.mean())

        for ci, (L, h, _) in enumerate(CELLS):
            attn_runs = np.array([attn[L, h, s:e].sum() for s, e in runs],
                                   dtype=np.float32)
            per_head_pos_attn[(L, h)].append(attn_runs)
            # Spearman correlations (per-prompt, then averaged)
            try:
                per_head_corrs[(L, h)]["value"].append(
                    spearmanr(attn_runs, ctx_v)[0])
                per_head_corrs[(L, h)]["dist_target"].append(
                    spearmanr(attn_runs, dist_target)[0])
                per_head_corrs[(L, h)]["dist_mean"].append(
                    spearmanr(attn_runs, dist_mean)[0])
                per_head_corrs[(L, h)]["rank"].append(
                    spearmanr(attn_runs, ranks)[0])
            except Exception:
                pass

    print("\n=== mean Spearman correlations (per head, across prompts) ===")
    print(f"{'cell':<8s} {'group':<22s} {'ρ_val':>7s} {'ρ_distT':>8s} "
          f"{'ρ_distμ':>8s} {'ρ_rank':>8s}")
    rows_summary = []
    for L, h, g in CELLS:
        c = per_head_corrs[(L, h)]
        v = np.nanmean(c["value"]) if c["value"] else np.nan
        dt = np.nanmean(c["dist_target"]) if c["dist_target"] else np.nan
        dm = np.nanmean(c["dist_mean"]) if c["dist_mean"] else np.nan
        rk = np.nanmean(c["rank"]) if c["rank"] else np.nan
        rows_summary.append({"L": L, "h": h, "group": g,
                              "rho_value": float(v), "rho_dist_target": float(dt),
                              "rho_dist_mean": float(dm), "rho_rank": float(rk)})
        print(f"L{L:>2d}H{h}    {g:<22s} {v:+.3f}  {dt:+.3f}   {dm:+.3f}   {rk:+.3f}")

    # ============== VIZ B: bar chart of correlations per head ==============
    fig, ax = plt.subplots(figsize=(13, 6))
    n_cells = len(CELLS)
    x = np.arange(n_cells)
    w = 0.21
    metrics = [
        ("rho_value", "value (max-selector if >0, min if <0)", "#1f77b4"),
        ("rho_dist_target", "|x − target| (comparator if <0, anti if >0)", "#d62728"),
        ("rho_dist_mean", "|x − μ| (outlier-attender if >0)", "#2ca02c"),
        ("rho_rank", "rank (recency if >0, primacy if <0)", "#9467bd"),
    ]
    for i, (key, label, color) in enumerate(metrics):
        vals = [r[key] for r in rows_summary]
        ax.bar(x + (i - 1.5) * w, vals, w, label=label, color=color,
                edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{L}H{h}" for L, h, _ in CELLS], rotation=0, fontsize=10)
    for tick_label, (_, _, g) in zip(ax.get_xticklabels(), CELLS):
        tick_label.set_color(GROUP_COLORS[g])
    ax.set_ylabel("mean Spearman ρ across prompts")
    ax.set_title(f"{SHORT} — per-head: how does last-token attention to a "
                  f"context number correlate with the number's properties?\n"
                  f"red x-tick = causal trio")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_png_b = REPO / "figures" / f"p2r_attn_value_correlation_{SHORT}.png"
    fig.savefig(out_png_b, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png_b}")

    # ============== VIZ A: row-norm overlay EXCLUDING BOS+tail ==============
    # Re-pick the same representative z=0 prompt as before
    zs = npz["zs"]
    rep_pi = int(np.argmin(np.abs(zs)))
    rep = js["prompts"][rep_pi]
    rep_attn = npz[f"attn_p{rep_pi}"]
    cats = rep["cats"]
    tokens = rep["tokens"]
    T_rep = rep["T"]

    # Mask: keep only positions that are NOT bos and NOT tail
    keep_mask = np.array([c not in ("bos", "tail") for c in cats])
    # Also exclude trailing newlines/colons that aren't informative
    print(f"\nrepresentative prompt: idx={rep['prompt_idx']} z={rep['z']:+.2f}")
    print(f"  T={T_rep}, kept={keep_mask.sum()} positions")

    cell_attn = np.zeros((len(CELLS), T_rep), dtype=np.float32)
    for ci, (L, h, _) in enumerate(CELLS):
        cell_attn[ci] = rep_attn[L, h]
    # Row-normalize using only kept positions' max
    row_max = np.array([cell_attn[ci, keep_mask].max() for ci in range(len(CELLS))])
    row_max = np.clip(row_max, 1e-9, None)
    cell_attn_norm = cell_attn / row_max[:, None]
    # Now mask out (set to NaN) the bos and tail positions so they show grey
    cell_attn_norm = np.where(keep_mask[None, :], cell_attn_norm, np.nan)

    fig, axes = plt.subplots(2, 1,
                              figsize=(max(14, T_rep * 0.13),
                                        0.45 * len(CELLS) + 2.5),
                              gridspec_kw={"height_ratios": [len(CELLS), 1.5]},
                              sharex=True)
    ax = axes[0]
    cmap = plt.cm.magma.copy()
    cmap.set_bad("#dddddd")  # masked = grey
    im = ax.imshow(cell_attn_norm, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(CELLS)))
    ax.set_yticklabels([f"L{L}H{h}" for L, h, _ in CELLS], fontsize=10)
    for tick_label, (_, _, g) in zip(ax.get_yticklabels(), CELLS):
        tick_label.set_color(GROUP_COLORS[g])
    ax.set_title(f"{SHORT} — last-token attention with BOS+tail masked out\n"
                  f"(z={rep['z']:+.2f}, x={rep['x']:.0f}, μ={rep.get('mu', 0):.0f}, "
                  f"σ={rep.get('sigma', 1):.1f}; row-normalized over kept positions)",
                  fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01,
                  label="attn (row-norm, kept-only)")

    # Token strip
    cat_colors = {
        "context_num": "#1f77b4", "target_num": "#d62728", "unit": "#2ca02c",
        "person_label": "#9467bd", "colon": "#7f7f7f", "newline": "#ffd700",
        "tail": "#ff7f0e", "bos": "#000000", "other": "#cccccc",
    }
    ax = axes[1]
    for t, c in enumerate(cats):
        ax.add_patch(plt.Rectangle((t - 0.5, 0), 1, 1,
                                    facecolor=cat_colors.get(c, "#cccccc"),
                                    edgecolor="white", linewidth=0.5))
    for t, tk in enumerate(tokens):
        s = tk.replace("\n", "\\n").strip()
        if not s:
            continue
        ax.text(t, 0.5 if t % 2 == 0 else 0.15,
                 s[:6], ha="center", va="center",
                 fontsize=6, rotation=90,
                 color="white" if cats[t] in
                 ("target_num", "tail", "person_label", "bos") else "black")
    ax.set_xlim(-0.5, T_rep - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("token position")
    handles = [plt.Rectangle((0, 0), 1, 1, color=cat_colors[c], label=c)
                for c in ["context_num", "target_num", "unit", "person_label",
                          "colon", "newline", "tail"]]
    ax.legend(handles=handles, loc="upper center", ncol=7, fontsize=8,
               bbox_to_anchor=(0.5, -0.5))
    fig.tight_layout()
    out_png_a = REPO / "figures" / f"p2r_attn_overlay_no_bos_{SHORT}.png"
    fig.savefig(out_png_a, dpi=150, bbox_inches="tight")
    print(f"wrote {out_png_a}")

    # Save summary JSON
    out_json = REPO / "results" / f"p2r_attn_value_correlation_{SHORT}.json"
    out_json.write_text(json.dumps({"cells": rows_summary}, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
