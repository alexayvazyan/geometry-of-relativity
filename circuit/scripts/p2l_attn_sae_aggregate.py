"""Phase 2L — aggregate SAE decomposition results across (model, feature, k).

Reads every results/p2l_attn_sae_*.json (and matching residuals NPZ when
present) and produces:

  1. CSV summary  — one row per (model, feature, k) with baseline r(LD,z),
     n_active features, top-3 features, primary-head share, primary-head
     z-correlation rank.
  2. Top-feature reuse table — which features appear in top-N across
     (feature × k) for each model. High overlap → general z-encoder; low
     overlap → context-specific.
  3. Plot: r(feat, z) vs r(feat, LD) scatter, faceted by (model, feature, k).
     Features in the upper-right (high z-corr, high LD-corr) are on the
     z → LD pipeline.

Output:
  results/p2l_aggregate_summary.csv
  results/p2l_aggregate_feature_reuse.json
  figures/p2l_aggregate_z_vs_ld.png
  figures/p2l_aggregate_head_zcorr_grid.png

Usage:
    python p2l_attn_sae_aggregate.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_cell(json_path: Path, npz_path: Path | None = None) -> dict:
    d = json.loads(json_path.read_text())
    if npz_path is not None and npz_path.exists():
        d["_npz"] = np.load(npz_path)
    return d


def cell_summary(d: dict) -> dict:
    top = d["top_features"][:5]
    base = d.get("baseline_LD_z_corr")
    base_ld = d.get("baseline_LD_mean")
    ph = d["primary_head"]
    # For each top feature, how many have primary head as argmax (attr or zcorr)
    ph_attr_top = sum(1 for f in top if f.get("argmax_attr_head") == ph)
    ph_zcorr_top = sum(1 for f in top if f.get("argmax_zcorr_head") == ph)
    return {
        "model": d["short"], "feature": d["feature"], "k": d["k"],
        "primary_head": ph,
        "n_active": d["sae"]["n_active"],
        "avg_l0": d["sae"]["avg_l0_per_prompt"],
        "baseline_r_LD_z": base, "baseline_LD_mean": base_ld,
        "top1_feat": top[0]["feat_idx"], "top1_r_z": top[0]["r_z"],
        "top1_r_ld": top[0].get("r_ld"),
        "top1_argmax_attr": top[0].get("argmax_attr_head"),
        "top1_argmax_zcorr": top[0].get("argmax_zcorr_head"),
        "ph_in_top5_attr": ph_attr_top,
        "ph_in_top5_zcorr": ph_zcorr_top,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob",
                    default="p2l_attn_sae_gemma*.json")
    args = ap.parse_args()

    res_dir = REPO / "results"
    cells = []
    for jp in sorted(res_dir.glob(args.results_glob)):
        if "_residuals" in jp.stem:
            continue
        npz_path = jp.with_name(jp.stem + "_residuals.npz")
        cells.append((jp, load_cell(jp, npz_path)))

    if not cells:
        raise SystemExit("no cells found")

    # 1. CSV summary
    rows = [cell_summary(d) for _, d in cells]
    csv_path = res_dir / "p2l_aggregate_summary.csv"
    keys = list(rows[0].keys())
    with csv_path.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r.get(k):.4f}" if isinstance(r.get(k), float)
                else str(r.get(k, ""))
                for k in keys) + "\n")
    print(f"wrote {csv_path}")
    print()
    # Print to stdout for quick read
    for r in rows:
        print(f"  {r['model']:13s}  {r['feature']:6s} k={r['k']:>2d}  "
              f"n_active={r['n_active']:>3d}  "
              f"r(LD,z)={r['baseline_r_LD_z']:+.3f}  "
              f"top1=feat{r['top1_feat']:>5d} r_z={r['top1_r_z']:+.2f} "
              f"r_ld={r['top1_r_ld']:+.2f}  "
              f"argmax_attr=H{r['top1_argmax_attr']} "
              f"argmax_zcorr=H{r['top1_argmax_zcorr']}  "
              f"primary in top5: attr {r['ph_in_top5_attr']}/5  "
              f"zcorr {r['ph_in_top5_zcorr']}/5")

    # 2. Feature reuse: per (model, feature×k) sets
    reuse_per_model: dict = defaultdict(lambda: defaultdict(set))
    for _, d in cells:
        m = d["short"]
        key = f"{d['feature']}_k{d['k']}"
        reuse_per_model[m][key] = set(f["feat_idx"] for f in d["top_features"][:10])
    reuse_summary: dict = {}
    for m, sets in reuse_per_model.items():
        keys = list(sets.keys())
        n = len(keys)
        # Pairwise overlap
        overlap = {}
        for i in range(n):
            for j in range(i, n):
                inter = len(sets[keys[i]] & sets[keys[j]])
                overlap[f"{keys[i]} ∩ {keys[j]}"] = inter
        # Features appearing in ≥ 2 settings
        all_feats = set().union(*sets.values())
        shared = {f: sum(1 for s in sets.values() if f in s) for f in all_feats}
        shared = {f: c for f, c in sorted(shared.items(), key=lambda kv: -kv[1])
                   if c >= 2}
        reuse_summary[m] = {
            "settings": keys,
            "pairwise_overlap_top10": overlap,
            "feats_in_multiple_settings": shared,
        }
    reuse_path = res_dir / "p2l_aggregate_feature_reuse.json"
    reuse_path.write_text(json.dumps(reuse_summary, indent=2))
    print(f"\nwrote {reuse_path}")
    for m, info in reuse_summary.items():
        print(f"\n  {m}: features in ≥2 of {len(info['settings'])} settings: "
              f"{len(info['feats_in_multiple_settings'])}")
        for f, c in list(info['feats_in_multiple_settings'].items())[:10]:
            print(f"    feat {f}: {c}/{len(info['settings'])} settings")

    # 3. Plot — r(feat, z) vs r(feat, LD), per cell
    n_cells = len(cells)
    cols = min(5, n_cells)
    rows_p = (n_cells + cols - 1) // cols
    fig, axes = plt.subplots(rows_p, cols, figsize=(4 * cols, 3.4 * rows_p),
                              squeeze=False)
    for i, (_, d) in enumerate(cells):
        ax = axes[i // cols, i % cols]
        rzs = [f["r_z"] for f in d["top_features"]]
        rlds = [f.get("r_ld", 0) for f in d["top_features"]]
        ph = d["primary_head"]
        is_ph = [f.get("argmax_zcorr_head") == ph for f in d["top_features"]]
        col = ["C3" if x else "C7" for x in is_ph]
        ax.scatter(rzs, rlds, c=col, s=40, edgecolor="black", linewidth=0.5)
        for f, r_z, r_ld in zip(d["top_features"][:5], rzs[:5], rlds[:5]):
            ax.annotate(str(f["feat_idx"]), (r_z, r_ld), fontsize=6,
                         xytext=(3, 2), textcoords="offset points")
        ax.axhline(0, color="black", lw=0.4)
        ax.axvline(0, color="black", lw=0.4)
        ax.set_xlabel("r(feat, z)")
        ax.set_ylabel("r(feat, LD)")
        base = d.get("baseline_LD_z_corr", float("nan"))
        ax.set_title(
            f"{d['short']} {d['feature']} k={d['k']}\n"
            f"n_active={d['sae']['n_active']}  baseline r(LD,z)={base:+.2f}",
            fontsize=9)
        ax.grid(alpha=0.3)
    for i in range(n_cells, rows_p * cols):
        axes[i // cols, i % cols].axis("off")
    fig.suptitle(
        "Phase 2L — feature z-corr vs LD-corr (red = primary-head argmax_zcorr)",
        fontsize=12)
    fig.tight_layout()
    out_png = REPO / "figures" / "p2l_aggregate_z_vs_ld.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png}")

    # 4. Per-model head signed-zcorr grid (rows = features, cols = heads)
    by_model: dict = defaultdict(list)
    for _, d in cells:
        by_model[d["short"]].append(d)
    fig2, axes = plt.subplots(1, len(by_model),
                                figsize=(7 * len(by_model), 5),
                                squeeze=False)
    for ci, (m, ds) in enumerate(by_model.items()):
        ax = axes[0, ci]
        rows_h = []
        labels = []
        for d in ds:
            for f in d["top_features"][:5]:
                rows_h.append(f["head_zcorr"])
                labels.append(f"{d['feature'][:1]}k{d['k']}/f{f['feat_idx']}")
        if not rows_h:
            ax.axis("off")
            continue
        H = np.array(rows_h)
        vmax = max(0.05, float(np.max(np.abs(H))))
        im = ax.imshow(H, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=+vmax)
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("head")
        ax.set_xticks(np.arange(H.shape[1]))
        ax.set_xticklabels([f"H{h}" for h in range(H.shape[1])], fontsize=8)
        primary = ds[0]["primary_head"]
        ax.axvline(primary - 0.5, color="black", lw=1)
        ax.axvline(primary + 0.5, color="black", lw=1)
        ax.set_title(f"{m} — top-5 features per setting × head signed z-corr\n"
                      f"(black band = primary head H{primary})", fontsize=10)
        fig2.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="r")
    fig2.tight_layout()
    out_png2 = REPO / "figures" / "p2l_aggregate_head_zcorr_grid.png"
    fig2.savefig(out_png2, dpi=130, bbox_inches="tight")
    print(f"wrote {out_png2}")


if __name__ == "__main__":
    main()
