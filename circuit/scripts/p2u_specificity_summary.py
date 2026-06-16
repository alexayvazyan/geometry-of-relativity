"""Cross-task specificity summary plot.

Reads per-task JSONs produced by p2u_specificity.py and plots:
  - one grouped-bar panel per task on the left (Δr for labelled tasks)
  - one neutral-text KL panel on the right

Bars per group: trio (red), 7-cell ext (orange), hub-alone (blue),
random-3 mean ± std (gray).

Usage:
  python3 scripts/p2u_specificity_summary.py --short gemma2-9b \
      --tasks relativity-it arithmetic truth neutral_text
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

TASK_DISPLAY = {
    "relativity-it": "relativity\n(height)",
    "relativity-weight": "relativity\n(weight)",
    "relativity-speed": "relativity\n(speed)",
    "arithmetic": "arithmetic",
    "truth": "truth",
    "truth-base": "truth\n(2-shot ICL)",
    "refusal": "refusal",
    "neutral_text": "neutral\ntext (KL)",
}


def _classify(label: str) -> str:
    if label.startswith("trio"):
        return "trio"
    if "extended" in label:
        return "ext"
    if "hub alone" in label:
        return "hub"
    if label.startswith("random"):
        return "random"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-9b")
    ap.add_argument("--tasks", nargs="+", required=True)
    args = ap.parse_args()

    summaries = {}
    for t in args.tasks:
        p = REPO / "results" / f"p2u_specificity_{args.short}_{t}.json"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        d = json.loads(p.read_text())
        cats = {"trio": [], "ext": [], "hub": [], "random": []}
        for r in d["runs"]:
            cats[_classify(r["label"])].append(r)
        summaries[t] = {
            "baseline_r": d["baseline"].get("r_LD_label"),
            "cats": cats,
        }

    # Split tasks: those with Δr (labelled) vs neutral_text (KL only)
    labelled = [t for t in args.tasks
                  if summaries[t]["cats"]["trio"]
                  and summaries[t]["cats"]["trio"][0].get("delta_r") is not None]
    kl_only = [t for t in args.tasks if t not in labelled]

    # Two separate figures: Δr (labelled tasks) and KL (neutral_text).
    fig_dr, ax_dr = plt.subplots(figsize=(11, 6.5))

    # ---- Δr grouped bars ----
    n = len(labelled)
    x = np.arange(n)
    w = 0.20
    # Order from left to right: top-7, top-3, top-1, random-3.
    cat_order = ["ext", "trio", "hub", "random"]
    cat_colors = {"trio": "tab:red", "ext": "tab:orange",
                   "hub": "tab:blue", "random": "0.55"}
    cat_labels = {"ext": "top 7", "trio": "top 3", "hub": "top 1",
                   "random": "random 3 (mean ± std, 5 seeds)"}
    cat_offsets = {c: (i - (len(cat_order) - 1) / 2) * w
                    for i, c in enumerate(cat_order)}

    drawn_legend = set()
    for c in cat_order:
        vals = []
        errs = []
        for t in labelled:
            runs = summaries[t]["cats"][c]
            drs = [r["delta_r"] for r in runs if r.get("delta_r") is not None]
            if c == "random":
                if drs:
                    vals.append(float(np.mean(drs)))
                    errs.append(float(np.std(drs)))
                else:
                    vals.append(0); errs.append(0)
            else:
                vals.append(drs[0] if drs else 0)
                errs.append(0)
        bars = ax_dr.bar(x + cat_offsets[c], vals, w, color=cat_colors[c],
                          edgecolor="black", linewidth=0.5,
                          yerr=errs if c == "random" else None,
                          capsize=4 if c == "random" else 0,
                          label=cat_labels[c])
        for bar, v in zip(bars, vals):
            if abs(v) > 0.01:
                ax_dr.text(bar.get_x() + bar.get_width() / 2,
                            v + (-0.02 if v < 0 else 0.01),
                            f"{v:+.2f}", ha="center", va="top" if v < 0 else "bottom",
                            fontsize=14, color=cat_colors[c],
                            fontweight="bold")

    ax_dr.axhline(0, color="black", lw=0.6)
    ax_dr.set_xticks(x)
    ax_dr.set_xticklabels([TASK_DISPLAY.get(t, t) for t in labelled],
                           fontsize=18)
    ax_dr.tick_params(axis="y", labelsize=16)
    ax_dr.set_ylabel("Δcorr(LD, label)", fontsize=20)
    ax_dr.set_title("Cell-set ablation effect on z-correlation, by task",
                     fontsize=18)
    ax_dr.legend(fontsize=15, loc="lower right")
    ax_dr.grid(axis="y", alpha=0.3)
    # Y-limits with a touch of headroom
    all_vals = [v for c in cat_order
                for v in [
                    (summaries[t]["cats"][c][0].get("delta_r")
                     if c != "random" and summaries[t]["cats"][c]
                     else float(np.mean([r["delta_r"]
                                         for r in summaries[t]["cats"][c]
                                         if r.get("delta_r") is not None]))
                     if c == "random" and summaries[t]["cats"][c]
                     else 0)
                    for t in labelled]]
    y_min = min(all_vals + [0]) - 0.08
    ax_dr.set_ylim(y_min, 0.05)

    short_display = {"gemma2-2b": "Gemma2-2B", "gemma2-9b": "Gemma2-9B",
                       "gemma2-2b-it": "Gemma2-2B-it",
                       "gemma2-9b-it": "Gemma2-9B-it"}.get(args.short, args.short)
    fig_dr.suptitle(
        f"{short_display} — Specificity of the height-derived "
        f"DLA top-N cells across tasks",
        fontsize=20, y=1.00)
    fig_dr.tight_layout()
    out_dr = REPO / "figures" / f"p2u_specificity_summary_{args.short}_dr.png"
    fig_dr.savefig(out_dr, dpi=140, bbox_inches="tight")
    print(f"wrote {out_dr}")

    # ---- Separate figure: neutral text KL ----
    if kl_only:
        t = kl_only[0]
        cats = summaries[t]["cats"]
        fig_kl, ax_kl = plt.subplots(figsize=(7.0, 5.5))
        bar_x = np.arange(4)
        bar_w = 0.6
        # Same left-to-right order as the Δr panel: top-7, top-3, top-1, random-3.
        bar_vals = [
            cats["ext"][0]["kl_mean"] if cats["ext"] else 0,
            cats["trio"][0]["kl_mean"] if cats["trio"] else 0,
            cats["hub"][0]["kl_mean"] if cats["hub"] else 0,
            float(np.mean([r["kl_mean"] for r in cats["random"]])) if cats["random"] else 0,
        ]
        bar_errs = [0, 0, 0,
                    float(np.std([r["kl_mean"] for r in cats["random"]]))
                    if cats["random"] else 0]
        bar_colors_lst = ["tab:orange", "tab:red", "tab:blue", "0.55"]
        bar_labels_lst = ["top 7", "top 3", "top 1", "random 3"]
        bars = ax_kl.bar(bar_x, bar_vals, bar_w, color=bar_colors_lst,
                          edgecolor="black", linewidth=0.5,
                          yerr=bar_errs, capsize=4)
        for bar, v in zip(bars, bar_vals):
            ax_kl.text(bar.get_x() + bar.get_width() / 2,
                        v + max(bar_vals) * 0.03,
                        f"{v:.4f}", ha="center", va="bottom",
                        fontsize=15, fontweight="bold")
        ax_kl.set_xticks(bar_x)
        ax_kl.set_xticklabels(bar_labels_lst, fontsize=18)
        ax_kl.tick_params(axis="y", labelsize=15)
        ax_kl.set_ylabel("mean KL [nats / prompt]", fontsize=19)
        ax_kl.set_title(f"{short_display} — neutral text KL under cell-set ablation",
                         fontsize=18)
        ax_kl.grid(axis="y", alpha=0.3)
        ax_kl.set_ylim(0, max(bar_vals) * 1.4)
        fig_kl.tight_layout()
        out_kl = REPO / "figures" / f"p2u_specificity_summary_{args.short}_kl.png"
        fig_kl.savefig(out_kl, dpi=140, bbox_inches="tight")
        print(f"wrote {out_kl}")


if __name__ == "__main__":
    main()
