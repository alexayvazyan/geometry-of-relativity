"""Plot cross-feature steering trajectories.

Reads results/p2f_xfeat_steering_<model>_ref<ref>_eval<eval>_k<k>.json and
plots three trajectories in raw-corr phase space:
  - meandiff_xfeat (REF-derived d_L applied to EVAL prompts) — purple dotted
  - manifold_zonly_xfeat (REF-derived z-only M applied to EVAL prompts) — red dashed
  - meandiff_native (EVAL-derived d_L; positive control if present) — green dashed

Output:
  figures/p2v_xfeat_steering_<model>_ref<ref>_eval<eval>.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_runs(d: dict, prefix: str, alpha_max: float) -> dict:
    runs = []
    for k, v in d["results"].items():
        if not k.startswith(prefix):
            continue
        a = v.get("alpha")
        if a is None or a > alpha_max + 1e-9:
            continue
        runs.append((float(a), float(v["r_ld_zeff"]), float(v["r_ld_x"])))
    if not runs:
        return None
    runs.sort()
    return {
        "alphas": [r[0] for r in runs],
        "r_zs": [r[1] for r in runs],
        "r_xs": [r[2] for r in runs],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma2-9b")
    ap.add_argument("--reference-pair", default="height")
    ap.add_argument("--eval-pair", default="weight")
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--alpha-max", type=float, default=1.0)
    ap.add_argument("--figsize", nargs=2, type=float, default=(9.0, 7.5))
    args = ap.parse_args()

    in_path = (REPO / "results"
               / f"p2f_xfeat_steering_{args.model}_ref{args.reference_pair}"
                 f"_eval{args.eval_pair}_k{args.k}.json")
    if not in_path.exists():
        raise SystemExit(f"missing {in_path}")
    d = json.loads(in_path.read_text())

    meandiff_x = load_runs(d, "meandiff_xfeat_", args.alpha_max)
    manifold_x = load_runs(d, "manifold_zonly_xfeat_", args.alpha_max)
    meandiff_n = load_runs(d, "meandiff_native_", args.alpha_max)

    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    xlim = (-0.20, 1.05)
    ylim = (-0.20, 1.05)

    rel_x = xlim[0] + 0.04
    ax.annotate("", xy=(rel_x, ylim[1] - 0.01),
                xytext=(rel_x, ylim[1] - 0.18),
                arrowprops=dict(arrowstyle="->", color="C0", lw=2.0))
    ax.text(rel_x, ylim[1] - 0.22, "RELATIVISTIC",
             color="C0", fontsize=18, fontweight="bold",
             ha="center", va="top", rotation=90)
    obj_y = ylim[0] + 0.04
    ax.annotate("", xy=(xlim[1] - 0.01, obj_y),
                xytext=(xlim[1] - 0.18, obj_y),
                arrowprops=dict(arrowstyle="->", color="C2", lw=2.0))
    ax.text(xlim[1] - 0.22, obj_y, "OBJECTIVE",
             color="C2", fontsize=18, fontweight="bold",
             ha="right", va="center")

    label_offsets = {0.5: (10, 0), 0.75: (10, 0), 1.0: (10, 0)}

    def _draw(sweep, color, ls, label, offsets):
        if sweep is None:
            return
        ax.plot(sweep["r_xs"], sweep["r_zs"], color=color, linestyle=ls,
                lw=2.6, zorder=5, label=label)
        ax.scatter(sweep["r_xs"], sweep["r_zs"], facecolor="white",
                    edgecolor=color, s=70, linewidth=1.8, zorder=11)
        for a, rx, rz in zip(sweep["alphas"], sweep["r_xs"], sweep["r_zs"]):
            if a in offsets:
                dx, dy = offsets[a]
                ax.annotate(rf"$\alpha$={a:g}", (rx, rz),
                            xytext=(dx, dy), textcoords="offset points",
                            fontsize=14, color=color,
                            ha=("right" if dx < 0 else "left"),
                            va="center", fontweight="bold")

    _draw(manifold_x, "tab:red", "--",
           f"manifold M (z-only), {args.reference_pair}→{args.eval_pair}",
           {0.5: (10, 4), 0.75: (10, 0), 1.0: (10, -4)})
    _draw(meandiff_x, "tab:purple", ":",
           f"mean-diff $\\hat d_L$, {args.reference_pair}→{args.eval_pair}",
           {0.5: (-12, -8), 0.75: (-12, -8), 1.0: (-12, -8)})
    _draw(meandiff_n, "tab:green", "-.",
           f"mean-diff $\\hat d_L$, {args.eval_pair}-native (control)",
           {0.5: (10, 4), 0.75: (10, 4), 1.0: (10, 0)})

    base = manifold_x or meandiff_x or meandiff_n
    if base is not None and len(base["alphas"]) > 0:
        bx, by = base["r_xs"][0], base["r_zs"][0]
        ax.scatter([bx], [by], facecolor="black", edgecolor="black",
                    s=85, zorder=12)
        ax.annotate("baseline", (bx, by),
                    xytext=(10, 10), textcoords="offset points",
                    fontsize=15, color="black", fontweight="bold",
                    ha="left", va="bottom")

    ax.axhline(0, color="black", lw=0.4, alpha=0.3)
    ax.axvline(0, color="black", lw=0.4, alpha=0.3)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)
    ax.tick_params(axis="both", which="major", labelsize=17)
    ax.set_xlabel(r"corr(LD, $x$)", fontsize=19)
    ax.set_ylabel(r"corr(LD, $z$)", fontsize=19)
    short_display = {"gemma2-2b": "Gemma2-2B", "gemma2-9b": "Gemma2-9B"}.get(
        args.model, args.model)
    ax.set_title(
        f"{short_display} — {args.reference_pair}→{args.eval_pair} steering transfer "
        f"(k={args.k})",
        fontsize=17)
    ax.legend(fontsize=13, loc="lower right", framealpha=0.95)

    fig.tight_layout()
    out = (REPO / "figures"
           / f"p2v_xfeat_steering_{args.model}_ref{args.reference_pair}"
             f"_eval{args.eval_pair}.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
