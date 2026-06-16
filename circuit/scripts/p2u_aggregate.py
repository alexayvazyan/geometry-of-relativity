"""Phase 2U aggregate — cross-task specificity figure.

Reads results/p2u_specificity_*_<task>.json for each task and produces a
side-by-side comparison of the trio's effect across tasks.

Top row: Δr (relative drop in r(LD, label)) per task × cell-set.
Bottom row: total KL(baseline || resample) per task × cell-set.

Random controls are aggregated as mean ± std across N_RANDOM_CONTROLS seeds.

Output: figures/p2u_specificity_summary_<short>.png + summary CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent

DEFAULT_TASKS = ["relativity-it", "arithmetic", "truth", "refusal", "neutral_text"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b-it")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    args = ap.parse_args()

    # Layout: trio | 7-cell | hub | random-mean±std
    GROUP_LABELS = ["trio", "7-cell", "hub (L16H4)", "random-3 (5 seeds)"]
    GROUP_COLORS = ["C3", "C1", "C0", "C7"]

    tasks_data = []
    for task in args.tasks:
        path = REPO / "results" / f"p2u_specificity_{args.short}_{task}.json"
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        d = json.loads(path.read_text())
        baseline_r = d["baseline"]["r_LD_label"]
        has_label = baseline_r is not None
        runs = d["runs"]
        # Group runs by cell-set kind via label prefix
        trio = next(r for r in runs if r["label"].startswith("trio"))
        ext = next(r for r in runs if r["label"].startswith("7-cell"))
        hub = next(r for r in runs if r["label"].startswith("L16H4 hub"))
        rand = [r for r in runs if r["label"].startswith("random-3")]

        rand_kl = np.array([r["kl_total"] for r in rand])
        if has_label:
            rand_dr = np.array([r["delta_r"] for r in rand])
            delta_r = [trio["delta_r"], ext["delta_r"], hub["delta_r"],
                        float(rand_dr.mean())]
            delta_r_err = [0, 0, 0, float(rand_dr.std(ddof=1))]
            rand_dr_max = float(rand_dr.max(initial=0))
            rand_dr_min = float(rand_dr.min(initial=0))
        else:
            delta_r = [None, None, None, None]
            delta_r_err = [0, 0, 0, 0]
            rand_dr_max = rand_dr_min = float("nan")

        tasks_data.append({
            "task": task,
            "n": d["n_prompts"],
            "baseline_r": baseline_r,
            "has_label": has_label,
            "delta_r": delta_r,
            "delta_r_err": delta_r_err,
            "kl_total": [trio["kl_total"], ext["kl_total"], hub["kl_total"],
                          float(rand_kl.mean())],
            "kl_total_err": [0, 0, 0, float(rand_kl.std(ddof=1))],
            "rand_dr_max": rand_dr_max,
            "rand_dr_min": rand_dr_min,
            "rand_kl_max": float(rand_kl.max(initial=0)),
        })

    # Top panel uses only labelled tasks; bottom panel uses all
    label_tasks = [td for td in tasks_data if td["has_label"]]
    n_tasks_top = len(label_tasks)
    n_tasks_bot = len(tasks_data)
    n_groups = len(GROUP_LABELS)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    width = 0.18
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2) * width

    # Top: Δr (label-tasks only)
    ax = axes[0]
    x_top = np.arange(n_tasks_top)
    for gi, (glabel, gcolor) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        ys = [td["delta_r"][gi] for td in label_tasks]
        errs = [td["delta_r_err"][gi] for td in label_tasks]
        bars = ax.bar(x_top + offsets[gi], ys, width, color=gcolor,
                       edgecolor="black", label=glabel,
                       yerr=errs, capsize=3)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width() / 2,
                     y - 0.02 * (1 if y < 0 else -1),
                     f"{y:+.2f}", ha="center",
                     va="top" if y < 0 else "bottom", fontsize=7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Δr(LD, label)  =  r_resample − r_baseline")
    ax.set_title(f"Specificity battery on {args.short}: trio drops r on relativity, "
                  f"barely on others")
    ax.set_xticks(x_top)
    ax.set_xticklabels([f"{td['task']}\n(baseline r={td['baseline_r']:+.2f}, n={td['n']})"
                         for td in label_tasks], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # Bottom: total KL (all tasks including neutral_text)
    ax = axes[1]
    x_bot = np.arange(n_tasks_bot)
    for gi, (glabel, gcolor) in enumerate(zip(GROUP_LABELS, GROUP_COLORS)):
        ys = [td["kl_total"][gi] for td in tasks_data]
        errs = [td["kl_total_err"][gi] for td in tasks_data]
        bars = ax.bar(x_bot + offsets[gi], ys, width, color=gcolor,
                       edgecolor="black", label=glabel,
                       yerr=errs, capsize=3)
    ax.set_ylabel("Σ KL(baseline || resample)  [nats]")
    ax.set_title("Total output-distribution disruption across the prompt set "
                  "(neutral_text = no task structure, no chat template)")
    ax.set_xticks(x_bot)
    ax.set_xticklabels([f"{td['task']}\n(n={td['n']})"
                         for td in tasks_data], fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Highlight the neutral_text column with a subtle background
    for i, td in enumerate(tasks_data):
        if td["task"] == "neutral_text":
            ax.axvspan(i - 0.5, i + 0.5, color="orange", alpha=0.08, zorder=0)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2u_specificity_summary_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"wrote {out_png}")

    # CSV summary
    out_csv = REPO / "results" / f"p2u_specificity_summary_{args.short}.csv"
    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["task", "n", "baseline_r",
                     "dr_trio", "dr_7cell", "dr_hub",
                     "dr_random_mean", "dr_random_std",
                     "dr_random_max_abs",
                     "kl_trio", "kl_7cell", "kl_hub",
                     "kl_random_mean", "kl_random_std",
                     "trio_specificity_ratio (|Δr_trio| / max(|Δr_random|, eps))"])
        for td in tasks_data:
            if td["has_label"]:
                spec_ratio = abs(td["delta_r"][0]) / max(
                    abs(td["rand_dr_max"]), abs(td["rand_dr_min"]), 1e-6)
                rand_max_str = (
                    f"{max(abs(td['rand_dr_max']), abs(td['rand_dr_min'])):+.4f}")
                w.writerow([td["task"], td["n"], f"{td['baseline_r']:+.4f}",
                             f"{td['delta_r'][0]:+.4f}",
                             f"{td['delta_r'][1]:+.4f}",
                             f"{td['delta_r'][2]:+.4f}",
                             f"{td['delta_r'][3]:+.4f}",
                             f"{td['delta_r_err'][3]:+.4f}",
                             rand_max_str,
                             f"{td['kl_total'][0]:.3f}",
                             f"{td['kl_total'][1]:.3f}",
                             f"{td['kl_total'][2]:.3f}",
                             f"{td['kl_total'][3]:.3f}",
                             f"{td['kl_total_err'][3]:.3f}",
                             f"{spec_ratio:.2f}"])
            else:
                w.writerow([td["task"], td["n"], "N/A",
                             "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
                             f"{td['kl_total'][0]:.3f}",
                             f"{td['kl_total'][1]:.3f}",
                             f"{td['kl_total'][2]:.3f}",
                             f"{td['kl_total'][3]:.3f}",
                             f"{td['kl_total_err'][3]:.3f}",
                             "N/A"])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
