#!/usr/bin/env python3
"""Regenerate the section-5 circuit figures on CPU from circuit/results/ (no model needed).

Each job runs a plotter in circuit/scripts/ (it reads circuit/results/, writes circuit/figures/),
then the produced PNGs are copied into paper/figures/ under the names the paper references.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "circuit" / "scripts"
CFIG = ROOT / "circuit" / "figures"
PAPER_FIG = ROOT / "paper" / "figures"
PY = sys.executable

# (script, args, [output basenames produced in circuit/figures to copy into paper/figures])
JOBS: list[tuple[str, list[str], list[str]]] = [
    ("p2u_n_sweep_xfeat_replot.py", ["--short", "gemma2-9b"],
        ["p2u_n_sweep_xfeat_gemma2-9b_dcorr.png", "p2u_n_sweep_xfeat_gemma2-9b_kl.png"]),
    ("p2u_specificity_summary.py",
        ["--short", "gemma2-9b", "--tasks", "relativity-weight", "arithmetic", "truth-base", "neutral_text"],
        ["p2u_specificity_summary_gemma2-9b_dr.png"]),
    ("p2o_attention_modes_comparison.py", ["--short", "gemma2-9b"],
        ["p2o_attention_modes_comparison_gemma2-9b.png"]),
    ("p2s_dla_vs_cossigma_plot_9b.py", [],
        ["p2s_dla_vs_cossigma_sweep_gemma2-9b.png"]),
    ("p2s_dla_xfeat_summary.py", ["--short", "gemma2-9b"],
        ["p2s_dla_xfeat_summary_gemma2-9b.png"]),
    ("p2v_xk_corr_grid.py", ["--short", "gemma2-9b", "--feature", "height", "--ks", "1", "5", "15"],
        ["p2v_xk_corr_grid_gemma2-9b_height.png"]),
    ("p2v_xk_delta_r_overlay.py", ["--short", "gemma2-9b", "--feature", "height", "--ks", "1", "5", "15"],
        ["p2v_xk_deltar_overlay_gemma2-9b_height.png"]),
    ("p2v_steering_with_arclength.py", ["--short", "gemma2-9b", "--feature", "height", "--k", "15"],
        ["p2v_steering_with_arclength_gemma2-9b_height.png"]),
    ("plot_p2d_phase_k_sweep.py",
        ["--models", "gemma2-9b", "--ks", "0", "1", "2", "5", "15", "--pair", "height",
         "--out", str(CFIG / "p2d_phase_k_sweep_9b.png")],
        ["p2d_phase_k_sweep_9b.png"]),
    ("p2v_xmodel_appendix_composites.py", [],
        ["xmodel_geometry_grid.png", "xmodel_interventions_grid.png"]),
]


def main() -> None:
    CFIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    copied, problems = [], []
    for script, args, outs in JOBS:
        print(f">> {script} {' '.join(args)}")
        r = subprocess.run([PY, str(SCRIPTS / script), *args], capture_output=True, text=True)
        if r.returncode != 0:
            print((r.stdout or "")[-400:])
            print((r.stderr or "")[-900:])
            problems.append(f"{script} (exit {r.returncode})")
            continue
        for o in outs:
            src = CFIG / o
            if src.exists():
                shutil.copy2(src, PAPER_FIG / o)
                copied.append(o)
            else:
                problems.append(f"{o} (not produced)")
    print(f"\ncopied {len(copied)} circuit figures into paper/figures/")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        sys.exit(1)


if __name__ == "__main__":
    main()
