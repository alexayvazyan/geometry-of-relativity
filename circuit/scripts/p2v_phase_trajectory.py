"""Phase 2V — phase-diagram trajectory under greedy ablation.

Plots the trajectory in PARTIAL-correlation phase space, matching the
convention of Jaehoon's geometry-of-relativity work (and Phase 2O
p2o_phase_plot.py). Axes:
  x = partial r(LD, x | z) = LD's x-encoding controlling for z
  y = partial r(LD, z | x) = LD's z-encoding controlling for x

Now overlays multiple ranking trajectories (e.g., σ_DLA·ρ+ vs |cos|+) on
the same phase space, each colored by N (heads ablated) along its line.

Partial correlations computed analytically from per-N (r_LDz, r_LDx) and
the fixed r(z, x) of the stimulus distribution:

  pc(LD, z | x) = (r_LDz - r_LDx · r_zx) / sqrt((1 - r_LDx²)(1 - r_zx²))
  pc(LD, x | z) = (r_LDx - r_LDz · r_zx) / sqrt((1 - r_LDz²)(1 - r_zx²))

No new GPU work; just re-computes from saved sweep JSON.

Output:
  figures/p2v_phase_trajectory_partial_<short>.png
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def partial_corr(r_LD_z: float, r_LD_x: float, r_zx: float) -> tuple[float, float]:
    """Return (pc(LD, x | z), pc(LD, z | x))."""
    den_zx = math.sqrt(max(1e-12, (1 - r_LD_x ** 2) * (1 - r_zx ** 2)))
    pc_z = (r_LD_z - r_LD_x * r_zx) / den_zx
    den_xz = math.sqrt(max(1e-12, (1 - r_LD_z ** 2) * (1 - r_zx ** 2)))
    pc_x = (r_LD_x - r_LD_z * r_zx) / den_xz
    return pc_x, pc_z


def load_trajectory(json_path: Path) -> dict:
    d = json.loads(json_path.read_text())
    base_r_z = d["baseline_r_LD_z"]
    base_r_x = d["baseline_r_LD_x"]
    Ns = [0] + [r["N"] for r in d["selected_runs"]]
    r_zs = [base_r_z] + [r["r"] for r in d["selected_runs"]]
    r_xs = [base_r_x] + [r["r_x"] for r in d["selected_runs"]]
    return {
        "feature": d["feature"],
        "k": d["k"],
        "cells": d["selected_cells"],
        "Ns": Ns,
        "r_zs": r_zs,
        "r_xs": r_xs,
    }


def load_manifold_sweep(json_path: Path, alpha_max: float = 1.0) -> dict:
    """Load manifold-shift α-sweep and return α-ordered (r_z, r_x) up to alpha_max.

    Only keys prefixed with `manifold_` are included — newer p2e JSONs may also
    contain `primal_a*` entries which are loaded by load_primal_sweep instead.
    """
    d = json.loads(json_path.read_text())
    runs = []
    for k, v in d["results"].items():
        if not k.startswith("manifold_"):
            continue
        a = v.get("alpha")
        if a is None or a > alpha_max + 1e-9:
            continue
        runs.append((float(a), float(v["r_ld_zeff"]), float(v["r_ld_x"])))
    runs.sort()
    return {
        "feature": d["pair"],
        "k": d["k"],
        "layer": d["layer"],
        "alphas": [r[0] for r in runs],
        "r_zs": [r[1] for r in runs],
        "r_xs": [r[2] for r in runs],
    }


def load_primal_sweep(json_path: Path, alpha_max: float = 1.0) -> dict | None:
    """Load primal (proj_out) α-sweep from a p2e_alpha_sweep_*.json file.

    Looks for keys 'primal_a0.25', 'primal_a0.50', etc. — emitted by the
    extended p2e_alpha_sweep.py that runs both manifold and primal modes.
    Returns None if no primal_* keys are present.
    """
    d = json.loads(json_path.read_text())
    runs = []
    for k, v in d["results"].items():
        if not k.startswith("primal_"):
            continue
        a = v.get("alpha")
        if a is None or a > alpha_max + 1e-9:
            continue
        runs.append((float(a), float(v["r_ld_zeff"]), float(v["r_ld_x"])))
    if not runs:
        return None
    runs.sort()
    return {
        "feature": d["pair"],
        "k": d["k"],
        "layer": d["layer"],
        "alphas": [r[0] for r in runs],
        "r_zs": [r[1] for r in runs],
        "r_xs": [r[2] for r in runs],
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--short", default="gemma2-2b")
    ap.add_argument("--suffixes", nargs="+",
                    default=["positive_dla_phase"],
                    help="suffixes (without .json) of n-sweep result files to overlay")
    ap.add_argument("--labels", nargs="+",
                    default=[r"$\sigma_{DLA}\cdot|\rho|$ ablation"],
                    help="display labels matching --suffixes (one per trajectory)")
    ap.add_argument("--cmaps", nargs="+",
                    default=["viridis"],
                    help="matplotlib colormaps per trajectory")
    ap.add_argument("--manifold-suffix", default=None,
                    help="if set, overlay 2E α-sweep from "
                         "results/p2e_alpha_sweep_<short>_<feature>_k<k>.json "
                         "(uses feature/k of first ablation trajectory)")
    ap.add_argument("--manifold-alpha-max", type=float, default=1.0,
                    help="upper bound on α to overlay")
    args = ap.parse_args()

    if len(args.labels) != len(args.suffixes) or len(args.cmaps) != len(args.suffixes):
        raise SystemExit("--labels and --cmaps must each match --suffixes in length")

    trajs = []
    for suf in args.suffixes:
        in_json = REPO / "results" / f"p2o_n_sweep_{args.short}_{suf}.json"
        if not in_json.exists():
            raise SystemExit(f"missing {in_json}")
        trajs.append(load_trajectory(in_json))

    # Stimulus r(z, x) fixed across runs (same feature/k expected)
    feature = trajs[0]["feature"]
    k = trajs[0]["k"]
    for t in trajs[1:]:
        if t["feature"] != feature or t["k"] != k:
            raise SystemExit("trajectories disagree on feature/k — cannot overlay")
    stim = REPO / "data" / "p2_shot_sweep" / f"{feature}_k{k}.jsonl"
    rows = [json.loads(l) for l in stim.open()]
    z_arr = np.array([float(r.get("z_eff", r.get("z", 0))) for r in rows],
                     dtype=np.float32)
    x_arr = np.array([float(r["x"]) for r in rows], dtype=np.float32)
    r_zx = float(np.corrcoef(z_arr, x_arr)[0, 1])
    print(f"r(z, x) over {len(rows)} {feature} k={k} prompts = {r_zx:+.3f}")

    # Compute partial-corr trajectories
    for t in trajs:
        pcs = [partial_corr(rz, rx, r_zx)
               for rz, rx in zip(t["r_zs"], t["r_xs"])]
        t["pc_xs"] = [p[0] for p in pcs]
        t["pc_zs"] = [p[1] for p in pcs]

    def _anchor_to_baseline(sweep: dict) -> None:
        """Replace the α=0 point with the ablation N=0 baseline (or prepend one)
        so the steering trajectory shares its origin with the ablation curve."""
        if abs(sweep["alphas"][0]) < 1e-9:
            sweep["r_zs"][0] = trajs[0]["r_zs"][0]
            sweep["r_xs"][0] = trajs[0]["r_xs"][0]
        else:
            sweep["alphas"] = [0.0] + sweep["alphas"]
            sweep["r_zs"] = [trajs[0]["r_zs"][0]] + sweep["r_zs"]
            sweep["r_xs"] = [trajs[0]["r_xs"][0]] + sweep["r_xs"]
        pcs = [partial_corr(rz, rx, r_zx)
                for rz, rx in zip(sweep["r_zs"], sweep["r_xs"])]
        sweep["pc_xs"] = [p[0] for p in pcs]
        sweep["pc_zs"] = [p[1] for p in pcs]

    def _print_steering_trajectory(name: str, sweep: dict) -> None:
        print(f"\n=== {name} trajectory (L{sweep.get('layer')}) ===")
        print(f"{'α':>5s}  {'r_z':>7s}  {'r_x':>7s}  "
              f"{'pc(x|z)':>8s}  {'pc(z|x)':>8s}")
        for i, a in enumerate(sweep["alphas"]):
            print(f"{a:>5.2f}  {sweep['r_zs'][i]:+7.3f}  "
                  f"{sweep['r_xs'][i]:+7.3f}  "
                  f"{sweep['pc_xs'][i]:+8.3f}  "
                  f"{sweep['pc_zs'][i]:+8.3f}")

    # Optional manifold (dual) steering overlay (Phase 2E)
    manifold = None
    m_path = REPO / "results" / f"p2e_alpha_sweep_{args.short}_{feature}_k{k}.json"
    if m_path.exists():
        manifold = load_manifold_sweep(m_path, args.manifold_alpha_max)
        _anchor_to_baseline(manifold)
        _print_steering_trajectory("dual (manifold) steering", manifold)
    else:
        print(f"manifold sweep not found at {m_path}; skipping overlay")

    # Optional primal (proj_out) steering overlay — read from same p2e file
    # if it includes primal_a* keys (extended p2e_alpha_sweep.py).
    primal = None
    if m_path.exists():
        primal = load_primal_sweep(m_path, args.manifold_alpha_max)
        if primal is not None:
            _anchor_to_baseline(primal)
            _print_steering_trajectory("primal (proj_out) steering", primal)
        else:
            print(f"no primal_a* keys in {m_path.name}; skipping primal overlay")

    # Print tables
    for label, t in zip(args.labels, trajs):
        print(f"\n=== trajectory: {label} ===")
        print(f"{'N':>3s}  {'r_z':>7s}  {'r_x':>7s}  "
              f"{'pc(x|z)':>8s}  {'pc(z|x)':>8s}  cell added")
        for i, N in enumerate(t["Ns"]):
            if N == 0:
                cell_label = "(baseline)"
            else:
                c = t["cells"][N - 1]
                cell_label = f"L{c['layer']}H{c['head']}"
            print(f"{N:>3d}  {t['r_zs'][i]:+7.3f}  {t['r_xs'][i]:+7.3f}  "
                  f"{t['pc_xs'][i]:+8.3f}  {t['pc_zs'][i]:+8.3f}  {cell_label}")

    fig, ax = plt.subplots(figsize=(13, 10.5))
    ax.tick_params(axis="both", which="major", labelsize=18)

    xlim = (-0.30, 1.05)
    ylim = (-0.30, 1.05)

    # Region labels (corner anchors of the phase space)
    ax.text(xlim[0] + 0.04, 0.97, "RELATIVISTIC",
            color="C0", fontsize=22, fontweight="bold", va="top")
    ax.text(xlim[1] - 0.04, ylim[0] + 0.04, "OBJECTIVE",
            color="C2", fontsize=22, fontweight="bold", ha="right")

    # Plot each trajectory with its own colormap
    cbar_handles = []
    for label, cmap_name, t in zip(args.labels, args.cmaps, trajs):
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=0, vmax=max(t["Ns"]))
        pts = np.array(list(zip(t["pc_xs"], t["pc_zs"])))
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        seg_colors = np.array(t["Ns"][:-1])
        lc = LineCollection(segs, cmap=cmap, norm=norm, lw=3.0, zorder=4)
        lc.set_array(seg_colors)
        ax.add_collection(lc)
        sc = ax.scatter(t["pc_xs"], t["pc_zs"], c=t["Ns"], cmap=cmap, norm=norm,
                        s=55, edgecolor="black", zorder=10, linewidth=0.5,
                        label=label)
        cbar_handles.append((label, sc))

    # Annotate notable Ns on the first trajectory (DLA)
    primary = trajs[0]
    annotate_Ns = {0: "N=0",
                   1: f"N=1 (+L{primary['cells'][0]['layer']}H{primary['cells'][0]['head']})",
                   4: "N=4", 8: "N=8", 16: "N=16", 32: "N=32"}
    for i, N in enumerate(primary["Ns"]):
        if N in annotate_Ns:
            ax.annotate(annotate_Ns[N],
                        (primary["pc_xs"][i], primary["pc_zs"][i]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=17, color="black",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", edgecolor="gray",
                                  alpha=0.85))

    # Endpoint label on second trajectory if it diverges
    if len(trajs) > 1:
        t = trajs[1]
        ax.annotate(f"{args.labels[1]}  N={t['Ns'][-1]}",
                    (t["pc_xs"][-1], t["pc_zs"][-1]),
                    xytext=(8, -16), textcoords="offset points",
                    fontsize=17, color="black",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="gray",
                              alpha=0.85))

    # Steering overlays (dashed/dotted, distinguished by colour and linestyle)
    label_offsets = {0.25: (10, -2), 0.5: (10, -2),
                     0.75: (10, -2), 1.0: (10, -2),
                     1.25: (10, -2), 1.5: (10, -2)}

    def _draw_steering(sweep: dict, color: str, linestyle: str,
                       legend_label: str) -> None:
        ax.plot(sweep["pc_xs"], sweep["pc_zs"],
                color=color, linestyle=linestyle, lw=2.2, zorder=5,
                label=legend_label)
        ax.scatter(sweep["pc_xs"], sweep["pc_zs"],
                   facecolor="white", edgecolor=color, s=60,
                   linewidth=1.6, zorder=11)
        for a, px, pz in zip(sweep["alphas"], sweep["pc_xs"],
                              sweep["pc_zs"]):
            if a in label_offsets:
                dx, dy = label_offsets[a]
                ax.annotate(rf"$\alpha$={a:g}", (px, pz),
                            xytext=(dx, dy), textcoords="offset points",
                            fontsize=17, color=color,
                            ha="left", va="center", fontweight="bold")

    if manifold is not None:
        _draw_steering(manifold, "tab:red", "--", "dual steering")
    if primal is not None:
        _draw_steering(primal, "tab:purple", ":", "primal steering")
    if manifold is not None or primal is not None:
        ax.legend(fontsize=18, loc="upper right")

    ax.axhline(0, color="black", lw=0.4, alpha=0.3)
    ax.axvline(0, color="black", lw=0.4, alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)

    # Two narrow colorbars stacked vertically on the right
    for j, (label, sc) in enumerate(cbar_handles):
        cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02 + 0.04 * j)
        cbar.set_label(f"N heads ablated  ({label})", fontsize=17)
        cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel(r"$pc$(LD, x | z)  →  objective", fontsize=20)
    ax.set_ylabel(r"$pc$(LD, z | x)  →  relativistic", fontsize=20)
    ax.set_title(f"{args.short} — phase trajectories  "
                 f"({feature} k={k}, corr(z,x)={r_zx:+.2f})",
                 fontsize=18)

    fig.tight_layout()
    out_png = REPO / "figures" / f"p2v_phase_trajectory_partial_{args.short}.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
