"""Vision phase diagram — n_ref × model grid in (r(LD,x), r(LD,z)) plane.

Mirrors plot_p2d_phase_grid.py style:
  - Each cell is a phase plane with shaded regions
    (RELATIVISTIC top-left, OBJECTIVE bottom-right, BIASED near origin,
    plus ANTI-RELATIVISTIC bottom-left for vision where the readout flips)
  - Baseline (green ●), manifold α∈{0.5,0.75,1.0} (blue ◆ progressively darker),
    proj_out α∈{...} (orange ◆ progressively darker)
  - Arrows from baseline → each intervention endpoint
  - ⟨LD⟩ annotated next to baseline

Reads results/<in_name>/aggregate.json (default vphase_grid).
Writes figures/vphase_diagram.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_cells(in_dir: Path) -> list[dict]:
    agg = in_dir / "aggregate.json"
    if agg.exists():
        return json.loads(agg.read_text())["cells"]
    return [json.loads(p.read_text()) for p in sorted(in_dir.glob("*.json"))
            if p.name != "aggregate.json"]


def parse_settings(settings: dict) -> dict:
    out: dict = {"baseline": settings.get("baseline"),
                  "manifold": [], "proj_out": []}
    for k, v in settings.items():
        if k.startswith("manifold_a"):
            out["manifold"].append((float(k.split("a")[-1]), v))
        elif k.startswith("proj_out_a"):
            out["proj_out"].append((float(k.split("a")[-1]), v))
    out["manifold"].sort(key=lambda t: t[0])
    out["proj_out"].sort(key=lambda t: t[0])
    return out


def setup_phase_axes(ax, xlim, ylim) -> None:
    # Region shading — RELATIVISTIC (top-left), OBJECTIVE (bottom-right),
    # BIASED (near origin / upper-right), ANTI-RELATIVISTIC (bottom-left).
    xmin, xmax = xlim
    ymin, ymax = ylim
    # Light blue for RELATIVISTIC (r_z > 0, r_x ≤ 0)
    ax.add_patch(Rectangle((xmin, 0), 0 - xmin, ymax - 0,
                            alpha=0.07, color="C0", zorder=0))
    # Light green for OBJECTIVE (r_z ≤ 0, r_x > 0)
    ax.add_patch(Rectangle((0, ymin), xmax - 0, 0 - ymin,
                            alpha=0.07, color="C2", zorder=0))
    # Light red for ANTI-RELATIVISTIC (r_z < 0, r_x < 0)
    ax.add_patch(Rectangle((xmin, ymin), 0 - xmin, 0 - ymin,
                            alpha=0.08, color="C3", zorder=0))
    ax.text(xmin + 0.02, ymax - 0.02, "RELATIVISTIC", color="C0",
            fontsize=8.5, fontweight="bold", va="top")
    ax.text(xmax - 0.02, ymin + 0.02, "OBJECTIVE", color="C2",
            fontsize=8.5, fontweight="bold", ha="right")
    ax.text(xmin + 0.02, ymin + 0.02, "ANTI-RELATIVISTIC",
            color="C3", fontsize=8.5, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.axvline(0, color="black", linewidth=0.4, alpha=0.4)
    ax.grid(alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default=str(REPO / "results" / "vphase_grid"))
    ap.add_argument("--out",
                    default=str(REPO / "figures" / "vphase_diagram.png"))
    ap.add_argument("--xlim", type=float, nargs=2, default=None,
                    help="r(LD,x) axis range; auto if omitted")
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    help="r(LD,z) axis range; auto if omitted")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    cells = load_cells(in_dir)

    by_key: dict[tuple[int, str], dict] = {}
    n_refs: list[int] = []
    shorts: list[str] = []
    for c in cells:
        if c["n_ref"] not in n_refs:
            n_refs.append(c["n_ref"])
        if c["model_short"] not in shorts:
            shorts.append(c["model_short"])
        by_key[(c["n_ref"], c["model_short"])] = c
    n_refs.sort()

    # Auto axis ranges from all r values across cells
    if args.xlim is None or args.ylim is None:
        rxs, rzs = [], []
        for c in cells:
            settings = c.get("result", {}).get("settings", {})
            for v in settings.values():
                rxs.append(v["corr_LD_x"])
                rzs.append(v["corr_LD_z"])
        if rxs and rzs:
            xpad = 0.08
            ypad = 0.08
            xlim_auto = (min(rxs + [0]) - xpad, max(rxs + [0]) + xpad)
            ylim_auto = (min(rzs + [0]) - ypad, max(rzs + [0]) + ypad)
        else:
            xlim_auto = (-1, 1)
            ylim_auto = (-1, 1)
        xlim = args.xlim if args.xlim else xlim_auto
        ylim = args.ylim if args.ylim else ylim_auto
    else:
        xlim, ylim = tuple(args.xlim), tuple(args.ylim)

    nrows = len(n_refs)
    ncols = len(shorts)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.5 * ncols, 4.5 * nrows),
                              squeeze=False)

    # Manifold blues / proj_out oranges for α progression
    manifold_colors = ["#9ec5e8", "#4a8fc7", "#1f4e79"]   # light → dark blue
    proj_colors = ["#fbc28a", "#ed8936", "#a85b15"]      # light → dark orange

    for r, n_ref in enumerate(n_refs):
        for c, short in enumerate(shorts):
            ax = axes[r, c]
            setup_phase_axes(ax, xlim, ylim)
            cell = by_key.get((n_ref, short))
            if cell is None or "settings" not in cell.get("result", {}):
                err = (cell or {}).get("result", {}).get("error", "missing")
                ax.text(0.5, 0.5, err, ha="center", va="center",
                        transform=ax.transAxes, color="red", fontsize=11)
                ax.set_title(f"{short}  |  n_ref={n_ref}", fontsize=11)
                continue

            s = parse_settings(cell["result"]["settings"])
            base = s["baseline"]
            bx = base["corr_LD_x"]
            by_ = base["corr_LD_z"]
            ax.scatter(bx, by_, s=240, color="tab:green", marker="o",
                        edgecolor="black", linewidth=1.0, zorder=5,
                        label="baseline")
            ax.annotate(f"baseline\n⟨LD⟩={base['ld_mean']:+.2f}\n"
                         f"pc(z|x)={base['partial_LD_z_given_x']:+.2f}",
                         (bx, by_), xytext=(8, 6), textcoords="offset points",
                         fontsize=7.2, color="tab:green", fontweight="bold")

            for ai, (a, m) in enumerate(s["manifold"]):
                col = manifold_colors[min(ai, len(manifold_colors) - 1)]
                rx, rz = m["corr_LD_x"], m["corr_LD_z"]
                ax.scatter(rx, rz, s=170, color=col, marker="D",
                            edgecolor="black", linewidth=0.7, zorder=4)
                ax.annotate("", xy=(rx, rz), xytext=(bx, by_),
                             arrowprops=dict(arrowstyle="->", color=col,
                                              alpha=0.55, lw=1.0),
                             zorder=3)
                if ai == len(s["manifold"]) - 1:
                    ax.annotate(
                        f"manifold α={a}\n⟨LD⟩={m['ld_mean']:+.2f}\n"
                        f"pc={m['partial_LD_z_given_x']:+.2f}",
                        (rx, rz), xytext=(8, 4),
                        textcoords="offset points",
                        fontsize=7.0, color=col, fontweight="bold")

            for ai, (a, m) in enumerate(s["proj_out"]):
                col = proj_colors[min(ai, len(proj_colors) - 1)]
                rx, rz = m["corr_LD_x"], m["corr_LD_z"]
                ax.scatter(rx, rz, s=170, color=col, marker="s",
                            edgecolor="black", linewidth=0.7, zorder=4)
                ax.annotate("", xy=(rx, rz), xytext=(bx, by_),
                             arrowprops=dict(arrowstyle="->", color=col,
                                              alpha=0.55, lw=1.0,
                                              linestyle=(0, (3, 2))),
                             zorder=3)
                if ai == len(s["proj_out"]) - 1:
                    ax.annotate(
                        f"proj_out α={a}\n⟨LD⟩={m['ld_mean']:+.2f}\n"
                        f"pc={m['partial_LD_z_given_x']:+.2f}",
                        (rx, rz), xytext=(8, -32),
                        textcoords="offset points",
                        fontsize=7.0, color=col, fontweight="bold")

            ax.set_title(
                f"{short.replace('gemma4-', '').upper()}  |  n_ref={n_ref}  "
                f"(L{cell['layer']}/{cell['n_layers']})",
                fontsize=10)
            if r == nrows - 1:
                ax.set_xlabel(r"$r$(LD, x)  →  objective")
            if c == 0:
                ax.set_ylabel(r"$r$(LD, z)  →  relativistic")

    fig.suptitle(
        "Vision phase diagram — shot-count × model migration in "
        "(r(LD,x), r(LD,z)) plane\n"
        "(green ● = baseline; blue ◆ = manifold α∈{0.5,0.75,1.0}; "
        "orange ■ = proj_out α∈{0.5,0.75,1.0})",
        y=1.0, fontsize=12)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
