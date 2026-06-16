"""Vision phase grid plot — n_refs (rows) × models (cols) phase-trajectory grid.

Reads results/<out_name>/aggregate.json (or any directory of per-cell JSONs)
and produces a single PNG with one trajectory subplot per (n_ref, model) cell.

Each subplot:
  x = α (intervention strength)
  y = partial r(z|x) on test fold
  blue solid: manifold-shift
  orange dashed: proj_out
  baseline drawn at α=0
  annotation: baseline r_z, baseline <LD>, partial r at α=1 for both methods

Usage:
    python vphase_plot.py --in results/vphase_grid --out figures/vphase_grid.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def load_cells(in_dir: Path) -> list[dict]:
    agg_path = in_dir / "aggregate.json"
    if agg_path.exists():
        return json.loads(agg_path.read_text())["cells"]
    return [json.loads(p.read_text()) for p in sorted(in_dir.glob("*.json"))
            if p.name != "aggregate.json"]


def parse_settings(settings: dict) -> dict:
    """Returns {'baseline': {...}, 'manifold': [(α, metrics), ...],
                  'proj_out': [(α, metrics), ...]}."""
    out = {"baseline": settings.get("baseline"),
            "manifold": [], "proj_out": []}
    for k, v in settings.items():
        if k.startswith("manifold_a"):
            a = float(k.split("a")[-1])
            out["manifold"].append((a, v))
        elif k.startswith("proj_out_a"):
            a = float(k.split("a")[-1])
            out["proj_out"].append((a, v))
    out["manifold"].sort(key=lambda t: t[0])
    out["proj_out"].sort(key=lambda t: t[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default=str(REPO / "results" / "vphase_grid"))
    ap.add_argument("--out",
                    default=str(REPO / "figures" / "vphase_grid_partial_r_zx.png"))
    ap.add_argument("--metric", default="partial_LD_z_given_x",
                    choices=["partial_LD_z_given_x", "corr_LD_z", "ld_mean"])
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    cells = load_cells(in_dir)

    # group by (n_ref, model_short)
    by_key: dict[tuple[int, str], dict] = {}
    n_refs = sorted({c["n_ref"] for c in cells})
    shorts = []
    for c in cells:
        if c["model_short"] not in shorts:
            shorts.append(c["model_short"])
        by_key[(c["n_ref"], c["model_short"])] = c

    nrows = len(n_refs)
    ncols = len(shorts)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(4.0 * ncols, 3.2 * nrows),
                              sharey=True, squeeze=False)

    for r, n_ref in enumerate(n_refs):
        for c, short in enumerate(shorts):
            ax = axes[r, c]
            cell = by_key.get((n_ref, short))
            if cell is None or "settings" not in cell.get("result", {}):
                err = (cell or {}).get("result", {}).get("error", "missing")
                ax.text(0.5, 0.5, err, ha="center", va="center",
                         transform=ax.transAxes, color="red")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{short}  n_ref={n_ref}", fontsize=10)
                continue
            s = parse_settings(cell["result"]["settings"])
            base_metric = s["baseline"][args.metric]

            ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
            ax.scatter([0], [base_metric], marker="v", color="black",
                        zorder=5, label=f"baseline ({base_metric:+.2f})")
            if s["manifold"]:
                xs = [0] + [a for a, _ in s["manifold"]]
                ys = [base_metric] + [m[args.metric] for _, m in s["manifold"]]
                ax.plot(xs, ys, "-o", color="C0", lw=2, label="manifold")
            if s["proj_out"]:
                xs = [0] + [a for a, _ in s["proj_out"]]
                ys = [base_metric] + [m[args.metric] for _, m in s["proj_out"]]
                ax.plot(xs, ys, "--s", color="C1", lw=2, label="proj_out")

            base = s["baseline"]
            sub = (f"baseline r_z={base['corr_LD_z']:+.2f}  "
                    f"<LD>={base['ld_mean']:+.1f}\n"
                    f"L={cell['layer']}/{cell['n_layers']}  "
                    f"+{base['n_x_pos_slope']}/-{base['n_x_neg_slope']}")
            ax.set_title(f"{short}  n_ref={n_ref}\n{sub}", fontsize=9)
            ax.set_xlabel("α")
            if c == 0:
                ax.set_ylabel(args.metric)
            if args.ylim:
                ax.set_ylim(*args.ylim)
            ax.grid(True, alpha=0.3)
            if r == 0 and c == ncols - 1:
                ax.legend(fontsize=8, loc="best")

    fig.suptitle("Vision phase grid — partial r(LD, z | x) under α-sweep "
                 "(rows = n_ref, cols = model)", fontsize=12)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"wrote {out_path}")

    # also dump a table summary
    table_path = out_path.with_suffix(".csv")
    rows = []
    for c in cells:
        if "settings" not in c.get("result", {}):
            continue
        s = parse_settings(c["result"]["settings"])
        base = s["baseline"]
        row = {
            "model": c["model_short"], "n_ref": c["n_ref"],
            "layer": c["layer"], "n_layers": c["n_layers"],
            "base_r_z": base["corr_LD_z"],
            "base_r_x": base["corr_LD_x"],
            "base_pc_zx": base["partial_LD_z_given_x"],
            "base_ld_mean": base["ld_mean"],
            "base_pos_slopes": base["n_x_pos_slope"],
            "base_neg_slopes": base["n_x_neg_slope"],
        }
        for a, m in s["manifold"]:
            row[f"manifold_a{a}_pc"] = m["partial_LD_z_given_x"]
            row[f"manifold_a{a}_ld"] = m["ld_mean"]
        for a, m in s["proj_out"]:
            row[f"proj_out_a{a}_pc"] = m["partial_LD_z_given_x"]
            row[f"proj_out_a{a}_ld"] = m["ld_mean"]
        rows.append(row)

    if rows:
        keys = list(rows[0].keys())
        with table_path.open("w") as f:
            f.write(",".join(keys) + "\n")
            for row in rows:
                f.write(",".join(f"{row.get(k, ''):.4f}" if isinstance(row.get(k), float)
                                  else str(row.get(k, ""))
                                  for k in keys) + "\n")
        print(f"wrote {table_path}")


if __name__ == "__main__":
    main()
