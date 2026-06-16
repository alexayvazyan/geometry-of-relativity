"""Phase 3 stimulus generator — relative-context size on a single canvas.

Composes:
  - N_REF reference squares, sizes drawn from a discrete normal around (μ, σ),
    placed in a grid in the upper region of the canvas. All grey.
  - 1 target square of side x, placed bottom-center, drawn black.
  - Light visual separator between the reference grid and the target region
    (e.g., a horizontal grey rule).

Each generated image carries a JSON sidecar with the (x, μ, σ, z, ref_sizes,
seed) tuple, in line with the v11 trial JSONL convention.

Z-score:  z = (x − μ) / σ.
Mirrors v11 dense grid hyperparameters: 20 x-values × 20 z-values × small N
seeds, with implausible μ values dropped.

Usage:
    python p3_render_relative_size.py --out-dir ../stimuli/relsize_v0 \\
                                       --n-x 12 --n-z 12 --n-seeds 3
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---- defaults: chosen so the v11 z-grid is roughly symmetric in pixel space ----

CANVAS = 448             # square canvas, matches gemma4-speed-probe e1b
TARGET_REGION_FRAC = 0.40  # bottom 40% holds the target
REF_REGION_FRAC = 0.55     # top 55% holds the reference grid
SEPARATOR_BAND_FRAC = 0.05  # 5% middle is a pale separator

DEFAULT_X_MIN = 24
DEFAULT_X_MAX = 120
DEFAULT_SIGMA = 16   # fixed σ — same convention as text height (σ=10 cm)
DEFAULT_N_REF = 12   # reference squares per stimulus

# Z-grid mirrors v11
Z_MIN = -3.0
Z_MAX = +3.0


@dataclass(frozen=True)
class StimSpec:
    x: int           # target square side, pixels
    mu: float        # reference-mean square side
    sigma: float     # reference σ
    z: float         # (x - mu) / sigma
    seed: int


def is_plausible(spec: StimSpec, x_min: int, x_max: int) -> bool:
    """Reject specs whose μ would put reference squares outside reasonable size."""
    # Reference squares are sampled from N(μ, σ); we don't want sub-1px or
    # squares larger than the per-ref-cell allotment.
    if spec.mu - 2 * spec.sigma < 4:
        return False
    if spec.mu + 2 * spec.sigma > x_max + spec.sigma:
        return False
    return True


def sample_ref_sizes(spec: StimSpec, n_ref: int, rng: random.Random) -> list[int]:
    """Draw n_ref reference square sides from N(μ, σ), rounded to ints, clipped."""
    out: list[int] = []
    for _ in range(n_ref):
        v = rng.gauss(spec.mu, spec.sigma)
        v = int(round(max(4, v)))
        out.append(v)
    return out


def render_stim(
    spec: StimSpec,
    ref_sizes: list[int],
    canvas: int = CANVAS,
) -> Image.Image:
    """Render the composite stimulus.

    Layout:
      [reference grid in upper REF_REGION_FRAC]
      [thin separator band]
      [target square, centered horizontally, in bottom TARGET_REGION_FRAC]
    """
    img = Image.new("L", (canvas, canvas), color=255)  # white background
    draw = ImageDraw.Draw(img)

    ref_h = int(canvas * REF_REGION_FRAC)
    sep_h = int(canvas * SEPARATOR_BAND_FRAC)
    tgt_h = canvas - ref_h - sep_h
    sep_y0 = ref_h
    sep_y1 = ref_h + sep_h

    # Separator band: light grey
    draw.rectangle([0, sep_y0, canvas - 1, sep_y1 - 1], fill=230)

    # Reference grid: pack n_ref squares in a rough grid in [0, ref_h].
    # Use a fixed 4×3 layout for n_ref=12; otherwise compute cols×rows.
    n_ref = len(ref_sizes)
    cols = math.ceil(math.sqrt(n_ref * canvas / max(1, ref_h)))
    rows = math.ceil(n_ref / cols)
    cell_w = canvas // cols
    cell_h = ref_h // rows
    rng_layout = random.Random(spec.seed * 7919 + 1)
    for i, side in enumerate(ref_sizes):
        r = i // cols
        c = i % cols
        cx = c * cell_w + cell_w // 2
        cy = r * cell_h + cell_h // 2
        # Add small jitter so the reference grid doesn't look mechanical
        jitter_x = rng_layout.randint(-cell_w // 8, cell_w // 8)
        jitter_y = rng_layout.randint(-cell_h // 8, cell_h // 8)
        cx += jitter_x
        cy += jitter_y
        x0 = cx - side // 2
        y0 = cy - side // 2
        x1 = x0 + side
        y1 = y0 + side
        # Clip to top region
        x0 = max(2, x0); y0 = max(2, y0)
        x1 = min(canvas - 2, x1); y1 = min(sep_y0 - 2, y1)
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=120)  # mid-grey

    # Target square: black, centered horizontally in the bottom region
    tgt_cx = canvas // 2
    tgt_cy = sep_y1 + tgt_h // 2
    tx0 = tgt_cx - spec.x // 2
    ty0 = tgt_cy - spec.x // 2
    tx1 = tx0 + spec.x
    ty1 = ty0 + spec.x
    tx0 = max(2, tx0); ty0 = max(sep_y1 + 2, ty0)
    tx1 = min(canvas - 2, tx1); ty1 = min(canvas - 2, ty1)
    draw.rectangle([tx0, ty0, tx1 - 1, ty1 - 1], fill=0)  # black

    return img.convert("RGB")


def gen_grid(
    n_x: int,
    n_z: int,
    n_seeds: int,
    x_min: int,
    x_max: int,
    sigma: float,
) -> list[StimSpec]:
    """Generate the (x, z) × seed grid, dropping implausible cells."""
    xs = np.linspace(x_min, x_max, n_x).round().astype(int)
    zs = np.linspace(Z_MIN, Z_MAX, n_z).round(2)
    out: list[StimSpec] = []
    for x in xs:
        for z in zs:
            mu = x - sigma * z
            for seed in range(n_seeds):
                spec = StimSpec(x=int(x), mu=float(mu), sigma=float(sigma),
                                z=float(z), seed=seed)
                if not is_plausible(spec, x_min, x_max):
                    continue
                out.append(spec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-x", type=int, default=12)
    ap.add_argument("--n-z", type=int, default=12)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--x-min", type=int, default=DEFAULT_X_MIN)
    ap.add_argument("--x-max", type=int, default=DEFAULT_X_MAX)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    ap.add_argument("--n-ref", type=int, default=DEFAULT_N_REF)
    ap.add_argument("--canvas", type=int, default=CANVAS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    specs = gen_grid(args.n_x, args.n_z, args.n_seeds,
                     args.x_min, args.x_max, args.sigma)
    print(f"generating {len(specs)} stimuli (n_x={args.n_x}, n_z={args.n_z}, "
          f"seeds={args.n_seeds}; some cells dropped if mu out of band)")

    manifest_path = out_dir / "stimuli.jsonl"
    n_written = 0
    with manifest_path.open("w") as f:
        for i, spec in enumerate(specs):
            rng = random.Random(0xBEEF + spec.seed * 31337 + i)
            ref_sizes = sample_ref_sizes(spec, args.n_ref, rng)
            img = render_stim(spec, ref_sizes, canvas=args.canvas)
            stem = f"x{spec.x:03d}_z{spec.z:+0.2f}_s{spec.seed}_{i:05d}"
            img_path = img_dir / f"{stem}.png"
            img.save(img_path, optimize=True)
            row = {
                "id": stem,
                "image_path": str(img_path.relative_to(out_dir)),
                "x": spec.x,
                "mu": spec.mu,
                "sigma": spec.sigma,
                "z": spec.z,
                "seed": spec.seed,
                "ref_sizes": ref_sizes,
                "canvas": args.canvas,
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1
    print(f"wrote {n_written} stimuli + manifest to {manifest_path}")
    print(f"images in {img_dir}")


if __name__ == "__main__":
    main()
