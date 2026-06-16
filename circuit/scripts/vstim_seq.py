"""Vision V1 — sequential single-square stimuli for relativity probing.

For each (x, z) cell × seed, generate N_REF + 1 PNG images:
  - N_REF reference images, each one black square centered on white canvas,
    side ~ N(μ, σ), rounded and clipped to a plausible range.
  - 1 target image, single square of side x, centered.

z = (x − μ) / σ.

Stimuli are arranged so the model sees a sequence of "Image 1 ... Image N,
Target" and is asked to characterize the target's size. This mirrors the
v11 text design where 15 context heights are listed before the target.

Output:
  <out>/stimuli.jsonl  — one row per prompt with metadata + image filenames
  <out>/images/<id>/frame_{0..N}.png — N reference + 1 target

Usage:
    python vstim_seq.py --out ../stimuli/vsize_v0 --n-x 10 --n-z 10 --n-seeds 5
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CANVAS = 224           # Gemma 4 vision native resolution
DEFAULT_N_REF = 8      # reference images per stimulus
DEFAULT_X_MIN = 16
DEFAULT_X_MAX = 96
DEFAULT_SIGMA = 12.0
Z_MIN = -2.5
Z_MAX = +2.5

REF_SIZE_MIN = 4       # absolute floor (still visible)
REF_SIZE_MAX = 140     # absolute ceiling (well within 224 canvas)
PLAUSIBILITY_K_SIGMA = 2.0  # require μ ± k·σ to stay in [MIN, MAX]


@dataclass(frozen=True)
class StimSpec:
    x: int
    mu: float
    sigma: float
    z: float
    seed: int


def make_square_frame(canvas: int, side: int, bg: int = 255, fg: int = 0
                      ) -> Image.Image:
    """Single black square (side `side`) centered on white canvas."""
    arr = np.full((canvas, canvas), bg, dtype=np.uint8)
    x0 = (canvas - side) // 2
    y0 = (canvas - side) // 2
    x1 = x0 + side
    y1 = y0 + side
    arr[y0:y1, x0:x1] = fg
    return Image.fromarray(arr, mode="L").convert("RGB")


def is_plausible_mu(mu: float, sigma: float, x_min: int, x_max: int) -> bool:
    """Most reference draws (±k·σ) should fit in [REF_SIZE_MIN, REF_SIZE_MAX]."""
    lo = mu - PLAUSIBILITY_K_SIGMA * sigma
    hi = mu + PLAUSIBILITY_K_SIGMA * sigma
    return lo >= REF_SIZE_MIN and hi <= REF_SIZE_MAX


def sample_ref_sizes(mu: float, sigma: float, n_ref: int,
                      rng: random.Random) -> list[int]:
    out: list[int] = []
    for _ in range(n_ref):
        v = rng.gauss(mu, sigma)
        v = int(round(v))
        v = max(REF_SIZE_MIN, min(REF_SIZE_MAX, v))
        out.append(v)
    return out


def gen_grid(n_x: int, n_z: int, n_seeds: int,
             x_min: int, x_max: int, sigma: float) -> list[StimSpec]:
    xs = np.linspace(x_min, x_max, n_x).round().astype(int)
    zs = np.linspace(Z_MIN, Z_MAX, n_z).round(2)
    specs: list[StimSpec] = []
    for x in xs:
        for z in zs:
            mu = float(x) - sigma * z
            if not is_plausible_mu(mu, sigma, x_min, x_max):
                continue
            for seed in range(n_seeds):
                specs.append(StimSpec(x=int(x), mu=mu, sigma=sigma,
                                       z=float(z), seed=seed))
    return specs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-x", type=int, default=10)
    ap.add_argument("--n-z", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-ref", type=int, default=DEFAULT_N_REF)
    ap.add_argument("--canvas", type=int, default=CANVAS)
    ap.add_argument("--x-min", type=int, default=DEFAULT_X_MIN)
    ap.add_argument("--x-max", type=int, default=DEFAULT_X_MAX)
    ap.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    img_root = args.out / "images"
    img_root.mkdir(exist_ok=True)

    specs = gen_grid(args.n_x, args.n_z, args.n_seeds, args.x_min,
                     args.x_max, args.sigma)
    print(f"generating {len(specs)} stimuli (n_x={args.n_x}, n_z={args.n_z}, "
          f"seeds={args.n_seeds}); some cells dropped if mu out of band")

    manifest_path = args.out / "stimuli.jsonl"
    n_written = 0
    with manifest_path.open("w") as f:
        for i, spec in enumerate(specs):
            stem = f"x{spec.x:03d}_z{spec.z:+0.2f}_s{spec.seed}_{i:05d}"
            stim_dir = img_root / stem
            stim_dir.mkdir(exist_ok=True)
            rng = random.Random(0xBEEF + spec.seed * 31337 + i)
            ref_sizes = sample_ref_sizes(spec.mu, spec.sigma, args.n_ref, rng)
            for k, side in enumerate(ref_sizes):
                make_square_frame(args.canvas, side).save(
                    stim_dir / f"ref_{k:02d}.png", optimize=True)
            make_square_frame(args.canvas, spec.x).save(
                stim_dir / "target.png", optimize=True)
            row = {
                "id": stem,
                "stim_dir": str(stim_dir.relative_to(args.out)),
                "ref_filenames": [f"ref_{k:02d}.png" for k in range(args.n_ref)],
                "target_filename": "target.png",
                "x": spec.x,
                "mu": spec.mu,
                "sigma": spec.sigma,
                "z": spec.z,
                "seed": spec.seed,
                "ref_sizes": ref_sizes,
                "n_ref": args.n_ref,
                "canvas": args.canvas,
                "low_word": "small",
                "high_word": "big",
            }
            f.write(json.dumps(row) + "\n")
            n_written += 1
    print(f"wrote {n_written} stimuli + manifest to {manifest_path}")


if __name__ == "__main__":
    main()
