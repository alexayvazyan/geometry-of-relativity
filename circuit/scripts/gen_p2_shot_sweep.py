"""Phase 2A — generate prompts for the shot-count behavioral sweep.

For each pair × cell × k:
  - Reuse Jaehoon's v11 (x, z, cell_seed) grid logic, but vary `k` (= number of
    context items before the target).
  - Same seed across k values → context is *prefix-nested* (k=2 prompt is the
    first 2 items of the k=15 prompt). Lets us read shot-count effects without
    changing realizations.
  - At k=0 we strip the `{items}\n` prefix from the implicit template so the
    surface form is "Person 1: <x> ... This person is" — same template family,
    no context. (This is *not* the existing zero-shot template, which uses
    different prose like "A person who is X cm is".)

Per prompt we record:
  x        : target raw value
  z        : population z = (x - mu) / sigma  (the value used to derive context)
  z_eff    : sample-mean-based z = (x - mean(context)) / sigma_eff
  mu       : population mu used to sample context
  mu_eff   : sample mean of context (= mu when k=0 → NaN)
  sigma    : population sigma
  sigma_eff: sample stdev of context (NaN when k<2)
  k        : number of context items (0..15)

Usage:
  python scripts/gen_p2_shot_sweep.py --pairs height weight speed \
                                      --k 0 1 2 4 8 15 \
                                      --n-seeds 3 \
                                      --n-x 20 --n-z 20

Output:
  data/p2_shot_sweep/<pair>_k<k>.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent  # relativity_ablation/
GOR = REPO.parent / "geometry-of-relativity"
sys.path.insert(0, str(GOR / "scripts" / "vast_remote"))
from extract_v4_adjpairs import (  # noqa: E402
    PAIRS,
    LOG_SPACE_PAIRS,
    Pair,
    build_implicit_items,
    fmt_num,
    compute_z,
    sample_context,
)

OUT = REPO / "data" / "p2_shot_sweep"

PAIRS_BY_NAME = {p.name: p for p in PAIRS}

# Match v11_dense grid bounds.
Z_MIN, Z_MAX = -3.0, 3.0


def x_grid(pair: Pair, n_x: int) -> list[float]:
    lo = float(min(pair.target_values))
    hi = float(max(pair.target_values))
    if pair.name in LOG_SPACE_PAIRS:
        log_lo = math.log(lo * 0.7)
        log_hi = math.log(hi * 1.5)
        xs = np.exp(np.linspace(log_lo, log_hi, n_x))
        return [round(x, 0) if x >= 100 else round(x, 1) for x in xs.tolist()]
    span = hi - lo
    x_lo = max(0.5, lo - 0.1 * span)
    return np.linspace(x_lo, hi + 0.1 * span, n_x).round(1).tolist()


def z_grid(n_z: int) -> list[float]:
    return np.linspace(Z_MIN, Z_MAX, n_z).round(2).tolist()


def derive_mu(pair: Pair, x: float, z: float) -> float:
    if pair.name in LOG_SPACE_PAIRS:
        return x * (pair.sigma ** (-z))
    return x - pair.sigma * z


def is_plausible_mu(pair: Pair, mu: float) -> bool:
    if pair.name in LOG_SPACE_PAIRS:
        return mu > 0
    lo = float(min(pair.target_values))
    hi = float(max(pair.target_values))
    span = hi - lo
    return (lo - 0.5 * span) <= mu <= (hi + 0.5 * span)


def make_prompt_k(pair: Pair, x: float, mu: float, seed: int, k: int) -> tuple[str, list[float]]:
    """Build the implicit prompt with `k` context items.

    Returns (prompt_text, sampled_context_values).
    """
    if k == 0:
        target_only_template = pair.format_prompt_implicit.replace("{items}\n", "")
        prompt = target_only_template.format(n_last=1, x_str=fmt_num(x))
        return prompt, []
    # Reuse v11 sampler with n=k. Because random.Random is deterministic per
    # seed, the first k of the n=15 sample is identical — prompts nest cleanly.
    low = pair.target_values[0] * 0.4
    high = pair.target_values[-1] * 2.5
    sample = sample_context(
        mu, pair.sigma, seed, n=k, low=low, high=high,
        log_space=(pair.name in LOG_SPACE_PAIRS),
    )
    # Reuse the per-pair item formatting (cm / kg / km/h / etc.) by calling
    # build_implicit_items with the same seed and slicing — but since
    # build_implicit_items also calls sample_context internally, we want to
    # avoid double-sampling. Easiest: rebuild items directly here, mirroring
    # build_implicit_items's pair-name switch.
    items = _format_items(pair, sample)
    items_block = "\n".join(items)
    prompt = pair.format_prompt_implicit.format(
        items=items_block,
        n_last=k + 1,
        x_str=fmt_num(x),
    )
    return prompt, [float(v) for v in sample]


def _format_items(pair: Pair, sample: list[float]) -> list[str]:
    name = pair.name
    if name == "height":
        return [f"Person {i+1}: {int(v)} cm" for i, v in enumerate(sample)]
    if name == "age":
        return [f"Person {i+1}: {int(v)} years old" for i, v in enumerate(sample)]
    if name == "weight":
        return [f"Person {i+1}: {int(v)} kg" for i, v in enumerate(sample)]
    if name == "size":
        return [f"Object {i+1}: {int(v)} cm across" for i, v in enumerate(sample)]
    if name == "speed":
        return [f"Vehicle {i+1}: {int(v)} km/h" for i, v in enumerate(sample)]
    if name == "wealth":
        return [f"Person {i+1} earns ${int(v)}/year" for i, v in enumerate(sample)]
    if name == "experience":
        return [f"Worker {i+1}: {int(v)} years experience" for i, v in enumerate(sample)]
    if name == "bmi_abs":
        return [f"Person {i+1}: BMI {v:.1f}" for i, v in enumerate(sample)]
    return [f"Item {i+1}: {v}" for i, v in enumerate(sample)]


def gen_for_pair_k(pair: Pair, k: int, n_x: int, n_z: int, n_seeds: int) -> list[dict]:
    """Build the full trial list for one (pair, k) pair."""
    xs = x_grid(pair, n_x)
    zs = z_grid(n_z)

    # Seed offset: per-pair from v11_dense (zlib.crc32 of pair name) — match
    # the convention so we don't reuse seeds with v11.
    import zlib
    pair_seed_offset = zlib.crc32(pair.name.encode()) & 0xFFFFFFFF

    trials: list[dict] = []

    if k == 0:
        # No context → only x varies; z and seed are irrelevant.
        for xi, x in enumerate(xs):
            prompt, _ = make_prompt_k(pair, x, mu=float("nan"), seed=0, k=0)
            trials.append({
                "id": f"p2_{pair.name}_k0_{xi:03d}",
                "pair": pair.name,
                "k": 0,
                "x": float(x),
                "z": float("nan"),
                "z_eff": float("nan"),
                "mu": float("nan"),
                "mu_eff": float("nan"),
                "sigma": float(pair.sigma),
                "sigma_eff": float("nan"),
                "seed": 0,
                "cell_seed": 0,
                "low_word": pair.low_word,
                "high_word": pair.high_word,
                "prompt": prompt,
                "context_values": [],
            })
        return trials

    # k >= 1: full (x, z, cell_seed) grid.
    idx = 0
    for x in xs:
        for z in zs:
            mu = derive_mu(pair, float(x), float(z))
            if not is_plausible_mu(pair, mu):
                continue
            for cs in range(n_seeds):
                seed = pair_seed_offset + cs
                prompt, ctx = make_prompt_k(pair, float(x), mu, seed, k)
                ctx_arr = np.asarray(ctx, dtype=np.float64)
                if pair.name in LOG_SPACE_PAIRS:
                    log_ctx = np.log(np.maximum(ctx_arr, 1e-12))
                    mu_eff_log = float(log_ctx.mean()) if k >= 1 else float("nan")
                    mu_eff = float(math.exp(mu_eff_log))
                    sigma_eff = float(log_ctx.std(ddof=1)) if k >= 2 else float("nan")
                    z_eff = (math.log(float(x)) - mu_eff_log) / math.log(pair.sigma)
                else:
                    mu_eff = float(ctx_arr.mean())
                    sigma_eff = float(ctx_arr.std(ddof=1)) if k >= 2 else float("nan")
                    z_eff = (float(x) - mu_eff) / pair.sigma
                trials.append({
                    "id": f"p2_{pair.name}_k{k}_{idx:06d}",
                    "pair": pair.name,
                    "k": k,
                    "x": float(x),
                    "z": float(z),
                    "z_eff": float(z_eff),
                    "mu": float(mu),
                    "mu_eff": mu_eff,
                    "sigma": float(pair.sigma),
                    "sigma_eff": sigma_eff,
                    "seed": int(seed),
                    "cell_seed": int(cs),
                    "low_word": pair.low_word,
                    "high_word": pair.high_word,
                    "prompt": prompt,
                    "context_values": [float(v) for v in ctx],
                })
                idx += 1
    return trials


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=["height", "weight", "speed"],
                   help="pairs to generate (default: height weight speed)")
    p.add_argument("--k", nargs="+", type=int, default=[0, 1, 2, 4, 8, 15])
    p.add_argument("--n-x", type=int, default=20)
    p.add_argument("--n-z", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=3,
                   help="cell_seeds per (x, z) cell (k>=1)")
    p.add_argument("--out", type=str, default=str(OUT))
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for pair_name in args.pairs:
        pair = PAIRS_BY_NAME[pair_name]
        for k in args.k:
            trials = gen_for_pair_k(pair, k, args.n_x, args.n_z, args.n_seeds)
            out_path = out_dir / f"{pair_name}_k{k}.jsonl"
            with out_path.open("w") as f:
                for t in trials:
                    f.write(json.dumps(t) + "\n")
            print(f"[{pair_name}] k={k:2d} → {len(trials):>5d} trials → {out_path}")
            summary.append((pair_name, k, len(trials)))

    print(f"\nTotal: {sum(c for _, _, c in summary)} trials across "
          f"{len(args.pairs)} pairs × {len(args.k)} k-values.")


if __name__ == "__main__":
    main()
