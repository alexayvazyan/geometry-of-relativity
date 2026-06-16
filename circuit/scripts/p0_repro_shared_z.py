"""Phase 0 — reproduce v11.5 §16.1 shared-z direction from cached v11 NPZs.

Inputs (expected layout):
    DATA_ROOT/v11/<model_short>/<pair>/<model_short>_<pair>_v11_residuals.npz
    DATA_ROOT/data_gen/v11_<pair>_trials.jsonl

Run:
    python p0_repro_shared_z.py \\
        --data-root /home/alexander/research_projects/geometry-of-relativity \\
        --model-short gemma2-2b \\
        --layer 20

Outputs:
    results/p0_shared_z_<model_short>_L<layer>.json
    A short stdout table of pairwise cos and shared/primal cos.

Pass criterion (per FINDINGS §3): pairwise mean cos within 0.02 of the
published v11.5 §16.1 values (0.559 for 2B at L20, 0.516 for 9B at L33).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ALL_PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]

# Published v11.5 §16.1 values for sanity-checking
PUBLISHED_PAIRWISE_COS = {("gemma2-2b", 20): 0.559, ("gemma2-9b", 33): 0.516}
PUBLISHED_SHARED_VS_OWN = {  # cos(w_shared_proc, primal_z[pair])
    # WARNING: README/FINDINGS tables list `ratio_shared_to_within` (steering
    # efficacy), not these cosines. The cosines below come directly from
    # geometry-of-relativity/results/v11_5/<model>/shared_z_analysis.json.
    ("gemma2-2b", 20): {
        "height": 0.8887, "age": 0.7816, "weight": 0.8970, "size": 0.8107,
        "speed": 0.6983, "wealth": 0.7397, "experience": 0.7183, "bmi_abs": 0.7245,
    },
    ("gemma2-9b", 33): {  # NOTE: not yet verified against the 9B JSON
        "height": 0.75, "weight": 0.80, "size": 0.66, "wealth": 0.70,
        "bmi_abs": 0.65, "age": 0.56, "speed": 0.42, "experience": 0.50,
    },
}


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def primal_z_from_npz(npz_path: Path, layer: int) -> np.ndarray | None:
    if not npz_path.exists():
        return None
    d = np.load(npz_path)
    h = d["activations"][:, layer, :].astype(np.float64)
    z = d["z"]
    high = h[z > +1.0].mean(0)
    low = h[z < -1.0].mean(0)
    return high - low


def build_w_shared(primals: list[np.ndarray]) -> dict[str, np.ndarray]:
    P = np.stack(primals)  # (n_pairs, d_model)
    w_mean = unit(P.mean(0))
    # Procrustes: sign-flip each pair to align with w_mean, then re-mean
    P_aligned = np.array([p if (p @ w_mean) >= 0 else -p for p in P])
    w_proc = unit(P_aligned.mean(0))
    return {"mean": w_mean, "proc": w_proc}


def pairwise_cos(primals: list[np.ndarray]) -> float:
    units = [unit(p) for p in primals]
    pairs = []
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            pairs.append(float(units[i] @ units[j]))
    return float(np.mean(pairs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="root containing results/v11/<model>/<pair>/_residuals.npz")
    ap.add_argument("--model-short", choices=["gemma2-2b", "gemma2-9b"], required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent.parent / "results"))
    args = ap.parse_args()

    data_root = Path(args.data_root)
    model_short = args.model_short
    L = args.layer

    primals: list[np.ndarray] = []
    pair_names: list[str] = []
    for pair in ALL_PAIRS:
        npz = data_root / "results" / "v11" / model_short / pair / \
              f"{model_short}_{pair}_v11_residuals.npz"
        d = primal_z_from_npz(npz, L)
        if d is None:
            print(f"  [skip] no NPZ for {pair} at {npz}")
            continue
        primals.append(d)
        pair_names.append(pair)

    if len(primals) < 2:
        raise SystemExit(
            f"Need at least 2 pair NPZs to reproduce. Found {len(primals)}.\n"
            f"Run a fetch first — see FINDINGS §1 for the dependency note."
        )

    pw_cos = pairwise_cos(primals)
    shared = build_w_shared(primals)
    cos_with_proc = {p: float(unit(primals[i]) @ shared["proc"])
                     for i, p in enumerate(pair_names)}

    # Compare to published values
    pub_pw = PUBLISHED_PAIRWISE_COS.get((model_short, L))
    pub_shared = PUBLISHED_SHARED_VS_OWN.get((model_short, L), {})
    pw_diff = pw_cos - pub_pw if pub_pw is not None else None

    print(f"\n=== Phase 0 reproduction: {model_short} L{L}  ({len(pair_names)} pairs) ===")
    print(f"pairs: {pair_names}")
    print(f"pairwise mean cos: {pw_cos:+.3f}"
          + (f"   (published {pub_pw:+.3f}, diff {pw_diff:+.3f})" if pub_pw is not None else ""))
    print("\ncos(w_shared_proc, primal_z[pair]):")
    print(f"{'pair':<12} {'this run':>10} {'published':>10} {'diff':>8}")
    fail_count = 0
    for p in pair_names:
        this = cos_with_proc[p]
        pub = pub_shared.get(p)
        if pub is None:
            print(f"  {p:<10} {this:>+10.3f}        --       --")
        else:
            diff = this - pub
            mark = "  OK" if abs(diff) < 0.01 else " FAIL"
            if abs(diff) >= 0.01:
                fail_count += 1
            print(f"  {p:<10} {this:>+10.4f}  {pub:>+8.4f}  {diff:>+7.4f}{mark}")

    print(f"\n{fail_count}/{len(pair_names)} pairs differ from published by ≥0.01")

    out = {
        "model_short": model_short,
        "layer": L,
        "pairs": pair_names,
        "pairwise_cos_mean": pw_cos,
        "published_pairwise_cos_mean": pub_pw,
        "cos_w_shared_proc_vs_primal": cos_with_proc,
        "published_cos_w_shared_proc_vs_primal": pub_shared,
        "pairs_differing_geq_0p05": fail_count,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"p0_shared_z_{model_short}_L{L}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
