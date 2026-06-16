"""V15: leave-one-out shared directions for z vs raw x.

This runner tests whether a shared direction built without the target concept
still steers that held-out concept, and whether that generalization is stronger
for relative standing z than for raw magnitude x.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "vast_remote"))
from _token_utils import first_token_id  # noqa: E402

MODEL_BY_SHORT = {
    "gemma2-2b": "google/gemma-2-2b",
    "gemma2-9b": "google/gemma-2-9b",
}
LATE_BY_SHORT = {
    "gemma2-2b": 20,
    "gemma2-9b": 33,
}
PAIRS = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]
DIR_KINDS = ["z", "x"]

RESULT_DIR = REPO / "results" / "v15"
FIG_DIR = REPO / "figures" / "v15"
PAPER_FIG_DIR = REPO / "paper" / "icml2026_draft" / "figures"
JSON_PATH = RESULT_DIR / "shared_direction_loo_z_vs_x.json"


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n


def repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def load_npz(model_short: str, pair: str) -> np.lib.npyio.NpzFile:
    path = REPO / "results" / "v11" / model_short / pair / f"{model_short}_{pair}_v11_residuals.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {repo_relative(path)}. Fetch V11 activations first with "
            "`python scripts/fetch_from_hf.py --only v11 --only prompts`."
        )
    return np.load(path)


def load_trials(pair: str) -> list[dict]:
    path = REPO / "data_gen" / f"v11_{pair}_trials.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {repo_relative(path)}. Fetch prompts first with "
            "`python scripts/fetch_from_hf.py --only prompts --data-kind jsonl`."
        )
    return [json.loads(line) for line in path.open()]


def seed0_trials(pair: str, max_n: int | None) -> list[dict]:
    rows = []
    seen = set()
    for row in load_trials(pair):
        if row.get("cell_seed", row.get("seed")) != 0:
            continue
        key = (round(float(row["x"]), 4), round(float(row["z"]), 4))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    if max_n is not None and len(rows) > max_n:
        idx = np.linspace(0, len(rows) - 1, max_n).round().astype(int)
        rows = [rows[int(i)] for i in idx]
    return rows


def direction(model_short: str, pair: str, layer: int, kind: str) -> np.ndarray:
    data = load_npz(model_short, pair)
    h = data["activations"][:, layer, :].astype(np.float64)
    if kind == "z":
        z = data["z"].astype(np.float64)
        hi = z > 1.0
        lo = z < -1.0
    elif kind == "x":
        x = data["x"].astype(np.float64)
        q25, q75 = np.quantile(x, [0.25, 0.75])
        hi = x >= q75
        lo = x <= q25
    else:
        raise ValueError(kind)
    if not hi.any() or not lo.any():
        raise ValueError(f"{pair}/{kind} has empty high or low split")
    return h[hi].mean(axis=0) - h[lo].mean(axis=0)


def sign_aligned_mean(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("cannot build shared direction from zero vectors")
    init = unit(np.mean(np.stack([unit(v) for v in vectors]), axis=0))
    aligned = [unit(v) if float(unit(v) @ init) >= 0 else -unit(v) for v in vectors]
    return unit(np.mean(np.stack(aligned), axis=0))


def get_layers(model):
    m = model
    for attr in ("model", "language_model", "text_model"):
        if hasattr(m, attr) and hasattr(getattr(m, attr), "layers"):
            return getattr(m, attr).layers
        if hasattr(m, attr):
            m = getattr(m, attr)
    raise AttributeError("decoder layers not found")


def steer_ld(
    model,
    tok,
    rows: list[dict],
    steer_direction: np.ndarray,
    layer: int,
    alpha: float,
    batch_size: int,
    max_seq: int,
) -> np.ndarray:
    import torch

    d = torch.tensor(unit(steer_direction), dtype=torch.bfloat16, device=model.device)
    layers = get_layers(model)

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + alpha * d
        return (h,) + output[1:] if isinstance(output, tuple) else h

    handle = layers[layer].register_forward_hook(hook)
    vals = np.zeros(len(rows), dtype=np.float32)
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            enc = tok(
                [r["prompt"] for r in batch],
                return_tensors="pt",
                padding="max_length",
                max_length=max_seq,
                truncation=True,
            ).to(model.device)
            hi = [first_token_id(tok, r["high_word"]) for r in batch]
            lo = [first_token_id(tok, r["low_word"]) for r in batch]
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[:, -1, :].float()
            vals[start : start + len(batch)] = np.array(
                [float(logits[i, hi[i]] - logits[i, lo[i]]) for i in range(len(batch))],
                dtype=np.float32,
            )
    finally:
        handle.remove()
    return vals


def steering_slope(model, tok, rows: list[dict], steer_direction: np.ndarray, layer: int, args) -> float:
    pos = steer_ld(model, tok, rows, steer_direction, layer, args.alpha, args.batch_size, args.max_seq)
    neg = steer_ld(model, tok, rows, steer_direction, layer, -args.alpha, args.batch_size, args.max_seq)
    return float((pos - neg).mean() / (2.0 * args.alpha))


def signed_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return float("nan")
    return float(numerator / denominator)


def matrix_to_array(matrix: dict[str, dict[str, float]], pairs: list[str]) -> np.ndarray:
    return np.array([[matrix[target][source] for source in pairs] for target in pairs], dtype=np.float64)


def mean_offdiag(matrix: dict[str, dict[str, float]], pairs: list[str]) -> float:
    arr = matrix_to_array(matrix, pairs)
    mask = ~np.eye(len(pairs), dtype=bool)
    return float(np.nanmean(arr[mask]))


def build_summary(result: dict) -> dict:
    pairs = result["pairs"]
    complete_pairs = [
        p
        for p in pairs
        if "loo_shared_ratio" in result["by_pair"][p]["z"]
        and "loo_shared_ratio" in result["by_pair"][p]["x"]
    ]
    loo_z = [result["by_pair"][p]["z"]["loo_shared_ratio"] for p in complete_pairs]
    loo_x = [result["by_pair"][p]["x"]["loo_shared_ratio"] for p in complete_pairs]
    full_z_complete = all(
        len(result["full_matrices"]["z"][target]) == len(pairs) for target in pairs
    )
    full_x_complete = all(
        len(result["full_matrices"]["x"][target]) == len(pairs) for target in pairs
    )
    unexpected = [
        p
        for p in complete_pairs
        if np.isfinite(result["by_pair"][p]["x"]["loo_shared_ratio"])
        and np.isfinite(result["by_pair"][p]["z"]["loo_shared_ratio"])
        and result["by_pair"][p]["x"]["loo_shared_ratio"] >= result["by_pair"][p]["z"]["loo_shared_ratio"] - 0.05
    ]
    return {
        "n_pass_loo_z_ratio_gt_0p5": int(np.sum(np.array(loo_z, dtype=np.float64) > 0.5)),
        "n_pass_loo_x_ratio_gt_0p5": int(np.sum(np.array(loo_x, dtype=np.float64) > 0.5)),
        "mean_loo_z_ratio": float(np.nanmean(loo_z)),
        "mean_loo_x_ratio": float(np.nanmean(loo_x)),
        "mean_offdiag_full_matrix_z": mean_offdiag(result["full_matrices"]["z"], pairs)
        if full_z_complete
        else float("nan"),
        "mean_offdiag_full_matrix_x": mean_offdiag(result["full_matrices"]["x"], pairs)
        if full_x_complete
        else float("nan"),
        "n_complete_pairs": len(complete_pairs),
        "x_transfers_about_as_well_as_z": unexpected,
    }


def run_experiment(args) -> dict:
    layer = args.layer if args.layer is not None else LATE_BY_SHORT[args.model_short]
    model_id = MODEL_BY_SHORT[args.model_short]
    pairs = args.pairs
    print(f"[v15] model={args.model_short} layer={layer} alpha={args.alpha}", flush=True)

    directions = {
        kind: {pair: unit(direction(args.model_short, pair, layer, kind)) for pair in pairs}
        for kind in DIR_KINDS
    }
    shared_all = {
        kind: sign_aligned_mean([directions[kind][pair] for pair in pairs]) for kind in DIR_KINDS
    }
    shared_loo = {
        kind: {
            target: sign_aligned_mean(
                [directions[kind][source] for source in pairs if source != target]
            )
            for target in pairs
        }
        for kind in DIR_KINDS
    }

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[v15] loading model", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    result = {
        "model_id": model_id,
        "model_short": args.model_short,
        "layer": layer,
        "alpha": args.alpha,
        "pairs": pairs,
        "prompt_count_by_target": {},
        "by_pair": {pair: {kind: {} for kind in DIR_KINDS} for pair in pairs},
        "full_matrices": {kind: {target: {} for target in pairs} for kind in DIR_KINDS},
        "loo_target_summaries": {kind: {} for kind in DIR_KINDS},
        "direction_cosines": {
            kind: {
                target: {
                    "own_vs_all_shared": float(directions[kind][target] @ shared_all[kind]),
                    "own_vs_loo_shared": float(directions[kind][target] @ shared_loo[kind][target]),
                }
                for target in pairs
            }
            for kind in DIR_KINDS
        },
    }

    for target in pairs:
        start_time = time.time()
        rows = seed0_trials(target, args.prompts_per_target)
        result["prompt_count_by_target"][target] = len(rows)
        print(f"[v15] target={target} prompts={len(rows)}", flush=True)

        for kind in DIR_KINDS:
            own = steering_slope(model, tok, rows, directions[kind][target], layer, args)
            all_shared = steering_slope(model, tok, rows, shared_all[kind], layer, args)
            loo_shared = steering_slope(model, tok, rows, shared_loo[kind][target], layer, args)
            result["by_pair"][target][kind] = {
                "own_slope": own,
                "all_shared_slope": all_shared,
                "loo_shared_slope": loo_shared,
                "all_shared_ratio": signed_ratio(all_shared, own),
                "loo_shared_ratio": signed_ratio(loo_shared, own),
            }
            result["loo_target_summaries"][kind][target] = {
                "own_slope": own,
                "all_shared_ratio": signed_ratio(all_shared, own),
                "loo_shared_ratio": signed_ratio(loo_shared, own),
            }

            for source in pairs:
                result["full_matrices"][kind][target][source] = steering_slope(
                    model, tok, rows, directions[kind][source], layer, args
                )

        z_ratio = result["by_pair"][target]["z"]["loo_shared_ratio"]
        x_ratio = result["by_pair"][target]["x"]["loo_shared_ratio"]
        print(
            f"[v15]   {target:11s} loo ratio z={z_ratio:+.2f} x={x_ratio:+.2f} "
            f"({time.time() - start_time:.1f}s)",
            flush=True,
        )
        result["summary"] = build_summary(result)
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        JSON_PATH.write_text(json.dumps(result, indent=2))

    result["summary"] = build_summary(result)
    JSON_PATH.write_text(json.dumps(result, indent=2))
    print(f"[v15] wrote {repo_relative(JSON_PATH)}", flush=True)
    return result


def plot_heatmap(ax, arr: np.ndarray, row_labels: list[str], col_labels: list[str], title: str, cmap: str):
    finite = arr[np.isfinite(arr)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            text = "" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
    return im


def render_plots(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    pairs = result["pairs"]
    display = [p.replace("_abs", "") for p in pairs]

    z_ratios = np.array([result["by_pair"][p]["z"]["loo_shared_ratio"] for p in pairs])
    x_ratios = np.array([result["by_pair"][p]["x"]["loo_shared_ratio"] for p in pairs])
    x = np.arange(len(pairs))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.bar(x - width / 2, z_ratios, width, label="relative z", color="#4C72B0")
    ax.bar(x + width / 2, x_ratios, width, label="raw x", color="#DD8452")
    ax.axhline(0.5, color="#555555", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("LOO shared / own steering")
    ax.set_ylim(min(-0.1, float(np.nanmin([z_ratios, x_ratios])) - 0.1), max(1.1, float(np.nanmax([z_ratios, x_ratios])) + 0.15))
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper left")
    fig.tight_layout()
    ratio_path = FIG_DIR / "shared_direction_loo_ratios_z_vs_x.png"
    paper_path = PAPER_FIG_DIR / "fig_results_shared_direction_loo_z_vs_x_clean.png"
    fig.savefig(ratio_path, dpi=200)
    fig.savefig(paper_path, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for ax, kind, title in zip(axes, DIR_KINDS, ["source d_z", "source d_x"]):
        arr = matrix_to_array(result["full_matrices"][kind], pairs)
        im = plot_heatmap(ax, arr, display, display, title, "coolwarm")
        ax.set_xlabel("source concept")
        ax.set_ylabel("target concept")
    fig.colorbar(im, ax=axes, shrink=0.75, label="steering slope")
    fig.savefig(FIG_DIR / "shared_direction_full_matrix_z_vs_x.png", dpi=180)
    plt.close(fig)

    loo_arrs = {}
    for kind in DIR_KINDS:
        loo_arrs[kind] = np.array(
            [
                [
                    1.0,
                    result["by_pair"][p][kind]["all_shared_ratio"],
                    result["by_pair"][p][kind]["loo_shared_ratio"],
                ]
                for p in pairs
            ],
            dtype=np.float64,
        )
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0), constrained_layout=True)
    for ax, kind, title in zip(axes, DIR_KINDS, ["relative z", "raw x"]):
        im = plot_heatmap(ax, loo_arrs[kind], display, ["own", "all", "LOO"], title, "coolwarm")
        ax.set_ylabel("held-out target")
    fig.colorbar(im, ax=axes, shrink=0.75, label="shared / own ratio")
    fig.savefig(FIG_DIR / "shared_direction_loo_matrix_z_vs_x.png", dpi=180)
    plt.close(fig)

    print(f"[v15] wrote {repo_relative(ratio_path)}", flush=True)
    print(f"[v15] wrote {repo_relative(paper_path)}", flush=True)


def print_summary(result: dict) -> None:
    summary = result["summary"]
    prompt_counts = result["prompt_count_by_target"]
    counts = sorted(set(prompt_counts.values()))
    count_text = str(counts[0]) if len(counts) == 1 else json.dumps(prompt_counts, sort_keys=True)
    print("\n[v15] summary")
    print(f"  model={result['model_short']} layer={result['layer']} alpha={result['alpha']}")
    print(f"  prompt count per target={count_text}")
    print(f"  LOO d_z ratio > 0.5: {summary['n_pass_loo_z_ratio_gt_0p5']}/{len(result['pairs'])}")
    print(f"  LOO d_x ratio > 0.5: {summary['n_pass_loo_x_ratio_gt_0p5']}/{len(result['pairs'])}")
    print(f"  mean LOO d_z ratio={summary['mean_loo_z_ratio']:+.3f}")
    print(f"  mean LOO d_x ratio={summary['mean_loo_x_ratio']:+.3f}")
    print(f"  mean off-diagonal d_z transfer={summary['mean_offdiag_full_matrix_z']:+.3f}")
    print(f"  mean off-diagonal d_x transfer={summary['mean_offdiag_full_matrix_x']:+.3f}")
    unexpected = summary["x_transfers_about_as_well_as_z"]
    print(f"  d_x about as strong as d_z: {', '.join(unexpected) if unexpected else 'none'}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-short", default="gemma2-9b", choices=sorted(MODEL_BY_SHORT))
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=288)
    ap.add_argument("--prompts-per-target", type=int, default=None)
    ap.add_argument("--pairs", nargs="+", default=PAIRS, choices=PAIRS)
    ap.add_argument("--plot-only", action="store_true", help="render figures from the saved JSON")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        if not JSON_PATH.exists():
            raise SystemExit(f"Missing {repo_relative(JSON_PATH)}")
        result = json.loads(JSON_PATH.read_text())
    else:
        result = run_experiment(args)
    render_plots(result)
    print_summary(result)


if __name__ == "__main__":
    main()
