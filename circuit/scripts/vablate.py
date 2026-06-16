"""Vision V4 — manifold + proj_out ablation on vision-relativity grid.

Loads activations from `vextract_<short>_residuals.npz`, builds per-pair
primal_z and cell-mean lookup at chosen layer from train fold, then runs
α-sweep with manifold-shift and per-pair proj_out hooks. Same metrics as
text version (r_z, r_x, ⟨LD⟩) plus per-x slope diagnostics.

Train/test split is by cell_seed: train ∈ {0..n_train-1}, test = rest.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                            BitsAndBytesConfig)

REPO = Path(__file__).resolve().parent.parent


def get_decoder_layers(model):
    for path in [
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("model", "model", "layers"),
        ("model", "model", "language_model", "layers"),
    ]:
        m = model
        ok = True
        for attr in path:
            if hasattr(m, attr):
                m = getattr(m, attr)
            else:
                ok = False; break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def first_token_id(tok, w: str) -> int:
    return tok.encode(" " + w, add_special_tokens=False)[-1]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def make_partial_proj_hook(d_t, alpha):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h * d_t).sum(-1, keepdim=True)
        h = h - alpha * proj * d_t
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_alpha_shift_hook(state, alpha):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        delta = state["delta"]  # (1, d) for batch=1
        h = h + alpha * delta[None, None, :]
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def per_x_slope(ld, z, x) -> dict:
    """Cell-mean per-x slope of LD vs z. Returns dict[x] = (slope, r, mean_LD)."""
    cells = defaultdict(list)
    for i in range(len(ld)):
        cells[(round(float(x[i]), 4), round(float(z[i]), 4))].append(float(ld[i]))
    cell = {(xk, zk): float(np.mean(v)) for (xk, zk), v in cells.items()}
    by_x = defaultdict(list)
    for (xk, zk), m in cell.items():
        by_x[xk].append((zk, m))
    out = {}
    for xk, vals in by_x.items():
        zs_x = np.array([t[0] for t in vals])
        ms = np.array([t[1] for t in vals])
        if len(zs_x) < 3:
            continue
        r = float(np.corrcoef(ms, zs_x)[0, 1])
        slope = float(np.polyfit(zs_x, ms, 1)[0])
        out[float(xk)] = {"slope": slope, "r": r, "mean_ld": float(np.mean(ms)),
                            "n_z": len(zs_x)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-short", required=True)
    ap.add_argument("--name", required=True,
                    help="name of the vextract NPZ (sans suffix)")
    ap.add_argument("--stim", default="stimuli/vsize_v0/stimuli.jsonl")
    ap.add_argument("--layer", type=int, default=33,
                    help="L20-equivalent late layer; for E4B (42L) try L33")
    ap.add_argument("--high-word", default="big")
    ap.add_argument("--low-word", default="small")
    ap.add_argument("--prompt", default="The square in the last image is")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 0.75, 1.0])
    ap.add_argument("--n-train-seeds", type=int, default=3,
                    help="cell_seeds 0..n_train-1 for train fold; rest = test")
    ap.add_argument("--quant-4bit", action="store_true",
                    help="load model in 4-bit (bitsandbytes); for 31B-it on 32GB VRAM")
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()

    out_name = args.out_name or f"vablate_{args.model_short}_L{args.layer}"
    out_dir = REPO / "results"

    # Load extraction
    npz_path = out_dir / f"{args.name}_residuals.npz"
    print(f"[vab] loading {npz_path}...")
    d = np.load(npz_path)
    A = d["activations"]  # (N, n_layers+1, d_model) — note: hidden_states has +1
    z = d["z"].astype(np.float64)
    x = d["x"].astype(np.float64)
    seed = d["seed"]
    ld_baseline = d["ld"].astype(np.float64)
    n = A.shape[0]
    print(f"[vab]   N={n}  shape={A.shape}  layer={args.layer}")

    # The layer index in `A` includes embedding (index 0). So the output of
    # decoder layer L is at A[:, L+1, :]. We hook decoder_layers[L] which
    # also produces the post-L activation. Our `--layer` arg refers to the
    # decoder layer index (0-indexed); A index is layer+1.
    L_arg = args.layer
    h_layer = A[:, L_arg + 1, :].astype(np.float64)
    print(f"[vab]   hidden shape at layer L{L_arg}: {h_layer.shape}")

    # Train/test split by cell_seed
    n_train = args.n_train_seeds
    train_mask = seed < n_train
    test_mask = ~train_mask
    print(f"[vab]   train fold: {train_mask.sum()} prompts (cell_seed<{n_train})  "
          f"test fold: {test_mask.sum()} prompts")

    # primal_z from train fold
    z_tr = z[train_mask]
    h_tr = h_layer[train_mask]
    primal = (h_tr[z_tr > +1.0].mean(0) - h_tr[z_tr < -1.0].mean(0))
    print(f"[vab]   primal_z norm = {np.linalg.norm(primal):.2f}")

    # Cell-mean lookup from train fold (per (x_round, z_round))
    cells: dict[tuple[float, float], list[np.ndarray]] = defaultdict(list)
    for i in np.where(train_mask)[0]:
        key = (round(float(x[i]), 4), round(float(z[i]), 4))
        cells[key].append(h_layer[i])
    lookup = {k: np.mean(np.stack(v), axis=0) for k, v in cells.items()}

    # Per-prompt manifold displacement for test prompts
    by_x: dict[float, list[tuple[float, np.ndarray]]] = defaultdict(list)
    for (xk, zk), v in lookup.items():
        by_x[xk].append((zk, v))
    test_idx = np.where(test_mask)[0]
    Δ = np.zeros((len(test_idx), h_layer.shape[1]), dtype=np.float64)
    n_missing = 0
    for j, i in enumerate(test_idx):
        xi = round(float(x[i]), 4)
        zi = round(float(z[i]), 4)
        same_x = by_x.get(xi, [])
        if not same_x:
            n_missing += 1
            continue
        z_t, mu_target = min(same_x, key=lambda kv: abs(kv[0] - 0.0))
        z_s, mu_source = min(same_x, key=lambda kv: abs(kv[0] - zi))
        Δ[j] = mu_target - mu_source
    if n_missing:
        print(f"[vab]   warning: {n_missing}/{len(test_idx)} test prompts had "
              f"no same-x cell in train fold")

    norms = np.linalg.norm(Δ, axis=1)
    print(f"[vab]   manifold ||Δ||: mean={norms.mean():.2f}  "
          f"median={np.median(norms):.2f}  max={norms.max():.2f}")

    # Load model + processor for forward passes
    print(f"[vab] loading {args.model} (4bit={args.quant_4bit})...")
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model,
                                          token=os.environ.get("HF_TOKEN"))
    load_kwargs: dict = dict(device_map="auto", token=os.environ.get("HF_TOKEN"))
    if args.quant_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    else:
        load_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs).eval()
    print(f"[vab]   loaded in {time.time() - t0:.1f}s")

    layers = get_decoder_layers(model)
    img_token = getattr(proc, "image_token", "<|image|>")
    tok = proc.tokenizer
    high_id = first_token_id(tok, args.high_word)
    low_id = first_token_id(tok, args.low_word)

    # Direction tensors on device
    d_t = torch.tensor(unit(primal), dtype=torch.bfloat16, device=model.device)

    # Load test prompts (re-load images per stimulus)
    stim_path = REPO / args.stim
    rows = [json.loads(line) for line in stim_path.open()]
    stim_root = stim_path.parent

    # Filter by test_idx
    rows_test = [rows[i] for i in test_idx]
    z_test = z[test_idx]
    x_test = x[test_idx]
    print(f"[vab]   test set: {len(rows_test)} prompts")

    @torch.no_grad()
    def fwd_LD(rows_subset, hook_fn=None, per_prompt_delta=None):
        """If per_prompt_delta is not None, hook_fn is created via state-shift."""
        state = {"delta": None}
        if per_prompt_delta is not None:
            hook_fn = make_alpha_shift_hook(state, alpha=1.0)
        handle = layers[L_arg].register_forward_hook(hook_fn) if hook_fn else None
        ld = np.zeros(len(rows_subset), dtype=np.float32)
        try:
            for j, row in enumerate(rows_subset):
                sd = stim_root / row["stim_dir"]
                refs = [Image.open(sd / fn).convert("RGB")
                         for fn in row["ref_filenames"]]
                tgt = Image.open(sd / row["target_filename"]).convert("RGB")
                n_imgs = len(refs) + 1
                text = " ".join([img_token] * n_imgs) + " " + args.prompt
                inp = proc(text=text, images=refs + [tgt],
                            return_tensors="pt").to(model.device)
                if per_prompt_delta is not None:
                    state["delta"] = torch.tensor(per_prompt_delta[j],
                                                    dtype=torch.bfloat16,
                                                    device=model.device)
                logits = model(**inp, use_cache=False).logits[0, -1].float()
                ld[j] = float(logits[high_id] - logits[low_id])
        finally:
            if handle is not None:
                handle.remove()
        return ld

    results = {
        "model_id": args.model,
        "model_short": args.model_short,
        "layer": L_arg,
        "alphas": list(args.alphas),
        "n_train_seeds": n_train,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "primal_z_norm": float(np.linalg.norm(primal)),
        "manifold_norm_mean": float(norms.mean()),
        "settings": {},
    }

    def record(name, ld_arr):
        r_z = float(np.corrcoef(ld_arr, z_test)[0, 1])
        r_x = float(np.corrcoef(ld_arr, x_test)[0, 1])
        ld_mean = float(ld_arr.mean())
        # partial r(LD, z | x)
        r_zx = float(np.corrcoef(z_test, x_test)[0, 1])
        denom = np.sqrt(max(1e-12, (1 - r_x ** 2) * (1 - r_zx ** 2)))
        pc = (r_z - r_x * r_zx) / denom
        # per-x slopes
        px = per_x_slope(ld_arr, z_test, x_test)
        pos = sum(1 for v in px.values() if v["slope"] > 0)
        neg = sum(1 for v in px.values() if v["slope"] < 0)
        results["settings"][name] = {
            "corr_LD_z": r_z, "corr_LD_x": r_x, "partial_LD_z_given_x": pc,
            "ld_mean": ld_mean, "n_x_pos_slope": pos, "n_x_neg_slope": neg,
            "per_x": {str(k): v for k, v in px.items()},
        }
        print(f"[vab]   {name:<20} r_z={r_z:+.3f}  r_x={r_x:+.3f}  "
              f"pc(z|x)={pc:+.3f}  <LD>={ld_mean:+.2f}  "
              f"x-slopes: +{pos}/-{neg}", flush=True)

    # Baseline (no hook)
    print("\n[vab] === baseline ===")
    t1 = time.time()
    ld_b = fwd_LD(rows_test)
    record("baseline", ld_b)
    print(f"[vab]   ({time.time()-t1:.0f}s)")

    # α-sweep
    for alpha in args.alphas:
        if alpha == 0:
            continue
        print(f"\n[vab] === α={alpha} ===")
        t1 = time.time()
        ld = fwd_LD(rows_test, per_prompt_delta=alpha * Δ)
        record(f"manifold_a{alpha:.2f}", ld)
        print(f"[vab]   manifold ({time.time()-t1:.0f}s)")

        t1 = time.time()
        ld = fwd_LD(rows_test, hook_fn=make_partial_proj_hook(d_t, alpha))
        record(f"proj_out_a{alpha:.2f}", ld)
        print(f"[vab]   proj_out ({time.time()-t1:.0f}s)")

    out_path = out_dir / f"{out_name}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[vab] wrote {out_path}")


if __name__ == "__main__":
    main()
