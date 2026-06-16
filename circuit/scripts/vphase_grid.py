"""Vision phase grid — shot-count × model sweep on Gemma 4 IT family.

For each (model, n_ref) cell:
  1. Generate stim grid in-memory (n_x × n_z × n_seeds, plausibility filtered).
  2. Forward pass per stimulus → residual at chosen layer + big/small logits.
  3. Train/test split by seed; build primal_z + per-x manifold lookup.
  4. α-sweep on test fold: baseline, manifold-shift, proj_out at α ∈ args.alphas.
  5. Save per-cell JSON with r_z, r_x, partial r(z|x), <LD>, per-x slope counts.

Outputs:
  results/<out_name>/<model_short>_n{n_ref}_L{layer}.json
  results/<out_name>/<model_short>_n{n_ref}_L{layer}_residuals.npz
  results/<out_name>/aggregate.json  (after final cell)

Defaults: models {E2B-it, E4B-it, 31B-it}, n_refs {1, 4, 8}, alphas {0.5, 0.75, 1.0}.
31B-it loaded in 4-bit by default (bf16 OOMs).

Compute budget on RTX 5090: ~1hr for full 3×3 grid (E2B fast, E4B medium,
31B slow at 4-bit). OOM cells are recorded with error and skipped.

Usage:
    python vphase_grid.py                                 # full sweep
    python vphase_grid.py --models google/gemma-4-E2B-it \\
        --shorts gemma4-e2b-it --n-refs 1 --limit-stim 20  # smoke
    python vphase_grid.py --skip-completed                 # resume
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                            BitsAndBytesConfig)

REPO = Path(__file__).resolve().parent.parent

# ---------- stimulus generation (mirrors vstim_seq.py) ----------

CANVAS = 224
DEFAULT_X_MIN = 16
DEFAULT_X_MAX = 96
DEFAULT_SIGMA = 12.0
Z_MIN = -2.5
Z_MAX = +2.5
REF_SIZE_MIN = 4
REF_SIZE_MAX = 140
PLAUSIBILITY_K_SIGMA = 2.0


@dataclass(frozen=True)
class StimSpec:
    x: int
    mu: float
    sigma: float
    z: float
    seed: int


def make_square_frame(canvas: int, side: int, bg: int = 255, fg: int = 0) -> Image.Image:
    arr = np.full((canvas, canvas), bg, dtype=np.uint8)
    x0 = (canvas - side) // 2
    arr[x0:x0 + side, x0:x0 + side] = fg
    return Image.fromarray(arr, mode="L").convert("RGB")


def sample_ref_sizes(mu: float, sigma: float, n_ref: int,
                     rng: random.Random) -> list[int]:
    out: list[int] = []
    for _ in range(n_ref):
        v = rng.gauss(mu, sigma)
        v = max(REF_SIZE_MIN, min(REF_SIZE_MAX, int(round(v))))
        out.append(v)
    return out


def gen_grid(n_x: int, n_z: int, n_seeds: int, x_min: int, x_max: int,
             sigma: float) -> list[StimSpec]:
    xs = np.linspace(x_min, x_max, n_x).round().astype(int)
    zs = np.linspace(Z_MIN, Z_MAX, n_z).round(2)
    specs: list[StimSpec] = []
    for x in xs:
        for z in zs:
            mu = float(x) - sigma * z
            lo = mu - PLAUSIBILITY_K_SIGMA * sigma
            hi = mu + PLAUSIBILITY_K_SIGMA * sigma
            if not (lo >= REF_SIZE_MIN and hi <= REF_SIZE_MAX):
                continue
            for seed in range(n_seeds):
                specs.append(StimSpec(int(x), mu, sigma, float(z), seed))
    return specs


def build_stim_images(specs: list[StimSpec], n_ref: int, canvas: int) -> list[dict]:
    """Returns list of dicts with PIL images and metadata."""
    out: list[dict] = []
    for i, sp in enumerate(specs):
        rng = random.Random(0xBEEF + sp.seed * 31337 + i)
        ref_sides = sample_ref_sizes(sp.mu, sp.sigma, n_ref, rng)
        refs = [make_square_frame(canvas, s) for s in ref_sides]
        tgt = make_square_frame(canvas, sp.x)
        out.append({
            "refs": refs, "target": tgt, "x": sp.x, "z": sp.z,
            "seed": sp.seed, "mu": sp.mu, "sigma": sp.sigma,
            "ref_sizes": ref_sides,
            "id": f"x{sp.x:03d}_z{sp.z:+.2f}_s{sp.seed}_{i:05d}",
        })
    return out


# ---------- model utilities ----------

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
                ok = False
                break
        if ok and hasattr(m, "__getitem__"):
            return m
    raise RuntimeError("could not locate decoder layers")


def first_token_id(tok, w: str) -> int:
    return tok.encode(" " + w, add_special_tokens=False)[-1]


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


# ---------- ablation hooks ----------

def make_proj_hook(d_t, alpha: float):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        proj = (h * d_t).sum(-1, keepdim=True)
        h = h - alpha * proj * d_t
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


def make_shift_hook(state):
    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + state["delta"][None, None, :]
        return (h,) + output[1:] if isinstance(output, tuple) else h
    return hook


# ---------- diagnostics ----------

def per_x_slope_summary(ld, z, x):
    cells: dict = defaultdict(list)
    for i in range(len(ld)):
        cells[(round(float(x[i]), 4), round(float(z[i]), 4))].append(float(ld[i]))
    cell = {k: float(np.mean(v)) for k, v in cells.items()}
    by_x: dict = defaultdict(list)
    for (xk, zk), m in cell.items():
        by_x[xk].append((zk, m))
    pos = neg = 0
    for xk, vals in by_x.items():
        if len(vals) < 3:
            continue
        zs = np.array([t[0] for t in vals])
        ms = np.array([t[1] for t in vals])
        slope = float(np.polyfit(zs, ms, 1)[0])
        if slope > 0:
            pos += 1
        elif slope < 0:
            neg += 1
    return pos, neg


# ---------- per-cell pipeline ----------

def run_cell(model, proc, layers, layer_idx: int, stims: list[dict],
             alphas: list[float], n_train_seeds: int, prompt: str,
             high_id: int, low_id: int, log_prefix: str = ""):
    img_token = getattr(proc, "image_token", "<|image|>")
    n = len(stims)

    # capture last-token residual at layer_idx via forward hook
    captured = {"h": None}

    def cap_hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h[0, -1].detach().float().cpu().numpy()
        return output

    h_buf: list[np.ndarray] = []
    high_log = np.zeros(n, dtype=np.float32)
    low_log = np.zeros(n, dtype=np.float32)

    handle = layers[layer_idx].register_forward_hook(cap_hook)
    t0 = time.time()
    try:
        for i, st in enumerate(stims):
            n_imgs = len(st["refs"]) + 1
            text = " ".join([img_token] * n_imgs) + " " + prompt
            with torch.inference_mode():
                inp = proc(text=text, images=st["refs"] + [st["target"]],
                            return_tensors="pt").to(model.device)
                out = model(**inp, use_cache=False)
                h_buf.append(captured["h"].copy())
                logits = out.logits[0, -1].float()
                high_log[i] = float(logits[high_id])
                low_log[i] = float(logits[low_id])
            if (i + 1) % 50 == 0 or i == n - 1:
                rate = (i + 1) / max(1e-3, time.time() - t0)
                eta = (n - i - 1) / max(1e-3, rate)
                print(f"{log_prefix}extract {i+1}/{n}  {rate:.2f} p/s  eta={eta:.0f}s",
                      flush=True)
    finally:
        handle.remove()

    H = np.stack(h_buf)
    z_arr = np.array([s["z"] for s in stims], dtype=np.float64)
    x_arr = np.array([s["x"] for s in stims], dtype=np.float64)
    seed_arr = np.array([s["seed"] for s in stims], dtype=np.int32)

    train_mask = seed_arr < n_train_seeds
    test_mask = ~train_mask
    test_idx = np.where(test_mask)[0]
    z_test = z_arr[test_idx]
    x_test = x_arr[test_idx]

    # primal_z from train
    z_tr = z_arr[train_mask]
    H_tr = H[train_mask]
    primal = H_tr[z_tr > +1.0].mean(0) - H_tr[z_tr < -1.0].mean(0)
    primal_norm = float(np.linalg.norm(primal))

    # per-x manifold lookup (target z=0)
    cells: dict = defaultdict(list)
    for i in np.where(train_mask)[0]:
        key = (round(float(x_arr[i]), 4), round(float(z_arr[i]), 4))
        cells[key].append(H[i])
    lookup = {k: np.stack(v).mean(0) for k, v in cells.items()}
    by_x: dict = defaultdict(list)
    for (xk, zk), v in lookup.items():
        by_x[xk].append((zk, v))
    Δ = np.zeros((len(test_idx), H.shape[1]), dtype=np.float64)
    n_missing = 0
    for j, i in enumerate(test_idx):
        xi = round(float(x_arr[i]), 4)
        zi = round(float(z_arr[i]), 4)
        same_x = by_x.get(xi, [])
        if not same_x:
            n_missing += 1
            continue
        _, mu_tgt = min(same_x, key=lambda kv: abs(kv[0]))
        _, mu_src = min(same_x, key=lambda kv: abs(kv[0] - zi))
        Δ[j] = mu_tgt - mu_src

    d_t = torch.tensor(unit(primal), dtype=torch.bfloat16, device=model.device)

    def fwd_test(hook_fn=None, per_prompt_delta: np.ndarray | None = None):
        state = {"delta": None}
        if per_prompt_delta is not None:
            hook_fn = make_shift_hook(state)
        h_handle = (layers[layer_idx].register_forward_hook(hook_fn)
                     if hook_fn is not None else None)
        ld = np.zeros(len(test_idx), dtype=np.float32)
        try:
            for j, i in enumerate(test_idx):
                st = stims[i]
                n_imgs = len(st["refs"]) + 1
                text = " ".join([img_token] * n_imgs) + " " + prompt
                if per_prompt_delta is not None:
                    state["delta"] = torch.tensor(per_prompt_delta[j],
                                                    dtype=torch.bfloat16,
                                                    device=model.device)
                with torch.inference_mode():
                    inp = proc(text=text, images=st["refs"] + [st["target"]],
                                return_tensors="pt").to(model.device)
                    logits = model(**inp, use_cache=False).logits[0, -1].float()
                    ld[j] = float(logits[high_id] - logits[low_id])
        finally:
            if h_handle is not None:
                h_handle.remove()
        return ld

    def metrics(ld_arr: np.ndarray) -> dict:
        r_z = float(np.corrcoef(ld_arr, z_test)[0, 1])
        r_x = float(np.corrcoef(ld_arr, x_test)[0, 1])
        r_zx = float(np.corrcoef(z_test, x_test)[0, 1])
        denom = math.sqrt(max(1e-12, (1 - r_x ** 2) * (1 - r_zx ** 2)))
        pc = (r_z - r_x * r_zx) / denom
        pos, neg = per_x_slope_summary(ld_arr, z_test, x_test)
        return {
            "corr_LD_z": r_z, "corr_LD_x": r_x, "partial_LD_z_given_x": pc,
            "ld_mean": float(ld_arr.mean()), "ld_std": float(ld_arr.std()),
            "n_x_pos_slope": pos, "n_x_neg_slope": neg,
        }

    settings: dict = {}
    print(f"{log_prefix}baseline...", flush=True)
    t1 = time.time()
    ld_b = fwd_test()
    settings["baseline"] = metrics(ld_b)
    m = settings["baseline"]
    print(f"{log_prefix}  baseline  r_z={m['corr_LD_z']:+.3f}  r_x={m['corr_LD_x']:+.3f}  "
          f"pc={m['partial_LD_z_given_x']:+.3f}  <LD>={m['ld_mean']:+.2f}  "
          f"+{m['n_x_pos_slope']}/-{m['n_x_neg_slope']}  ({time.time()-t1:.0f}s)",
          flush=True)

    for a in alphas:
        if a == 0:
            continue
        t1 = time.time()
        ld_m = fwd_test(per_prompt_delta=a * Δ)
        settings[f"manifold_a{a:.2f}"] = metrics(ld_m)
        m = settings[f"manifold_a{a:.2f}"]
        print(f"{log_prefix}  manifold a={a}  r_z={m['corr_LD_z']:+.3f}  "
              f"pc={m['partial_LD_z_given_x']:+.3f}  <LD>={m['ld_mean']:+.2f}  "
              f"+{m['n_x_pos_slope']}/-{m['n_x_neg_slope']}  ({time.time()-t1:.0f}s)",
              flush=True)

        t1 = time.time()
        ld_p = fwd_test(hook_fn=make_proj_hook(d_t, a))
        settings[f"proj_out_a{a:.2f}"] = metrics(ld_p)
        m = settings[f"proj_out_a{a:.2f}"]
        print(f"{log_prefix}  proj_out a={a}  r_z={m['corr_LD_z']:+.3f}  "
              f"pc={m['partial_LD_z_given_x']:+.3f}  <LD>={m['ld_mean']:+.2f}  "
              f"+{m['n_x_pos_slope']}/-{m['n_x_neg_slope']}  ({time.time()-t1:.0f}s)",
              flush=True)

    return ({
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_missing": int(n_missing),
        "primal_z_norm": primal_norm,
        "manifold_norm_mean": float(np.linalg.norm(Δ, axis=1).mean()),
        "settings": settings,
    }, H, z_arr, x_arr, seed_arr)


# ---------- main ----------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["google/gemma-4-E2B-it",
                             "google/gemma-4-E4B-it",
                             "google/gemma-4-31B-it"])
    ap.add_argument("--shorts", nargs="+",
                    default=["gemma4-e2b-it", "gemma4-e4b-it", "gemma4-31b-it"])
    ap.add_argument("--n-refs", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--n-x", type=int, default=10)
    ap.add_argument("--n-z", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--n-train-seeds", type=int, default=3)
    ap.add_argument("--layer-frac", type=float, default=0.78,
                    help="layer index = round(frac * (n_layers - 1)); "
                         "matches existing E4B-it L33 (33/42 = 0.785)")
    ap.add_argument("--layer-overrides", nargs="+", default=["gemma4-31b-it=56"],
                    help="model_short=L pairs; default pins 31B-it to L56 "
                         "(matches Phase 1 vision result)")
    ap.add_argument("--quant-4bit-models", nargs="+", default=["gemma4-31b-it"])
    ap.add_argument("--high-word", default="big")
    ap.add_argument("--low-word", default="small")
    ap.add_argument("--prompt", default="The square in the last image is")
    ap.add_argument("--out-name", default="vphase_grid")
    ap.add_argument("--skip-completed", action="store_true",
                    help="re-use existing per-cell JSONs (resume mode)")
    ap.add_argument("--limit-stim", type=int, default=None,
                    help="cap N stimuli (smoke testing)")
    args = ap.parse_args()

    if len(args.models) != len(args.shorts):
        raise SystemExit("--models and --shorts must be same length")

    out_root = REPO / "results" / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    layer_overrides: dict[str, int] = {}
    for kv in args.layer_overrides:
        k, v = kv.split("=")
        layer_overrides[k] = int(v)

    aggregate = {"args": vars(args), "cells": []}

    for model_id, short in zip(args.models, args.shorts):
        quant_4bit = short in args.quant_4bit_models
        print(f"\n=== loading {model_id} (4bit={quant_4bit}) ===", flush=True)
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(
            model_id, token=os.environ.get("HF_TOKEN"))
        load_kwargs: dict = dict(device_map="auto",
                                  token=os.environ.get("HF_TOKEN"))
        if quant_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            load_kwargs["dtype"] = torch.bfloat16
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, **load_kwargs).eval()
        except Exception as e:
            print(f"  FAILED to load {model_id}: {e}", flush=True)
            continue
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
        layers = get_decoder_layers(model)
        n_layers = len(layers)
        layer_idx = layer_overrides.get(
            short, int(round(args.layer_frac * (n_layers - 1))))
        layer_idx = max(0, min(n_layers - 1, layer_idx))
        print(f"  n_layers={n_layers}, hooking decoder layer L{layer_idx}",
              flush=True)

        tok = proc.tokenizer
        high_id = first_token_id(tok, args.high_word)
        low_id = first_token_id(tok, args.low_word)

        for n_ref in args.n_refs:
            cell_name = f"{short}_n{n_ref}_L{layer_idx}"
            cell_path = out_root / f"{cell_name}.json"
            if args.skip_completed and cell_path.exists():
                print(f"\n--- {cell_name}: SKIP (exists) ---", flush=True)
                aggregate["cells"].append(json.loads(cell_path.read_text()))
                continue
            print(f"\n--- {cell_name} ---", flush=True)
            specs = gen_grid(args.n_x, args.n_z, args.n_seeds,
                              DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_SIGMA)
            if args.limit_stim:
                specs = specs[:args.limit_stim]
            print(f"  {len(specs)} stimuli at n_ref={n_ref}", flush=True)
            stims = build_stim_images(specs, n_ref, CANVAS)

            try:
                cell, H, z_arr, x_arr, seed_arr = run_cell(
                    model, proc, layers, layer_idx, stims, args.alphas,
                    args.n_train_seeds, args.prompt, high_id, low_id,
                    log_prefix=f"  [{cell_name}] ")
            except torch.cuda.OutOfMemoryError as e:
                print(f"  OOM on {cell_name}: {e}", flush=True)
                torch.cuda.empty_cache()
                rec = {"name": cell_name, "model_id": model_id,
                        "model_short": short, "n_ref": n_ref, "layer": layer_idx,
                        "n_layers": n_layers,
                        "result": {"error": "OOM", "msg": str(e)}}
                cell_path.write_text(json.dumps(rec, indent=2))
                aggregate["cells"].append(rec)
                continue

            rec = {"name": cell_name, "model_id": model_id,
                    "model_short": short, "n_ref": n_ref, "layer": layer_idx,
                    "n_layers": n_layers, "result": cell}
            cell_path.write_text(json.dumps(rec, indent=2))
            aggregate["cells"].append(rec)
            np.savez(out_root / f"{cell_name}_residuals.npz",
                     H=H.astype(np.float16), z=z_arr, x=x_arr, seed=seed_arr,
                     layer=layer_idx)
            print(f"  saved {cell_path.name}", flush=True)
            torch.cuda.empty_cache()

        del model, proc
        gc.collect()
        torch.cuda.empty_cache()

    (out_root / "aggregate.json").write_text(json.dumps(aggregate, indent=2))
    print(f"\nwrote aggregate to {out_root / 'aggregate.json'}")


if __name__ == "__main__":
    main()
