"""Re-extract baseline LDs per stimulus for vphase_grid cells.

vphase_grid.py kept residuals + summary metrics, but not per-stim LD arrays
(needed for LD-vs-z scatter plots à la p2a_ld_vs_z). This script does a
single baseline forward pass per (model, n_ref) cell and writes a small
NPZ with per-stim x, z, seed, ld.

Output: results/vphase_grid/<cell>_baseline_lds.npz

No ablation, no residual capture — fast pass.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                            BitsAndBytesConfig)

# reuse stimulus-gen helpers from vphase_grid
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vphase_grid import (build_stim_images, gen_grid, first_token_id,
                          DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_SIGMA, CANVAS)

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["google/gemma-4-E2B-it",
                             "google/gemma-4-E4B-it",
                             "google/gemma-4-31B-it"])
    ap.add_argument("--shorts", nargs="+",
                    default=["gemma4-e2b-it", "gemma4-e4b-it", "gemma4-31b-it"])
    ap.add_argument("--n-refs", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--n-x", type=int, default=10)
    ap.add_argument("--n-z", type=int, default=10)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--quant-4bit-models", nargs="+", default=["gemma4-31b-it"])
    ap.add_argument("--high-word", default="big")
    ap.add_argument("--low-word", default="small")
    ap.add_argument("--prompt", default="The square in the last image is")
    ap.add_argument("--out-name", default="vphase_grid")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    out_root = REPO / "results" / args.out_name
    out_root.mkdir(parents=True, exist_ok=True)

    for model_id, short in zip(args.models, args.shorts):
        # Match the layer used by vphase_grid for the cell name
        # (we need this to match output naming convention; load matching cell JSONs)
        pending = []
        for n_ref in args.n_refs:
            existing = sorted(out_root.glob(f"{short}_n{n_ref}_L*.json"))
            if not existing:
                print(f"  skip {short} n={n_ref}: no matching cell JSON")
                continue
            cell_name = existing[0].stem
            out_path = out_root / f"{cell_name}_baseline_lds.npz"
            if args.skip_existing and out_path.exists():
                print(f"  skip {cell_name}: {out_path.name} exists")
                continue
            pending.append((n_ref, cell_name, out_path))

        if not pending:
            continue

        quant_4bit = short in args.quant_4bit_models
        print(f"\n=== loading {model_id} (4bit={quant_4bit}) ===", flush=True)
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(model_id,
                                              token=os.environ.get("HF_TOKEN"))
        load_kwargs: dict = dict(device_map="auto",
                                  token=os.environ.get("HF_TOKEN"))
        if quant_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else:
            load_kwargs["dtype"] = torch.bfloat16
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, **load_kwargs).eval()
        print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

        tok = proc.tokenizer
        high_id = first_token_id(tok, args.high_word)
        low_id = first_token_id(tok, args.low_word)
        img_token = getattr(proc, "image_token", "<|image|>")

        for n_ref, cell_name, out_path in pending:
            print(f"\n--- {cell_name} ---", flush=True)
            specs = gen_grid(args.n_x, args.n_z, args.n_seeds,
                              DEFAULT_X_MIN, DEFAULT_X_MAX, DEFAULT_SIGMA)
            stims = build_stim_images(specs, n_ref, CANVAS)
            n = len(stims)
            ld_arr = np.zeros(n, dtype=np.float32)
            high_log = np.zeros(n, dtype=np.float32)
            low_log = np.zeros(n, dtype=np.float32)
            t1 = time.time()
            for i, st in enumerate(stims):
                n_imgs = len(st["refs"]) + 1
                text = " ".join([img_token] * n_imgs) + " " + args.prompt
                with torch.inference_mode():
                    inp = proc(text=text, images=st["refs"] + [st["target"]],
                                return_tensors="pt").to(model.device)
                    logits = model(**inp, use_cache=False).logits[0, -1].float()
                    high_log[i] = float(logits[high_id])
                    low_log[i] = float(logits[low_id])
                    ld_arr[i] = high_log[i] - low_log[i]
                if (i + 1) % 50 == 0 or i == n - 1:
                    rate = (i + 1) / max(1e-3, time.time() - t1)
                    eta = (n - i - 1) / max(1e-3, rate)
                    print(f"  {i+1}/{n}  {rate:.2f} p/s  eta={eta:.0f}s",
                          flush=True)

            x_arr = np.array([s["x"] for s in stims], dtype=np.float32)
            z_arr = np.array([s["z"] for s in stims], dtype=np.float32)
            seed_arr = np.array([s["seed"] for s in stims], dtype=np.int32)
            ids = np.array([s["id"] for s in stims])
            np.savez(out_path, x=x_arr, z=z_arr, seed=seed_arr, ids=ids,
                      ld=ld_arr, high_logit=high_log, low_logit=low_log)
            r_z = float(np.corrcoef(ld_arr, z_arr)[0, 1])
            r_x = float(np.corrcoef(ld_arr, x_arr)[0, 1])
            print(f"  saved {out_path.name}  "
                  f"r_z={r_z:+.3f}  r_x={r_x:+.3f}  <LD>={ld_arr.mean():+.2f}",
                  flush=True)

        del model, proc
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
