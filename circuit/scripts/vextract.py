"""Vision V3 — extract residuals + LD on full vision-relativity grid.

Loads a Gemma 4 multimodal model. For each stimulus in stimuli.jsonl,
builds an inline-image prompt, captures last-token residuals at every
decoder layer, and reads out high/low logits (default big/small).

Outputs:
  results/<out_name>_residuals.npz  -- activations[N, n_layers, d_model] fp16
                                       + x, z, mu, sigma, seed, ld_logits
  results/<out_name>_meta.json      -- model meta + tokenization info
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (AutoProcessor, AutoModelForImageTextToText,
                            BitsAndBytesConfig)

REPO = Path(__file__).resolve().parent.parent


def get_decoder_layers(model):
    """Locate the decoder-layer module list across model variants."""
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
            return m, ".".join(path)
    raise RuntimeError("could not locate decoder layers")


def first_token_id(tok, word: str) -> int:
    return tok.encode(" " + word, add_special_tokens=False)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-short", required=True)
    ap.add_argument("--stim", default="stimuli/vsize_v0/stimuli.jsonl")
    ap.add_argument("--prompt",
                    default="The square in the last image is")
    ap.add_argument("--high-word", default="big")
    ap.add_argument("--low-word", default="small")
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap N stimuli for testing")
    ap.add_argument("--quant-4bit", action="store_true",
                    help="load model in 4-bit (bitsandbytes) — needed for 31B-it on 32GB VRAM")
    args = ap.parse_args()

    out_name = args.out_name or f"vextract_{args.model_short}"
    out_dir = REPO / "results"
    out_dir.mkdir(exist_ok=True)

    print(f"[vext] loading {args.model} (4bit={args.quant_4bit})...", flush=True)
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
    print(f"[vext]   loaded in {time.time() - t0:.1f}s", flush=True)

    layers, layer_path = get_decoder_layers(model)
    n_layers = len(layers)
    print(f"[vext]   decoder layers at {layer_path} (n={n_layers})")

    # Discover d_model via a probe forward pass after we have a stim
    img_token = getattr(proc, "image_token", "<|image|>")
    tok = proc.tokenizer
    high_id = first_token_id(tok, args.high_word)
    low_id = first_token_id(tok, args.low_word)
    print(f"[vext]   high='{args.high_word}'(id={high_id})  "
          f"low='{args.low_word}'(id={low_id})")

    # Load stimuli
    stim_path = REPO / args.stim
    rows = [json.loads(line) for line in stim_path.open()]
    if args.limit:
        rows = rows[:args.limit]
    n = len(rows)
    stim_root = stim_path.parent
    print(f"[vext]   {n} stimuli from {stim_path}")

    # Forward pass loop — batch size 1 (multimodal inputs are heterogeneous)
    activations: list[np.ndarray] = []  # each (n_layers+1, d_model)
    high_logits = np.zeros(n, dtype=np.float32)
    low_logits = np.zeros(n, dtype=np.float32)
    next_argmax = np.zeros(n, dtype=np.int32)

    t1 = time.time()
    for i, row in enumerate(rows):
        sd = stim_root / row["stim_dir"]
        refs = [Image.open(sd / fn).convert("RGB") for fn in row["ref_filenames"]]
        tgt = Image.open(sd / row["target_filename"]).convert("RGB")
        n_imgs = len(refs) + 1
        text = " ".join([img_token] * n_imgs) + " " + args.prompt

        with torch.inference_mode():
            inp = proc(text=text, images=refs + [tgt],
                        return_tensors="pt").to(model.device)
            out = model(**inp, output_hidden_states=True, use_cache=False)
            # last token across all hidden states (includes embedding + n_layers)
            hs = torch.stack([h[0, -1] for h in out.hidden_states])
            activations.append(hs.float().cpu().numpy().astype(np.float16))
            logits_last = out.logits[0, -1].float()
            high_logits[i] = float(logits_last[high_id])
            low_logits[i] = float(logits_last[low_id])
            next_argmax[i] = int(logits_last.argmax().item())

        if (i + 1) % 25 == 0 or i == n - 1:
            rate = (i + 1) / max(1e-3, time.time() - t1)
            eta = (n - i - 1) / max(1e-3, rate)
            print(f"[vext]   {i+1}/{n}  rate={rate:.2f} p/s  eta={eta:.0f}s",
                  flush=True)

    # Stack to (N, n_hidden, d_model). hidden_states has length n_layers+1.
    A = np.stack(activations)  # (N, n_layers+1, d_model)
    print(f"[vext] activations shape: {A.shape}")

    ids = np.array([r["id"] for r in rows])
    xs = np.array([r["x"] for r in rows], dtype=np.float32)
    zs = np.array([r["z"] for r in rows], dtype=np.float32)
    mus = np.array([r["mu"] for r in rows], dtype=np.float32)
    sigmas = np.array([r["sigma"] for r in rows], dtype=np.float32)
    seeds = np.array([r["seed"] for r in rows], dtype=np.int32)
    ld = high_logits - low_logits

    out_npz = out_dir / f"{out_name}_residuals.npz"
    np.savez(out_npz,
             activations=A,
             ids=ids, x=xs, z=zs, mu=mus, sigma=sigmas, seed=seeds,
             high_logit=high_logits, low_logit=low_logits, ld=ld,
             next_argmax=next_argmax,
             layer_indices=np.arange(A.shape[1]))
    print(f"[vext] wrote {out_npz}  ({out_npz.stat().st_size/1e6:.1f} MB)")

    meta = {
        "model_id": args.model,
        "model_short": args.model_short,
        "n_stimuli": n,
        "n_layers": n_layers,
        "d_model": int(A.shape[2]),
        "high_word": args.high_word,
        "low_word": args.low_word,
        "high_id": high_id,
        "low_id": low_id,
        "prompt": args.prompt,
        "stim_path": str(stim_path),
        "decoder_layers_path": layer_path,
        "elapsed_sec": round(time.time() - t1, 1),
    }
    (out_dir / f"{out_name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[vext] meta: {meta}")

    # Behavioral baseline
    r_z = float(np.corrcoef(ld, zs)[0, 1])
    r_x = float(np.corrcoef(ld, xs)[0, 1])
    print(f"\n[vext] BEHAVIORAL  corr(LD, z) = {r_z:+.3f}")
    print(f"[vext] BEHAVIORAL  corr(LD, x) = {r_x:+.3f}")
    print(f"[vext] LD mean = {float(ld.mean()):+.2f}  std = {float(ld.std()):+.2f}")


if __name__ == "__main__":
    main()
