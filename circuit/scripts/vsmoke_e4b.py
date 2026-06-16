"""Vision V2 — smoke test on Gemma 4 E4B.

For two opposite-z stimuli (z very negative, z very positive at same x),
load E4B as image-text model, pass the 8 references + target with a
"the square in the last image is" prompt, decode top-K next tokens, and
compute LD = logit(big) − logit(small).

If LD shifts strongly in the predicted direction across the two stimuli
(LD lower for z=−2.5, LD higher for z=+2.5), the relativity signal is
present in vision and we can scale up.

Also probes the decoder-layer path (model.model.language_model.layers per
Alex's existing 31B work) so we know what layer-index to hook later.
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
from transformers import AutoProcessor, AutoModelForImageTextToText

REPO = Path(__file__).resolve().parent.parent

CANDIDATE_HIGH = ["big", "large", "huge"]
CANDIDATE_LOW = ["small", "tiny", "little"]


def load_stim(stim_root: Path, row: dict) -> tuple[list[Image.Image], Image.Image]:
    sd = stim_root / row["stim_dir"]
    refs = [Image.open(sd / fn).convert("RGB") for fn in row["ref_filenames"]]
    tgt = Image.open(sd / row["target_filename"]).convert("RGB")
    return refs, tgt


def first_token_id(tok, word: str) -> int:
    """Token id when the word appears as a continuation (with leading space)."""
    ids = tok.encode(" " + word, add_special_tokens=False)
    return ids[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B")
    ap.add_argument("--stim", default="stimuli/vsize_v0/stimuli.jsonl")
    ap.add_argument("--n-pairs", type=int, default=4,
                    help="how many extreme-pair stimuli to test")
    args = ap.parse_args()

    print(f"[vsmoke] loading {args.model}...", flush=True)
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto").eval()
    print(f"[vsmoke]   loaded in {time.time()-t0:.1f}s", flush=True)

    # Decoder-layer path discovery
    paths_to_try = [
        ("model.model.language_model.layers",
         lambda m: m.model.language_model.layers),
        ("model.language_model.layers", lambda m: m.language_model.layers),
        ("model.model.layers", lambda m: m.model.layers),
    ]
    layers = None
    for name, fn in paths_to_try:
        try:
            layers = fn(model)
            print(f"[vsmoke] decoder layers found at: {name} (n={len(layers)})")
            break
        except AttributeError:
            continue
    if layers is None:
        raise RuntimeError("could not locate decoder layers")

    img_token = getattr(proc, "image_token", "<|image|>")
    tok = proc.tokenizer

    # Decide token ids for high/low candidates
    print(f"\n[vsmoke] token ids for candidate adjectives:")
    for w in CANDIDATE_HIGH + CANDIDATE_LOW:
        tid = first_token_id(tok, w)
        decoded = tok.decode([tid])
        print(f"  {w:>8} -> id={tid}  decoded={decoded!r}")

    # Load manifest
    manifest_path = REPO / args.stim
    rows = [json.loads(line) for line in manifest_path.open()]
    # Sort by (x, z) so we can pick extremes
    rows_by_x: dict[int, list[dict]] = {}
    for r in rows:
        rows_by_x.setdefault(r["x"], []).append(r)
    # Pick stimuli where the same x has both z<0 and z>0 in the grid
    extreme_pairs = []
    for x, group in rows_by_x.items():
        zs_avail = sorted({r["z"] for r in group})
        if not zs_avail or min(zs_avail) >= 0 or max(zs_avail) <= 0:
            continue
        z_low = min(zs_avail)
        z_high = max(zs_avail)
        # Use seed=0 from each
        r_low = next(r for r in group if r["z"] == z_low and r["seed"] == 0)
        r_high = next(r for r in group if r["z"] == z_high and r["seed"] == 0)
        extreme_pairs.append((r_low, r_high))
    extreme_pairs = extreme_pairs[: args.n_pairs]

    print(f"\n[vsmoke] testing {len(extreme_pairs)} (x_fixed, z_low, z_high) triples...")

    stim_root = (REPO / args.stim).parent

    @torch.no_grad()
    def fwd_logits(refs, tgt, prompt_text, mode="inline"):
        """mode='inline' uses raw img-token strings; mode='chat' uses
        apply_chat_template with image content blocks."""
        n_imgs = len(refs) + 1
        if mode == "inline":
            text = " ".join([img_token] * n_imgs) + " " + prompt_text
            inp = proc(text=text, images=refs + [tgt],
                        return_tensors="pt").to(model.device)
        elif mode == "chat":
            content = [{"type": "image", "image": img} for img in refs + [tgt]]
            content.append({"type": "text", "text": prompt_text})
            messages = [{"role": "user", "content": content}]
            inp = proc.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
        else:
            raise ValueError(mode)
        out = model(**inp, use_cache=False)
        logits = out.logits[0, -1].float().cpu().numpy()
        return logits

    high_id = first_token_id(tok, "big")
    low_id  = first_token_id(tok, "small")

    PROMPT_VARIANTS = [
        ("inline_simple", "inline",
          "The square in the last image is"),
        ("inline_explicit", "inline",
          "Above are several images of black squares of varying sizes. "
          "Compared to the others, the square in the last image is"),
        ("chat_compare", "chat",
          "Compared to the previous squares, the square in the last image is"),
        ("chat_relative", "chat",
          "Looking at the squares above, the final square is relatively"),
    ]

    for v_name, mode, prompt in PROMPT_VARIANTS:
        print(f"\n[vsmoke] === variant: {v_name}  (mode={mode}) ===")
        print(f"[vsmoke] prompt: {prompt!r}")
        print(f"{'x':>4} {'z_low':>6} {'z_high':>7} {'LD@low':>9} {'LD@high':>9} "
              f"{'ΔLD':>7}")
        first = True
        for r_low, r_high in extreme_pairs:
            refs_low, tgt_low = load_stim(stim_root, r_low)
            refs_high, tgt_high = load_stim(stim_root, r_high)
            try:
                logits_low = fwd_logits(refs_low, tgt_low, prompt, mode=mode)
                logits_high = fwd_logits(refs_high, tgt_high, prompt, mode=mode)
            except Exception as e:
                print(f"  variant {v_name} failed: {e}")
                break
            LD_low = float(logits_low[high_id] - logits_low[low_id])
            LD_high = float(logits_high[high_id] - logits_high[low_id])
            print(f"{r_low['x']:>4} {r_low['z']:>+6.2f} {r_high['z']:>+7.2f} "
                  f"{LD_low:>+9.3f} {LD_high:>+9.3f} {LD_high - LD_low:>+7.3f}")
            if first:
                top5 = np.argsort(-logits_low)[:5]
                tops = "  ".join(f"{tok.decode([int(tid)])!r}={logits_low[tid]:.1f}"
                                  for tid in top5)
                print(f"  top5@first[z_low]: {tops}")
                first = False


if __name__ == "__main__":
    main()
