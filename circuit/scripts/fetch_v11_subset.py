"""Pull the subset of v11 NPZs and trial JSONL files needed for Phase 0/1.

The geometry-of-relativity repo's fetch_from_hf.py FOLDERS dict does not
include v11; this script does the equivalent fetch directly via
snapshot_download with allow_patterns. Symlinks files into the canonical
locations expected by p0_repro_shared_z.py.

Run:
    python fetch_v11_subset.py --pairs height weight speed \\
                               --models gemma2-2b
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "xrong1729/mech-interp-relativity-activations"
GOR_ROOT = Path("/home/alexander/research_projects/geometry-of-relativity")


def fetch_pair(model_short: str, pair: str, token: str | None) -> int:
    """Pull <pair> NPZ for one model. Returns number of files linked."""
    hf_path_glob = f"v11/{model_short}/{pair}/*.npz"
    cache_root = snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=[hf_path_glob], token=token,
    )
    src_dir = Path(cache_root) / "v11" / model_short / pair
    if not src_dir.exists():
        print(f"  [no match] {hf_path_glob}")
        return 0
    dst_dir = GOR_ROOT / "results" / "v11" / model_short / pair
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_dir.iterdir():
        if not f.is_file():
            continue
        dst = dst_dir / f.name
        if dst.exists() or dst.is_symlink():
            continue
        os.symlink(f.resolve(), dst)
        n += 1
    return n


def fetch_trial_jsonl(pair: str, token: str | None) -> int:
    """Pull v11_<pair>_trials.jsonl into geometry-of-relativity/data_gen/."""
    glob = f"prompts/v11_{pair}_trials.jsonl"
    cache_root = snapshot_download(
        repo_id=REPO_ID, repo_type="dataset",
        allow_patterns=[glob], token=token,
    )
    src = Path(cache_root) / "prompts" / f"v11_{pair}_trials.jsonl"
    if not src.exists():
        print(f"  [no match] {glob}")
        return 0
    dst_dir = GOR_ROOT / "data_gen"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"v11_{pair}_trials.jsonl"
    if dst.exists() or dst.is_symlink():
        return 0
    os.symlink(src.resolve(), dst)
    return 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+",
                    default=["height", "weight", "speed"],
                    help="adjective pairs to fetch (subset of the 8)")
    ap.add_argument("--models", nargs="+",
                    default=["gemma2-2b"],
                    choices=["gemma2-2b", "gemma2-9b"])
    ap.add_argument("--also-jsonl", action="store_true",
                    help="also fetch the prompts/v11_<pair>_trials.jsonl files")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if token is None:
        # Fall back to local cached token
        token_file = Path("~/.cache/huggingface/token").expanduser()
        if token_file.exists():
            token = token_file.read_text().strip()

    total_npz = 0
    for model in args.models:
        for pair in args.pairs:
            print(f"fetching v11/{model}/{pair}/*.npz")
            n = fetch_pair(model, pair, token)
            print(f"  linked {n} files")
            total_npz += n

    total_jsonl = 0
    if args.also_jsonl:
        for pair in args.pairs:
            print(f"fetching prompts/v11_{pair}_trials.jsonl")
            n = fetch_trial_jsonl(pair, token)
            print(f"  linked {n} files")
            total_jsonl += n

    print(f"\ndone. {total_npz} NPZs and {total_jsonl} JSONLs linked.")


if __name__ == "__main__":
    main()
