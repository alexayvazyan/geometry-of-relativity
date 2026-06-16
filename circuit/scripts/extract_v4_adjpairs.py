"""Compatibility shim.

The circuit scripts were written against `extract_v4_adjpairs` in the old repo. The canonical
prompt engine now lives in `data_gen/prompts.py`; this module re-exports the symbols those scripts
import so they run self-contained in this repo.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # repo root, for data_gen

from data_gen.prompts import (  # noqa: E402
    LOG_SPACE_PAIRS,
    PAIRS,
    PAIRS_BY_NAME,
    Pair,
    build_implicit_items,
    compute_z,
    derive_mu,
    fmt_num,
    make_explicit_prompt,
    make_implicit_prompt,
    make_zero_shot_prompt,
    sample_context,
)

__all__ = [
    "PAIRS", "PAIRS_BY_NAME", "LOG_SPACE_PAIRS", "Pair", "build_implicit_items",
    "compute_z", "derive_mu", "fmt_num", "sample_context",
    "make_implicit_prompt", "make_explicit_prompt", "make_zero_shot_prompt",
]
