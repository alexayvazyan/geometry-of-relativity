"""Canonical model IDs, global seed, and circuit constants.

Domain and prompt specifications (the 8 adjective pairs, their sigma, ranges, and templates) are
the single responsibility of ``data_gen/prompts.py`` (``PAIRS``). Import them from there rather
than redefining anything here — this module is only about *which model* and *which seed*.
"""

from __future__ import annotations

# Global RNG seed. Context samples are additionally seeded per (domain, mu, seed) cell so the
# implicit context is deterministic across runs; see data_gen/prompts.py::sample_context.
SEED = 0

# Primary model for all main-text results.
PRIMARY_MODEL = "google/gemma-2-9b"

# HF ids. gemma-2-2b is the legacy model some early figures were produced on.
MODELS: dict[str, str] = {
    "gemma-2-9b": "google/gemma-2-9b",      # primary
    "gemma-2-2b": "google/gemma-2-2b",      # legacy / early figures
    # Cross-model replication (appendix xmodel grids)
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "olmo-2-7b": "allenai/OLMo-2-1124-7B",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "qwen3-14b": "Qwen/Qwen3-14B",
}

# Layer where the shared z-direction / manifold steering is applied on gemma-2-9b.
STEER_LAYER_9B = 33

# DLA-ranked causal trio on gemma-2-9b, as (layer, head).
CAUSAL_TRIO_9B = [(23, 14), (26, 4), (31, 3)]
