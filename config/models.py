"""Canonical model IDs, global seed, and adjective-domain specs for the paper.

Single source of truth so every extract/analyze/figure script agrees on models, seeds, and the
domain grid (paper Table 1). Import from here rather than hardcoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Global RNG seed. Context samples are additionally seeded per (domain, mu, sigma) cell so the
# implicit context is deterministic across runs; see data_gen/prompts.py.
SEED = 0

# --- Models -----------------------------------------------------------------------------------

# Primary model for all main-text results.
PRIMARY_MODEL = "google/gemma-2-9b"

# HF ids. The paper's headline circuit (trio L23H14/L26H4/L31H3) is on gemma-2-9b; gemma-2-2b is
# the legacy model some intermediate artifacts were produced on.
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
# DLA-ranked causal trio (gemma-2-9b), as (layer, head).
CAUSAL_TRIO_9B = [(23, 14), (26, 4), (31, 3)]


# --- Domains (paper Table 1) ------------------------------------------------------------------

@dataclass(frozen=True)
class Domain:
    name: str
    unit: str
    low: str          # low-end adjective
    high: str         # high-end adjective
    sigma: float      # context spread (linear units, or log-factor when log_space)
    x_min: float
    x_max: float
    log_space: bool = False  # wealth/income normalises in log space


DOMAINS: dict[str, Domain] = {
    "height":     Domain("height", "cm", "short", "tall", 10.0, 147.0, 183.0),
    "age":        Domain("age", "yr", "young", "old", 5.0, 16.0, 64.0),
    "weight":     Domain("weight", "kg", "light", "heavy", 8.0, 45.0, 105.0),
    "size":       Domain("size", "cm diameter", "small", "big", 6.0, 0.5, 65.5),
    "speed":      Domain("speed", "km/h", "slow", "fast", 15.0, 7.0, 163.0),
    "income":     Domain("income", "USD/yr", "poor", "rich", math.log(2), 14_000.0, 900_000.0, log_space=True),
    "experience": Domain("experience", "years", "novice", "expert", 4.0, 0.5, 27.4),
    "bmi":        Domain("bmi", "kg/m^2", "thin", "obese", 3.0, 14.9, 40.1),
}
