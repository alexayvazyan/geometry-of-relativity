"""Canonical prompt engine for the gradable-adjective relativity experiments.

Consolidates the prompt logic that was previously duplicated across:
  - geometry-of-relativity/scripts/vast_remote/extract_v4_adjpairs.py  (the PAIRS source)
  - geometry-of-relativity/scripts/gen_v11_dense.py                    (dense grid)
  - relativity_ablation/scripts/gen_p2_shot_sweep.py                   (k-shot variant)
  - geometry-of-relativity/src/data_gen.py                            (early v2 API)

The prompt strings here are byte-for-byte the ones used for the paper's Gemma-2-9B runs, so
cached results stay valid. Each domain defines an implicit-list template (the format the paper's
main results use), plus explicit and zero-shot variants.

Workstream B extends this module with *naturalistic* frames (rosters, salary sheets, …); it does
not change the legacy frames below.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Wealth normalises in log-space: z = (log x - log mu) / log(sigma_factor).
LOG_SPACE_PAIRS: set[str] = {"wealth"}


@dataclass(frozen=True)
class Pair:
    name: str                       # canonical key; matches result filenames
    domain: str
    unit: str
    low_word: str                   # low-end adjective ("short")
    high_word: str                  # high-end adjective ("tall")
    target_values: tuple[float, ...]  # 5 discrete x values (v4); also fix the dense-grid bounds
    mu_values: tuple[float, ...]
    sigma: float                    # additive stdev; multiplicative factor for log-space pairs
    format_prompt_implicit: str
    format_prompt_explicit: str
    format_prompt_zero: str
    value_label: str


PAIRS: list[Pair] = [
    Pair("height", "height", "cm", "short", "tall",
         (150.0, 160.0, 165.0, 170.0, 180.0), (145.0, 155.0, 165.0, 175.0, 185.0), 10.0,
         "{items}\nPerson {n_last}: {x_str} cm. This person is",
         "In a group where most people's heights cluster around {mu_str} cm "
         "(give or take {sigma_str} cm), a person who is {x_str} cm is",
         "A person who is {x_str} cm is", "height"),
    Pair("age", "age", "years old", "young", "old",
         (20.0, 30.0, 40.0, 50.0, 60.0), (25.0, 35.0, 40.0, 45.0, 55.0), 5.0,
         "{items}\nPerson {n_last}: {x_str} years old. This person is",
         "In a group where most people's ages cluster around {mu_str} years "
         "(give or take {sigma_str} years), a person who is {x_str} years old is",
         "A person who is {x_str} years old is", "age"),
    Pair("weight", "weight", "kg", "light", "heavy",
         (50.0, 65.0, 75.0, 85.0, 100.0), (55.0, 65.0, 75.0, 85.0, 95.0), 8.0,
         "{items}\nPerson {n_last}: {x_str} kg. This person is",
         "In a group where most people's weights cluster around {mu_str} kg "
         "(give or take {sigma_str} kg), a person who weighs {x_str} kg is",
         "A person who weighs {x_str} kg is", "weight"),
    Pair("size", "size", "cm diameter", "small", "big",
         (5.0, 15.0, 25.0, 40.0, 60.0), (10.0, 20.0, 30.0, 40.0, 55.0), 6.0,
         "{items}\nObject {n_last}: {x_str} cm across. This object is",
         "In a group of objects whose sizes cluster around {mu_str} cm "
         "(give or take {sigma_str} cm), an object that is {x_str} cm across is",
         "An object that is {x_str} cm across is", "size"),
    Pair("speed", "speed", "km/h", "slow", "fast",
         (20.0, 50.0, 80.0, 110.0, 150.0), (30.0, 60.0, 80.0, 100.0, 140.0), 15.0,
         "{items}\nVehicle {n_last}: {x_str} km/h. This vehicle is",
         "In a group of vehicles whose speeds cluster around {mu_str} km/h "
         "(give or take {sigma_str} km/h), a vehicle going {x_str} km/h is",
         "A vehicle going {x_str} km/h is", "speed"),
    Pair("wealth", "wealth", "USD annual income", "poor", "rich",
         (20000.0, 50000.0, 100000.0, 250000.0, 600000.0),
         (30000.0, 70000.0, 150000.0, 400000.0, 1000000.0), 2.0,
         "{items}\nPerson {n_last} earns ${x_str}/year. This person is",
         "In a group where most people earn around ${mu_str}/year "
         "(incomes spread by roughly a factor of {sigma_str}), a person earning ${x_str}/year is",
         "A person earning ${x_str}/year is", "wealth"),
    Pair("experience", "experience", "years experience", "novice", "expert",
         (1.0, 5.0, 10.0, 15.0, 25.0), (2.0, 7.0, 12.0, 18.0, 25.0), 4.0,
         "{items}\nWorker {n_last}: {x_str} years experience. This worker is",
         "In a team where most have around {mu_str} years experience "
         "(give or take {sigma_str}), a worker with {x_str} years experience is",
         "A worker with {x_str} years experience is", "experience"),
    # Absolute-adjective control (clinically anchored): expected NOT to show relativity.
    Pair("bmi_abs", "BMI", "BMI", "thin", "obese",
         (17.0, 22.0, 27.0, 32.0, 38.0), (20.0, 25.0, 28.0, 30.0, 33.0), 3.0,
         "{items}\nPerson {n_last}: BMI {x_str}. This person is",
         "In a group where most people's BMIs cluster around {mu_str} "
         "(give or take {sigma_str}), a person with BMI {x_str} is",
         "A person with BMI {x_str} is", "bmi"),
]

PAIRS_BY_NAME: dict[str, Pair] = {p.name: p for p in PAIRS}


# --- scalar helpers ---------------------------------------------------------------------------

def fmt_num(v: float) -> str:
    """Render a number without a trailing .0 for integer-valued floats."""
    if v == int(v):
        return str(int(v))
    return f"{v:.1f}"


def compute_z(pair: Pair, x: float, mu: float) -> float:
    """Context-normalized standing. Log-space for wealth."""
    if pair.name in LOG_SPACE_PAIRS:
        if x <= 0 or mu <= 0 or pair.sigma <= 1.0:
            raise ValueError(f"log-space pair {pair.name} needs x>0, mu>0, sigma>1")
        return (math.log(x) - math.log(mu)) / math.log(pair.sigma)
    return (x - mu) / pair.sigma


def derive_mu(pair: Pair, x: float, z: float) -> float:
    """Inverse of compute_z: the context mean that yields target z at value x."""
    if pair.name in LOG_SPACE_PAIRS:
        return x * (pair.sigma ** (-z))
    return x - pair.sigma * z


# --- context sampling -------------------------------------------------------------------------

def sample_context(mu: float, sigma: float, seed: int, n: int = 15,
                   low: float | None = None, high: float | None = None,
                   log_space: bool = False) -> list[float]:
    """Sample `n` reference values around `mu`. Deterministic in `seed`.

    Normal(mu, sigma), or LogNormal(log mu, log sigma_factor) for log-space pairs. Because the RNG
    is seeded, the first k of an n=15 draw is identical, so k-shot prompts nest cleanly.
    """
    rng = random.Random(seed)
    out: list[float] = []
    for _ in range(n):
        if log_space:
            v = math.exp(rng.gauss(math.log(mu), math.log(sigma)))
        else:
            v = rng.gauss(mu, sigma)
        if low is not None:
            v = max(low, v)
        if high is not None:
            v = min(high, v)
        out.append(round(v, 1) if abs(v) < 100 else round(v))
    return out


def _format_items(pair: Pair, sample: list[float]) -> list[str]:
    """Per-domain rendering of the context list (the single source for item surface form)."""
    n = pair.name
    if n == "height":
        return [f"Person {i+1}: {int(v)} cm" for i, v in enumerate(sample)]
    if n == "age":
        return [f"Person {i+1}: {int(v)} years old" for i, v in enumerate(sample)]
    if n == "weight":
        return [f"Person {i+1}: {int(v)} kg" for i, v in enumerate(sample)]
    if n == "size":
        return [f"Object {i+1}: {int(v)} cm across" for i, v in enumerate(sample)]
    if n == "speed":
        return [f"Vehicle {i+1}: {int(v)} km/h" for i, v in enumerate(sample)]
    if n == "wealth":
        return [f"Person {i+1} earns ${int(v)}/year" for i, v in enumerate(sample)]
    if n == "experience":
        return [f"Worker {i+1}: {int(v)} years experience" for i, v in enumerate(sample)]
    if n == "bmi_abs":
        return [f"Person {i+1}: BMI {v:.1f}" for i, v in enumerate(sample)]
    return [f"Item {i+1}: {v}" for i, v in enumerate(sample)]


def _sample_bounds(pair: Pair) -> tuple[float, float]:
    return pair.target_values[0] * 0.4, pair.target_values[-1] * 2.5


def build_implicit_items(pair: Pair, mu: float, seed: int, n: int = 15) -> list[str]:
    low, high = _sample_bounds(pair)
    sample = sample_context(mu, pair.sigma, seed, n, low, high,
                            log_space=(pair.name in LOG_SPACE_PAIRS))
    return _format_items(pair, sample)


# --- prompt builders --------------------------------------------------------------------------

def make_implicit_prompt(pair: Pair, x: float, mu: float, seed: int, k: int = 15,
                         return_context: bool = False):
    """Implicit-list prompt with `k` context items (k=0 → target only, same template family).

    With return_context=True, also returns the sampled context values (for k>0).
    """
    if k == 0:
        template = pair.format_prompt_implicit.replace("{items}\n", "")
        prompt = template.format(n_last=1, x_str=fmt_num(x))
        return (prompt, []) if return_context else prompt
    low, high = _sample_bounds(pair)
    sample = sample_context(mu, pair.sigma, seed, n=k, low=low, high=high,
                            log_space=(pair.name in LOG_SPACE_PAIRS))
    items_block = "\n".join(_format_items(pair, sample))
    prompt = pair.format_prompt_implicit.format(items=items_block, n_last=k + 1, x_str=fmt_num(x))
    return (prompt, [float(v) for v in sample]) if return_context else prompt


def make_explicit_prompt(pair: Pair, x: float, mu: float) -> str:
    return pair.format_prompt_explicit.format(
        mu_str=fmt_num(mu), sigma_str=fmt_num(pair.sigma), x_str=fmt_num(x))


def make_zero_shot_prompt(pair: Pair, x: float) -> str:
    return pair.format_prompt_zero.format(x_str=fmt_num(x))


# ============================ Naturalistic frames (Workstream B) ============================
# Realistic-prose variants of the implicit-list frame. The SAME sampled context values are rendered
# with named people in a believable scenario, ending at the same adjective cliff ("... is"), so
# (x, z, mu, sigma) are identical to make_implicit_prompt and only the surface naturalism changes.
# Used to test whether z-dominance survives off the toy "Person i: 150 cm" format.

_NAMES = (
    "Maya", "Liam", "Aisha", "Diego", "Hannah", "Omar", "Yuki", "Sofia", "Noah", "Priya",
    "Ethan", "Zara", "Lucas", "Nina", "Ravi", "Chloe", "Mateo", "Leah", "Kai", "Amara",
    "Theo", "Iris", "Dani", "Bex",
)

# per-domain scenario: g=group noun, item=per-person clause, target=closing clause ending at "is".
_NATURALISTIC: dict[str, dict[str, str]] = {
    "height":     {"g": "the varsity basketball squad", "item": "{n} is {v} cm",
                   "target": "{n}, who just joined {g}, is {x} cm. Compared with the squad, {n} is"},
    "age":        {"g": "the hiking club", "item": "{n} is {v}",
                   "target": "{n}, a new member of {g}, is {x}. Compared with the club, {n} is"},
    "weight":     {"g": "the rowing crew", "item": "{n} weighs {v} kg",
                   "target": "{n}, the newest member of {g}, weighs {x} kg. For the crew, {n} is"},
    "size":       {"g": "a batch of components", "item": "one part is {v} cm across",
                   "target": "a new part in {g} is {x} cm across. For this batch, it is"},
    "speed":      {"g": "this afternoon's heat", "item": "{n}'s car clocked {v} km/h",
                   "target": "a late entrant in {g} clocked {x} km/h. For this heat, that car is"},
    "wealth":     {"g": "the engineering team", "item": "{n} earns ${v} a year",
                   "target": "{n}, a new hire on {g}, earns ${x} a year. Compared with the team, {n} is"},
    "experience": {"g": "the studio", "item": "{n} has {v} years of experience",
                   "target": "{n}, who just joined {g}, has {x} years of experience. For the studio, {n} is"},
    "bmi_abs":    {"g": "a group of patients", "item": "{n} has a BMI of {v}",
                   "target": "{n}, a new patient in {g}, has a BMI of {x}. Among the group, {n} is"},
}

# Neutral target clauses: same naturalistic named-prose, but no attribute-priming scenario, so the
# only difference from the primed frame is the world-knowledge context (B3 control).
_NAT_NEUTRAL_TARGET: dict[str, str] = {
    "height":     "{n} is {x} cm. Compared with the others, {n} is",
    "age":        "{n} is {x}. Compared with the others, {n} is",
    "weight":     "{n} weighs {x} kg. Compared with the others, {n} is",
    "size":       "a new part is {x} cm across. Compared with the rest, it is",
    "speed":      "{n}'s car clocked {x} km/h. Compared with the others, that car is",
    "wealth":     "{n} earns ${x} a year. Compared with the others, {n} is",
    "experience": "{n} has {x} years of experience. Compared with the others, {n} is",
    "bmi_abs":    "{n} has a BMI of {x}. Compared with the others, {n} is",
}


def _nat_value(pair: Pair, v: float) -> str:
    return f"{v:.1f}" if pair.name == "bmi_abs" else str(int(v))


def make_naturalistic_prompt(pair: Pair, x: float, mu: float, seed: int, k: int = 15,
                             style: str = "primed", return_context: bool = False):
    """Naturalistic-prose version of make_implicit_prompt: identical sampled context, named entities,
    ending at the same '... is' adjective cliff. style='primed' uses an attribute-relevant scenario
    (e.g. a basketball squad for height); style='neutral' drops the priming scenario."""
    cfg = _NATURALISTIC[pair.name]
    low, high = _sample_bounds(pair)
    sample = sample_context(mu, pair.sigma, seed, n=k, low=low, high=high,
                            log_space=(pair.name in LOG_SPACE_PAIRS))
    names = list(_NAMES)
    random.Random(seed).shuffle(names)
    items = [cfg["item"].format(n=names[i % len(names)], v=_nat_value(pair, v))
             for i, v in enumerate(sample)]
    target_name = names[k % len(names)]
    target_tmpl = cfg["target"] if style == "primed" else _NAT_NEUTRAL_TARGET[pair.name]
    body = ". ".join(items) + ".\n" + target_tmpl.format(n=target_name, x=fmt_num(x), g=cfg["g"])
    return (body, [float(v) for v in sample]) if return_context else body
