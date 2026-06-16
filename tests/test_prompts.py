"""Golden-string regression tests for the unified prompt engine.

These lock the legacy implicit-list format that the paper's Gemma-2-9B results were generated on,
so the consolidation (and any future naturalistic-frame work) can't silently change it.
"""

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data_gen import prompts as P  # noqa: E402


def test_height_k15_golden():
    h = P.PAIRS_BY_NAME["height"]
    p = P.make_implicit_prompt(h, 170, 160, 0, k=15)
    lines = p.splitlines()
    assert lines[0] == "Person 1: 169 cm"
    assert lines[-1] == "Person 16: 170 cm. This person is"
    assert p.count("Person ") == 16


def test_k0_is_target_only():
    h = P.PAIRS_BY_NAME["height"]
    assert P.make_implicit_prompt(h, 170, 160, 0, k=0) == "Person 1: 170 cm. This person is"


def test_kshot_context_nests():
    # k-shot prompts share a seed, so the first k of a longer draw are identical.
    h = P.PAIRS_BY_NAME["height"]
    _, c3 = P.make_implicit_prompt(h, 170, 160, 0, k=3, return_context=True)
    _, c8 = P.make_implicit_prompt(h, 170, 160, 0, k=8, return_context=True)
    assert c8[:3] == c3


def test_wealth_logspace_z():
    w = P.PAIRS_BY_NAME["wealth"]
    z = P.compute_z(w, 200000, 100000)  # x = 2*mu, sigma factor = 2  ->  z = 1
    assert abs(z - 1.0) < 1e-9
    assert abs(P.derive_mu(w, 200000, z) - 100000) < 1e-6


def test_derive_mu_roundtrip_all_pairs():
    for p in P.PAIRS:
        x, mu = p.target_values[2], p.mu_values[1]
        z = P.compute_z(p, x, mu)
        assert abs(P.derive_mu(p, x, z) - mu) < 1e-6, p.name
