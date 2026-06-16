"""Phase 2F — iterative manifold-transport delta computation.

Helper module. Provides `iterative_manifold_delta`, the path-summed analogue
of `p2e_residual_interventions.manifold_delta`. The chord delta in Phase 2E
is

    Δ_chord = M[x_bin, z_target_bin] − M[x_bin, z_source_bin]

The iterative delta sums local cell-mean differences along a one-step-at-a-time
walk through populated bins on the same x-row, from the source bin toward the
nearest-to-zero populated bin:

    Δ_iter = Σ_k  (M[x_bin, b_{k+1}] − M[x_bin, b_k])

Mathematically, when the source bin is itself populated this sum telescopes
to Δ_chord. The two diverge ONLY when the source bin is empty: the chord
falls back to the count-weighted x-row marginal, while the iterative version
"snaps" the source to the nearest populated bin and walks from there.

The compensating diagnostic (path length vs chord norm, per-prompt
`||Δ_iter − Δ_chord||`) is exposed in the metadata returned alongside the
delta array, so callers can quantify how much non-additivity exists in
practice.

This file does NOT modify Phase 2E in-place; it imports `build_cell_mean_lookup`
and is used as a drop-in delta source by `p2f_iterative_alpha_sweep.py` and
`p2f_xk_iterative_alpha_sweep.py`, which pass the resulting deltas through
the existing `run_intervention(mode="manifold", deltas=...)` hook.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IterativeMeta:
    n_steps: np.ndarray            # int32 (n,) — number of bin transitions
    path_norm: np.ndarray          # float32 (n,) — sum of ||local step|| along path
    snap_used: np.ndarray          # bool (n,) — source bin was empty, snapped
    fallback_used: np.ndarray      # bool (n,) — entire x-slice empty, delta=0
    diff_to_chord_norm: np.ndarray # float32 (n,) — ||Δ_iter − Δ_chord||
    path_bins: list                # list[list[int]] — per-prompt visited bins

    def summary(self) -> dict:
        return {
            "mean_steps": float(self.n_steps.mean()),
            "max_steps": int(self.n_steps.max()),
            "median_steps": float(np.median(self.n_steps)),
            "mean_path_norm": float(self.path_norm.mean()),
            "max_path_norm": float(self.path_norm.max()),
            "frac_snap_used": float(self.snap_used.mean()),
            "frac_fallback_used": float(self.fallback_used.mean()),
            "mean_diff_to_chord_norm": float(self.diff_to_chord_norm.mean()),
            "max_diff_to_chord_norm": float(self.diff_to_chord_norm.max()),
            "frac_diff_above_1e-6": float((self.diff_to_chord_norm > 1e-6).mean()),
        }


def iterative_manifold_delta(M, counts, x_edges, z_edges, x_marg, x, z,
                              chord_deltas=None):
    """Per-prompt iterative manifold-transport delta.

    Parameters mirror `p2e_residual_interventions.manifold_delta`. If
    `chord_deltas` (output of the chord version) is supplied, the per-prompt
    `||Δ_iter − Δ_chord||` diagnostic is populated; otherwise zeros.

    Returns
    -------
    deltas : np.ndarray, shape (n, d), float32
        Same dimension as the chord delta — drop-in replacement.
    meta : IterativeMeta
        Per-prompt path metadata + summary aggregates.
    """
    n_x_bins, n_z_bins, d = M.shape
    n = len(x)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    deltas = np.zeros((n, d), dtype=np.float32)
    n_steps = np.zeros(n, dtype=np.int32)
    path_norm = np.zeros(n, dtype=np.float32)
    snap_used = np.zeros(n, dtype=bool)
    fallback_used = np.zeros(n, dtype=bool)
    path_bins: list = [None] * n

    for i in range(n):
        xb = int(np.clip(np.digitize(x[i], x_edges) - 1, 0, n_x_bins - 1))
        zb = int(np.clip(np.digitize(z[i], z_edges) - 1, 0, n_z_bins - 1))

        valid = counts[xb] > 0
        if not valid.any():
            # Entire x-slice is unpopulated. Mirror chord behaviour: zero delta.
            fallback_used[i] = True
            path_bins[i] = []
            continue

        valid_idx = np.where(valid)[0]

        # If source bin is empty, snap to the nearest populated bin first.
        if counts[xb, zb] == 0:
            snap_used[i] = True
            zb = int(valid_idx[np.argmin(np.abs(valid_idx - zb))])

        # Target bin: populated bin whose centre is closest to z=0.
        tgt_b = int(valid_idx[np.argmin(np.abs(z_centers[valid_idx]))])

        if zb == tgt_b:
            path_bins[i] = [zb]
            continue

        # Restrict path to populated bins between zb and tgt_b inclusive,
        # ordered from zb toward tgt_b.
        lo, hi = (zb, tgt_b) if zb < tgt_b else (tgt_b, zb)
        path_idx = [int(b) for b in valid_idx if lo <= b <= hi]
        if zb > tgt_b:
            path_idx = path_idx[::-1]
        path_bins[i] = path_idx

        delta_total = np.zeros(d, dtype=np.float64)
        norm_acc = 0.0
        for j in range(len(path_idx) - 1):
            cur = M[xb, path_idx[j]]
            nxt = M[xb, path_idx[j + 1]]
            step = nxt - cur
            delta_total += step
            norm_acc += float(np.linalg.norm(step))

        deltas[i] = delta_total.astype(np.float32)
        n_steps[i] = len(path_idx) - 1
        path_norm[i] = norm_acc

    if chord_deltas is None:
        diff_to_chord = np.zeros(n, dtype=np.float32)
    else:
        diff_to_chord = np.linalg.norm(deltas - chord_deltas, axis=-1).astype(np.float32)

    meta = IterativeMeta(
        n_steps=n_steps,
        path_norm=path_norm,
        snap_used=snap_used,
        fallback_used=fallback_used,
        diff_to_chord_norm=diff_to_chord,
        path_bins=path_bins,
    )
    return deltas, meta


# ---------------------------------------------------------------------------
# Arc-length-parameterised manifold transport (α as flow parameter, not scalar)
# ---------------------------------------------------------------------------


def precompute_arclength_paths(M, counts, x_edges, z_edges, x_marg, x, z):
    """For each prompt, build the populated-bin path source→target on its
    x-row and cache (Ms, arc_u) so per-α deltas reduce to interpolation.

    Returns a list of dicts with keys:
      Ms        : (n_path, d) array of cell-mean residuals along the path
      arc_u     : (n_path,) cumulative arc length, normalised to [0, 1]
      total_len : float (Σ ||M[b_{k+1}] − M[b_k]||)
      n_steps   : int (n_path − 1, or 0 if path is trivial)
      snap_used : bool
      fallback_used : bool
    """
    n_x_bins, n_z_bins, d = M.shape
    n = len(x)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2
    paths = []
    for i in range(n):
        xb = int(np.clip(np.digitize(x[i], x_edges) - 1, 0, n_x_bins - 1))
        zb = int(np.clip(np.digitize(z[i], z_edges) - 1, 0, n_z_bins - 1))
        valid = counts[xb] > 0
        snap_used = False
        fallback_used = False
        if not valid.any():
            paths.append({
                "Ms": np.zeros((1, d), dtype=np.float32),
                "arc_u": np.array([0.0], dtype=np.float32),
                "total_len": 0.0, "n_steps": 0,
                "snap_used": False, "fallback_used": True,
            })
            continue

        valid_idx = np.where(valid)[0]
        if counts[xb, zb] == 0:
            snap_used = True
            zb = int(valid_idx[np.argmin(np.abs(valid_idx - zb))])
        tgt_b = int(valid_idx[np.argmin(np.abs(z_centers[valid_idx]))])

        if zb == tgt_b:
            paths.append({
                "Ms": M[xb, [zb]].astype(np.float32, copy=True),
                "arc_u": np.array([0.0], dtype=np.float32),
                "total_len": 0.0, "n_steps": 0,
                "snap_used": snap_used, "fallback_used": False,
            })
            continue

        lo, hi = (zb, tgt_b) if zb < tgt_b else (tgt_b, zb)
        path_idx = [int(b) for b in valid_idx if lo <= b <= hi]
        if zb > tgt_b:
            path_idx = path_idx[::-1]

        Ms = np.stack([M[xb, b].astype(np.float64) for b in path_idx], axis=0)
        seg_lens = np.linalg.norm(np.diff(Ms, axis=0), axis=-1)
        total_len = float(seg_lens.sum())
        if total_len < 1e-12:
            arc_u = np.linspace(0.0, 1.0, len(Ms))
        else:
            cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
            arc_u = cum / total_len

        paths.append({
            "Ms": Ms.astype(np.float32, copy=True),
            "arc_u": arc_u.astype(np.float32),
            "total_len": total_len,
            "n_steps": len(Ms) - 1,
            "snap_used": snap_used,
            "fallback_used": fallback_used,
        })
    return paths


def arclength_delta_at_alpha(paths, alpha, d):
    """Per-prompt delta at arc-length-fraction α along each prompt's path.

    α ∈ [0, 1] interpolates ON the manifold polyline (linear interp between
    two adjacent populated bin means, weighted by arc length). α = 0 → stay at
    source (zero delta). α = 1 → reach target (== chord delta). For α > 1, we
    extrapolate beyond target along the chord direction (M_α = M_target +
    (α − 1) · (M_target − M_source)) so the existing α grid (1.25, 1.5, 2.0)
    keeps a defined behaviour.
    """
    n = len(paths)
    deltas = np.zeros((n, d), dtype=np.float32)
    for i, p in enumerate(paths):
        Ms = p["Ms"]
        if Ms.shape[0] < 2:
            continue
        arc_u = p["arc_u"]
        if alpha <= 0.0:
            M_alpha = Ms[0]
        elif alpha >= 1.0:
            chord = Ms[-1] - Ms[0]
            M_alpha = Ms[-1] + (alpha - 1.0) * chord
        else:
            k = int(np.searchsorted(arc_u, alpha, side="right") - 1)
            k = max(0, min(k, len(Ms) - 2))
            denom = arc_u[k + 1] - arc_u[k]
            lam = 0.0 if denom < 1e-12 else float((alpha - arc_u[k]) / denom)
            M_alpha = Ms[k] + lam * (Ms[k + 1] - Ms[k])
        deltas[i] = (M_alpha - Ms[0]).astype(np.float32)
    return deltas


def arclength_path_summary(paths) -> dict:
    n_steps = np.array([p["n_steps"] for p in paths], dtype=np.int32)
    total_len = np.array([p["total_len"] for p in paths], dtype=np.float32)
    snap_used = np.array([p["snap_used"] for p in paths], dtype=bool)
    fallback_used = np.array([p["fallback_used"] for p in paths], dtype=bool)
    return {
        "mean_steps": float(n_steps.mean()),
        "median_steps": float(np.median(n_steps)),
        "max_steps": int(n_steps.max()),
        "mean_total_len": float(total_len.mean()),
        "max_total_len": float(total_len.max()),
        "frac_snap_used": float(snap_used.mean()),
        "frac_fallback_used": float(fallback_used.mean()),
        "frac_zero_path": float((n_steps == 0).mean()),
    }
