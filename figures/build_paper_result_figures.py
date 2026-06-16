#!/usr/bin/env python3
"""Build paper-facing result figures from existing repo artifacts.

These figures intentionally remove internal experiment labels from visible
titles. Source provenance stays in LaTeX comments/captions and git history.
"""

from __future__ import annotations

import json
import importlib.util
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG.mkdir(parents=True, exist_ok=True)

PAIR_LABELS = {
    "age": "Age",
    "bmi_abs": "BMI",
    "experience": "Experience",
    "height": "Height",
    "size": "Size",
    "speed": "Speed",
    "wealth": "Income",
    "weight": "Weight",
}

PAIR_ORDER = ["height", "age", "weight", "size", "speed", "wealth", "experience", "bmi_abs"]

DIST_LABELS = {
    "normal": "Normal",
    "uniform": "Uniform",
    "beta_u": "U-shaped beta",
    "beta_low": "Low-skew beta",
    "beta_high": "High-skew beta",
    "bimodal": "Bimodal",
}

ORDER_LABELS = {
    "random": "Random",
    "ascending": "Ascending",
    "descending": "Descending",
    "alternating_low_high": "Alternating",
    "near_target_first": "Near target first",
    "near_target_last": "Near target last",
}

OOD_LABELS = {
    "base": "In-range worlds",
    "world_extreme_low": "Low extreme worlds",
    "world_extreme_high": "High extreme worlds",
}

OOD_MAIN_LABELS = {
    "base": "In-range",
    "world_extreme_low": "Low world",
    "world_extreme_high": "High world",
}


def open_on_white(path: Path) -> Image.Image:
    """Open an image and composite transparency over white."""
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, "white")
    bg.alpha_composite(img)
    return bg.convert("RGB")


def crop_dense_height_heatmap() -> None:
    src = ROOT / "figures" / "v10" / "behavioral_logit_diff_xz.png"
    img = open_on_white(src)
    # Preserve the heatmap pixels from the audited source artifact, but rebuild
    # paper-facing axes so labels remain legible at one-column width.
    heatmap = img.crop((78, 69, 722, 649))

    fig, ax = plt.subplots(figsize=(3.35, 2.48), dpi=300)
    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="upper",
        extent=[145, 190, -3, 3],
    )
    ax.set_xticks(np.arange(145, 191, 5))
    ax.set_yticks(np.arange(-3, 4, 1))
    ax.set_xlabel(r"$x$ (height in cm)", fontsize=9.8, labelpad=2.5)
    ax.set_ylabel(r"$z=(x-\mu)/\sigma$", fontsize=9.8, labelpad=2.5)
    ax.tick_params(labelsize=8.8, pad=2)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-2.8, vmax=2.8))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.047, pad=0.045)
    cbar.set_label(r"Mean $\Delta_{\mathrm{logit}}$", fontsize=9.4, labelpad=4)
    cbar.ax.tick_params(labelsize=8.4, pad=1.5)
    fig.tight_layout(pad=0.15)
    fig.savefig(PAPER_FIG / "fig_results_dense_height_heatmap_clean.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def build_pca_all_pairs() -> None:
    pairs = [(pair, PAIR_LABELS[pair]) for pair in PAIR_ORDER]
    fig, axes = plt.subplots(4, 2, figsize=(6.7, 8.2), dpi=180)
    for ax, (pair, title) in zip(axes.flat, pairs):
        src = ROOT / "figures" / "v11" / "pca" / f"{pair}_gemma2-9b_2d_L33.png"
        img = open_on_white(src)
        # Remove per-plot internal title. Keep axes, labels, points, and colorbar.
        img = img.crop((0, 38, img.width, img.height))
        ax.imshow(img)
        ax.set_title(title, fontsize=8, pad=1)
        ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.savefig(PAPER_FIG / "fig_results_pca_all_pairs_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_kshot_evolution_2x3() -> None:
    src = ROOT / "internal" / "kshot" / "phase" / "figures" / "p2a_ld_vs_z_height_gemma2-9b.png"
    img = open_on_white(src)
    # Drop the global title and original per-panel titles; add clean labels.
    # Coordinates are fixed for the committed source artifact.
    y0, y1 = 155, 455
    x_ranges = [
        (0, 470),
        (485, 880),
        (895, 1285),
        (1298, 1690),
        (1705, 2098),
        (2110, 2495),
    ]
    panels = [img.crop((x0, y0, x1, y1)) for x0, x1 in x_ranges]

    titles = [
        "No context",
        "1 example",
        "2 examples",
        "4 examples",
        "8 examples",
        "15 examples",
    ]
    target_w = 500
    resized = []
    for panel in panels:
        scale = target_w / panel.width
        target_h = int(panel.height * scale)
        resized.append(panel.resize((target_w, target_h), Image.Resampling.LANCZOS))

    pad_x, pad_y, title_h = 28, 24, 42
    row_h = max(p.height for p in resized)
    canvas = Image.new("RGB", (3 * target_w + 4 * pad_x, 2 * (row_h + title_h) + 3 * pad_y), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    for i, (panel, title) in enumerate(zip(resized, titles)):
        row, col = divmod(i, 3)
        x = pad_x + col * (target_w + pad_x)
        y = pad_y + row * (row_h + title_h + pad_y)
        bbox = draw.textbbox((0, 0), title, font=font)
        draw.text((x + (target_w - (bbox[2] - bbox[0])) / 2, y), title, fill="black", font=font)
        canvas.paste(panel, (x, y + title_h))
    canvas.save(PAPER_FIG / "fig_results_kshot_ld_evolution_2x3.png")


def build_layer_sweep_summary() -> None:
    readout = json.loads((ROOT / "results" / "v12" / "layer_sweep_9b.json").read_text())
    steering = json.loads((ROOT / "results" / "v12" / "layer_sweep_9b_steering.json").read_text())

    layers = np.array([r["layer"] for r in next(iter(readout["pairs"].values()))["layer_records"]])
    r2_by_pair = np.array(
        [[r["r2_cv_z"] for r in pair_data["layer_records"]] for pair_data in readout["pairs"].values()]
    )
    r2_mean = r2_by_pair.mean(axis=0)
    r2_std = r2_by_pair.std(axis=0)

    steering_layers = np.array(steering["layers"])
    steering_keys = [
        ("primal_z", "primal z", "C0"),
        ("probe_z", "probe z", "C3"),
        ("random_null", "random", "0.45"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(3.55, 4.35), dpi=220)
    ax = axes[0]
    ax.plot(layers, r2_mean, "o-", color="C0", markersize=2.5, linewidth=1.5)
    ax.fill_between(layers, r2_mean - r2_std, r2_mean + r2_std, color="C0", alpha=0.15, linewidth=0)
    ax.set_title("Linear readout of relative standing", fontsize=8)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel(r"CV $R^2(z)$", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for key, label, color in steering_keys:
        vals = np.array(
            [[pair_data[str(layer)][key] for layer in steering_layers] for pair_data in steering["by_pair"].values()]
        )
        mean = vals.mean(axis=0)
        std = vals.std(axis=0)
        ax.plot(steering_layers, mean, "o-", color=color, label=label, markersize=2.5, linewidth=1.5)
        ax.fill_between(steering_layers, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title("Causal steering effect by layer", fontsize=8)
    ax.set_xlabel("Layer", fontsize=7)
    ax.set_ylabel("Logit-difference slope", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    fig.tight_layout(pad=0.25)
    fig.savefig(PAPER_FIG / "fig_results_layer_sweep_summary_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_layer_z_x_encode_use() -> None:
    metrics = json.loads((ROOT / "results" / "v14_1" / "fig5" / "fig5_layer_x_z_metrics.json").read_text())
    layers = np.array(metrics["layers"], dtype=int)
    pairs = metrics["pairs"]

    def matrix(key: str) -> np.ndarray:
        return np.array(
            [[metrics["by_pair_layer"][pair][str(layer)][key] for layer in layers] for pair in pairs],
            dtype=float,
        )

    r2z = matrix("cv_r2_z")
    r2x = matrix("cv_r2_x")
    steer_z = matrix("primal_z_steering_slope")
    steer_x = matrix("primal_x_steering_slope")

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.45), dpi=260)

    ax = axes[0]
    for vals, color, label in [
        (r2z, "C0", r"$R^2(z)$"),
        (r2x, "C1", r"$R^2(x)$"),
    ]:
        center = np.nanmean(vals, axis=0)
        lo = np.nanpercentile(vals, 10, axis=0)
        hi = np.nanpercentile(vals, 90, axis=0)
        ax.plot(layers, center, "o-", color=color, label=label, markersize=2.6, linewidth=1.45)
        ax.fill_between(layers, lo, hi, color=color, alpha=0.14, linewidth=0)
    ax.set_xlabel("Layer", fontsize=7.5)
    ax.set_ylabel(r"CV $R^2$", fontsize=7.5)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="lower right")

    ax = axes[1]
    for vals, color, label in [
        (steer_z, "C0", r"$d_z$"),
        (steer_x, "C1", r"$d_x$"),
    ]:
        center = np.nanmean(vals, axis=0)
        lo = np.nanpercentile(vals, 10, axis=0)
        hi = np.nanpercentile(vals, 90, axis=0)
        ax.plot(layers, center, "o-", color=color, label=label, markersize=2.6, linewidth=1.45)
        ax.fill_between(layers, lo, hi, color=color, alpha=0.14, linewidth=0)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Layer", fontsize=7.5)
    ax.set_ylabel("LD slope", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    fig.tight_layout(pad=0.35)
    fig.savefig(PAPER_FIG / "fig_results_layer_z_x_encode_use_clean.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def standardized_distribution_samples() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 1200
    samples = {
        "normal": rng.normal(0.0, 1.0, n),
        "uniform": rng.uniform(-math.sqrt(3), math.sqrt(3), n),
        "beta_u": rng.beta(0.45, 0.45, n) * 2.0 - 1.0,
        "beta_low": rng.beta(2.0, 5.0, n) * 2.0 - 1.0,
        "beta_high": rng.beta(5.0, 2.0, n) * 2.0 - 1.0,
        "bimodal": rng.choice([-1.0, 1.0], size=n) * 1.15 + rng.normal(0.0, 0.22, n),
    }
    for key, vals in list(samples.items()):
        vals = np.asarray(vals, dtype=float)
        vals = vals - vals.mean()
        samples[key] = vals / vals.std(ddof=1)
    return samples


def build_distribution_shapes() -> None:
    samples = standardized_distribution_samples()
    fig, axes = plt.subplots(2, 3, figsize=(3.35, 2.25), dpi=280, sharex=True, sharey=True)
    for ax, key in zip(axes.flat, ["normal", "uniform", "beta_u", "beta_low", "beta_high", "bimodal"]):
        ax.hist(samples[key], bins=24, color="#4C78A8", alpha=0.78)
        ax.set_title(DIST_LABELS[key], fontsize=5.8, pad=1.5)
        ax.tick_params(labelsize=5.2, pad=1)
        ax.grid(axis="y", alpha=0.18)
    for ax in axes[-1, :]:
        ax.set_xlabel("Standardized value", fontsize=5.4, labelpad=1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Count", fontsize=5.4, labelpad=1)
    fig.tight_layout(pad=0.18)
    fig.savefig(PAPER_FIG / "fig_results_distribution_shapes_clean.png", bbox_inches="tight", dpi=280)
    plt.close(fig)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def binned_mean_rows(rows: list[dict], condition_key: str, conditions: list[str]) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for pair in PAIR_ORDER:
        for condition in conditions:
            buckets: dict[float, list[float]] = {}
            for row in rows:
                if row["pair"] != pair or row.get(condition_key) != condition:
                    continue
                if int(row.get("n_context", 31)) != 31:
                    continue
                z = round(float(row["z"]), 2)
                buckets.setdefault(z, []).append(float(row["ld"]))
            xs = np.array(sorted(buckets), dtype=float)
            ys = np.array([np.mean(buckets[float(x)]) for x in xs], dtype=float)
            out[(pair, condition)] = (xs, ys)
    return out


def plot_ld_by_z_grid(
    rows: list[dict],
    condition_key: str,
    conditions: list[str],
    labels: dict[str, str],
    out_name: str,
    selected_pairs: list[str] | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] = (7.05, 8.25),
    legend_ncol: int | None = None,
    top_rect: float = 0.965,
    title_fs: float = 10,
    label_fs: float = 9,
    tick_fs: float = 8,
    legend_fs: float = 9,
    marker_size: float = 3.2,
    line_width: float = 1.7,
) -> None:
    series = binned_mean_rows(rows, condition_key, conditions)
    pairs = selected_pairs or PAIR_ORDER
    nrows = math.ceil(len(pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=240, sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    colors = plt.cm.tab10(np.linspace(0, 1, len(conditions)))
    for ax, pair in zip(axes_arr.flat, pairs):
        for color, condition in zip(colors, conditions):
            xs, ys = series[(pair, condition)]
            if len(xs) == 0:
                continue
            ax.plot(xs, ys, "o-", markersize=marker_size, linewidth=line_width, color=color, label=labels[condition])
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.45)
        ax.set_title(PAIR_LABELS[pair], fontsize=title_fs)
        ax.tick_params(labelsize=tick_fs)
        ax.grid(alpha=0.22)
    for ax in axes_arr.flat[len(pairs) :]:
        ax.axis("off")
    for ax in axes_arr[-1, :]:
        ax.set_xlabel(r"Relative standing $z$", fontsize=label_fs)
    for ax in axes_arr[:, 0]:
        ax.set_ylabel(r"$\Delta_{\mathrm{logit}}$", fontsize=label_fs)
    handles, legend_labels = axes_arr[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=legend_ncol or min(len(conditions), 3),
        frameon=False,
        fontsize=legend_fs,
    )
    fig.tight_layout(rect=(0, 0, 1, top_rect), pad=0.35)
    fig.savefig(PAPER_FIG / out_name, bbox_inches="tight", dpi=240)
    plt.close(fig)


def build_order_main_figure(order_rows: list[dict]) -> None:
    conditions = ["random", "ascending", "descending", "alternating_low_high"]
    series = binned_mean_rows(order_rows, "order_kind", conditions)
    pairs = ["height", "weight"]
    colors = {
        "random": "#1f77b4",
        "ascending": "#9467bd",
        "descending": "#8c564b",
        "alternating_low_high": "#17becf",
    }

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.8), dpi=260, sharex=True)
    for ax, pair in zip(axes, pairs):
        for condition in conditions:
            xs, ys = series[(pair, condition)]
            ax.plot(
                xs,
                ys,
                "o-",
                markersize=2.2,
                linewidth=1.15,
                color=colors[condition],
                label=ORDER_LABELS[condition],
            )
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.45)
        ax.set_title(PAIR_LABELS[pair], fontsize=8.0, pad=2)
        ax.set_ylabel(r"$\Delta_{\mathrm{logit}}$", fontsize=7.0, labelpad=1.5)
        ax.tick_params(labelsize=6.4, pad=1.5)
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel(r"Relative standing $z$", fontsize=7.0, labelpad=1.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=6.1,
        handlelength=1.2,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.3)
    fig.savefig(PAPER_FIG / "fig_results_order_ld_by_z_main.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def build_distribution_bimodal_main_figure(distribution_rows: list[dict]) -> None:
    samples = standardized_distribution_samples()
    conditions = ["normal", "bimodal"]
    series = binned_mean_rows(distribution_rows, "dist_kind", conditions)
    pairs = ["height", "weight"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(3.35, 2.75),
        dpi=260,
        gridspec_kw={"width_ratios": [0.82, 1.35]},
    )

    for ax, key in zip(axes[:, 0], conditions):
        ax.hist(samples[key], bins=22, color="#4C78A8", alpha=0.78)
        ax.set_title(DIST_LABELS[key], fontsize=7.2, pad=2)
        ax.set_ylabel("Count", fontsize=6.2, labelpad=1)
        ax.tick_params(labelsize=5.8, pad=1)
        ax.grid(axis="y", alpha=0.18)
    axes[1, 0].set_xlabel("Standardized value", fontsize=6.2, labelpad=1)

    colors = {"normal": "#1f77b4", "bimodal": "#d62728"}
    for ax, pair in zip(axes[:, 1], pairs):
        for condition in conditions:
            xs, ys = series[(pair, condition)]
            ax.plot(
                xs,
                ys,
                "o-",
                markersize=2.0,
                linewidth=1.05,
                color=colors[condition],
                label=DIST_LABELS[condition],
            )
        ax.axhline(0, color="black", linewidth=0.55, alpha=0.45)
        ax.set_title(PAIR_LABELS[pair], fontsize=7.2, pad=2)
        ax.set_ylabel(r"$\Delta_{\mathrm{logit}}$", fontsize=6.2, labelpad=1)
        ax.tick_params(labelsize=5.8, pad=1)
        ax.grid(alpha=0.22)
    axes[1, 1].set_xlabel(r"Relative standing $z$", fontsize=6.2, labelpad=1)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
        fontsize=6.1,
        handlelength=1.2,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.25, w_pad=0.55, h_pad=0.55)
    fig.savefig(PAPER_FIG / "fig_results_distribution_bimodal_main.png", bbox_inches="tight", dpi=260)
    plt.close(fig)


def build_robustness_figures() -> None:
    affine_rows = load_jsonl(ROOT / "results" / "v14_1" / "affine_ood" / "affine_ood_rows.jsonl")
    distribution_rows = load_jsonl(ROOT / "results" / "v14" / "distribution" / "distribution_rows.jsonl")
    order_rows = load_jsonl(ROOT / "results" / "v14_1" / "order" / "order_rows.jsonl")

    build_order_main_figure(order_rows)
    build_distribution_bimodal_main_figure(distribution_rows)

    plot_ld_by_z_grid(
        affine_rows,
        "ood_condition",
        ["base", "world_extreme_low", "world_extreme_high"],
        OOD_MAIN_LABELS,
        "fig_results_ood_ld_by_z_main.png",
        selected_pairs=["height", "weight", "age"],
        ncols=1,
        figsize=(3.35, 4.95),
        legend_ncol=3,
        top_rect=0.965,
        title_fs=8.2,
        label_fs=7.2,
        tick_fs=6.8,
        legend_fs=7.0,
        marker_size=2.3,
        line_width=1.25,
    )
    plot_ld_by_z_grid(
        affine_rows,
        "ood_condition",
        ["base", "world_extreme_low", "world_extreme_high"],
        OOD_LABELS,
        "fig_results_ood_ld_by_z_full_clean.png",
    )
    plot_ld_by_z_grid(
        distribution_rows,
        "dist_kind",
        ["normal", "uniform", "beta_u", "beta_low", "beta_high", "bimodal"],
        DIST_LABELS,
        "fig_results_distribution_ld_by_z_clean.png",
        figsize=(7.05, 8.65),
        legend_ncol=3,
        top_rect=0.895,
        title_fs=8.2,
        label_fs=7.4,
        tick_fs=6.6,
        legend_fs=7.2,
        marker_size=2.0,
        line_width=1.1,
    )
    plot_ld_by_z_grid(
        order_rows,
        "order_kind",
        ["random", "ascending", "descending", "alternating_low_high", "near_target_first", "near_target_last"],
        ORDER_LABELS,
        "fig_results_order_ld_by_z_clean.png",
        figsize=(7.05, 8.65),
        legend_ncol=3,
        top_rect=0.895,
        title_fs=8.2,
        label_fs=7.4,
        tick_fs=6.6,
        legend_fs=7.2,
        marker_size=2.0,
        line_width=1.1,
    )


def crop_relative_objective_phase() -> None:
    src = ROOT / "internal" / "kshot" / "phase" / "figures" / "p2d_phase_grid_partial.png"
    out = PAPER_FIG / "fig_results_relative_objective_phase_clean.png"
    out_9b = PAPER_FIG / "fig_results_relative_objective_phase_9b_clean.png"
    img = open_on_white(src)
    # Drop the global internal title, preserving panel titles, axes, and annotations.
    cropped = img.crop((0, 76, img.width, img.height))
    cropped.save(out)

    # One-column paper version: keep the 9B row and wrap the five shot counts.
    # Coordinates preserve each source panel's title, axes, and annotations.
    boxes = [
        (70, 430, 382, 842),
        (434, 430, 746, 842),
        (798, 430, 1110, 842),
        (1162, 430, 1474, 842),
        (1485, 430, 2018, 842),
    ]
    panels = [cropped.crop(box) for box in boxes]
    target_w = 365
    resized = []
    for panel in panels:
        scale = target_w / panel.width
        resized.append(panel.resize((target_w, int(panel.height * scale)), Image.Resampling.LANCZOS))
    pad_x, pad_y = 22, 18
    panel_h = max(p.height for p in resized)
    canvas_w = 3 * target_w + 4 * pad_x
    canvas_h = 2 * panel_h + 3 * pad_y
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    for i, panel in enumerate(resized):
        row, col = divmod(i, 3)
        x = pad_x + col * (target_w + pad_x)
        y = pad_y + row * (panel_h + pad_y)
        canvas.paste(panel, (x, y))
    canvas.save(out_9b)


def build_relative_objective_phase_9b() -> None:
    """Replot the phase-plane result in a one-column-friendly layout."""
    module_path = ROOT / "internal" / "kshot" / "phase" / "scripts" / "plot_p2d_phase_grid_partial.py"
    spec = importlib.util.spec_from_file_location("phase_partial", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    phase = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(phase)

    model = "gemma2-9b"
    pair = "height"
    ks = [0, 1, 2, 5, 15]
    r_zx_per_k = phase.get_r_zx_per_k(pair, ks=[1, 2, 5, 15])

    fig, axes = plt.subplots(2, 3, figsize=(3.55, 4.65), dpi=220)
    axes_flat = axes.ravel()

    p2d = json.loads((ROOT / "internal" / "kshot" / "phase" / "results" / "p2d_l0all_per_k_gemma2-9b_height.json").read_text())["results"]
    p2d_k1 = json.loads((ROOT / "internal" / "kshot" / "phase" / "results" / "p2d_partial_l0_gemma2-9b_height_k1.json").read_text())
    p2e_k15 = json.loads((ROOT / "internal" / "kshot" / "phase" / "results" / "p2e_residual_interventions_gemma2-9b_height_k15.json").read_text())["results"]

    def partial_point(row: dict, k: int) -> tuple[float, float]:
        if k == 0:
            return float(row["r_ld_x"]), 0.0
        if "p_lx_z" in row and "p_lz_x" in row:
            return float(row["p_lx_z"]), float(row["p_lz_x"])
        p_z, p_x = phase.partial_corr(float(row["r_ld_zeff"]), float(row["r_ld_x"]), r_zx_per_k[k])
        return p_x, p_z

    def baseline_row(k: int) -> dict:
        if k == 1:
            return p2d_k1["baseline"]
        if k == 15:
            return p2e_k15["baseline"]
        return p2d[f"k{k}"]["baseline"]

    def l0_row(k: int) -> dict:
        if k == 1:
            return p2d_k1["l0_all"]
        if k == 15:
            return phase.l0_partials(model, pair, k, r_zx_per_k[k])
        return p2d[f"k{k}"]["l0_all"]

    for ax, k in zip(axes_flat, ks):
        ax.add_patch(plt.Rectangle((0.0, 0.0), 0.5, 0.5, alpha=0.10, color="C3", zorder=0))
        ax.add_patch(plt.Rectangle((0.5, 0.0), 0.5, 0.5, alpha=0.10, color="C2", zorder=0))
        ax.add_patch(plt.Rectangle((0.0, 0.5), 0.5, 0.5, alpha=0.10, color="C0", zorder=0))
        ax.add_patch(plt.Rectangle((0.5, 0.5), 0.5, 0.5, alpha=0.10, color="gold", zorder=0))
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="black", linewidth=0.3, alpha=0.4)
        ax.axvline(0.5, color="black", linewidth=0.3, alpha=0.4)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.grid(alpha=0.2)
        ax.set_title(f"k={k}", fontsize=7)
        base_xy = partial_point(baseline_row(k), k)
        l0_xy = partial_point(l0_row(k), k)
        ax.scatter(*base_xy, s=48, color="tab:green", edgecolor="black", linewidth=0.6, zorder=4)
        ax.text(base_xy[0] + 0.02, base_xy[1] + 0.03, "base", color="tab:green", fontsize=5.5, weight="bold")
        ax.scatter(*l0_xy, s=50, color="tab:red", marker="X", edgecolor="black", linewidth=0.5, zorder=5)
        ax.text(l0_xy[0] + 0.02, l0_xy[1] - 0.07, "L0 all", color="tab:red", fontsize=5.5, weight="bold")
        ax.annotate("", xy=l0_xy, xytext=base_xy, arrowprops=dict(arrowstyle="->", color="black", alpha=0.45, lw=0.6))

        if k == 15:
            interventions = phase.residual_intervention_partials(model, pair, k, r_zx_per_k[k])
            for mode, label, color in [
                ("mean_ablate", "mean", "tab:orange"),
                ("proj_out", "proj", "tab:purple"),
                ("manifold_a075", "manifold", "tab:cyan"),
            ]:
                row = interventions.get(mode)
                if row is None:
                    continue
                point = (row["p_lx_z"], row["p_lz_x"])
                ax.scatter(*point, s=34, color=color, marker="D", edgecolor="black", linewidth=0.4, zorder=4)
                ax.text(point[0] + 0.02, point[1] + 0.02, label, color=color, fontsize=5.2, weight="bold")
                if base_xy is not None:
                    ax.annotate("", xy=point, xytext=base_xy, arrowprops=dict(arrowstyle="->", color=color, alpha=0.45, lw=0.5))

        ax.tick_params(labelsize=5.5)
        ax.set_xlabel("")
        ax.set_ylabel("")

    for ax in axes_flat[len(ks):]:
        ax.axis("off")
    axes[0, 0].set_ylabel("relative axis", fontsize=6)
    axes[1, 0].set_ylabel("relative axis", fontsize=6)
    axes[1, 0].set_xlabel("objective axis", fontsize=6)
    axes[1, 1].set_xlabel("objective axis", fontsize=6)

    fig.tight_layout(pad=0.25)
    fig.savefig(PAPER_FIG / "fig_results_relative_objective_phase_9b_replot_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_cross_pair_transfer() -> None:
    src = ROOT / "results" / "v11" / "gemma2-9b" / "cross_pair_transfer_dense.json"
    data = json.loads(src.read_text())
    pairs = data["pairs"]
    mat = np.array(
        [[data["transfer_slope_target_by_source"][target][source] for source in pairs] for target in pairs],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(3.55, 3.35), dpi=220)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.09, vmax=0.09)
    labels = [PAIR_LABELS[p] for p in pairs]
    ax.set_xticks(range(len(pairs)), labels=labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(pairs)), labels=labels, fontsize=6)
    ax.set_xlabel("Source concept", fontsize=7)
    ax.set_ylabel("Target concept", fontsize=7)
    for i in range(len(pairs)):
        for j in range(len(pairs)):
            ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Logit-difference slope", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.15)
    fig.savefig(PAPER_FIG / "fig_results_cross_pair_transfer_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_z_vs_x_transfer() -> None:
    src = ROOT / "results" / "v13" / "x_transfer" / "cross_pair_transfer_x_8x8.json"
    summary_src = ROOT / "results" / "v13" / "x_transfer" / "cross_pair_transfer_x_vs_z_summary.json"
    data = json.loads(src.read_text())
    summary = json.loads(summary_src.read_text())["families"]
    pairs = data["pairs"]
    labels = [PAIR_LABELS[p] for p in pairs]
    matrices = data["matrices"]

    z_mat = np.array([[matrices["primal_z"][target][source] for source in pairs] for target in pairs], dtype=float)
    x_mat = np.array(
        [[matrices["primal_x_naive"][target][source] for source in pairs] for target in pairs],
        dtype=float,
    )

    vmax = max(float(np.nanmax(np.abs(z_mat))), float(np.nanmax(np.abs(x_mat))), 0.09)
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 1.95), dpi=240, constrained_layout=True)
    for ax, mat, title in [
        (axes[0], z_mat, "$z$ direction"),
        (axes[1], x_mat, "raw-$x$ direction"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=7)
        ax.set_xticks(range(len(pairs)), labels=labels, rotation=45, ha="right", fontsize=4.6)
        ax.set_yticks(range(len(pairs)), labels=labels if ax is axes[0] else [], fontsize=4.6)
        ax.set_xlabel("Source", fontsize=5.5)
        if ax is axes[0]:
            ax.set_ylabel("Target", fontsize=5.5)

    cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.012)
    cbar.set_label("Slope", fontsize=5.5)
    cbar.ax.tick_params(labelsize=4.8)

    fig.savefig(PAPER_FIG / "fig_results_z_vs_x_transfer_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(3.45, 5.55), dpi=220, constrained_layout=True)
    for ax, mat, title in [
        (axes[0], z_mat, "$z$ direction"),
        (axes[1], x_mat, "raw-$x$ direction"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(range(len(pairs)), labels=labels, rotation=45, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(pairs)), labels=labels, fontsize=5.5)
        ax.set_xlabel("Source", fontsize=6)
        ax.set_ylabel("Target", fontsize=6)
    cbar = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cbar.set_label("Steering slope", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    fig.savefig(PAPER_FIG / "fig_results_z_vs_x_transfer_stacked_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def crop_residual_vs_lexical_transfer() -> None:
    src = ROOT / "results" / "v12_2" / "residual_vs_lexical_transfer.json"
    out = PAPER_FIG / "fig_results_residual_vs_lexical_transfer_matrices_clean.png"
    data = json.loads(src.read_text())
    pairs = data["pairs"]
    labels = [PAIR_LABELS[pair] for pair in pairs]
    matrices = data["matrices"]
    panel_specs = [
        ("full", "Full direction"),
        ("lexical_projection", "Lexical projection"),
        ("lexical_residual", "Lexical residual"),
        ("random_null", "Random null"),
    ]
    mats = [
        np.array([[matrices[key][target][source] for source in pairs] for target in pairs], dtype=float)
        for key, _ in panel_specs
    ]
    vmax = max(float(np.nanmax(np.abs(mat))) for mat in mats)
    vmax = max(vmax, 0.12)

    fig = plt.figure(figsize=(3.55, 3.35), dpi=240)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.055], wspace=0.22, hspace=0.46)
    axes = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
                     [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]])
    for ax, mat, (_, title) in zip(axes.flat, mats, panel_specs):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=6.8, pad=1)
        ax.set_xticks(range(len(pairs)), labels=labels, rotation=45, ha="right", fontsize=4.5)
        ax.set_yticks(range(len(pairs)), labels=labels if ax in axes[:, 0] else [], fontsize=4.5)
        ax.tick_params(length=2, pad=1)
    axes[0, 0].set_ylabel("Target", fontsize=5.5)
    axes[1, 0].set_ylabel("Target", fontsize=5.5)
    axes[1, 0].set_xlabel("Source", fontsize=5.5)
    axes[1, 1].set_xlabel("Source", fontsize=5.5)
    cax = fig.add_subplot(gs[:, 2])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Steering slope", fontsize=5.5)
    cbar.ax.tick_params(labelsize=4.8)
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_shared_direction_steering() -> None:
    src = ROOT / "results" / "v11_5" / "gemma2-9b" / "shared_z_analysis.json"
    data = json.loads(src.read_text())
    pairs = data["pairs"]
    labels = [PAIR_LABELS[p] for p in pairs]
    within = np.array([data["by_pair_steering"][p]["within_slope"] for p in pairs])
    shared = np.array([data["by_pair_steering"][p]["shared_slope_proc"] for p in pairs])
    ratios = np.array([data["by_pair_steering"][p]["ratio_shared_to_within"] for p in pairs])

    x = np.arange(len(pairs))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.55, 2.35), dpi=220)
    ax.bar(x - width / 2, within, width, label="own z")
    ax.bar(x + width / 2, shared, width, label="shared z")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Logit-difference slope", fontsize=7)
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=6)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, float(np.max(np.maximum(within, shared)) + 0.014))
    for xi, y, ratio in zip(x, np.maximum(within, shared), ratios):
        color = "C2" if ratio >= 0.5 else "C3"
        ax.text(xi, y + 0.004, f"{ratio:.0%}", ha="center", va="bottom", fontsize=5.5, color=color)
    fig.tight_layout(pad=0.2)
    fig.savefig(PAPER_FIG / "fig_results_shared_direction_steering_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def build_lexical_transfer_summary() -> None:
    src = ROOT / "results" / "v12_2" / "residual_vs_lexical_transfer_summary.json"
    data = json.loads(src.read_text())["families"]
    keys = ["full", "lexical_projection", "lexical_residual", "random_null"]
    labels = ["Full z", "Lexical\nprojection", "Residual z", "Random"]
    diag = np.array([data[k]["mean_diagonal"] for k in keys])
    off = np.array([data[k]["mean_off_diagonal"] for k in keys])
    ci = np.array([data[k]["off_diagonal_mean_ci95"] for k in keys])
    off_err = np.vstack([off - ci[:, 0], ci[:, 1] - off])

    x = np.arange(len(keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.55, 2.25), dpi=220)
    ax.bar(x - width / 2, diag, width, label="same concept")
    ax.bar(x + width / 2, off, width, yerr=off_err, capsize=2, label="other concepts")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylabel("Logit-difference slope", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(pad=0.2)
    fig.savefig(PAPER_FIG / "fig_results_lexical_transfer_summary_clean.png", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    crop_dense_height_heatmap()
    build_pca_all_pairs()
    build_kshot_evolution_2x3()
    build_layer_sweep_summary()
    build_layer_z_x_encode_use()
    build_distribution_shapes()
    build_robustness_figures()
    crop_relative_objective_phase()
    try:
        build_relative_objective_phase_9b()
    except Exception as exc:  # legacy phase-replot variant, not used by the current paper
        print(
            "[skip] build_relative_objective_phase_9b: needs "
            "internal/kshot/phase/data/p2_shot_sweep/*.jsonl "
            f"(regenerable via gen_p2_shot_sweep); {exc}"
        )
    build_shared_direction_steering()
    build_cross_pair_transfer()
    build_z_vs_x_transfer()
    crop_residual_vs_lexical_transfer()
    build_lexical_transfer_summary()
    print(f"Wrote paper result figures to {PAPER_FIG}")


if __name__ == "__main__":
    main()
