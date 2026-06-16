"""Cross-model appendix composites for the relativity ablation paper.

Composes per-model figures from
`geometry-of-relativity/figures/v13/<model>/...` into 2 cross-model appendix
figures, each a 2-column × 5-row grid (one row per replication model).

Output:
  overleaf/figures/xmodel_geometry_grid.png
  overleaf/figures/xmodel_interventions_grid.png

Source figures (Jaehoon's v13 pipeline, 5 models):
  qwen__qwen2.5_3b
  meta_llama__llama_3.2_3b
  allenai__olmo_2_1124_7b
  eleutherai__pythia_2.8b
  qwen__qwen3_14b

Layout per composite:
  - row label (model display name) on left margin
  - column titles at top
  - per-row: two source PNGs scaled to a common row height
"""
from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]  # circuit/
GOR_FIG = ROOT / "inputs" / "v13"           # per-model v13 source panels (Jaehoon's pipeline)
OUT_DIR = ROOT / "figures"

MODELS = [
    ("qwen__qwen2.5_3b",          "Qwen 2.5 3B"),
    ("meta_llama__llama_3.2_3b",  "Llama 3.2 3B"),
    ("allenai__olmo_2_1124_7b",   "OLMo 2 7B"),
    ("eleutherai__pythia_2.8b",   "Pythia 2.8B"),
    ("qwen__qwen3_14b",           "Qwen 3 14B"),
]


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def _scale_to_height(im: Image.Image, h: int) -> Image.Image:
    w = round(im.width * (h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def compose(out_path: Path,
            cols: list[tuple[str, str]],
            row_height: int,
            title: str,
            label_w: int = 220,
            col_pad: int = 24,
            row_pad: int = 18,
            top_pad: int = 90,
            col_title_pad: int = 50) -> None:
    """cols: list of (figure_filename, column_title)."""
    n_cols = len(cols)
    # First pass: load + scale all imgs and compute column widths
    grid: list[list[Image.Image]] = []
    col_widths = [0] * n_cols
    for short, _display in MODELS:
        row_imgs = []
        for fname, _ctitle in cols:
            p = GOR_FIG / short / fname
            if not p.exists():
                raise SystemExit(f"missing {p}")
            im = Image.open(p).convert("RGB")
            im = _scale_to_height(im, row_height)
            row_imgs.append(im)
        for j, im in enumerate(row_imgs):
            col_widths[j] = max(col_widths[j], im.width)
        grid.append(row_imgs)

    total_w = label_w + sum(col_widths) + (n_cols + 1) * col_pad
    n_rows = len(MODELS)
    total_h = top_pad + col_title_pad + n_rows * (row_height + row_pad) + row_pad

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    col_font = _font(20)
    row_font = _font(22)

    # Title
    draw.text((col_pad, 20), title, fill="black", font=title_font)

    # Column titles
    x_cursor = label_w + col_pad
    for j, (_fname, ctitle) in enumerate(cols):
        cx = x_cursor + col_widths[j] // 2
        bbox = draw.textbbox((0, 0), ctitle, font=col_font)
        text_w = bbox[2] - bbox[0]
        draw.text((cx - text_w // 2, top_pad), ctitle,
                  fill="black", font=col_font)
        x_cursor += col_widths[j] + col_pad

    # Rows
    y_cursor = top_pad + col_title_pad
    for i, (_short, display) in enumerate(MODELS):
        # Row label centered vertically
        bbox = draw.textbbox((0, 0), display, font=row_font)
        text_h = bbox[3] - bbox[1]
        draw.text((col_pad, y_cursor + row_height // 2 - text_h // 2),
                  display, fill="black", font=row_font)
        # Paste each image into its column slot
        x_cursor = label_w + col_pad
        for j, im in enumerate(grid[i]):
            # center image horizontally within column slot
            offset_x = x_cursor + (col_widths[j] - im.width) // 2
            canvas.paste(im, (offset_x, y_cursor))
            x_cursor += col_widths[j] + col_pad
        y_cursor += row_height + row_pad

    canvas.save(out_path, "PNG", optimize=True)
    print(f"wrote {out_path}  ({canvas.width}x{canvas.height})")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figure A — geometry & behavior
    compose(
        out_path=OUT_DIR / "xmodel_geometry_grid.png",
        cols=[
            ("fig_pca_montage.png",
             "PCA decomposition of cell-mean activations (8 concepts)"),
            ("fig_behavioral_height.png",
             "Dense (x, z) LD grid (height)"),
        ],
        row_height=520,
        title="Z-score relativity geometry and behavioral readout replicate across small models",
        label_w=240,
    )

    # Figure B — interventions
    compose(
        out_path=OUT_DIR / "xmodel_interventions_grid.png",
        cols=[
            ("fig_dla_ablation.png",
             "DLA top-3 trio Δr per concept (red) vs random-3 control (gray)"),
            ("fig_cross_pair_matrix.png",
             "Cross-pair steering (slope ratio: source dir × target prompts)"),
        ],
        row_height=540,
        title="Causal interventions (attention ablation, cross-pair steering) replicate across small models",
        label_w=240,
    )


if __name__ == "__main__":
    main()
