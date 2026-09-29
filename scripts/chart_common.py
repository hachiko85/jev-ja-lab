"""Shared chart theme (plain white background) and the optional branded footer
(logo, author, project+version, repo URL, QR code). The final comparison charts use the
plain theme without the footer."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

import matplotlib.image as mpimg
import qrcode
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/hachiko85/jev-ja-lab"
AUTHOR = "hachiko85"
LOGO_PATH = ROOT / "assets/branding/watermark_circle.png"
_QR_CACHE_PATH = ROOT / "assets/branding/.repo_qr_cache.png"

FIG_BG = "#ffffff"
PLOT_BG = "#ffffff"
TEXT = "#1f2328"
MUTED = "#57606a"
GRID = "#d0d7de"


@lru_cache(maxsize=1)
def project_name_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data.get("project", {})
    name = project.get("name", "jev-ja-lab")
    version = project.get("version", "0.0.0")
    return f"{name} v{version}"


def _ensure_qr() -> Path:
    if not _QR_CACHE_PATH.is_file():
        _QR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # qrcode.make() returns a 1-bit image; matplotlib's imshow applies a
        # colormap to single-channel data instead of treating it as B/W, so
        # convert to plain RGB before saving.
        img = qrcode.make(REPO_URL, border=1).convert("RGB")
        img.save(_QR_CACHE_PATH)
    return _QR_CACHE_PATH


def apply_font(plt) -> str:
    from matplotlib import font_manager

    font_candidates = [
        "BIZ UDGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Noto Sans JP", "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in font_candidates if f in available), "DejaVu Sans")
    plt.rcParams["font.family"] = font
    return font


def draw_footer(fig, *, watermark: bool = True) -> None:
    """Single-line footer: [circular logo] author  ...  project vX / repo URL [QR].

    Logo+author sit at the left, project/version/URL+QR at the right, all on
    one row so the footer stays a thin strip regardless of figure width.
    """
    row_y = 0.012
    row_h = 0.05

    if watermark and LOGO_PATH.is_file():
        try:
            logo_img = mpimg.imread(LOGO_PATH)
            logo_ax = fig.add_axes([0.012, row_y, row_h * 0.85, row_h * 0.85])
            logo_ax.imshow(logo_img)
            logo_ax.axis("off")
            fig.text(
                0.012 + row_h * 0.85 + 0.008, row_y + row_h * 0.425, AUTHOR,
                ha="left", va="center", color=MUTED, fontsize=8.5,
            )
        except Exception:
            pass

    qr_size = row_h * 0.85
    qr_x = 0.985 - qr_size
    text_right_edge = qr_x - 0.01
    fig.text(
        text_right_edge, row_y + row_h * 0.425,
        f"{project_name_version()} / {REPO_URL}",
        ha="right", va="center", color=MUTED, fontsize=7.5,
    )
    try:
        qr_img = mpimg.imread(_ensure_qr())
        qr_ax = fig.add_axes([qr_x, row_y, qr_size, qr_size])
        qr_ax.imshow(qr_img)
        qr_ax.axis("off")
    except Exception:
        pass


def _make_circular_logo(source: Path, dest: Path, size: int = 256) -> None:
    """One-off helper (not called at chart-build time) to regenerate the
    circular-cropped logo from a square source image."""
    img = Image.open(source).convert("RGBA")
    edge = min(img.size)
    img = img.crop((0, 0, edge, edge)).resize((size, size), Image.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    from PIL import ImageDraw

    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    img.putalpha(mask)
    img.save(dest)
