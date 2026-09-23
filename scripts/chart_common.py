"""Shared chart branding: project name/version footer, repo QR code, watermark.

Used by scripts/build_final_charts.py (and any other comparison chart) so
every generated figure carries the same footer: project name+version,
repo label+URL, a QR code linking to the repo, and an optional faint
watermark image (assets/branding/watermark.png) in the bottom-left corner.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

import matplotlib.image as mpimg
import qrcode

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/hachiko85/jev-ja-lab"
REPO_LABEL = "hachiko85/jev-ja-lab"
WATERMARK_PATH = ROOT / "assets/branding/watermark.png"
_QR_CACHE_PATH = ROOT / "assets/branding/.repo_qr_cache.png"

FIG_BG = "#12151c"
PLOT_BG = "#1b2028"
TEXT = "#e8eaed"
MUTED = "#9aa2b1"
GRID = "#333a46"


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
        qrcode.make(REPO_URL, border=1).save(_QR_CACHE_PATH)
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
    """Bottom-right: project name+version, repo label+URL, QR code (QR sits
    in its own reserved corner so text never overlaps it).
    Optional faint watermark in the bottom-left corner.
    """
    text_right_edge = 0.895
    fig.text(
        text_right_edge, 0.05, project_name_version(),
        ha="right", va="bottom", color=TEXT, fontsize=8.5, fontweight="bold",
    )
    fig.text(
        text_right_edge, 0.027, REPO_LABEL,
        ha="right", va="bottom", color=MUTED, fontsize=6.5,
    )
    fig.text(
        text_right_edge, 0.008, REPO_URL,
        ha="right", va="bottom", color=MUTED, fontsize=6.5,
    )
    try:
        qr_img = mpimg.imread(_ensure_qr())
        qr_ax = fig.add_axes([0.92, 0.008, 0.07, 0.07])
        qr_ax.imshow(qr_img)
        qr_ax.axis("off")
    except Exception:
        pass
    if watermark and WATERMARK_PATH.is_file():
        try:
            wm_img = mpimg.imread(WATERMARK_PATH)
            wm_ax = fig.add_axes([0.012, 0.008, 0.06, 0.06], zorder=0)
            wm_ax.imshow(wm_img, alpha=0.35)
            wm_ax.axis("off")
        except Exception:
            pass
