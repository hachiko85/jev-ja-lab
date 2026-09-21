from __future__ import annotations

import argparse
import json
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from openjev_ja.visualize.radar import RadarConfigError, render_radar

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "reports" / "model-evaluation-dashboard" / "dist"
EXPORT_CONFIG = ROOT / "configs" / "visualization" / "dashboard-export.yaml"
COLORS = ["#59e1c2", "#72a8ff", "#f5b85b", "#f0799b", "#a98cff", "#2fb7a5"]


def _text(value: Any, name: str, limit: int = 120) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"invalid {name}")
    return value.strip()


def _chart_payload(raw: dict[str, Any], work: Path) -> tuple[Path, Path]:
    axes = raw.get("axes")
    series = raw.get("series")
    if not isinstance(axes, list) or not 3 <= len(axes) <= 30:
        raise ValueError("axes must contain 3-30 items")
    if not isinstance(series, list) or not 1 <= len(series) <= 6:
        raise ValueError("series must contain 1-6 items")
    axis_rows = []
    axis_ids: set[str] = set()
    for axis in axes:
        if not isinstance(axis, dict):
            raise ValueError("invalid axis")
        axis_id = _text(axis.get("id"), "axis id", 80)
        if axis_id in axis_ids:
            raise ValueError("duplicate axis id")
        axis_ids.add(axis_id)
        axis_rows.append({"id": axis_id, "label": _text(axis.get("label"), "axis label")})
    records = []
    chart_series = []
    for index, item in enumerate(series):
        if not isinstance(item, dict):
            raise ValueError("invalid series")
        model_id = _text(item.get("id"), "model id", 100)
        name = _text(item.get("name"), "model name")
        values = item.get("values")
        if not isinstance(values, list) or len(values) != len(axis_rows):
            raise ValueError("series values do not match axes")
        for axis, value in zip(axis_rows, values, strict=True):
            number = float(value)
            if not 0 <= number <= 1:
                raise ValueError("score must be between 0 and 1")
            records.append({"model": model_id, "axis": axis["id"], "score": number})
        chart_series.append(
            {
                "name": name,
                "color": COLORS[index],
                "source": {
                    "path": "values.json",
                    "format": "json",
                    "where": {"model": model_id},
                    "axis_column": "axis",
                    "metric": "score",
                },
            }
        )
    output_format = raw.get("format", "png")
    if output_format not in {"png", "svg"}:
        raise ValueError("format must be png or svg")
    theme = raw.get("theme", "dark")
    if theme not in {"dark", "light"}:
        raise ValueError("theme must be dark or light")
    base = yaml.safe_load(EXPORT_CONFIG.read_text(encoding="utf-8"))
    base.update(
        {
            "theme": theme,
            "title": _text(raw.get("title", base["title"]), "title", 160),
            "axes": axis_rows,
            "series": chart_series,
            "output": f"chart.{output_format}",
            "figure_size": [max(11, min(18, 8 + len(axis_rows) * 0.45)), 11],
        }
    )
    base.pop("subtitle", None)
    values_path = work / "values.json"
    config_path = work / "radar.yaml"
    values_path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(
        yaml.safe_dump(base, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return config_path, work / f"chart.{output_format}"


def _render_ranking(raw: dict[str, Any], work: Path) -> Path:
    rows = raw.get("rows")
    if not isinstance(rows, list) or not 1 <= len(rows) <= 100:
        raise ValueError("rows must contain 1-100 items")
    parsed = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid ranking row")
        value = float(row.get("value"))
        if value < 0:
            raise ValueError("ranking value must be non-negative")
        parsed.append((_text(row.get("name"), "model name"), value, bool(row.get("partial"))))
    output_format = raw.get("format", "png")
    if output_format not in {"png", "svg"}:
        raise ValueError("format must be png or svg")
    theme_name = raw.get("theme", "dark")
    if theme_name not in {"dark", "light"}:
        raise ValueError("theme must be dark or light")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_candidates = ["BIZ UDGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo"]
    plt.rcParams["font.family"] = next(
        (font for font in font_candidates if font in available_fonts), "DejaVu Sans"
    )

    dark = theme_name == "dark"
    figure_color = "#12151c" if dark else "white"
    plot_color = "#1b2028" if dark else "#f4f5f7"
    text_color = "#e8eaed" if dark else "#1a1a1a"
    muted_color = "#9aa2b1" if dark else "#666666"
    height = max(6.5, 0.42 * len(parsed) + 2.1)
    figure, axis = plt.subplots(figsize=(12, height))
    figure.patch.set_facecolor(figure_color)
    axis.set_facecolor(plot_color)
    names = [f"{name}{' *' if partial else ''}" for name, _, partial in parsed][::-1]
    values = [value for _, value, _ in parsed][::-1]
    colors = ["#f5b85b" if partial else "#59e1c2" for _, _, partial in parsed][::-1]
    bars = axis.barh(names, values, color=colors, alpha=0.9)
    is_latency = raw.get("metric") == "latency"
    axis.set_xlim(0, max(values) * 1.14 if values else 1)
    for bar, value in zip(bars, values, strict=True):
        label = f"{value:.1f} ms" if is_latency else f"{value:.1%}"
        axis.text(
            value + axis.get_xlim()[1] * 0.012,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            color=text_color,
            fontsize=10,
        )
    axis.tick_params(colors=text_color, labelsize=10)
    axis.xaxis.set_visible(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="x", color="#333a46" if dark else "#cbd1d8", alpha=0.5)
    figure.suptitle(
        _text(raw.get("title"), "title", 180),
        y=0.975,
        color=text_color,
        fontsize=20,
        fontweight="bold",
    )
    note = "* 暫定値: 完了済みPrimitiveのみの平均" if any(p for _, _, p in parsed) else ""
    if note:
        figure.text(0.5, 0.945, note, ha="center", color=muted_color, fontsize=9)
    figure.subplots_adjust(top=0.91, bottom=0.05, left=0.28, right=0.94)
    output = work / f"ranking.{output_format}"
    figure.savefig(output, dpi=360, bbox_inches="tight", facecolor=figure_color)
    plt.close(figure)
    return output


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/export-radar", "/api/export-ranking"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("invalid request size")
            raw = json.loads(self.rfile.read(length))
            if not isinstance(raw, dict):
                raise ValueError("body must be an object")
            with tempfile.TemporaryDirectory(prefix="openjev-radar-") as tmp:
                if self.path == "/api/export-radar":
                    config_path, output_path = _chart_payload(raw, Path(tmp))
                    render_radar(config_path)
                else:
                    output_path = _render_ranking(raw, Path(tmp))
                content = output_path.read_bytes()
                content_type = "image/png" if output_path.suffix == ".png" else "image/svg+xml"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                filename = f'openjev-{raw.get("view", "radar")}{output_path.suffix}'
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except (ValueError, TypeError, json.JSONDecodeError, RadarConfigError) as exc:
            body = json.dumps({"error": str(exc)}, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.BAD_REQUEST)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the OpenJEV dashboard and radar export API."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"OpenJEV dashboard: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
