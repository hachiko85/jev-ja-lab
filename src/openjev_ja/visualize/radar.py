from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class RadarConfigError(ValueError):
    """Raised when a radar configuration or result file is invalid."""


_THEMES: dict[str, dict[str, str]] = {
    "light": {
        "figure_color": "white",
        "plot_color": "#f4f5f7",
        "text_color": "#1a1a1a",
        "muted_color": "#666666",
        "tick_color": "#8b949e",
        "grid_color": "#cbd1d8",
        "spine_color": "#c3c9d0",
    },
    "dark": {
        "figure_color": "#12151c",
        "plot_color": "#1b2028",
        "text_color": "#e8eaed",
        "muted_color": "#9aa2b1",
        "tick_color": "#8b93a3",
        "grid_color": "#333a46",
        "spine_color": "#454d5c",
    },
}

_DEFAULT_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "Benchmark comparison",
        "note": "No text generation; one deterministic forward pass per item.",
    },
    "ja": {
        "title": "ベンチマーク比較",
        "note": "テキスト生成なし。項目ごとに決定論的な単回forward passで比較。",
    },
}


def _theme_colors(config: Mapping[str, Any]) -> dict[str, str]:
    theme_name = str(config.get("theme", "light")).lower()
    if theme_name not in _THEMES:
        raise RadarConfigError(f"unsupported theme: {theme_name!r} (use 'light' or 'dark')")
    colors = dict(_THEMES[theme_name])
    for key in colors:
        if key in config:
            colors[key] = str(config[key])
    return colors


def _localized_text(config: Mapping[str, Any], key: str) -> str | None:
    language = str(config.get("language", "en")).lower()
    return _DEFAULT_TEXT.get(language, _DEFAULT_TEXT["en"]).get(key)


def _require_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RadarConfigError(f"{location} must be a mapping")
    return value


def _resolve_path(config_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (config_dir / path).resolve()


def _nested_value(payload: Any, key: str) -> Any:
    current = payload
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise RadarConfigError(f"metric path not found: {key}")
        current = current[part]
    return current


def _records_from_file(path: Path, file_format: str) -> list[Mapping[str, Any]]:
    if file_format == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
            records = payload["rows"]
        else:
            records = [payload]
    elif file_format == "jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif file_format == "parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Install visualization dependencies: pip install -e '.[viz]'"
            ) from exc
        records = pd.read_parquet(path).to_dict(orient="records")
    else:
        raise RadarConfigError(f"unsupported result format: {file_format}")
    if not all(isinstance(record, Mapping) for record in records):
        raise RadarConfigError(f"result rows must be objects: {path}")
    return records


def _detect_format(path: Path, declared: str | None) -> str:
    if declared and declared != "auto":
        return declared.lower()
    suffix = path.suffix.lower()
    formats = {".json": "json", ".jsonl": "jsonl", ".parquet": "parquet"}
    try:
        return formats[suffix]
    except KeyError as exc:
        raise RadarConfigError(f"cannot detect result format from suffix: {path.name}") from exc


def _filter_records(
    records: list[Mapping[str, Any]], where: Mapping[str, Any], path: Path
) -> list[Mapping[str, Any]]:
    filtered = [
        record
        for record in records
        if all(_nested_value(record, str(key)) == expected for key, expected in where.items())
    ]
    if not filtered:
        raise RadarConfigError(f"no row matched filters in {path}")
    return filtered


def _normalize_value(value: Any, input_scale: str, location: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RadarConfigError(f"{location} must be numeric") from exc
    if input_scale == "percent":
        number /= 100.0
    elif input_scale != "fraction":
        raise RadarConfigError(f"unsupported input_scale: {input_scale}")
    if not 0.0 <= number <= 1.0:
        raise RadarConfigError(f"{location} must normalize to the 0-1 range; got {number}")
    return number


def _point_value(
    point: Mapping[str, Any], config_dir: Path, default_scale: str, location: str
) -> float:
    path = _resolve_path(config_dir, str(point["path"]))
    if not path.is_file():
        raise RadarConfigError(f"result file not found: {path}")
    file_format = _detect_format(path, point.get("format"))
    records = _records_from_file(path, file_format)
    where = _require_mapping(point.get("where", {}), f"{location}.where")
    records = _filter_records(records, where, path)
    if len(records) != 1:
        raise RadarConfigError(f"{location} matched {len(records)} rows; expected exactly one")
    metric = str(point.get("metric", "accuracy"))
    value = _nested_value(records[0], metric)
    return _normalize_value(value, str(point.get("input_scale", default_scale)), location)


def _series_from_points(
    series: Mapping[str, Any], axis_ids: list[str], config_dir: Path, default_scale: str
) -> list[float]:
    points = _require_mapping(series.get("points"), "series.points")
    missing = [axis_id for axis_id in axis_ids if axis_id not in points]
    if missing:
        raise RadarConfigError(f"series {series.get('name')!r} lacks axes: {', '.join(missing)}")
    return [
        _point_value(
            _require_mapping(points[axis_id], f"series.points.{axis_id}"),
            config_dir,
            str(series.get("input_scale", default_scale)),
            f"series.points.{axis_id}",
        )
        for axis_id in axis_ids
    ]


def _series_from_table(
    series: Mapping[str, Any], axis_ids: list[str], config_dir: Path, default_scale: str
) -> list[float]:
    source = _require_mapping(series.get("source"), "series.source")
    path = _resolve_path(config_dir, str(source["path"]))
    if not path.is_file():
        raise RadarConfigError(f"result file not found: {path}")
    records = _records_from_file(path, _detect_format(path, source.get("format")))
    where = _require_mapping(source.get("where", {}), "series.source.where")
    records = _filter_records(records, where, path)
    axis_column = str(source.get("axis_column", "dataset"))
    metric = str(source.get("metric", "accuracy"))
    by_axis: dict[str, Mapping[str, Any]] = {}
    for record in records:
        axis_id = str(_nested_value(record, axis_column))
        if axis_id in by_axis:
            raise RadarConfigError(f"duplicate axis {axis_id!r} in {path}")
        by_axis[axis_id] = record
    missing = [axis_id for axis_id in axis_ids if axis_id not in by_axis]
    if missing:
        raise RadarConfigError(f"table {path.name} lacks axes: {', '.join(missing)}")
    input_scale = str(source.get("input_scale", series.get("input_scale", default_scale)))
    return [
        _normalize_value(_nested_value(by_axis[axis_id], metric), input_scale, axis_id)
        for axis_id in axis_ids
    ]


def load_radar_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Install visualization dependencies: pip install -e '.[viz]'") from exc
    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = dict(_require_mapping(payload, "config"))
    if config.get("version", 1) != 1:
        raise RadarConfigError("only radar config version 1 is supported")
    return config, path.parent


def _validated_chart_data(
    config: Mapping[str, Any], config_dir: Path
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    axes_raw = config.get("axes")
    if not isinstance(axes_raw, Sequence) or isinstance(axes_raw, (str, bytes)):
        raise RadarConfigError("axes must be a list")
    axes = [_require_mapping(axis, "axes[]") for axis in axes_raw]
    if len(axes) < 3:
        raise RadarConfigError("a radar chart requires at least three axes")
    axis_ids = [str(axis["id"]) for axis in axes]
    if len(set(axis_ids)) != len(axis_ids):
        raise RadarConfigError("axis ids must be unique")
    axis_labels = [str(axis.get("label", axis["id"])) for axis in axes]
    series_raw = config.get("series")
    if not isinstance(series_raw, Sequence) or isinstance(series_raw, (str, bytes)):
        raise RadarConfigError("series must be a list")
    default_scale = str(config.get("input_scale", "fraction"))
    loaded: list[dict[str, Any]] = []
    for raw in series_raw:
        item = dict(_require_mapping(raw, "series[]"))
        if "name" not in item:
            raise RadarConfigError("every series requires a name")
        has_points, has_source = "points" in item, "source" in item
        if has_points == has_source:
            raise RadarConfigError(f"series {item['name']!r} requires exactly one of points/source")
        item["values"] = (
            _series_from_points(item, axis_ids, config_dir, default_scale)
            if has_points
            else _series_from_table(item, axis_ids, config_dir, default_scale)
        )
        loaded.append(item)
    if not loaded:
        raise RadarConfigError("at least one series is required")
    return axis_ids, axis_labels, loaded


def render_radar(
    config_path: str | Path,
    *,
    output_override: str | Path | None = None,
    show: bool = False,
) -> Path:
    config, config_dir = load_radar_config(config_path)
    _, labels, series = _validated_chart_data(config, config_dir)
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("Install visualization dependencies: pip install -e '.[viz]'") from exc
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_candidates = config.get(
        "font_family",
        ["BIZ UDGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo", "Noto Sans JP", "DejaVu Sans"],
    )
    if isinstance(font_candidates, str):
        font_candidates = [font_candidates]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    selected_font = next(
        (font for font in font_candidates if font in available_fonts), "DejaVu Sans"
    )
    plt.rcParams["font.family"] = selected_font
    theme = _theme_colors(config)
    count = len(labels)
    angles = [index * 2 * math.pi / count for index in range(count)]
    closed_angles = [*angles, angles[0]]
    width, height = config.get("figure_size", [10, 10])
    figure, axis = plt.subplots(figsize=(float(width), float(height)), subplot_kw={"polar": True})
    figure.patch.set_facecolor(theme["figure_color"])
    axis.set_facecolor(theme["plot_color"])
    axis.set_theta_offset(math.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles, labels=labels, fontsize=float(config.get("label_size", 12)))
    axis.tick_params(
        axis="x", colors=theme["text_color"], pad=float(config.get("label_pad", 18))
    )
    for angle, label in zip(angles, axis.get_xticklabels(), strict=True):
        horizontal = math.sin(angle)
        vertical = math.cos(angle)
        label.set_horizontalalignment(
            "left" if horizontal > 0.15 else "right" if horizontal < -0.15 else "center"
        )
        label.set_verticalalignment(
            "bottom" if vertical > 0.15 else "top" if vertical < -0.15 else "center"
        )
        label.set_linespacing(float(config.get("label_line_spacing", 1.15)))
    ticks = [float(value) for value in config.get("ticks", [0.2, 0.4, 0.6, 0.8, 1.0])]
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{value:.0%}" for value in ticks], color=theme["tick_color"])
    axis.grid(color=theme["grid_color"], linewidth=0.8, alpha=0.85)
    axis.spines["polar"].set_color(theme["spine_color"])
    for index, item in enumerate(series):
        values = [*item["values"], item["values"][0]]
        color = item.get("color", f"C{index}")
        axis.plot(
            closed_angles,
            values,
            color=color,
            linewidth=float(item.get("line_width", 2.2)),
            marker=item.get("marker", "o"),
            markersize=float(item.get("marker_size", 4.5)),
            label=str(item["name"]),
        )
        axis.fill(closed_angles, values, color=color, alpha=float(item.get("fill_alpha", 0.06)))
    default_title = _localized_text(config, "title") or "Benchmark comparison"
    title = str(config.get("title", default_title))
    figure.suptitle(
        title,
        fontsize=float(config.get("title_size", 20)),
        fontweight="bold",
        y=float(config.get("title_y", 0.975)),
        color=theme["text_color"],
    )
    subtitle = config.get("subtitle")
    if subtitle:
        figure.text(
            0.5,
            float(config.get("subtitle_y", 0.94)),
            str(subtitle),
            ha="center",
            color=theme["muted_color"],
            fontsize=float(config.get("subtitle_size", 10)),
        )
    legend_columns = int(config.get("legend_columns", min(3, len(series))))
    legend_rows = math.ceil(len(series) / legend_columns)
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=legend_columns,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_color(theme["text_color"])
    default_note = _localized_text(config, "note")
    note = config.get("note", default_note)
    if note:
        figure.text(
            0.5,
            float(config.get("note_y", 0.905)),
            str(note),
            ha="center",
            color=theme["muted_color"],
            fontsize=float(config.get("note_size", 8)),
        )
    bottom_margin = max(0.16, 0.16 + max(0, legend_rows - 2) * 0.035)
    figure.subplots_adjust(
        top=float(config.get("plot_top", 0.80)),
        bottom=float(config.get("plot_bottom", bottom_margin)),
        left=float(config.get("plot_left", 0.11)),
        right=float(config.get("plot_right", 0.89)),
    )
    raw_output = output_override or config.get("output")
    if not raw_output:
        raise RadarConfigError("output is required in YAML or --output")
    output = _resolve_path(config_dir, str(raw_output))
    if output.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise RadarConfigError("output suffix must be .png, .svg, or .pdf")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=int(config.get("dpi", 180)), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    return output
