from __future__ import annotations

import json

import pandas as pd
import yaml

from openjev_ja.visualize.radar import _validated_chart_data, load_radar_config, render_radar


def _write_config(tmp_path, source_name: str, output: str = "chart.png"):
    config = {
        "version": 1,
        "output": output,
        "axes": [
            {"id": "one", "label": "One"},
            {"id": "two", "label": "Two"},
            {"id": "three", "label": "Three"},
        ],
        "series": [
            {
                "name": "model",
                "source": {
                    "path": source_name,
                    "axis_column": "dataset",
                    "metric": "metrics.accuracy" if source_name.endswith(".json") else "accuracy",
                },
            }
        ],
    }
    path = tmp_path / "radar.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_json_table_source_and_render(tmp_path) -> None:
    rows = [
        {"dataset": "one", "metrics": {"accuracy": 0.2}},
        {"dataset": "two", "metrics": {"accuracy": 0.5}},
        {"dataset": "three", "metrics": {"accuracy": 0.8}},
    ]
    (tmp_path / "results.json").write_text(json.dumps(rows), encoding="utf-8")
    config_path = _write_config(tmp_path, "results.json")
    output = render_radar(config_path)
    assert output.is_file()
    assert output.read_bytes().startswith(b"\x89PNG")


def test_parquet_table_source(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {"dataset": "one", "accuracy": 20.0},
            {"dataset": "two", "accuracy": 50.0},
            {"dataset": "three", "accuracy": 80.0},
        ]
    )
    frame.to_parquet(tmp_path / "results.parquet")
    config_path = _write_config(tmp_path, "results.parquet")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["series"][0]["source"]["input_scale"] = "percent"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded, config_dir = load_radar_config(config_path)
    _, _, series = _validated_chart_data(loaded, config_dir)
    assert series[0]["values"] == [0.2, 0.5, 0.8]


def test_per_run_summary_points(tmp_path) -> None:
    for axis_id, accuracy in (("one", 0.1), ("two", 0.4), ("three", 0.9)):
        (tmp_path / f"{axis_id}.json").write_text(
            json.dumps({"accuracy": accuracy}), encoding="utf-8"
        )
    config = {
        "version": 1,
        "output": "points.svg",
        "axes": [{"id": axis_id} for axis_id in ("one", "two", "three")],
        "series": [
            {
                "name": "per-run",
                "points": {
                    axis_id: {"path": f"{axis_id}.json", "metric": "accuracy"}
                    for axis_id in ("one", "two", "three")
                },
            }
        ],
    }
    config_path = tmp_path / "points.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    loaded, config_dir = load_radar_config(config_path)
    _, _, series = _validated_chart_data(loaded, config_dir)
    assert series[0]["values"] == [0.1, 0.4, 0.9]
