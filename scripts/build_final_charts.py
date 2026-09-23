"""Build the final cross-method comparison table and charts under
results/eval-summary/: ranking bar, primitive breakdown, size-vs-accuracy
scatter, latency bar, and the 3-primitive radar. Run after
build_eval_summary.py (needs results/eval-summary/eval_summary.json).

Label convention for every chart: a two-tier label per method —
"<library/method name> [n-shot] [vVersion]" on top, "<base model> (<params>B)"
below it in a smaller/muted font. The second line is omitted when it would
just repeat the top line (e.g. a bare BERT/embedding model evaluated
directly). Jev's version comes from its own API response (`response_model`)
rather than being guessed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_common import FIG_BG, GRID, MUTED, PLOT_BG, TEXT, apply_font, draw_footer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/eval-summary"

COLORS = ["#4b83ad", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
          "#76b7b2", "#edc948", "#ff9da7", "#9c755f"]

# (model_id in eval_summary.json's `models[]`, or {noul,choice,score} model
# ids when the method is split per-primitive, or literal scores for a method
# with no local eval_summary entry e.g. Jev's own API-reported scores)
GROUPS = [
    {
        "method": "semif(2-shot)",
        "label_top": "semif 2-shot", "label_bottom": "Qwen3.5-4B (4.66B)",
        "noul": "semif-logit-fewshot-qwen3.5-4b-2shot-noul",
        "choice": "semif-logit-fewshot-qwen3.5-4b-2shot-choice",
        "score": "semif-logit-fewshot-qwen3.5-4b-2shot-score",
        "params_b": 4.66, "latency_ms": 90.00,
    },
    {
        "method": "semif-ja(zero-shot)",
        "label_top": "semif-ja 0-shot", "label_bottom": "Qwen3.5-4B (4.66B)",
        "single": "qwen3.5-4b", "params_b": 4.66, "latency_ms": 83.92,
    },
    {
        "method": "semif(zero-shot)",
        "label_top": "semif 0-shot", "label_bottom": "Qwen3.5-4B (4.66B)",
        "single": "semif-logit-qwen3.5-4b", "params_b": 4.66, "latency_ms": 65.87,
    },
    {
        "method": "AlexWortega_openjev 4B v2",
        "label_top": "AlexWortega_openjev 4B v2", "label_bottom": "Qwen3.5-4B (4.54B)",
        "single": "openjev-4b-v2", "params_b": 4.54, "latency_ms": 75.28,
    },
    {
        "method": "AlexWortega_openjev 0.8B",
        "label_top": "AlexWortega_openjev 0.8B v2", "label_bottom": "Qwen3.5-0.8B (0.85B)",
        "single": "openjev-0.8b", "params_b": 0.85, "latency_ms": 44.28,
    },
    {
        "method": "laya-multilingual",
        "label_top": "laya multilingual", "label_bottom": "ModernBERT・独自 (0.161B)",
        "noul": "laya-multilingual-noul", "choice": "laya-multilingual-choice",
        "score": "laya-multilingual-score", "params_b": 0.161, "latency_ms": 18.32,
    },
    {
        "method": "embedding: ruri-v3-310m",
        "label_top": "ruri-v3-310m", "label_bottom": "(0.315B)",
        "noul": "ruri-v3-310m-noul", "choice": "ruri-v3-310m-choice",
        "score": "ruri-v3-310m-score", "params_b": 0.315, "latency_ms": 1.35,
    },
    {
        "method": "bert: modernbert-ja-310m",
        "label_top": "modernbert-ja-310m", "label_bottom": "(0.315B)",
        "single": "modernbert-ja-310m", "params_b": 0.315, "latency_ms": 2.38,
    },
    {
        # Version pinned from the API's own response_model field, not guessed.
        "method": "Jev(jev-1.13.0)",
        "label_top": "Jev v1.13.0", "label_bottom": None,
        "noul": 0.7386120142176127, "choice": 0.8474204549250123, "score": 0.8821615110661686,
        "params_b": None, "latency_ms": 235.12,
        # Jev has no entry in eval-summary/eval_summary.json (API-only method,
        # never copied in by build_eval_summary.py) so per-dataset radar data
        # is read straight from its own run directory instead.
        "radar_run": "eval-jev-latest", "radar_model_id": "jev-latest",
    },
]

# Datasets common to every method above, per primitive (intersection checked
# by hand against each method's summary.<primitive>.json) — the axes for the
# per-primitive radar charts.
PRIMITIVE_DATASETS: dict[str, list[str]] = {
    "noul": [
        "jad_afc_false", "jad_afc_nei", "jad_afc_true", "janli_entailment",
        "jcola_in_domain", "jcola_out_of_domain", "jnli_contradiction",
        "jnli_entailment", "jnli_missing_evidence", "paws_x_ja", "textdetox_ja",
        "wrime_anger_binary", "wrime_joy_binary", "wrime_positive_binary",
    ],
    "choice": [
        "mmmlu_ja", "jmmlu", "jgpqa_diamond", "jcommonsenseqa", "xwinograd_ja",
        "mgsm_ja", "gsm8k_ja_mc4", "gsm8k_ja_mc10", "jnli",
    ],
    "score": [
        "wrime_joy", "wrime_anger", "wrime_sentiment", "synthetic_urgency",
        "synthetic_dissatisfaction", "synthetic_risk", "synthetic_relevance",
    ],
}
PRIMITIVE_METRIC = {
    "noul": "f1", "choice": "accuracy", "score": "normalized_quadratic_weighted_kappa",
}
DATASET_LABELS = {
    "jad_afc_false": "JAD-AFC False", "jad_afc_nei": "JAD-AFC NEI",
    "jad_afc_true": "JAD-AFC True", "janli_entailment": "JaNLI",
    "jcola_in_domain": "JCoLA In", "jcola_out_of_domain": "JCoLA Out",
    "jnli_contradiction": "JNLI矛盾", "jnli_entailment": "JNLI含意",
    "jnli_missing_evidence": "JNLI情報不足", "paws_x_ja": "PAWS-X",
    "textdetox_ja": "TextDetox", "wrime_anger_binary": "WRIME怒り有無",
    "wrime_joy_binary": "WRIME喜び有無", "wrime_positive_binary": "WRIMEポジ有無",
    "mmmlu_ja": "MMMLU", "jmmlu": "JMMLU", "jgpqa_diamond": "JGPQA",
    "jcommonsenseqa": "JCommonsenseQA", "xwinograd_ja": "XWinograd",
    "mgsm_ja": "MGSM", "gsm8k_ja_mc4": "GSM8K MC4", "gsm8k_ja_mc10": "GSM8K MC10",
    "jnli": "JNLI", "wrime_joy": "WRIME喜び", "wrime_anger": "WRIME怒り",
    "wrime_sentiment": "WRIME極性", "synthetic_urgency": "緊急度",
    "synthetic_dissatisfaction": "不満度", "synthetic_risk": "危険度",
    "synthetic_relevance": "関連度",
}


def build_table() -> list[dict]:
    summary = json.loads((OUT / "eval_summary.json").read_text(encoding="utf-8"))
    models = {m["id"]: m for m in summary["models"]}

    def prim_score(model_id: str, primitive: str) -> float | None:
        m = models.get(model_id)
        if m is None:
            return None
        p = m["primitives"].get(primitive)
        return p.get("score") if p else None

    rows = []
    for g in GROUPS:
        if "single" in g:
            mid = g["single"]
            noul = prim_score(mid, "noul")
            choice = prim_score(mid, "choice")
            score = prim_score(mid, "score")
        elif isinstance(g.get("noul"), str):
            noul = prim_score(g["noul"], "noul")
            choice = prim_score(g["choice"], "choice")
            score = prim_score(g["score"], "score")
        else:
            noul, choice, score = g["noul"], g["choice"], g["score"]
        vals = [v for v in (noul, choice, score) if v is not None]
        overall = sum(vals) / len(vals) if vals else None
        rows.append({
            "method": g["method"], "label_top": g["label_top"], "label_bottom": g["label_bottom"],
            "noul": noul, "choice": choice, "score": score, "overall": overall,
            "params_b": g["params_b"], "latency_ms": g["latency_ms"],
        })
    rows.sort(key=lambda r: -(r["overall"] or 0))
    table_path = OUT / "final_methods_table.json"
    table_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {table_path} ({len(rows)} rows)")
    return rows


def build_ranking_chart(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: r["overall"])
    apply_font(plt)
    highlight, bar_color = "#59a14f", "#4b83ad"
    overall = [r["overall"] for r in rows]
    colors = [highlight if i == len(rows) - 1 else bar_color for i in range(len(rows))]
    y_pos = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PLOT_BG)
    bars = ax.barh(y_pos, overall, color=colors, height=0.6)
    for bar, value in zip(bars, overall, strict=True):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                va="center", ha="left", color=TEXT, fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlim(0, max(overall) * 1.18)
    for pos, r in zip(y_pos, rows, strict=True):
        ax.text(-0.012, pos + 0.12, r["label_top"], transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=TEXT, fontsize=11)
        if r.get("label_bottom"):
            ax.text(-0.012, pos - 0.16, r["label_bottom"], transform=ax.get_yaxis_transform(),
                    ha="right", va="center", color=MUTED, fontsize=8)
    ax.set_xlabel("Overall(noul/choice/score 平均)", color=MUTED, fontsize=10)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    fig.suptitle("手法別ランキング", fontsize=17, fontweight="bold", color=TEXT, y=0.97)
    fig.text(0.5, 0.925, "3primitive(Noul/Choice/Score)の等加重平均、降順",
              ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.19, right=0.95)
    draw_footer(fig)
    out = OUT / "ranking-bar-chart.png"
    fig.savefig(out, dpi=180, facecolor=FIG_BG)
    plt.close(fig)
    print("wrote", out)


def build_primitive_chart(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: -r["overall"])
    apply_font(plt)
    colors = {"noul": "#e15759", "choice": "#4b83ad", "score": "#59a14f"}
    labels = {"noul": "Noul(F1)", "choice": "Choice(accuracy)", "score": "Score(QWK)"}
    n = len(rows)
    x = np.arange(n)
    width = 0.26

    fig, ax = plt.subplots(figsize=(17, 9))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PLOT_BG)
    for i, prim in enumerate(("noul", "choice", "score")):
        values = [r[prim] for r in rows]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, color=colors[prim], label=labels[prim])
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("スコア(0-1正規化)", color=MUTED, fontsize=10)
    ax.tick_params(axis="y", colors=MUTED)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    for xi, r in zip(x, rows, strict=True):
        ax.text(xi, -0.025, r["label_top"], transform=ax.get_xaxis_transform(),
                ha="right", va="top", color=TEXT, fontsize=9.5, rotation=25, rotation_mode="anchor")
        if r.get("label_bottom"):
            ax.text(xi, -0.11, r["label_bottom"], transform=ax.get_xaxis_transform(),
                    ha="right", va="top", color=MUTED, fontsize=7,
                    rotation=25, rotation_mode="anchor")
    legend = ax.legend(loc="upper right", frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.1))
    for text in legend.get_texts():
        text.set_color(TEXT)
    fig.suptitle("primitive別スコア比較", fontsize=17, fontweight="bold", color=TEXT, y=0.975)
    fig.text(0.5, 0.93, "Noul(F1) / Choice(accuracy) / Score(QWK) の手法別内訳",
              ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.87, bottom=0.28, left=0.055, right=0.98)
    draw_footer(fig)
    out = OUT / "primitive-comparison-bar-chart.png"
    fig.savefig(out, dpi=180, facecolor=FIG_BG)
    plt.close(fig)
    print("wrote", out)


def build_scatter_and_latency_charts(rows: list[dict]) -> None:
    apply_font(plt)

    scatter_rows = [r for r in rows if r["params_b"] is not None]
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PLOT_BG)
    for i, r in enumerate(scatter_rows):
        bottom = r.get("label_bottom")
        label = f"{r['label_top']} {bottom}" if bottom else r["label_top"]
        ax.scatter(r["params_b"], r["overall"], s=160, color=COLORS[i % len(COLORS)], zorder=3,
                   edgecolors=FIG_BG, linewidths=1.2)
        ha = "right" if r["params_b"] > 3.5 else "left"
        xoff = -10 if ha == "right" else 8
        ax.annotate(label, (r["params_b"], r["overall"]),
                    textcoords="offset points", xytext=(xoff, 6), color=TEXT, fontsize=9, ha=ha)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 9)
    ax.set_xlabel("パラメータ数(B、log scale)", color=MUTED, fontsize=10)
    ax.set_ylabel("Overall", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.8, which="both")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    fig.suptitle("モデルサイズ vs 精度", fontsize=17, fontweight="bold", color=TEXT, y=0.975)
    fig.text(0.5, 0.925, "Jev(パラメータ非公開)は対象外", ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.09, right=0.97)
    draw_footer(fig)
    out1 = OUT / "size-vs-accuracy-scatter.png"
    fig.savefig(out1, dpi=180, facecolor=FIG_BG)
    plt.close(fig)
    print("wrote", out1)

    rows_sorted = sorted(rows, key=lambda r: r["latency_ms"])
    latencies = [r["latency_ms"] for r in rows_sorted]
    y_pos = list(range(len(rows_sorted)))
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor(FIG_BG)
    ax.set_facecolor(PLOT_BG)
    bars = ax.barh(y_pos, latencies, color="#e15759", height=0.6)
    for bar, value in zip(bars, latencies, strict=True):
        ax.text(value * 1.06, bar.get_y() + bar.get_height() / 2, f"{value:.2f} ms",
                va="center", ha="left", color=TEXT, fontsize=10)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    for pos, r in zip(y_pos, rows_sorted, strict=True):
        ax.text(-0.012, pos + 0.12, r["label_top"], transform=ax.get_yaxis_transform(),
                ha="right", va="center", color=TEXT, fontsize=11)
        if r.get("label_bottom"):
            ax.text(-0.012, pos - 0.16, r["label_bottom"], transform=ax.get_yaxis_transform(),
                    ha="right", va="center", color=MUTED, fontsize=8)
    ax.set_xlabel("平均推論時間(ms/件、log scale)", color=MUTED, fontsize=10)
    ax.tick_params(axis="x", colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.8, which="both")
    for spine in ax.spines.values():
        spine.set_color(GRID)
    fig.suptitle("手法別 平均推論時間", fontsize=17, fontweight="bold", color=TEXT, y=0.97)
    fig.text(0.5, 0.925, "全データセット件数加重平均。Jevはネットワーク往復込み(API)、他はGPU",
              ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.22, right=0.95)
    draw_footer(fig)
    out2 = OUT / "latency-bar-chart.png"
    fig.savefig(out2, dpi=180, facecolor=FIG_BG)
    plt.close(fig)
    print("wrote", out2)


def _legend_name(r: dict) -> str:
    if r.get("label_bottom"):
        return f"{r['label_top']} [{r['label_bottom']}]"
    return r["label_top"]


def _render_radar(
    series: list[dict], axis_labels: list[str], *,
    title: str, subtitle: str, out_path: Path, label_fontsize: float = 13,
) -> None:
    apply_font(plt)
    theme = {
        "figure_color": FIG_BG, "plot_color": PLOT_BG, "text_color": TEXT,
        "muted_color": MUTED, "tick_color": "#8b93a3", "grid_color": GRID,
        "spine_color": "#454d5c",
    }
    count = len(axis_labels)
    angles = [i * 2 * math.pi / count for i in range(count)]
    closed_angles = [*angles, angles[0]]
    figure, axis = plt.subplots(figsize=(13, 13.8), subplot_kw={"polar": True})
    figure.patch.set_facecolor(theme["figure_color"])
    axis.set_facecolor(theme["plot_color"])
    axis.set_theta_offset(math.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_xticks(angles, labels=axis_labels, fontsize=label_fontsize)
    axis.tick_params(axis="x", colors=theme["text_color"], pad=16)
    for angle, label in zip(angles, axis.get_xticklabels(), strict=True):
        h, v = math.sin(angle), math.cos(angle)
        label.set_horizontalalignment("left" if h > 0.15 else "right" if h < -0.15 else "center")
        label.set_verticalalignment("bottom" if v > 0.15 else "top" if v < -0.15 else "center")
    ticks = [0.2, 0.4, 0.6, 0.8, 1.0]
    axis.set_ylim(0.0, 1.0)
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{v:.0%}" for v in ticks], color=theme["tick_color"])
    axis.grid(color=theme["grid_color"], linewidth=0.8, alpha=0.85)
    axis.spines["polar"].set_color(theme["spine_color"])
    for item in series:
        values = [*item["values"], item["values"][0]]
        axis.plot(closed_angles, values, color=item["color"], linewidth=2.2,
                  marker="o", markersize=4.5, label=item["name"])
        axis.fill(closed_angles, values, color=item["color"], alpha=0.06)
    figure.suptitle(title, fontsize=19, fontweight="bold", y=0.965, color=theme["text_color"])
    figure.text(0.5, 0.935, subtitle, ha="center", color=theme["muted_color"], fontsize=10)
    legend = axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=1, frameon=False)
    for text in legend.get_texts():
        text.set_color(theme["text_color"])
    figure.subplots_adjust(top=0.86, bottom=0.18, left=0.11, right=0.89)
    draw_footer(figure)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)
    print("wrote", out_path)


def build_radar_chart(rows: list[dict]) -> None:
    axis_ids = ("noul", "choice", "score")
    axis_labels = ["Noul\n(二値判定・F1)", "Choice\n(多肢選択・accuracy)", "Score\n(順序尺度・QWK)"]
    series = [
        {"name": _legend_name(r), "color": COLORS[i % len(COLORS)],
         "values": [r[axis] for axis in axis_ids]}
        for i, r in enumerate(rows)
    ]
    _render_radar(
        series, axis_labels, title="全手法比較",
        subtitle="Noul(F1) / Choice(accuracy) / Score(QWK)",
        out_path=OUT / "radar-final-all-methods.png",
    )
    yaml_out = OUT / "radar-final-all-methods.yaml"
    yaml_out.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "axes": list(axis_ids),
                "series": [{"name": s["name"], "values": s["values"]} for s in series],
            },
            allow_unicode=True, sort_keys=False,
        ),
        encoding="utf-8",
    )
    print("wrote", yaml_out)


def _dataset_scores(g: dict, primitive: str) -> dict[str, float]:
    run = g.get("radar_run", "eval-summary")
    model_id = g["radar_model_id"] if "radar_model_id" in g else g.get("single", g.get(primitive))
    path = ROOT / "results" / run / model_id / f"summary.{primitive}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    metric = PRIMITIVE_METRIC[primitive]
    return {r["dataset"]: r[metric] for r in rows}


def build_per_primitive_radars(rows: list[dict]) -> None:
    """One radar per primitive: axes = datasets common to all 9 methods
    within that primitive, series = the methods, values = the primitive's
    own metric (F1/accuracy/QWK) per dataset."""
    method_to_group = {g["method"]: g for g in GROUPS}
    titles = {"noul": "Noul詳細比較", "choice": "Choice詳細比較", "score": "Score詳細比較"}
    subtitles = {
        "noul": "データセット別F1(全手法共通14件)",
        "choice": "データセット別accuracy(全手法共通9件)",
        "score": "データセット別normalized QWK(全手法共通7件)",
    }
    for primitive in ("noul", "choice", "score"):
        axes = PRIMITIVE_DATASETS[primitive]
        axis_labels = [DATASET_LABELS.get(a, a) for a in axes]
        series = []
        for i, row in enumerate(rows):
            group = method_to_group[row["method"]]
            scores = _dataset_scores(group, primitive)
            series.append({
                "name": _legend_name(row), "color": COLORS[i % len(COLORS)],
                "values": [scores[a] for a in axes],
            })
        label_fontsize = 11 if len(axes) <= 9 else 9.5
        _render_radar(
            series, axis_labels, title=titles[primitive], subtitle=subtitles[primitive],
            out_path=OUT / f"radar-{primitive}-detail.png", label_fontsize=label_fontsize,
        )


def main() -> None:
    rows = build_table()
    build_ranking_chart(rows)
    build_primitive_chart(rows)
    build_scatter_and_latency_charts(rows)
    build_radar_chart(rows)
    build_per_primitive_radars(rows)


if __name__ == "__main__":
    main()
