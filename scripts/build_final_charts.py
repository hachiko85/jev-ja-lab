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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chart_common import FIG_BG, GRID, MUTED, PLOT_BG, TEXT, apply_font

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/eval-summary"

COLORS = ["#4b83ad", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
          "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#17becf", "#8c564b"]

# 2-shot variants (semif, decider) are evaluated but left out of the charts for now.
# (model_id in eval_summary.json's `models[]`, or {noul,choice,score} model
# ids when the method is split per-primitive, or literal scores for a method
# with no local eval_summary entry e.g. Jev's own API-reported scores)
GROUPS = [
    {
        "method": "semif-ja(zero-shot)",
        "helpsteer": "qwen3.5-4b",
        "label_top": "semif-ja 0-shot", "label_bottom": "Qwen3.5-4B (4.66B)",
        "single": "qwen3.5-4b", "params_b": 4.66, "latency_ms": 83.92,
    },
    {
        "method": "semif(zero-shot)",
        "helpsteer": "semif-logit-qwen3.5-4b",
        "label_top": "semif 0-shot", "label_bottom": "Qwen3.5-4B (4.66B)",
        "single": "semif-logit-qwen3.5-4b", "params_b": 4.66, "latency_ms": 65.87,
    },
    {
        "method": "AlexWortega_openjev 4B v2",
        "helpsteer": "openjev-4b-v2",
        "label_top": "AlexWortega_openjev 4B v2", "label_bottom": "Qwen3.5-4B (4.54B)",
        "single": "openjev-4b-v2", "params_b": 4.54, "latency_ms": 75.28,
    },
    {
        "method": "laya-multilingual",
        "helpsteer": "laya-multilingual-score",
        "label_top": "laya multilingual", "label_bottom": "ModernBERT・独自 (0.161B)",
        "noul": "laya-multilingual-noul", "choice": "laya-multilingual-choice",
        "score": "laya-multilingual-score", "params_b": 0.161, "latency_ms": 18.32,
    },
    {
        "method": "bert: modernbert-ja-310m",
        "helpsteer": "modernbert-ja-310m",
        "label_top": "modernbert-ja-310m", "label_bottom": "(0.315B)",
        "single": "modernbert-ja-310m", "params_b": 0.315, "latency_ms": 2.38,
    },
    {
        "method": "Hopper(zero-shot)",
        "helpsteer": "hopper-qwen3.5-4b-lora-score",
        "label_top": "Hopper 0-shot", "label_bottom": "Qwen3.5-4B + LoRA (4.66B)",
        "noul": "hopper-qwen3.5-4b-lora-noul", "choice": "hopper-qwen3.5-4b-lora-choice",
        "score": "hopper-qwen3.5-4b-lora-score", "params_b": 4.66, "latency_ms": 71.06,
    },
    {
        "method": "decider-4b v2",
        "helpsteer": "decider-4b-v2-score",
        "label_top": "decider v2", "label_bottom": "Qwen3.5-4B-Base (4.2B)",
        "noul": "decider-4b-v2-noul", "choice": "decider-4b-v2-choice",
        "score": "decider-4b-v2-score", "params_b": 4.2, "latency_ms": 70.43,
    },

    {
        "method": "CLM v0.1",
        "helpsteer": "clm-v0.1-8b-score",
        "label_top": "CLM v0.1", "label_bottom": "Qwen3-8B + heads (8.2B)",
        "noul": "clm-v0.1-8b-noul", "choice": "clm-v0.1-8b-choice",
        "score": "clm-v0.1-8b-score", "params_b": 8.2, "latency_ms": 89.33,
    },
    {
        # Version pinned from the API's own response_model field, not guessed.
        "method": "Jev(jev-1.13.0)",
        "helpsteer": "jev-latest", "helpsteer_run": "eval-helpsteer-jev",
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
    # jad_afc_* excluded: matches the dataset set used by the reference
    # dashboard screenshot this radar's axes/labels were aligned to.
    "noul": [
        "janli_entailment", "jcola_in_domain", "jcola_out_of_domain",
        "jnli_contradiction", "jnli_entailment", "jnli_missing_evidence",
        "paws_x_ja", "textdetox_ja", "wrime_anger_binary", "wrime_joy_binary",
        "wrime_positive_binary",
    ],
    "choice": [
        "gsm8k_ja_mc10", "gsm8k_ja_mc4", "xwinograd_ja", "jcommonsenseqa",
        "mmmlu_ja", "jgpqa_diamond", "mgsm_ja", "jnli", "jmmlu",
    ],
    "score": [
        "synthetic_dissatisfaction", "synthetic_relevance", "synthetic_risk",
        "synthetic_urgency", "wrime_anger", "wrime_joy", "wrime_sentiment",
    ],
}
PRIMITIVE_METRIC = {
    "noul": "f1", "choice": "accuracy", "score": "normalized_quadratic_weighted_kappa",
}
# "<dataset name>\n（<what it measures>）" — matches the reference dashboard's
# axis-label convention so datasets are identifiable without prior context.
DATASET_LABELS = {
    "janli_entailment": "JaNLI Entailment\n（難しい言い換えの含意判定）",
    "jcola_in_domain": "JCoLA In-domain\n（既知構文の文法判断）",
    "jcola_out_of_domain": "JCoLA Out-of-domain\n（未知構文の文法判断）",
    "jnli_contradiction": "JNLI Contradiction\n（前提と仮説が矛盾するか）",
    "jnli_entailment": "JNLI Entailment\n（前提から仮説を導けるか）",
    "jnli_missing_evidence": "JNLI Neutral\n（真偽を決める情報が不足か）",
    "paws_x_ja": "PAWS-X Japanese\n（2文が同じ意味か）",
    "textdetox_ja": "TextDetox Japanese\n（日本語の有害表現検出）",
    "wrime_anger_binary": "WRIME Anger Binary\n（読み手が怒りを感じるか）",
    "wrime_joy_binary": "WRIME Joy Binary\n（読み手が喜びを感じるか）",
    "wrime_positive_binary": "WRIME Positive Binary\n（読み手が肯定感情を持つか）",
    "gsm8k_ja_mc10": "GSM8K JA MC10\n（数値推論・10択）",
    "gsm8k_ja_mc4": "GSM8K JA MC4\n（数値推論・4択）",
    "xwinograd_ja": "XWinograd JA\n（文脈・照応解析）",
    "jcommonsenseqa": "JCommonsenseQA\n（日本語常識推論）",
    "mmmlu_ja": "MMMLU JA-JP\n（一般・学術知識）",
    "jgpqa_diamond": "JGPQA Diamond\n（高難度科学推論）",
    "mgsm_ja": "MGSM JA\n（日本語数学推論）",
    "jnli": "JNLI 3-class\n（日本語NLI・意味理解）",
    "jmmlu": "JMMLU\n（日本語・日本固有知識）",
    "synthetic_dissatisfaction": "Synthetic Dissatisfaction\n（顧客の不満度・5段階）",
    "synthetic_relevance": "Synthetic Relevance\n（質問と回答の関連度・5段階）",
    "synthetic_risk": "Synthetic Risk\n（事故・損失リスク・5段階）",
    "synthetic_urgency": "Synthetic Urgency\n（業務対応の緊急度・5段階）",
    "wrime_anger": "WRIME Anger\n（読み手の怒り強度・4段階）",
    "wrime_joy": "WRIME Joy\n（読み手の喜び強度・4段階）",
    "wrime_sentiment": "WRIME Sentiment\n（否定〜肯定の感情・5段階）",
}


HELPSTEER_AXES = ("correctness", "helpfulness", "verbosity", "complexity", "coherence")
HELPSTEER_LABELS = {
    "correctness": "Correctness\n（回答の正確性）",
    "helpfulness": "Helpfulness\n（回答の有用性）",
    "verbosity": "Verbosity\n（回答の詳細度）",
    "complexity": "Complexity\n（回答の複雑さ）",
    "coherence": "Coherence\n（回答の一貫性）",
}


def _helpsteer_scores(g: dict) -> dict[str, float]:
    """Per-axis normalized QWK on HelpSteer2-JA benchmark-v1 (2,500 rows), the extra Score
    indicator kept out of the 30-dataset Score aggregate. Empty until that run exists."""
    run = g.get("helpsteer_run", "eval-helpsteer-series")
    path = ROOT / "results" / run / g["helpsteer"] / "summary.score.json"
    if not path.is_file():
        return {}
    rows = {r["dataset"]: r for r in json.loads(path.read_text(encoding="utf-8"))}
    axes = {a: rows.get(f"helpsteer_{a}") for a in HELPSTEER_AXES}
    if any(v is None for v in axes.values()):
        return {}
    return {a: v[PRIMITIVE_METRIC["score"]] for a, v in axes.items()}


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
        helpsteer_axes = _helpsteer_scores(g)
        helpsteer = (
            sum(helpsteer_axes.values()) / len(helpsteer_axes) if helpsteer_axes else None
        )
        vals = [v for v in (noul, choice, score) if v is not None]
        overall = sum(vals) / len(vals) if vals else None
        rows.append({
            "method": g["method"], "label_top": g["label_top"], "label_bottom": g["label_bottom"],
            "noul": noul, "choice": choice, "score": score, "overall": overall,
            "helpsteer": helpsteer, "helpsteer_axes": helpsteer_axes,
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
    fig.suptitle("ライブラリ別総合スコア", fontsize=17, fontweight="bold", color=TEXT, y=0.97)
    fig.text(0.5, 0.925, "3primitive(Noul/Choice/Score)の等加重平均、降順",
              ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.11, left=0.19, right=0.95)
    out = OUT / "ranking-bar-chart.png"
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
    fig.suptitle("ライブラリ別平均推論時間", fontsize=17, fontweight="bold", color=TEXT, y=0.97)
    fig.text(0.5, 0.925, "全データセット件数加重平均。Jevはネットワーク往復込み(API)、他はGPU",
              ha="center", color=MUTED, fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.22, right=0.95)
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
    h_margin: float = 0.11,
) -> None:
    apply_font(plt)
    theme = {
        "figure_color": FIG_BG, "plot_color": PLOT_BG, "text_color": TEXT,
        "muted_color": MUTED, "tick_color": MUTED, "grid_color": GRID,
        "spine_color": "#8c959f",
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
    legend = axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1, frameon=False)
    for text in legend.get_texts():
        text.set_color(theme["text_color"])
    figure.subplots_adjust(top=0.86, bottom=0.20, left=h_margin, right=1 - h_margin)
    figure.savefig(out_path, dpi=180)
    plt.close(figure)
    print("wrote", out_path)


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
    titles = {"noul": "Noul", "choice": "Choice", "score": "Score"}
    metric_labels = {"noul": "F1", "choice": "accuracy", "score": "normalized QWK"}
    for primitive in ("noul", "choice", "score"):
        axes = PRIMITIVE_DATASETS[primitive]
        axis_labels = [DATASET_LABELS.get(a, a) for a in axes]
        # Score also carries the HelpSteer2-JA axes (extra indicator); only methods that
        # have that run can be drawn on the extended radar.
        with_helpsteer = primitive == "score" and all(r.get("helpsteer_axes") for r in rows)
        if with_helpsteer:
            axis_labels += [f"HelpSteer2 {HELPSTEER_LABELS[a]}" for a in HELPSTEER_AXES]
        series = []
        for i, row in enumerate(rows):
            group = method_to_group[row["method"]]
            scores = _dataset_scores(group, primitive)
            values = [scores[a] for a in axes]
            if with_helpsteer:
                values += [row["helpsteer_axes"][a] for a in HELPSTEER_AXES]
            series.append({
                "name": _legend_name(row), "color": COLORS[i % len(COLORS)],
                "values": values,
            })
        count = len(axis_labels)
        subtitle = f"データセット別{metric_labels[primitive]}(全ライブラリ共通{count}件)"
        if with_helpsteer:
            subtitle += " HelpSteer2-JAはbenchmark-v1・2,500件"
        _render_radar(
            series, axis_labels, title=titles[primitive], subtitle=subtitle,
            out_path=OUT / f"radar-{primitive}-detail.png", label_fontsize=10,
            h_margin=0.20,
        )


def main() -> None:
    rows = build_table()
    build_ranking_chart(rows)
    build_scatter_and_latency_charts(rows)
    build_per_primitive_radars(rows)


if __name__ == "__main__":
    main()
