# ruff: noqa: E501, E701, E702
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/model-evaluation-dashboard/dist/index.html"
TEMPLATE = ROOT / "reports/model-evaluation-dashboard/dashboard.template.html"
FULL_RUNS = ("eval-primitives-20260918", "eval-qwen3.5-0.8b-hub")
EXCLUDED_PRIMARY_DATASETS = {"jad_afc_true", "jad_afc_false", "jad_afc_nei"}

DATASET_LABELS = {
    "jnli_entailment": "JNLI・含意", "jnli_contradiction": "JNLI・矛盾", "jnli_missing_evidence": "JNLI・中立",
    "janli_entailment": "JaNLI・含意", "jcola_in_domain": "JCoLA・既知構文", "jcola_out_of_domain": "JCoLA・未知構文",
    "paws_x_ja": "PAWS-X・言い換え", "textdetox_ja": "TextDetox・有害表現", "jad_afc_true": "JAD-AFC・真",
    "jad_afc_false": "JAD-AFC・偽", "jad_afc_nei": "JAD-AFC・不明", "wrime_joy_binary": "WRIME・喜び判定",
    "wrime_anger_binary": "WRIME・怒り判定", "wrime_positive_binary": "WRIME・肯定判定", "mmmlu_ja": "MMMLU・学術知識",
    "jmmlu": "JMMLU・日本語知識", "jgpqa_diamond": "JGPQA・高度科学", "jcommonsenseqa": "JCommonsenseQA・常識",
    "xwinograd_ja": "XWinograd・照応", "mgsm_ja": "MGSM・応用数学", "gsm8k_ja_mc4": "GSM8K・数学4択",
    "gsm8k_ja_mc10": "GSM8K・数学10択", "jnli": "JNLI・3分類", "wrime_joy": "WRIME・喜び強度",
    "wrime_anger": "WRIME・怒り強度", "wrime_sentiment": "WRIME・感情極性", "synthetic_urgency": "合成・緊急度",
    "synthetic_dissatisfaction": "合成・不満度", "synthetic_risk": "合成・リスク", "synthetic_relevance": "合成・関連度",
}

DATASET_META = {
    "jnli_entailment": ("前提文から仮説文が論理的に導けるかを二値判定。", "子どもが公園で遊ぶ → 子どもが屋外にいる"),
    "jnli_contradiction": ("前提文と仮説文が互いに矛盾するかを二値判定。", "男性が走る → 誰も動いていない"),
    "jnli_missing_evidence": ("前提だけでは仮説の真偽を決められない中立関係を判定。", "女性が本を読む → その本は小説だ"),
    "janli_entailment": ("語順や否定に強い敵対的な日本語NLIで含意を判定。", "似た表現でも意味関係が変わる文対"),
    "jcola_in_domain": ("学習時と同種の構文で日本語文が文法的かを判定。", "私は昨日、駅で友人に会った。"),
    "jcola_out_of_domain": ("未知の構文領域で日本語文の文法的妥当性を判定。", "未学習タイプの語順・格・省略を含む文"),
    "paws_x_ja": ("語彙がよく似た2文が実際に同じ意味かを判定。", "犬が猫を追う / 猫が犬を追う"),
    "textdetox_ja": ("日本語文に侮辱・攻撃などの有害表現があるかを判定。", "相手への直接的な罵倒を含む文"),
    "jad_afc_true": ("根拠に照らして主張が真であるかを判定。", "根拠記事に支持される主張"),
    "jad_afc_false": ("根拠に照らして主張が偽であるかを判定。", "根拠記事に反する主張"),
    "jad_afc_nei": ("根拠だけでは主張を確認できないかを判定。陽性22/612で強く不均衡。", "記事にない人物属性を断定した主張"),
    "wrime_joy_binary": ("SNS文から読み手が喜びを感じるかを二値判定。", "やった、試験に合格した！"),
    "wrime_anger_binary": ("SNS文から読み手が怒りを感じるかを二値判定。陽性58/2500で強く不均衡。", "また電車が遅れて本当に腹が立つ。"),
    "wrime_positive_binary": ("SNS文の読み手感情が肯定的かを二値判定。", "今日はとても良い一日だった。"),
    "mmmlu_ja": ("MMLU日本語版。人文・社会・自然科学・専門職など57分野を4択評価。", "文学、歴史、法学、医学、計算機科学"),
    "jmmlu": ("日本語・日本文化を含む幅広い学術知識を多肢選択で評価。", "日本史、国語、理数、社会"),
    "jgpqa_diamond": ("専門家水準の科学知識と多段推論を難問の選択式で評価。", "物理・化学・生物の高度な推論"),
    "jcommonsenseqa": ("日常場面の一般常識と因果関係を5択で評価。", "雨の日の外出に必要な物は？"),
    "xwinograd_ja": ("代名詞や省略主語が何を指すか、文脈理解を評価。", "大きすぎたのはトロフィーか箱か？"),
    "mgsm_ja": ("日本語の小学校算数文章題。式を立てる能力を評価。", "単価と個数から合計金額を求める"),
    "gsm8k_ja_mc4": ("日本語GSM8Kを4択化した応用数学。", "割合・速さ・個数の文章題"),
    "gsm8k_ja_mc10": ("日本語GSM8Kを10択化。紛らわしい誤答の識別も評価。", "多段計算の最終値を10候補から選ぶ"),
    "jnli": ("前提と仮説の関係を含意・矛盾・中立の3択で分類。", "2文を読み3つの意味関係から選ぶ"),
    "wrime_joy": ("SNS文から読み手が感じる喜びの強さを4段階で推定。", "0=なし〜3=強い喜び"),
    "wrime_anger": ("SNS文から読み手が感じる怒りの強さを4段階で推定。", "0=なし〜3=強い怒り"),
    "wrime_sentiment": ("SNS文の感情極性を否定的〜肯定的の5段階で推定。", "非常に否定的〜非常に肯定的"),
    "synthetic_urgency": ("合成した業務文から対応の緊急度を5段階で推定。", "本番サービス停止中、至急復旧してほしい"),
    "synthetic_dissatisfaction": ("合成した顧客文から不満の強さを5段階で推定。", "三度問い合わせたが返金されない"),
    "synthetic_risk": ("合成した業務・安全文から事故や損失リスクを5段階で推定。", "アカウントが不正利用された可能性"),
    "synthetic_relevance": ("合成した質問と回答の関連度を5段階で推定。", "送料の質問に配送費を直接回答"),
}

PRIMITIVE_META = {
    "noul": {"label": "Noul（二値判断）", "description": "Yes/No型の判定能力。各データセットの陽性クラスF1をmacro平均。"},
    "choice": {"label": "Choice（多肢選択）", "description": "文学・一般常識・科学・照応・応用数学などの正答率をmacro平均。"},
    "score": {"label": "Score（段階評価）", "description": "感情強度や合成業務属性を順序尺度で評価。normalized quadratic weighted kappaをmacro平均。"},
}

CHART_LABELS = {
    "jnli_entailment": "JNLI Entailment\n（前提から仮説を導けるか）",
    "jnli_contradiction": "JNLI Contradiction\n（前提と仮説が矛盾するか）",
    "jnli_missing_evidence": "JNLI Neutral\n（真偽を決める情報が不足か）",
    "janli_entailment": "JaNLI Entailment\n（難しい言い換えの含意判定）",
    "jcola_in_domain": "JCoLA In-domain\n（既知構文の文法判断）",
    "jcola_out_of_domain": "JCoLA Out-of-domain\n（未知構文の文法判断）",
    "paws_x_ja": "PAWS-X Japanese\n（2文が同じ意味か）",
    "textdetox_ja": "TextDetox Japanese\n（日本語の有害表現検出）",
    "jad_afc_true": "JAD-AFC True\n（根拠が主張を支持するか）",
    "jad_afc_false": "JAD-AFC False\n（根拠が主張を反証するか）",
    "jad_afc_nei": "JAD-AFC NEI\n（主張の根拠が不足か）",
    "wrime_joy_binary": "WRIME Joy Binary\n（読み手が喜びを感じるか）",
    "wrime_anger_binary": "WRIME Anger Binary\n（読み手が怒りを感じるか）",
    "wrime_positive_binary": "WRIME Positive Binary\n（読み手が肯定感情を持つか）",
    "mmmlu_ja": "MMMLU JA-JP\n（一般・学術知識）", "jmmlu": "JMMLU\n（日本語・日本固有知識）",
    "jgpqa_diamond": "JGPQA Diamond\n（高難度科学推論）", "jcommonsenseqa": "JCommonsenseQA\n（日本語常識推論）",
    "xwinograd_ja": "XWinograd JA\n（文脈・照応解析）", "mgsm_ja": "MGSM JA\n（日本語数学推論）",
    "gsm8k_ja_mc4": "GSM8K JA MC4\n（数値推論・4択）", "gsm8k_ja_mc10": "GSM8K JA MC10\n（数値推論・10択）",
    "jnli": "JNLI 3-class\n（日本語NLI・意味理解）",
    "wrime_joy": "WRIME Joy\n（読み手の喜び強度・4段階）",
    "wrime_anger": "WRIME Anger\n（読み手の怒り強度・4段階）",
    "wrime_sentiment": "WRIME Sentiment\n（否定〜肯定の感情・5段階）",
    "synthetic_urgency": "Synthetic Urgency\n（業務対応の緊急度・5段階）",
    "synthetic_dissatisfaction": "Synthetic Dissatisfaction\n（顧客の不満度・5段階）",
    "synthetic_risk": "Synthetic Risk\n（事故・損失リスク・5段階）",
    "synthetic_relevance": "Synthetic Relevance\n（質問と回答の関連度・5段階）",
}

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def model_specs(model_id: str) -> tuple[str, float | None, str, list[str]]:
    key = model_id.lower()
    family = next((name for token, name in (("qwen3-swallow", "Qwen3-Swallow"), ("qwen3.5", "Qwen3.5"), ("qwen2.5", "Qwen2.5"), ("qwen3", "Qwen3"), ("gemma", "Gemma 4"), ("sarashina", "Sarashina2.2"), ("llm-jp", "LLM-jp 4")) if token in key), "Other")
    match = re.search(r"(?:^|[-_])(?:e)?(\d+(?:\.\d+)?)b(?:[-_]|$)", key)
    size = float(match.group(1)) if match else None
    if "thinking" in key:
        variant, tags = "Thinking", ["Thinking"]
    elif "qat" in key:
        variant, tags = "Instruct · QAT", ["Instruct"]
    elif "instruct" in key or "-it" in key:
        variant, tags = "Instruct", ["Instruct"]
    elif family in {"Qwen3", "Qwen3.5"} and not any(token in key for token in ("-base", "-cpt", "-sft", "-rl")):
        variant, tags = "Instruct + Thinking", ["Instruct", "Thinking"]
    elif any(token in key for token in ("-sft", "-rl", "-cpt")):
        variant = next(name for token, name in (("-sft", "SFT"), ("-rl", "RL"), ("-cpt", "CPT")) if token in key)
        tags = ["other"]
    else:
        variant, tags = "Base", ["Base"]
    return family, size, variant, tags

def primitive_model(run_name: str, row: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / "results" / run_name; model_id = row["id"]; family, size, variant, variant_tags = model_specs(model_id)
    primitives: dict[str, Any] = {}; latencies: list[tuple[float, int]] = []
    for primitive in ("noul", "choice", "score"):
        source = row.get("primitives", {}).get(primitive)
        if not source: continue
        datasets = {x["dataset"]: {"label": DATASET_LABELS.get(x["dataset"], x["dataset"]), "value": x["normalized_score"], "total": x["total"]} for x in source["datasets"] if x["dataset"] not in EXCLUDED_PRIMARY_DATASETS}
        detail = root / model_id / f"summary.{primitive}.json"
        if detail.exists():
            latencies += [(float(x.get("mean_latency_ms", 0)), int(x["total"])) for x in read_json(detail)]
        score = fmean(item["value"] for item in datasets.values()) if datasets else source["score"]
        primitives[primitive] = {"score": score, "status": source["status"], "datasets": datasets}
    items = sum(n for _, n in latencies)
    complete_scores = [p["score"] for p in primitives.values() if p["status"] == "complete"]
    return {"id": model_id, "label": row["label"].replace(" (Hub)", ""), "family": family, "size": size, "variant": variant, "variantTags": variant_tags, "source": run_name, "coverage": sum(p["status"] == "complete" for p in primitives.values()), "overall": fmean(complete_scores) if len(complete_scores)==3 else None, "primitives": primitives, "meanLatencyMs": sum(v*n for v,n in latencies)/items if items else None, "items": items, "note": None if row.get("overall_status") == "complete" else "一部評価"}

def legacy_model(path: Path) -> dict[str, Any]:
    model_id = path.parent.name; rows = read_json(path); family, size, variant, variant_tags = model_specs(model_id)
    datasets = {x["dataset"]: {"label": DATASET_LABELS.get(x["dataset"], x["dataset"]), "value": x["accuracy"], "total": x["total"]} for x in rows}; total = sum(int(x["total"]) for x in rows)
    return {"id": model_id, "label": model_id.replace("-", " ").title(), "family": family, "size": size, "variant": variant, "variantTags": variant_tags, "source": "eval-20260918", "coverage": 1, "overall": None, "primitives": {"choice": {"score": fmean(x["accuracy"] for x in rows), "status": "complete", "datasets": datasets}}, "meanLatencyMs": sum(float(x["mean_latency_ms"])*int(x["total"]) for x in rows)/total, "items": total, "note": "Choiceのみ。Noul / Scoreは未評価。"}

def jev_model() -> dict[str, Any]:
    root = ROOT / "results/eval-jev-latest/jev-latest"
    groups = {"noul": set(DATASET_LABELS)-EXCLUDED_PRIMARY_DATASETS-{"mmmlu_ja","jmmlu","jgpqa_diamond","jcommonsenseqa","xwinograd_ja","mgsm_ja","gsm8k_ja_mc4","gsm8k_ja_mc10","jnli","wrime_joy","wrime_anger","wrime_sentiment","synthetic_urgency","synthetic_dissatisfaction","synthetic_risk","synthetic_relevance"}, "choice": {"mmmlu_ja","jmmlu","jgpqa_diamond","jcommonsenseqa","xwinograd_ja","mgsm_ja","gsm8k_ja_mc4","gsm8k_ja_mc10","jnli"}, "score": {"wrime_joy","wrime_anger","wrime_sentiment","synthetic_urgency","synthetic_dissatisfaction","synthetic_risk","synthetic_relevance"}}
    metric = {"noul":"f1", "choice":"accuracy", "score":"normalized_quadratic_weighted_kappa"}; primitives={}; items=inp=out=0; latency=0.0
    for primitive, ids in groups.items():
        datasets={}; values=[]
        for dataset in sorted(ids):
            path=root/dataset/"summary.json"
            if not path.exists(): continue
            row=read_json(path); value=float(row[metric[primitive]]); datasets[dataset]={"label":DATASET_LABELS[dataset],"value":value,"total":row["total"]}; values.append(value)
            n=int(row["total"]); items+=n; inp+=int(row.get("input_tokens",0)); out+=int(row.get("output_tokens",0)); latency+=float(row["mean_latency_ms"])*n
        if values: primitives[primitive]={"score":fmean(values),"status":"complete" if len(values)==len(ids) else "partial","datasets":datasets}
    coverage=sum(len(primitives.get(p,{}).get("datasets",{}))==len(ids) for p,ids in groups.items()); scores=[primitives[p]["score"] for p in groups if primitives.get(p,{}).get("status")=="complete"]
    return {"id":"jev-latest","label":"Jev 1.13.0","family":"Jev","size":None,"variant":"API","variantTags":["other"],"source":"eval-jev-latest","coverage":coverage,"overall":fmean(scores) if scores else None,"overallProvisional":coverage<3,"primitives":primitives,"meanLatencyMs":latency/items if items else None,"items":items,"inputTokens":inp,"outputTokens":out,"estimatedCostUsd":inp/1_000_000*0.042,"note":None if coverage==3 else "暫定Overall。完了Primitiveのみの平均。"}

def main() -> int:
    models=[]; seen=set()
    for run in FULL_RUNS:
        path=ROOT/"results"/run/"eval_summary.json"
        if path.exists():
            for row in read_json(path).get("models",[]):
                if row["id"] not in seen: models.append(primitive_model(run,row)); seen.add(row["id"])
    consolidated=ROOT/"results/eval-20260920/eval_summary.json"
    if consolidated.exists():
        for row in read_json(consolidated).get("models",[]):
            if row["id"] not in seen: models.append(primitive_model("eval-20260920",row)); seen.add(row["id"])
    for path in sorted((ROOT/"results/eval-20260918").glob("*/summary.json")):
        if path.parent.name not in seen: models.append(legacy_model(path)); seen.add(path.parent.name)
    models.append(jev_model()); models.sort(key=lambda m:(m["family"],m["size"] or 999,m["variant"],m["label"]))
    payload={"generatedAt":datetime.now(UTC).isoformat(),"models":models,"datasetLabels":DATASET_LABELS,"chartLabels":CHART_LABELS,"datasetMeta":{k:{"description":v[0],"sample":v[1]} for k,v in DATASET_META.items()},"primitiveMeta":PRIMITIVE_META,"selectionNote":"評価結果が存在する全モデルを掲載。レーダーは最大6モデルを同時比較。"}
    html=TEMPLATE.read_text(encoding="utf-8").replace("/*__DASHBOARD_DATA__*/",f"window.DASHBOARD_DATA = {json.dumps(payload,ensure_ascii=False)};"); OUTPUT.parent.mkdir(parents=True,exist_ok=True); OUTPUT.write_text(html,encoding="utf-8"); print(f"{OUTPUT} ({len(models)} models)"); return 0

if __name__ == "__main__": raise SystemExit(main())
