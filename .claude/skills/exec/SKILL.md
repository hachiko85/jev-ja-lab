---
name: exec
description: ベンチマークではなく、ユーザーが与えた1件の質問・選択肢に対して指定手法(scorer)で1回だけJevの判定(Noul/Choice/Score)を実行したいときに使う。動作確認やデモ、単発の問い合わせに使う。
---

# exec: 指定手法での単発Jev実行

「この質問をXという手法で判定させて」のような、ベンチマークデータセットを介さない
アドホックな1回実行に使う。`eval_run`(データセット全体の評価)とは目的が異なる —
ここでは`(question, options)`の組を直接scorerに渡し、`predicted_index`/`probabilities`
を返すだけ。

## primitiveの選び方

入力される選択肢の性質で決める:

- **Noul**(Yes/No二値判断): `options = ["No", "Yes"]`(または`["いいえ", "はい"]`)
- **Choice**(複数候補から1つ): `options`は3つ以上の候補文字列のリスト
- **Score**(順序尺度): `options`は順序を持つ段階のリスト(例: `["不満なし","低い",
  "中程度","高い","非常に高い"]`)

## 手法(scorer)ごとの呼び出し方

いずれも`.score(question: str, options: list[str]) -> ScoreResult`で、
`ScoreResult`は`scores`/`probabilities`/`predicted_index`/`latency_ms`を持つ。
GPU/モデルロードが要るため、都度ロードせず1プロセス内で使い回す。

```python
# next-token logit(独自日本語プロンプト)
from openjev_ja.methods.next_token_logit import NextTokenLogitScorer
scorer = NextTokenLogitScorer("Qwen/Qwen3.5-4B", device="auto", dtype="bfloat16")

# semif系(SemIfのchatテンプレート+JSON構造化、zero/few-shot共通クラス)
from openjev_ja.methods.semif_logit import SemifLogitScorer
scorer = SemifLogitScorer("Qwen/Qwen3.5-4B", device="auto", dtype="bfloat16")
# few-shotにしたい場合は primitive= / datasets_root= / few_shot_count= を追加

# AlexWortega/openjev(NLI cross-encoder)
from openjev_ja.methods.nli_cross_encoder import NLICrossEncoderScorer
scorer = NLICrossEncoderScorer(
    "AlexWortega/openjev", subfolder="qwen3.5-4b-nli-v2",
    trust_remote_code=True, device="cuda", dtype="bfloat16",
)

# TypeSafe Jev API(本家、.envにTYPESAFE_API_KEY/TYPESAFE_API_URLが必要)
from openjev_ja.methods.typesafe_jev import JevScorer
scorer = JevScorer(model="jev-latest")

# HopitAI/hopper(LoRAをQwen3.5-4Bにマージ。primitiveは "noul"|"choice"|"score" のいずれか。研究・デモ用途のみ)
from openjev_ja.methods.hopper import HopperScorer
scorer = HopperScorer(primitive="choice", device="cuda")

# Mapika/decider-4b v2(`uv pip install --no-deps decider-ai`が前提)
from openjev_ja.methods.decider import DeciderScorer
scorer = DeciderScorer(primitive="choice", revision="v2", device="cuda")

# embedding / laya / laya-bert / jevlike は訓練済みheadファイル
# (head_path / checkpoint_path) が要るため、configs/eval/*.yaml内の
# 該当エントリからパスを確認して使う。

result = scorer.score(question, options)
print(result.predicted_index, options[result.predicted_index], result.probabilities)
```

実行は`uv run python -c "..."`のワンライナーか、`$TEMP`(スクラッチパッド)に一時
スクリプトを書いて`uv run python <path>`する。プロジェクトのソースにexec用の
恒久スクリプトを追加する必要はない(単発実行が目的のため)。

## 注意

- モデルロードに時間がかかる手法(4B級LLM)は、複数件まとめて聞かれたら1回のロードで
  使い回す。
- 結果を評価記録として残したい規模になったら、これは`exec`の範囲を超えている ——
  `eval_run`スキルに切り替えてデータセット化・`run_evaluation()`経由の実行を検討する。
