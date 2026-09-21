# Primitive評価設計

## 目的

評価データとJev Primitiveを分離し、同じ元データから複数の判断形式を構築する。

```text
Dataset → Dataset Adapter → Primitive Adapter → Evaluator → Primitive Summary
```

## Noul

Yes / Noの二値判断。Accuracy、Precision、Recall、F1、Brier Score、ECEを保存する。

現行設定ではJNLIを以下へ変換する。

- `jnli_entailment`：仮説が前提から支持されるか
- `jnli_contradiction`：仮説が前提と矛盾するか
- `jnli_missing_evidence`：前提だけでは判断材料が不足しているか

## Choice

複数候補から1つを選ぶ判断。Top-1 Accuracy、NLL、Brier Score、ECEを保存する。

現行設定:

- MMMLU JA-JP：一般・学術知識
- JMMLU：日本語・日本固有知識
- JGPQA Diamond：高難度科学推論
- JCommonsenseQA：日本語常識推論
- XWinograd JA：文脈・照応解析
- MGSM JA：日本語数学推論
- GSM8K JA MC4：数値推論・4択
- GSM8K JA MC10：数値推論・10択
- JNLI 3-class：日本語NLI・意味理解

## Score

順序付き尺度を選ぶ判断。MAE、RMSE、Spearman相関、Quadratic Weighted Kappaを保存する。

汎用JSONL adapterは `state` と `gold_score` を読み、YAMLの `criteria` を順序付き候補として使う。

本番設定では公式WRIME v2のtest splitを原データ無改変で使用する。

- `wrime_joy`：読み手が感じる喜びの強度（0〜3）
- `wrime_anger`：読み手が感じる怒りの強度（0〜3）
- `wrime_sentiment`：読み手が感じる感情極性（-2〜2を候補index 0〜4へ写像）

WRIMEのライセンスはCC BY-NC-ND 4.0。設定では取得時commitをrevisionとして固定する。配布元：`https://github.com/ids-cv/wrime`

## 集約

データセット件数の差がPrimitiveスコアを支配しないよう、次の順で集約する。

1. データセット別metricを0〜1へ正規化する。
2. Primitive内でデータセットmacro平均を計算する。
3. YAMLの `tasks.summary.weights` に従いPrimitive間を加重平均する。

未評価Primitiveは分母から除外し、結果へ `overall_status: partial` と `primitive_coverage` を保存する。現行本番結果は3 Primitiveすべて完了している。

## 実行段階

各Primitiveは独立CLIで実行できる。`--phase smoke` はデータセットごとの件数を `smoke_limit` へ制限し、`--phase production` は全件を処理する。

workflowはYAMLの `workflow.order` に従い、Noul、Choice、Score、Summaryを順次実行する。各段階の完了状態は `eval_<task>.json` へ保存する。
