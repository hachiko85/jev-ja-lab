# 実装計画

## 調査結果

- TypeSafe Jev: `POST /v1/systemone` に `state`、`model`、`questions` を渡す。Choice は `choice`、全選択肢の `probabilities`、`confidence` を返す。公式 Python SDK も存在するが、HTTP スキーマを明示的に検証できる薄いアダプターを採用する。
- Qwen3.5: Transformers 5.17 系が `Qwen3_5ForSequenceClassification` と生成モデルの forward API を提供する。raw baseline は生成せず、最終位置の固定回答ラベル logits のみ読む。
- OpenJev NLI: `contradiction / entailment / neutral` の 3 クラス。multiple-choice は各選択肢を hypothesis に変換し、独立した `P(entailment)` の最大値を選ぶ。
- Dataset: 各公開カードとスキーマを確認済み。JGPQA は利用規約同意と認証が必要。JMMLU は科目別 config。JNLI は validation split を評価に使う。

## Vertical slice

1. 共通型、MMMLU JA-JP adapter、MockScorer、JSON 結果保存。
2. direct-logit Qwen scorer と単一 token 検証。
3. 残りの dataset adapter と deterministic GSM8K distractor。
4. Jev Choice API adapter と mock HTTP test。
5. NLI cross-encoder scorer、日本語・英語 hypothesis template。
6. accuracy、latency、throughput、該当 scorer の calibration 指標。
7. unit/smoke test、lint、5 件 mock 実行。実モデル/API の smoke と full run は資格情報・計算資源確認後に実行。

## Primitive評価拡張

1. Dataset AdapterとPrimitive Adapterを分離し、JNLIから3種類のNoul評価を生成する。
2. Choiceは既存9データセットを分類して再利用する。
3. Scoreは順序付きcriteriaを持つ汎用JSONL adapterと専用metricを提供する。
4. `eval_noul`、`eval_choice`、`eval_score`、`eval_summary` を独立CLIにする。
5. YAMLでtask順序、有効化、モデル、データセット、smoke件数、並列度、集約指標を制御する。
6. smoke成功後にのみfull runを行い、Primitive別結果と部分評価を明示した総合値を保存する。

## 非対象

現フェーズでは fine-tuning と学習済み OpenJev-JA 推論を実装しない。`train` と `infer` は拡張境界のみ保持する。
