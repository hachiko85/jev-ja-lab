---
name: eval_run
description: 指定したモデル・手法(scorer)・データセットでベンチマーク評価を実行するときに使う。単発の小規模実行(CLI直叩き)とYAML定義のフルバッチ実行のどちらを組み立てるべきかを判断する。
---

# eval_run: ベンチマーク評価の実行

「〇〇モデルを△△手法で××データセットに対して評価して」という依頼に応える。まず
**単発実行**(`jev-ja-lab-eval`、1モデル・1手法・1〜数データセット)で足りるか、
**YAML経由の一括実行**(`jev-ja-lab-eval-workflow`、複数モデル・複数primitive)が
必要かを判断してから進める。目的が曖昧なら、まず単発実行で件数を絞って試すのが安全。

## 手法(scorer)とその実行経路

| scorer名 | 対応CLI | 備考 |
|---|---|---|
| `qwen-direct` (next-token logit、独自日本語プロンプト) | `jev-ja-lab-eval` | `--model`にHF repo_id、既定は`Qwen/Qwen3.5-4B` |
| `jev` (TypeSafe Jev API) | `jev-ja-lab-eval` | `.env`の`TYPESAFE_API_KEY`/`TYPESAFE_API_URL`必須 |
| `nli-cross-encoder` (AlexWortega/openjev) | `jev-ja-lab-eval` | `--model`省略時`AlexWortega/openjev`(サブフォルダ指定はorchestrate経由が必要) |
| `mock` | `jev-ja-lab-eval` | 動作確認専用、実際の判定はしない |
| `semif-logit` (semif zero-shot/few-shot) | `jev-ja-lab-orchestrate` / `-workflow` | `few_shot_count`はconfig側で指定 |
| `hopper` (HopitAI/hopper、Qwen3.5-4B + LoRA) | `jev-ja-lab-orchestrate` / `-workflow` | `configs/eval/hopper-series.yaml`。`peft`が必要(`uv pip install peft`)。アダプタは研究・デモ用途のみ |
| `decider` (Mapika/decider-4b v2) | `jev-ja-lab-orchestrate` / `-workflow` | `configs/eval/decider-series.yaml`。`uv pip install --no-deps decider-ai`(numpy固定を避けるため)。`revision`でv1/v2/main(v2.1)を切替 |
| `embedding` / `laya` / `jevlike` / `laya-bert` / `masked-lm` | `jev-ja-lab-orchestrate` / `-workflow` | frozen encoder系、`primitive`+`head_path`等の追加設定が要る |

`jev-ja-lab-eval`のCLIが直接サポートしない手法(embedding/laya/jevlike/laya-bert/
semif-logit/hopper/decider系)は、`configs/eval/*.yaml`をベースにしたorchestration経由でのみ実行できる。

## パターンA: 単発実行(1モデル・数データセット、素早く試したい)

```bash
uv run jev-ja-lab-eval \
  --dataset <データセットID、"all"で全部> \
  --scorer <qwen-direct|jev|nli-cross-encoder|mock> \
  --model <HF repo_idまたはローカルパス、省略可> \
  --limit <件数、省略で全件> \
  --device auto --dtype bfloat16
```
データセットIDは`src/openjev_ja/eval/datasets/adapters.py`の`DATASET_NAMES`を参照
(`mmmlu_ja`, `jmmlu`, `jgpqa_diamond`, `jcommonsenseqa`, `xwinograd_ja`, `mgsm_ja`,
`gsm8k_ja_mc4`, `gsm8k_ja_mc10`, `jnli`)。結果は`results/<dataset>/summary.json`。

## パターンB: YAML経由のフルバッチ実行(複数モデル/primitive、記録に残す評価)

1. **既存configの再利用を優先**する。`configs/eval/`に対象手法のconfigが既にあるか
   確認(`embedding-series.yaml`、`laya-series.yaml`、`semif-logit-series.yaml`等)。
2. モデル・データセットを絞りたい場合は、configの`runtime.selected_model_ids`か
   `tasks.<primitive>.model_ids`/`tasks.<primitive>.datasets`で絞る。既存configを
   直接編集せず、対象を絞った一時コピーを作るほうが安全(元configを壊さない)。
3. まず**smoke phase**で疎通確認:
   ```bash
   uv run jev-ja-lab-eval-workflow --config <config.yaml> --phase smoke
   ```
   `smoke_limit`件のみ実行される。エラーが出ないか、`results/<smoke_run_name>/`に
   結果が出るかを確認してから本番へ進む。
4. 問題なければ本番実行:
   ```bash
   uv run jev-ja-lab-eval-workflow --config <config.yaml> --phase production
   ```
   `resume: true`なら既存の`summary.json`があるデータセットは自動でスキップされる
   (壊れた結果を直したい場合は該当ディレクトリを消してから再実行)。
5. 新手法・新モデル向けにconfigを新規作成する場合は、`scripts/generate_*_config.py`
   系のスクリプトを参考にする(embedding/laya_bert/openjev/semif_logit系の実例あり)。

## 結果の確認・比較

- 個別結果: `results/<run_name>/<model_id>/<dataset>/{summary.json,predictions.jsonl}`
- 複数モデルの横断比較が必要なら、`scripts/build_eval_summary.py`で
  `results/eval-summary/`へ統合してから`scripts/build_final_charts.py`でグラフ化する
  (詳細は`results/eval-summary/README.md`)。

## 注意

- 評価を再実行する前に`git status`相当の確認は不要だが、**既存の`results/`を上書きする
  実行**(同じ`run_name`で`resume: false`、または対象ディレクトリを手動削除しての
  再実行)は、意図した対象だけを削除・上書きしているか確認してから進める。
- GPU評価は時間がかかる。件数の見積もり(`expected_items`、モデルサイズ)を先に見て、
  長時間かかりそうならバックグラウンド実行を検討する。
