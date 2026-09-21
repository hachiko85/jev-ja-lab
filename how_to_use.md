# 評価の使い方

本書は `configs/eval/primitives.20260918.yaml` を使ったNoul、Choice、Score評価と総合集計の手順を説明します。

## 1. 処理構成

```text
YAML読込
  → データセット読込・Primitive変換
  → モデルをGPUへロード
  → dataset単位で評価
  → Primitive集約
  → 総合集約
  → レーダーチャート生成
```

モデルはYAML記載順に処理します。同一モデル内では、`runtime.devices` に指定したGPUへデータセットを分配します。tensor parallelではなく、各ワーカーが同じモデルを1つずつロードします。

## 2. インストール

uv:

```bash
uv sync --extra eval --extra viz --extra orchestrate --extra dev
```

Python仮想環境:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[eval,viz,orchestrate,dev]"
```

Windows PowerShellではactivateだけ変更します。

```powershell
.venv\Scripts\Activate.ps1
```

## 3. YAML全体構造

```yaml
version: 1
runtime: {}
workflow: {}
tasks: {}
models: []
datasets: []
```

### `runtime`

| 変数 | 型 | 説明 |
|---|---|---|
| `run_name` | string | production結果の保存先名。`results/<run_name>/` を作成 |
| `smoke_run_name` | string | smoke結果の保存先名 |
| `resume` | bool | `true`なら既存 `summary.json` があるdatasetをスキップ |
| `seed` | int | distractor生成などで使う乱数seed |
| `models_root` | path | ローカルモデルの基準ディレクトリ |
| `datasets_root` | path | ローカルデータとHub cacheの基準ディレクトリ |
| `output_root` | path | 結果の基準ディレクトリ |
| `devices` | list | 使用device。例: `[cuda:0, cuda:1]`、CPUなら `[cpu]` |
| `parallelism` | int | 同時ワーカー数。通常はGPU数以下 |
| `batch_size` | int | モデル共通batch size |
| `worker_timeout_seconds` | int | 1ワーカーの最大待機秒数 |

推奨GPU設定:

```yaml
runtime:
  devices: [cuda:0, cuda:1]
  parallelism: 2
  batch_size: 32
```

1 GPUにつき1ワーカーを推奨します。同じGPU IDを重複指定すると複数モデルコピーが同一GPUへ載り、OOMしやすくなります。

### `workflow`

```yaml
workflow:
  order: [noul, choice, score, summary]
```

実行順を定義します。`summary` は各Primitiveの結果を読み、総合値と総合チャートを生成します。

### `tasks`

```yaml
tasks:
  noul:
    enabled: true
    model_ids: [qwen-local]
    datasets: [jnli_entailment, jcola_in_domain]
    aggregate_metric: f1
    smoke_limit: 5
    parallelism: 2
    chart:
      enabled: true
      name: radar-noul
      title: Noul evaluation
      subtitle: Binary decision F1
      metric: f1
```

| 変数 | 説明 |
|---|---|
| `enabled` | taskの有効・無効 |
| `model_ids` | task対象モデル。省略時は全モデル |
| `datasets` | task対象dataset ID。ここにない定義は実行しない |
| `aggregate_metric` | dataset集約に使う指標 |
| `smoke_limit` | smoke時のdataset別最大件数 |
| `parallelism` | task固有ワーカー数。省略時はruntime値 |
| `chart.enabled` | Primitiveチャート生成可否 |
| `chart.name` | YAML・画像の出力名。ディレクトリ指定不可 |
| `chart.title` | チャートタイトル |
| `chart.subtitle` | チャートサブタイトル |
| `chart.metric` | 描画するsummary指標 |
| `chart.theme` | `light`(既定)または`dark`。背景・グリッド・文字色一式を切替 |
| `chart.language` | `ja`を指定すると、`title`/`subtitle`を明示しない場合の既定文言や注記(note)が日本語になる |

`chart.theme`/`chart.language`は `tasks.summary.chart` にも同様に指定できます。色を個別調整したい場合は、生成された `radar-*.yaml` に `figure_color`/`plot_color`/`text_color`/`muted_color`/`tick_color`/`grid_color`/`spine_color` を直接追記すると`theme`のプリセット値を上書きできます。

総合評価:

```yaml
tasks:
  summary:
    enabled: true
    weights:
      noul: 1.0
      choice: 1.0
      score: 1.0
    chart:
      enabled: true
      name: radar-summary
```

総合値は問題数による一括加重ではありません。dataset別macro平均をPrimitive値とし、`weights` でPrimitive間を加重します。

## 4. モデル設定

### ローカルモデル

```yaml
runtime:
  models_root: /models

models:
  - id: qwen-local
    label: Qwen Local
    path: Qwen3-8B
    dtype: bfloat16
    batch_size: 32
    color: "#4b83ad"
```

実体は `/models/Qwen3-8B` です。`path` に絶対パスも指定できますが、移行性を保つには `models_root` からの相対パスを推奨します。

### Hugging Face Hubモデル

```yaml
models:
  - id: gemma-hub
    label: Gemma Hub
    repo_id: google/gemma-4-E4B-it
    revision: <commit-hash>
    dtype: bfloat16
    batch_size: 8
```

`path` と `repo_id` はどちらか一方を指定します。`revision` はbranch名、tag、commit hashを受け付けます。再現性重視ならcommit hashを指定してください。

gatedモデル:

```bash
export HF_TOKEN=hf_xxx
```

PowerShell:

```powershell
$env:HF_TOKEN = "hf_xxx"
```

tokenをYAMLやGitへ保存しないでください。

### モデル変数

| 変数 | 必須 | 説明 |
|---|---|---|
| `id` | 必須 | 結果ディレクトリ名。YAML内で一意 |
| `label` | 推奨 | チャート表示名 |
| `path` | 条件付き | ローカルモデルパス |
| `repo_id` | 条件付き | HubモデルID |
| `revision` | 推奨 | Hub revision |
| `metadata_revision` | 任意 | 結果へ明示記録するrevision |
| `dtype` | 任意 | `bfloat16`、`float16`、`float32`など |
| `batch_size` | 任意 | モデル固有値。runtime値を上書き |
| `color` | 任意 | チャート色 |
| `scorer` | 任意 | 使用scorerを明示指定。省略時はモデルの`config.architectures`から自動判定 |

### BERT系(マスク言語モデル)対応

LLM(causal LM)だけでなく、BERT/RoBERTa/ModernBERT/ELECTRA/DeBERTaなど「`...ForMaskedLM`」アーキテクチャのエンコーダモデルも`repo_id`指定のみで評価できます。`scorer`を省略すればHubの`config.json`から自動でLLM用(`qwen-direct`)かBERT用(`masked-lm`)かを判定します。

```yaml
models:
  - id: modernbert-ja-310m
    label: ModernBERT-ja 310M
    repo_id: sbintuitions/modernbert-ja-310m
    revision: <commit-hash>
    dtype: float32
  - id: bert-base-multilingual-cased
    label: BERT Base Multilingual Cased (Google)
    repo_id: google-bert/bert-base-multilingual-cased
    revision: <commit-hash>
    dtype: float32
```

BERT用scorer(`MaskedLMScorer`)はLLM用と同じ枠組みで動作します。プロンプトの末尾に`[MASK]`(モデルのmask tokenに合わせて自動選択)を置き、その位置のlogitを候補ラベル(A/B/C/...)で比較します。テキスト生成は行わず、1回のforward passのみです。候補選択肢の実テキスト(「はい」「いいえ」等)が複数tokenへ分割されても問題ありません。比較対象はA/B/C/...の1文字ラベルのみです。

`scorer: masked-lm` / `scorer: qwen-direct` を明示指定すれば自動判定を上書きできます。

## 5. データセット設定

共通部分:

```yaml
datasets:
  - id: example_noul
    task: noul
    label: Example Noul
    feature: 日本語二値判断
    expected_items: 1000
    revision: <commit-hash>
    source: {}
```

| 変数 | 説明 |
|---|---|
| `id` | dataset ID。taskの `datasets` から参照 |
| `task` | `noul`、`choice`、`score` |
| `label` | チャート表示名 |
| `feature` | 評価特徴。チャート軸に表示 |
| `expected_items` | GPU割当の負荷分散に使う概算件数 |
| `revision` | 結果metadataへ記録するデータrevision。Hub sourceにも継承 |
| `limit` | productionでも件数を制限する場合の上限 |
| `source` | 読込元とadapter設定 |

### ローカルデータセット

```yaml
runtime:
  datasets_root: /datasets

datasets:
  - id: local_toxicity
    task: noul
    label: Local Toxicity
    feature: 日本語有害性
    expected_items: 1000
    revision: local-v1
    source:
      format: parquet
      path: toxicity/test.parquet
      adapter: noul
      item_id_field: id
      gold_field: label
      positive_values: [1]
      question_template: "次の文章は有害ですか？\n文章: {text}"
```

実体は `/datasets/toxicity/test.parquet` です。`path` はglobも利用できますが、必ず1ファイルだけに一致させます。

対応format: `arrow`、`csv`、`tsv`、`jsonl`、`parquet`、`jmmlu_zip`。

### Hugging Face Hubデータセット

```yaml
datasets:
  - id: hub_jcola
    task: noul
    label: Hub JCoLA
    feature: 日本語文法
    expected_items: 865
    revision: <commit-hash>
    source:
      repo_id: organization/dataset-name
      config: ja
      split: validation
      adapter: noul
      item_id_field: id
      gold_field: label
      positive_values: [1]
      question_template: "次の文章は日本語として成立しますか？\n文章: {sentence}"
```

Hubデータは `datasets.load_dataset()` で取得し、`datasets_root/.hub-cache` に保存します。

| 変数 | 説明 |
|---|---|
| `source.repo_id` | Hub dataset ID |
| `source.config` | dataset config名。不要なら省略 |
| `source.split` | split名。省略時 `test` |
| `source.revision` | source固有revision。省略時はdataset直下の `revision` を使用 |

gated datasetは `HF_TOKEN` が必要です。完全オフライン実行では事前cacheまたはローカルファイル方式を使ってください。

### local-firstフォールバック

`path` と、下記いずれかの取得元(`repo_id`、`hub_source`、`adapter_dataset`)を同一 `source` に併記できます。動作順は次の通りです。

1. `path` が `datasets_root` 配下で実ファイルに解決できればそれを使う(ネットワーク不要、既存のローカル配置がそのまま高速に動作)
2. 解決できなければ、宣言済みの取得元から自動取得してから読み込む

つまり同じYAML1本で「ローカルに事前配置した環境」と「何もないクリーンな環境」の両方が動きます。`configs/eval/primitives.20260918.yaml` の標準datasetは大半がこの二重構成です。

### `source.hub_source`(単一ファイルをHub Datasetリポジトリから取得)

```yaml
source:
  format: parquet
  path: openjev-extra/janli/data.parquet
  hub_source:
    repo_id: organization/dataset-name
    filename: janli/data.parquet
    revision: <commit-hash>
  adapter: noul
  gold_field: label
  positive_values: [1]
  question_template: "..."
```

`huggingface_hub.hf_hub_download()` でリポジトリ内の1ファイルを `datasets_root` (HF標準cacheレイアウト)へ取得し、以降は通常のローカルformatローダー(`format`で指定した形式)で読み込みます。`path`の実体とは別の場所に保存されるため、`path`は「ローカル事前配置時の期待パス」を表すだけで、`hub_source`取得時には使われません。gatedリポジトリや未認証など取得に失敗した場合はそのdatasetのみスキップされ(`skipped.<task>.<phase>.json`に記録)、他のdatasetの評価は継続します。

### `source.adapter_dataset`(choice primitiveの組込みfetcherを再利用)

```yaml
source:
  path: openai___mmmlu/JA_JP/0.0.0/*/mmmlu-test.arrow
  format: arrow
  adapter_dataset: mmmlu_ja
```

`mmmlu_ja`、`jmmlu`、`jgpqa_diamond`、`jcommonsenseqa`、`xwinograd_ja`、`mgsm_ja`、`gsm8k_ja_mc4`、`gsm8k_ja_mc10`、`jnli` の9つは `src/openjev_ja/eval/datasets/adapters.py` に既存の取得・変換ロジックがあり、`adapter_dataset` にその名前を指定するとそれを再利用します(zip展開、GitHub API経由の取得、gatedリポジトリの分かりやすいエラーメッセージなどを含む)。JGLUEのJNLI由来dataset(`jnli`、`jnli_entailment`、`jnli_contradiction`、`jnli_missing_evidence`)はHugging Face非掲載のためこの経路のみで自動取得します。キャッシュ先は環境変数 `OPENJEV_DATASETS_DIR`(未設定時はHugging Faceの既定cache)です。

### Noul adapter

```yaml
source:
  adapter: noul
  gold_field: label
  positive_values: [yes, 1]
  question_template: "判定対象: {text}"
```

- `positive_values`: Yesとして扱う値
- `positive_min`: 数値が指定値以上ならYes。`positive_values`とのどちらか一方を指定
- `item_id_field`: ID列。既定値 `id`
- `question_template`: `{列名}` を埋め込む質問テンプレート
- `max_chars`: 質問文字数上限。0または省略で無制限

### Score adapter

```yaml
source:
  adapter: score
  question_field: text
  gold_field: score
  gold_thresholds: [0.2, 0.4, 0.6, 0.8]
  criteria: [無害, 低い, 中程度, 高い, 非常に高い]
  prompt: 次の文章の有害度を評価してください。
```

- `criteria`: 低い順に並べた2個以上の候補
- `gold_field`: 正解score列
- `gold_offset`: 元scoreへ加算する整数
- `gold_thresholds`: 連続値を段階値へ変換する境界
- `criteria_field`: 行ごとにcriteriaを持つ場合の列名
- `question_field`: 評価対象列
- `question_template`: 複数列を結合する場合に使用
- `prompt`: 評価指示
- `max_chars`: 評価対象文字数上限

### 行filter

```yaml
source:
  where:
    split: test
  where_in:
    label: [Supported, Refuted]
  where_not:
    status: invalid
```

- `where`: 完全一致
- `where_in`: 候補集合のいずれか
- `where_not`: 指定値を除外

### Choice adapter

既存Choice datasetはdataset IDごとのadapterを利用します。MMMLU、JMMLU、JGPQA、JCommonsenseQA、XWinograd、MGSM、GSM8K、JNLIのschemaに対応しています。新しいschemaを使う場合は `src/openjev_ja/eval/datasets/adapters.py` に変換規則を追加してください。

## 6. オーケストレーション

事前smoke:

```bash
uv run jev-ja-lab-eval-workflow \
  --config configs/eval/primitives.20260918.yaml \
  --phase smoke
```

全taskを順次実行:

```bash
uv run jev-ja-lab-eval-workflow \
  --config configs/eval/primitives.20260918.yaml \
  --phase production
```

Primitive単位:

```bash
uv run jev-ja-lab-eval-noul --config configs/eval/primitives.20260918.yaml --phase production
uv run jev-ja-lab-eval-choice --config configs/eval/primitives.20260918.yaml --phase production
uv run jev-ja-lab-eval-score --config configs/eval/primitives.20260918.yaml --phase production
uv run jev-ja-lab-eval-summary --config configs/eval/primitives.20260918.yaml --phase production
```

一部モデルだけ評価:

```yaml
tasks:
  choice:
    model_ids: [qwen-local, gemma-hub]
```

追加datasetだけ評価する場合、既存 `run_name` と `resume: true` を維持してtaskの `datasets` へIDを追加します。既存summaryは再利用され、追加分だけ実行されます。評価条件を変更した場合は結果混在を避けるため、新しい `run_name` を指定してください。

## 7. batch size調整

1. `--phase smoke` で開始
2. GPUメモリを監視
3. OOMしない範囲で `batch_size` を増加
4. モデル別差が大きい場合は `models[].batch_size` で上書き
5. 本番条件ではbatch sizeを固定

本プロジェクトの実績値はQwen系32、Gemma 4 E2B/E4B 8です。環境、入力長、transformers/PyTorch版で上限は変わります。

## 8. resumeと結果

`resume: true` は次のファイル存在を完了条件にします。

```text
results/<run-name>/<model-id>/<dataset-id>/summary.json
```

出力:

```text
results/<run-name>/
├─ eval_noul.json
├─ eval_choice.json
├─ eval_score.json
├─ eval_summary.json
├─ skipped.<task>.<phase>.json
├─ run_config.<task>.<phase>.json
├─ radar-*.yaml
├─ radar-*.png
└─ <model-id>/<dataset-id>/
   ├─ metadata.json
   ├─ predictions.jsonl
   └─ summary.json
```

## 9. オフライン実行

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Hub IDを初回から使う場合はオフライン設定を外してください。

## 10. よくある問題

| 状況 | 対応 |
|---|---|
| `source_not_found` | `datasets_root` と `source.path` を確認 |
| gated repository error | 利用規約同意後に `HF_TOKEN` を設定 |
| CUDA OOM | batch sizeを半減。1 GPU 1ワーカーを確認 |
| 結果が再計算されない | `resume: false` または新しい `run_name` を使用 |
| Hubへ接続しない | offline環境変数を解除 |
| Choice列が合わない | 対象schema用adapterを追加 |
| チャートが古い | summary taskを再実行 |
