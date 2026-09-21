# jev-ja-lab Project Specification

## 1. 目的

`jev-ja-lab` は、TypeSafe AI の Jev と AlexWortega/openjev の考え方を参考に、**日本語における「文章生成ではなく判断を返すモデル」**を評価・学習・実行するためのプロジェクト。

初期フェーズでは **評価基盤のみ実装**する。  
将来 `train`（NLI fine-tuning）、`infer`（OpenJev-JA推論）、`eval`（評価）、`tests` を独立して拡張できる構成にする。

---

## 2. 技術前提

### Jev

TypeSafe AI は Jev を System One Model と呼び、通常のLLMの文字列生成ではなく、

```text
unstructured state
    ↓
typed probabilistic decisions
```

を返すモデルとして公開している。

公式発表では、

- 新しいモデルアーキテクチャ
- parallel sampler
- RLCD (Reinforcement Learning for Calibrated Decisions)
- 型付き出力
- probability / confidence
- 逐次的な文章生成を行わない

ことが説明されている。[S1]

Jev の内部アーキテクチャや RLCD の詳細は公開されていないため、本プロジェクトでは内部再現を目的にしない。

### AlexWortega/openjev

AlexWortega/openjev は `Qwen3.5-4B` を **3-class NLI cross-encoder** として fine-tuning している。

```text
Premise + Hypothesis
        ↓
Qwen3.5-4B
        ↓
SequenceClassification head
        ↓
contradiction / entailment / neutral
```

モデルカードでは、

- `Qwen3_5ForSequenceClassification`
- last-token pooling
- 3 labels
- plain cross-entropy

と説明されている。[S2]

学習コードでは SNLI + MultiNLI を使用する。[S3]

```text
0 = contradiction
1 = entailment
2 = neutral
```

Alex の multiple-choice 評価は、各選択肢を hypothesis に変換し、

```text
premise    = question
hypothesis = "The correct answer is: {option}"
```

として `P(entailment)` が最大の選択肢を回答とする。[S4]

したがってこれは Jev の蒸留ではなく、

> NLI を汎用 decision primitive として multiple-choice / reranking に転用する方式

である。

### Qwen3.5-4B

初期モデルは `Qwen/Qwen3.5-4B` とする。

Qwen3.5 は 4B dense model で、多言語対応を公称し、Transformers は
`Qwen3_5ForSequenceClassification` を提供している。[S5][S6]

NLI fine-tuning 前のモデルには未学習 classification head を付けて評価しない。

初期 baseline は TheoLeeCJ/openjev と同様に、**生成を行わず選択肢ラベルの logits を直接読む方式**を利用する。[S7]

```text
Question + choices
        ↓
Qwen3.5-4B forward
        ↓
logit(A), logit(B), ...
        ↓
softmax over candidate labels
        ↓
choice probabilities
```

---

## 3. 初期評価対象

初期フェーズでは以下を比較する。

1. **TypeSafe Jev**
2. **Qwen/Qwen3.5-4B raw direct-logit baseline**
3. **AlexWortega/openjev**（参照用・任意）

将来追加する。

- jev-ja-lab fine-tuned model
- Gemma 4 E2B
- Gemma 4 E4B
- その他小型日本語対応モデル

---

## 4. 日本語評価データセット

### Core benchmark

| ID | Dataset | 用途 |
|---|---|---|
| `mmmlu_ja` | `openai/MMMLU`, `JA_JP` | MMLU と同一問題系の日本語評価 |
| `jmmlu` | `nlp-waseda/JMMLU` | 日本語一般知識・日本固有問題 |
| `jgpqa_diamond` | `llm-jp/jgpqa`, `gpqa_diamond` | 高難度科学推論 |
| `jcommonsenseqa` | `sbintuitions/JCommonsenseQA` | 日本語常識推論 |
| `xwinograd_ja` | `Muennighoff/xwinograd`, `jp` | 文脈・照応推論 |
| `mgsm_ja` | `jbross-ibm-research/mgsm`, `ja` | 日本語数学（250問） |
| `gsm8k_ja_rest` | `SakanaAI/gsm8k-ja-test_250-1319` | GSM8K残り1069問 |
| `jnli` | JGLUE JNLI | 日本語NLI能力 |

MMMLU は MMLU test set を専門翻訳者が日本語化したデータなので、英語版との比較に向く。[S8]

JMMLU は 7,536 問・56分野で、日本語化MMLUに加えて日本固有問題を含む。[S9]

JGPQA は GPQA の日本語訳で、機械翻訳後に外部専門家が確認・修正している。`gpqa_diamond` も公開されているが、Hugging Faceでは利用規約への同意が必要。[S10]

JCommonsenseQA は日本語ネイティブの5択常識QA。[S11]

XWinograd の `jp` は 959例。[S12]

MGSM は GSM8K の同一250問を日本語を含む各言語へ人手翻訳している。[S13]  
SakanaAI のデータは GSM8K test の残り1069問を日本語化しており、MGSM 250問と合わせて1319問を構成できる。[S14]

JNLI は `entailment / contradiction / neutral` を持つ日本語NLIで、OpenJev-JA学習後の直接評価にも使用する。[S15]

### GSM8K の Alex 互換評価

Alex と同様に、正解数値に deterministic distractor を加えて以下を作る。[S4]

- `gsm8k_ja_mc4`
- `gsm8k_ja_mc10`

生成規則と seed は固定し、結果再現性を保証する。

---

## 5. 評価方法

全 scorer を共通インターフェースにする。

```python
score(question, options) -> ScoreResult
```

`ScoreResult` は最低限、

```text
scores
predicted_index
latency_ms
metadata
```

を持つ。

### `QwenDirectScorer`

- `Qwen/Qwen3.5-4B`
- `.generate()` を使用しない
- `A/B/C/...` の固定 answer token の logits のみ取得
- candidate logits のみに softmax
- answer token が単一tokenであることを起動時に検証
- model revision / dtype / device / prompt hash を保存

### `JevScorer`

- TypeSafe の**最新公式 API schema**に従う
- API key は環境変数 `TYPESAFE_API_KEY`
- URL・keyをコードへハードコードしない
- Choice の全 option probability を保存
- HTTP response metadata と latency を保存
- rate limit / retry を実装するが、過剰な自動再試行はしない

### `NLICrossEncoderScorer`

Alex方式の参照実装。

各optionについて、

```text
premise    = question
hypothesis = Japanese template(option)
```

を作り、`P(entailment)` 最大を選択する。

デフォルト日本語 hypothesis:

```text
正しい答えは「{option}」である。
```

英語テンプレートとの比較を可能にすること。

---

## 6. 指標

最低限以下を保存する。

- Accuracy
- 正解数 / 全件数
- mean latency
- p50 / p95 latency
- throughput
- per-item score / probability
- model / dataset revision
- seed
- prompt template
- git commit

probability distribution が定義できる scorer では追加で、

- NLL
- Brier Score
- ECE

を計算する。

NLI の独立 `P(entailment)` は option 間で合計1になる保証がないため、**Alex互換 accuracy と calibration 指標を混同しない**。

---

## 7. プロジェクト構成

```text
jev-ja-lab/
├─ README.md
├─ PROJECT_SPEC.md
├─ pyproject.toml
├─ .env.example
├─ .gitignore
├─ configs/
│  ├─ eval/
│  ├─ train/
│  └─ infer/
├─ src/
│  └─ openjev_ja/
│     ├─ common/
│     ├─ eval/
│     │  ├─ datasets/
│     │  ├─ scorers/
│     │  ├─ metrics.py
│     │  └─ runner.py
│     ├─ train/
│     └─ infer/
├─ tests/
│  ├─ unit/
│  ├─ smoke/
│  └─ fixtures/
└─ results/
```

責務を混ぜない。

```text
eval  = benchmark / scorer / metrics
train = fine-tuning
infer = 学習済みjev-ja-labの実行
tests = unit / smoke / integration
```

現フェーズでは `train` / `infer` に実処理を実装しない。  
将来の拡張境界だけ維持する。

`results/`、model weights、dataset cache は Git 管理しない。

---

## 8. Quick Start

Python 3.11+ を標準とする。

### uv

```bash
git clone <repo>
cd jev-ja-lab

uv sync --extra eval --extra dev
uv run pytest
uv run jev-ja-lab-eval --help
```

### 標準 venv + pip

```bash
git clone <repo>
cd jev-ja-lab

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
pip install -e ".[eval,dev]"

pytest
jev-ja-lab-eval --help
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

依存関係の正本は `pyproject.toml` とし、uv専用コードを書かない。

---

# 9. Codexへのプロジェクト作成指示

以下をそのまま開発指示として使用する。

---

## Codex Task

`jev-ja-lab` という Python プロジェクトを新規作成してください。

目的は、日本語における Jev / raw Qwen / OpenJev 系モデルの判断能力を、共通benchmarkで再現可能に比較することです。

### 開発原則

1. **最初にこの `PROJECT_SPEC.md` と引用元を読み、実装前に仕様を整理する。**
2. いきなり全機能を作らず、小さい vertical slice 単位で進める。
3. 各stepの終了時に smoke test と unit test を実行する。
4. testが通ってから次のstepへ進む。
5. 最後にのみ full benchmark を実行する。
6. 現フェーズでは評価機能だけ実装し、学習処理を実装しない。
7. API key、token、絶対パスをハードコードしない。
8. dataset/model revision、seed、prompt、実行環境を結果へ保存して再現性を確保する。
9. 過剰な抽象化を避け、単純なPythonコードを優先する。
10. 既存の公式ライブラリ (`transformers`, `datasets`, `torch`) を優先し、独自実装を最小限にする。

### Step 0: 調査・計画

実装前に以下を確認する。

- TypeSafe Jev の最新API
- AlexWortega/openjev の `train.py`, `eval.py`
- Qwen3.5 Transformers API
- 各datasetのschema / split / license / access条件

その後、短い `IMPLEMENTATION_PLAN.md` を作る。

不明なAPI仕様を推測して実装しない。

### Step 1: Skeleton

以下を作る。

- `pyproject.toml`
- `README.md`
- `src/openjev_ja/{common,eval,train,infer}`
- `tests/{unit,smoke,fixtures}`
- `configs/{eval,train,infer}`

`uv` と通常の `venv + pip` の両方でinstall可能にする。

`ruff` と `pytest` を dev dependency にする。

### Step 2: 最小 vertical slice

最初は、

```text
MMMLU JA-JP
    +
MockScorer
```

だけ実装する。

5件程度で、

```bash
jev-ja-lab-eval --dataset mmmlu_ja --scorer mock --limit 5
```

が成功し、JSON結果を保存することを確認する。

### Step 3: Qwen baseline

`QwenDirectScorer` を追加する。

- default model: `Qwen/Qwen3.5-4B`
- generation禁止
- answer-token logits を直接取得
- candidate logits のみ softmax
- answer label tokenization を検証

まず10件だけGPU smoke testする。

```bash
jev-ja-lab-eval \
  --dataset mmmlu_ja \
  --scorer qwen-direct \
  --model Qwen/Qwen3.5-4B \
  --limit 10
```

### Step 4: Dataset adapters

1つずつ追加し、その都度adapter unit testを実行する。

```text
mmmlu_ja
jmmlu
jgpqa_diamond
jcommonsenseqa
xwinograd_ja
mgsm_ja
gsm8k_ja_mc4
gsm8k_ja_mc10
jnli
```

すべて内部では共通形式へ変換する。

```python
BenchmarkItem(
    id=...,
    question=...,
    options=[...],
    gold_index=...,
    metadata={...},
)
```

JGPQA は gated dataset なので、未認証時は明確なエラーを表示する。

### Step 5: Jev adapter

公式最新APIのみを根拠に `JevScorer` を実装する。

```bash
export TYPESAFE_API_KEY=...
```

API keyがない場合、test全体を失敗させず integration test のみskipする。

Mock HTTPによるunit testを必ず作る。

### Step 6: Alex互換 NLI scorer

`NLICrossEncoderScorer` を追加する。

初期確認用モデル:

```text
AlexWortega/openjev
```

各optionの `P(entailment)` をscoreとし、argmaxを回答にする。

このscorerは将来作成する `jev-ja-lab` checkpointも同じinterfaceで読み込めるようにする。

### Step 7: Metrics / Results

各runについて、

```text
results/<run-id>/
├─ metadata.json
├─ predictions.jsonl
└─ summary.json
```

を保存する。

最低限、

```text
accuracy
correct
total
mean_latency_ms
p50_latency_ms
p95_latency_ms
```

を出す。

可能なscorerのみ、

```text
NLL
Brier
ECE
```

を出す。

### Step 8: Tests

最低限以下を作る。

- dataset schema conversion tests
- metric tests
- scorer mock tests
- Qwen answer-token validation test
- Jev HTTP mock test
- deterministic GSM8K distractor test
- result serialization test

通常のテストはmodel downloadや外部APIを必要としないこと。

```bash
pytest
ruff check .
```

が成功すること。

### Step 9: Smoke test

本番評価前に、

```text
1 dataset
×
5〜10 samples
×
each available scorer
```

を実行する。

出力とgold mappingを目視確認する。

特に、

- 選択肢順序
- gold index
- 日本語文字化け
- probabilityとchoice
- GSM8K distractor
- Jev/Qwenで同じ問題が渡されていること

を確認する。

### Step 10: Production evaluation

smoke / unit test がすべて成功してから full evaluation を実行する。

Full run はCLIから、

```bash
jev-ja-lab-eval \
  --dataset all \
  --scorer qwen-direct \
  --model Qwen/Qwen3.5-4B
```

のように再実行可能にする。

巨大な中間データやmodelをrepositoryへcommitしない。

最後に、

- 実装内容
- 実行コマンド
- test結果
- 未解決事項
- full evaluationを実行する前提条件

を `README.md` に簡潔に記載する。

---

## 10. Sources

- **[S1] TypeSafe AI — Introducing System One Models and Jev**  
  https://typesafe.ai/blog/introducing-system-one-models-and-jev

- **[S2] AlexWortega/openjev model card**  
  https://huggingface.co/AlexWortega/openjev

- **[S3] AlexWortega/openjev — NLI training code**  
  https://huggingface.co/AlexWortega/openjev/blob/main/code/train.py

- **[S4] AlexWortega/openjev — evaluation code**  
  https://huggingface.co/AlexWortega/openjev/blob/main/code/eval.py

- **[S5] Qwen/Qwen3.5-4B**  
  https://huggingface.co/Qwen/Qwen3.5-4B

- **[S6] Transformers Qwen3.5 documentation**  
  https://huggingface.co/docs/transformers/main/model_doc/qwen3_5

- **[S7] TheoLeeCJ/openjev — direct option-logit baseline**  
  https://github.com/TheoLeeCJ/openjev

- **[S8] OpenAI MMMLU**  
  https://huggingface.co/datasets/openai/MMMLU

- **[S9] JMMLU**  
  https://huggingface.co/datasets/nlp-waseda/JMMLU

- **[S10] JGPQA**  
  https://huggingface.co/datasets/llm-jp/jgpqa

- **[S11] JCommonsenseQA**  
  https://huggingface.co/datasets/sbintuitions/JCommonsenseQA

- **[S12] XWinograd**  
  https://huggingface.co/datasets/Muennighoff/xwinograd

- **[S13] MGSM**  
  https://huggingface.co/datasets/jbross-ibm-research/mgsm

- **[S14] SakanaAI GSM8K Japanese remainder**  
  https://huggingface.co/datasets/SakanaAI/gsm8k-ja-test_250-1319

- **[S15] JGLUE / JNLI**  
  https://github.com/yahoojapan/JGLUE


