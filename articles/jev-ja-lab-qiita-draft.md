# jev-ja-lab: 日本語で「判断するモデル」を横断評価する

## アブストラクト

本記事では、TypeSafe AI の Jev（System One Model）と AlexWortega/openjev の考え方を参考に構築した日本語評価基盤 `jev-ja-lab` を紹介する。`jev-ja-lab` は、モデルに文章を生成させるのではなく「判断」だけを返させ、その判断能力を Noul（Yes/No判定）・Choice（多肢選択）・Score（順序尺度スコアリング）という3つの Primitive に統一して評価する。

評価はテキスト生成を一切行わず、候補ラベルに対応する単一トークンの logit を1回の forward pass で比較する direct-logit 方式を採る。当初は causal LM（Qwen系など）のみを対象としていたが、今回 BERT 系のエンコーダ（masked LM）にも対応を拡張し、`config.architectures` からモデルの種類を自動判定して適切な scorer を選択する仕組みを実装した。

本記事では、Jev の考え方の概要、jev-ja-lab がそれをどう実装に落とし込んでいるか、評価方法・使用データセット・スコアの付け方を説明したうえで、Qwen系8モデルとBERT系2モデル、計10モデルの評価結果をまとめる。

## 自己紹介

<!-- ここに自己紹介を記入してください -->

## Jevの簡単な説明

TypeSafe AI は Jev を **System One Model** と呼んでいる。通常のLLMのように文章を逐次生成するのではなく、

```text
unstructured state
    ↓
typed probabilistic decisions
```

という形で、非構造化データから型付き確率的判断を直接返すモデルとして公開されている。公式発表では次のような特徴が説明されている。

- 新しいモデルアーキテクチャ
- parallel sampler
- RLCD（Reinforcement Learning for Calibrated Decisions）
- 型付き出力・probability / confidence
- 逐次的な文章生成を行わない

ただし Jev の内部アーキテクチャや RLCD の詳細は公開されていない。そのため `jev-ja-lab` は Jev の内部再現を目的とせず、「判断を返すモデル」という考え方だけを設計の出発点にしている。

もう一つの参考実装が AlexWortega/openjev で、こちらは `Qwen3.5-4B` を **3-class NLI cross-encoder** として fine-tuning したものである。

```text
Premise + Hypothesis
        ↓
Qwen3.5-4B (SequenceClassification head)
        ↓
contradiction / entailment / neutral
```

多肢選択問題を評価する際は、各選択肢を hypothesis に変換する。

```text
premise    = question
hypothesis = "The correct answer is: {option}"
```

とし、`P(entailment)` が最大になる選択肢を回答として採用する。つまりこれは Jev の蒸留ではなく、「NLIを汎用の decision primitive として multiple-choice / reranking に転用する」という方式である。`jev-ja-lab` にも同じ発想の `NLICrossEncoderScorer`（entailment ラベルを持つ NLI cross-encoder 専用）を実装しており、任意の3-class NLI fine-tuned モデルを同じ枠組みで評価できる。

## 今回のopenjevの理論（LLMでどのように実装しているのか）

`jev-ja-lab` の評価基盤は、Jev や NLI cross-encoder のような専用 fine-tuning を前提としない。素の事前学習済みモデルに対して、**生成させず、選択肢ラベルの logit を直接読む** baseline 方式を採用している。

### プロンプト設計

Noul・Choice・Score のすべてを「候補から1つを選ぶ」問題として統一的に扱う。選択肢は `A`, `B`, `C`, ... のアルファベット1文字ラベルへ写像し、次のテンプレートでプロンプトを組み立てる。

```text
問題:
{question}

選択肢:
A. {option_A}
B. {option_B}
...

答え:
```

この方式の利点は、Noul の「はい/いいえ」や Score の「無害/低い/中程度/高い」のように選択肢の実テキストが複数トークンに分割されても関係ないことである。比較対象は常に `A`/`B`/`C`/... の**単一トークン**ラベルに限定されるため、事前にラベルが1トークンへ収まることだけを検証すればよい（`validate_answer_tokens`）。

### Causal LM（生成モデル）の場合

Qwen系などの causal LM では、プロンプトの直後に続く**次トークン**の logit を1回の forward pass で読む。

```text
Question + choices
        ↓
Qwen forward (1 pass, no generation)
        ↓
logit(A), logit(B), ...
        ↓
softmax → choice probabilities
```

`Qwen3.5` のような Vision-Language 系アーキテクチャ（`config.model_type` が `qwen3_5` / `gemma4` など）は `AutoModelForImageTextToText` で、それ以外は `AutoModelForCausalLM` でロードする分岐を持つ。

### Masked LM（BERT系エンコーダ）の場合

今回拡張した部分である。BERT/RoBERTa/ModernBERT/ELECTRA/DeBERTa のようなエンコーダモデルには「次トークン」という概念がない。そこで、プロンプトの末尾を `[MASK]`（モデルのmask tokenに合わせて動的に選択）に置き換え、その**マスク位置**の logit を同じように比較する。

```text
問題:
{question}

選択肢:
A. {option_A}
B. {option_B}
...

答え: [MASK]
```

```text
Question + choices + [MASK]
        ↓
BERT forward (1 pass, no generation)
        ↓
mask position の logit(A), logit(B), ...
        ↓
softmax → choice probabilities
```

causal LM 用の `QwenDirectScorer` と masked LM 用の `MaskedLMScorer` は、プロンプト組み立て（`format_choices`）とラベル検証（`validate_answer_tokens`）のロジックを共通モジュールとして共有しており、「回答位置がプロンプト末尾の次トークンか、プロンプト中の `[MASK]` か」という一点だけが異なる設計になっている。

### モデル種別の自動判定

YAML設定で `scorer` を明示しない場合、Hugging Face の `config.architectures` を見て自動的に scorer を選ぶ。`...ForMaskedLM` で終わるアーキテクチャは `MaskedLMScorer`、それ以外の causal / image-text-to-text 系は `QwenDirectScorer` に振り分けられる。これにより、`repo_id` を1行指定するだけで LLM でも BERT でも同じ評価パイプラインに乗せられる。

```yaml
models:
  - id: modernbert-ja-310m
    repo_id: sbintuitions/modernbert-ja-310m   # 自動的に masked-lm scorer が選ばれる
  - id: qwen3.5-0.8b
    repo_id: Qwen/Qwen3.5-0.8B                  # 自動的に causal LM scorer が選ばれる
```

## 評価方法

評価は次の3つの Primitive で構成される。

| Primitive | 内容 | 主指標 |
|---|---|---|
| **Noul** | Yes / No の二値判断 | F1（Precision/Recall/Brier Score/ECEも記録） |
| **Choice** | 複数候補から1つを選択 | Top-1 Accuracy（NLL/Brier Score/ECEも記録） |
| **Score** | 順序付き尺度のスコアリング | 正規化Quadratic Weighted Kappa（MAE/RMSE/Spearman相関も記録） |

処理全体は次のパイプラインで実行される。

```text
YAML読込
  → データセット読込・Primitive変換
  → モデルをGPUへロード
  → dataset単位で評価（1 forward pass、生成なし）
  → Primitive集約
  → 総合集約
  → レーダーチャート生成
```

各データセットの指標はまず 0〜1 へ正規化し、Primitive内でデータセット単位の macro 平均を取る。そのうえで `tasks.summary.weights`（本評価では Noul:Choice:Score = 1:1:1）に従って Primitive間を加重平均し、総合スコアとする。データセット件数の差が総合スコアを支配しないよう、件数ではなく Primitive／データセット単位で平等に扱う設計である。

## 評価に利用したデータセット

標準評価プロファイルは全30軸で構成される。

| Primitive | 軸数 | データセット |
|---|---:|---|
| Noul | 14 | JGLUE JNLI(3軸: entailment/contradiction/missing evidence)、JaNLI、JCoLA(in-domain/out-of-domain)、PAWS-X JA、TextDetox JA、JAD-AFC(3軸: true/false/NEI)、WRIME Binary(3軸: joy/anger/positive) |
| Choice | 9 | MMMLU JA-JP、JMMLU、JGPQA Diamond、JCommonsenseQA、XWinograd JA、MGSM JA、GSM8K JA(4択/10択)、JNLI 3-class |
| Score | 7 | WRIME(joy/anger/sentiment)、決定論的合成データ4種（urgency/dissatisfaction/risk/relevance） |

WRIME は CC BY-NC-ND 4.0 のため、評価実行のたびに配布元（GitHub `ids-cv/wrime`）から直接取得する方式を取り、再配布は行っていない。JGPQA Diamond のように Hugging Face 側で利用規約への同意が必要なデータセットは、未認証環境では自動的にスキップされ、他のデータセットの評価は継続される。

## スコアの付け方

- **Noul**: 候補は「いいえ」「はい」の2値。`A`/`B` ラベルの logit を softmax し、確率最大のラベルを予測とする。正解ラベルとの F1 を算出する。
- **Choice**: 候補は最大26択まで対応（`A`〜`Z`）。各候補ラベルの logit を softmax し、Top-1 Accuracy を算出する。
- **Score**: `criteria` として順序付き候補（例:「無害」「低い」「中程度」「高い」「非常に高い」）を用意し、正解スコアとの順序のズレを Quadratic Weighted Kappa で評価したうえで 0〜1 へ正規化する。

いずれも**テキスト生成を行わない**。1回の forward pass で得られる logit のみを使うため、サンプリング温度や生成長といった生成パラメータに評価結果が左右されない。

## 結果まとめ

Qwen系8モデル（Qwen3.5-0.8B、Qwen3-0.6B、Qwen2.5シリーズの0.5B/1.5B/3B ×base/Instruct）と、今回新たに対応した BERT系2モデル（ModernBERT-ja 310M、BERT Base Multilingual Cased）、計10モデルを同一条件で評価した。

| モデル | 総合 | Noul(F1) | Choice(Acc) | Score(QWK) |
|---|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | **0.6264** | 0.6031 | 0.4698 | 0.8064 |
| Qwen2.5-3B | 0.5758 | 0.5517 | 0.4379 | 0.7378 |
| Qwen2.5-1.5B-Instruct | 0.5248 | 0.5544 | 0.4238 | 0.5963 |
| Qwen2.5-0.5B-Instruct | 0.5052 | 0.5519 | 0.3431 | 0.6206 |
| Qwen2.5-1.5B | 0.5006 | 0.5524 | 0.3956 | 0.5539 |
| Qwen3.5-0.8B | 0.4981 | 0.4669 | 0.3488 | 0.6787 |
| Qwen2.5-0.5B | 0.4620 | 0.5463 | 0.3387 | 0.5008 |
| Qwen3-0.6B | 0.4566 | 0.4225 | 0.3317 | 0.6157 |
| ModernBERT-ja 310M | 0.4349 | 0.4318 | 0.3499 | 0.5231 |
| BERT Base Multilingual Cased | 0.3489 | 0.3106 | 0.2344 | 0.5017 |

primitive_coverage はすべてのモデルで `3/3`（Noul・Choice・Score すべて完走）。

観察できる傾向:

- 同じQwen2.5サイズ帯では Instruct 版が base 版を一貫して上回る（例: 3B → 0.5758 → 0.6264、0.5B → 0.4620 → 0.5052）。
- パラメータ数と総合スコアはおおむね相関するが、Qwen2.5-1.5B-Instruct(0.5248) が Qwen3.5-0.8B(0.4981) を上回るなど、世代・チューニングの差がサイズ差を逆転する場面もある。
- 今回追加した BERT系2モデルは、fine-tuning なしの zero-shot 評価にもかかわらず Choice primitive（0.3499 / 0.2344）で一部の小型 causal LM に匹敵する結果を示した。一方 Score primitive は causal LM 群より一段低い水準（0.52前後）にとどまった。
- ModernBERT-ja はBERT Base Multilingual Casedを全Primitiveで上回った。日本語特化の事前学習が効いていると考えられる。

<!-- ここにレーダーチャート（radar-summary.png等）を挿入 -->

以上、`jev-ja-lab` による日本語判断モデルの横断評価の概要と結果である。
