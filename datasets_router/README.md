---
license: cc-by-sa-4.0
language:
- ja
pretty_name: openjev-ja-eval
task_categories:
- text-classification
- multiple-choice
configs:
- config_name: noul
  data_files:
  - split: test
    path: noul/test.parquet
- config_name: choice
  data_files:
  - split: test
    path: choice/test.parquet
- config_name: score
  data_files:
  - split: test
    path: score/test.parquet
- config_name: all
  data_files:
  - split: test
    path: all/test.parquet
- config_name: router
  data_files:
  - split: test
    path: router/test.parquet
---

# openjev-ja-eval

> Japanese evaluation datasets for Noul / Choice / Score decision models

## What is this?

[jev-ja-lab](https://github.com/hachiko85/jev-ja-lab) で日本語の判断モデルを評価するためのデータセット集です。
判断タスクを Noul(二値判定)・Choice(選択式)・Score(段階評価)の 3 種類に分け、既存の公開データセット
15 件をまとめています。

MIT・Apache-2.0・CC BY・CC BY-SA 4.0 で公開されている 11 件は、共通の形式に揃えて本リポジトリに
**収録(ミラー)** し、集合物として **CC BY-SA 4.0** で配布します。それ以外の 4 件(JMMLU(CC BY-NC-ND 4.0)、WRIME ver2(CC BY-NC-ND 4.0)、PAWS-X (ja)(other (PAWS-X の利用条件に従う))、TextDetox multilingual toxicity (ja)(OpenRAIL++))は、再配布の
条件が合わないため本リポジトリには含めず、`manifest.json` に配布元・revision・split を記録して、取得時に
配布元から直接ダウンロードします(ルーター)。

## Subsets

| Subset | Description | Split | Rows | Datasets (mirror) |
|---|---|---|---:|---|
| `noul` | Noul(二値判定) | `test` | 11,408 | `jnli_entailment`, `jnli_contradiction`, `jnli_missing_evidence`, `janli_entailment`, `jcola_in_domain`, `jcola_out_of_domain`, `jad_afc_true`, `jad_afc_false`, `jad_afc_nei` |
| `choice` | Choice(選択式) | `test` | 20,942 | `mmmlu_ja`, `jcommonsenseqa`, `xwinograd_ja`, `mgsm_ja`, `gsm8k_ja_mc4`, `gsm8k_ja_mc10`, `jnli` |
| `score` | Score(段階評価) | `test` | 13,300 | `synthetic_urgency`, `synthetic_dissatisfaction`, `synthetic_risk`, `synthetic_relevance`, `helpsteer_correctness`, `helpsteer_helpfulness`, `helpsteer_verbosity`, `helpsteer_complexity`, `helpsteer_coherence` |
| `all` | All(上3つすべて) | `test` | 45,650 | `jnli_entailment`, `jnli_contradiction`, `jnli_missing_evidence`, `janli_entailment`, `jcola_in_domain`, `jcola_out_of_domain`, `jad_afc_true`, `jad_afc_false`, `jad_afc_nei`, `mmmlu_ja`, `jcommonsenseqa`, `xwinograd_ja`, `mgsm_ja`, `gsm8k_ja_mc4`, `gsm8k_ja_mc10`, `jnli`, `synthetic_urgency`, `synthetic_dissatisfaction`, `synthetic_risk`, `synthetic_relevance`, `helpsteer_correctness`, `helpsteer_helpfulness`, `helpsteer_verbosity`, `helpsteer_complexity`, `helpsteer_coherence` |
| `router` | ルーティング表(収録していないデータセットの配布元一覧) | `test` | 5 | |

評価の split は `test` です。配布元に test が無い、または test のラベルが非公開のもの(JCommonsenseQA・
JNLI・JCoLA)は validation / valid を `test` として収録し、元の split は `source_split` 列に残しています。

### Data fields

| Field | Type | Description |
|---|---|---|
| `id` | string | 行ID(データセットID + 元データ内の番号) |
| `primitive` | string | `Noul` / `Choice` / `Score` |
| `dataset_id` | string | 評価用データセットID(jev-ja-lab の設定と同一。例: `jnli_entailment`) |
| `source_dataset` | string | 出典データセット名 |
| `source_repo` | string | 出典の配布元(Hugging Face repo または GitHub URL) |
| `source_url` | string | 出典データセットのURL |
| `source_split` | string | 出典での split(`(none)` は split 区分なし) |
| `source_license` | string | 出典データセットのライセンス |
| `question` | string | モデルに与える質問・状態 |
| `options` | list[string] | 選択肢(Noul は `いいえ`・`はい`、Score は段階の説明) |
| `gold_index` | int | 正解の選択肢番号(Score は段階) |
| `metadata` | string | 元データ由来の補足情報(JSON 文字列) |

## Datasets

| Name | Description | Task | Distribution | Source | Split used | License | Rows |
|---|---|---|---|---|---|---|---:|
| `mmmlu_ja` MMMLU (JA_JP) | OpenAIが公開するMMLUの多言語版のうち日本語(JA_JP)。57科目の4択知識問題。 | choice | mirror | [huggingface.co/datasets/openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU) | test | MIT | 14,042 |
| `jmmlu` JMMLU | MMLUの日本語翻訳・日本固有問題を含む4択知識問題(全科目)。 | choice | router | [huggingface.co/datasets/nlp-waseda/JMMLU](https://huggingface.co/datasets/nlp-waseda/JMMLU) | test | CC BY-NC-ND 4.0 | 7,536 |
| `jcommonsenseqa` JCommonsenseQA | JGLUEの日本語常識推論5択。JGLUEのtestはラベル非公開のためvalidationを評価に使用。 | choice | mirror | [huggingface.co/datasets/sbintuitions/JCommonsenseQA](https://huggingface.co/datasets/sbintuitions/JCommonsenseQA) | validation | CC BY-SA 4.0 | 1,119 |
| `xwinograd_ja` XWinograd (ja) | 多言語Winograd Schema Challengeの日本語(jp)。文脈・照応解析の2択。 | choice | mirror | [huggingface.co/datasets/Muennighoff/xwinograd](https://huggingface.co/datasets/Muennighoff/xwinograd) | test | CC BY 4.0 | 959 |
| `mgsm_ja` MGSM (ja) | GSM8Kを多言語化した数学文章題の日本語版。評価では選択式に変換して使用。 | choice | mirror | [huggingface.co/datasets/jbross-ibm-research/mgsm](https://huggingface.co/datasets/jbross-ibm-research/mgsm) | test | CC BY-SA 4.0 | 250 |
| `gsm8k_ja` GSM8K-JA (test 250-1319) | SakanaAIによるGSM8Kの日本語テスト分割(250〜1319件目)。評価では4択(mc4)と10択(mc10)に変換して使用。 | choice | mirror | [huggingface.co/datasets/SakanaAI/gsm8k-ja-test_250-1319](https://huggingface.co/datasets/SakanaAI/gsm8k-ja-test_250-1319) | test | Apache-2.0 | 1,069 |
| `jnli` JNLI (JGLUE v1.1.0) | JGLUEの日本語自然言語推論(含意・矛盾・中立の3値)。Choiceとして3択、Noulとして各ラベルの二値判定に使用。testはラベル非公開のためvalidを評価に使用。 | noul・choice | mirror | [github.com/yahoojapan/JGLUE](https://github.com/yahoojapan/JGLUE) | valid | CC BY-SA 4.0 | 2,434 |
| `wrime` WRIME ver2 | 文の書き手・読み手の感情強度アノテーション付きSNSコーパス。Scoreとして読み手の喜び・怒り・感情極性の強度、Noulとして二値化した判定に使用。再配布に制限があるため本リポジトリには保存しない。 | noul・score | router | [github.com/ids-cv/wrime](https://github.com/ids-cv/wrime) | test | CC BY-NC-ND 4.0 | 2,500 |
| `janli` JaNLI | 語順・言い換えなど日本語特有の現象に絞った敵対的NLI。含意判定(Noul)に使用。 | noul | mirror | [huggingface.co/datasets/hpprc/janli](https://huggingface.co/datasets/hpprc/janli) | test | CC BY-SA 4.0 | 720 |
| `jcola` JCoLA | 日本語の文法容認性判断。既知構文(in-domain)と未知構文(out-of-domain)のvalidを使用。 | noul | mirror | [github.com/osekilab/JCoLA](https://github.com/osekilab/JCoLA) | valid | CC BY-SA 4.0 | 1,550 |
| `paws_x_ja` PAWS-X (ja) | 語順を入れ替えた敵対的な言い換え判定の多言語版、日本語(ja)。2文が同じ意味かの二値判定(Noul)。 | noul | router | [huggingface.co/datasets/google-research-datasets/paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x) | test | other (PAWS-X の利用条件に従う) | 2,000 |
| `textdetox_ja` TextDetox multilingual toxicity (ja) | 多言語有害表現データセットの日本語部分。有害表現検出の二値判定(Noul)。日本語は言語名のsplit(ja)のみでtrain/testの区別はない。 | noul | router | [huggingface.co/datasets/textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) | ja | OpenRAIL++ | 5,000 |
| `jad_afc` JAD-AFC | 富士通研究所の日本語ファクトチェック用データセット。真・偽・情報不足(NEI)の3種をそれぞれNoulの二値判定に使用。split区分なし(CSV1枚)。 | noul | mirror | [github.com/FujitsuResearch/japanese-dataset-for-automated-fact-checking](https://github.com/FujitsuResearch/japanese-dataset-for-automated-fact-checking) | (none) | Apache-2.0 | 612 |
| `synthetic_score` Synthetic Score (独自) | 緊急度・不満度・リスク・関連度の4軸を、決定的テンプレート(seed=42)で生成した5段階Score。1軸200件、計800件。 | score | mirror | [hachiko85/openjev-ja-eval/synthetic_score](https://huggingface.co/datasets/hachiko85/openjev-ja-eval/tree/main/synthetic_score) | test | MIT | 800 |
| `helpsteer2_ja_benchmark_v1` HelpSteer2-JA benchmark-v1 (抽出サブセット) | kunishou/HelpSteer2-20k-ja(19,958行)からprompt単位で重複なしに2,500件を抽出(seed=42、sha256順)。正確性・有用性・詳細度・複雑さ・一貫性の5軸(各0〜4)のScore。抽出手順と件数は同梱のbenchmark-v1.manifest.json参照。 | score | mirror | [hachiko85/openjev-ja-eval/helpsteer2_ja](https://huggingface.co/datasets/hachiko85/openjev-ja-eval/tree/main/helpsteer2_ja)。原典: [kunishou/HelpSteer2-20k-ja](https://huggingface.co/datasets/kunishou/HelpSteer2-20k-ja) | test | CC BY 4.0 | 2,500 |

収録していないデータセット(ルーター対象。`router` サブセットに配布元一覧があります):

- `jgpqa_diamond` JGPQA (diamond) ([https://huggingface.co/datasets/llm-jp/jgpqa](https://huggingface.co/datasets/llm-jp/jgpqa), CC BY 4.0): 配布元で利用規約への同意とHugging Faceトークンが必要なため、ルーターのサブセットには含めない。--include-gated指定時のみ、利用者自身のトークンで取得を試みる。

## How to download and use

収録データ(mirror)は通常の Hugging Face データセットとして読み込めます。ルーター対象(JMMLU・WRIME・JGPQA・PAWS-X・TextDetox)だけは、
同梱クライアントで配布元から取得します。リポジトリが private の間はトークン(環境変数 `HF_TOKEN` または `token=`)が必要です。

### Using `datasets`

各サブセット(`noul` / `choice` / `score` / `all`)は `test` split を持ちます。`dataset_id` で個別のデータセットに絞れます。

```python
from datasets import load_dataset

REPO = "hachiko85/openjev-ja-eval"

noul = load_dataset(REPO, "noul", split="test")           # primitive == "Noul" のすべて
jnli = noul.filter(lambda r: r["dataset_id"] == "jnli_entailment")
print(jnli[0]["source_dataset"], jnli[0]["question"], jnli[0]["options"], jnli[0]["gold_index"])

choice = load_dataset(REPO, "choice", split="test")
everything = load_dataset(REPO, "all", split="test")       # Noul + Choice + Score

# 収録していないデータセットの配布元一覧(ルーティング表)
routed = load_dataset(REPO, "router", split="test")
```

### Using `huggingface_hub`

```python
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

path = hf_hub_download("hachiko85/openjev-ja-eval", "choice/test.parquet", repo_type="dataset")
table = pq.read_table(path)
print(table.num_rows, table.column_names)
```

コマンドラインでも取得できます:

```bash
hf download hachiko85/openjev-ja-eval --repo-type dataset --include "choice/*" "LICENSES/*" --local-dir ./openjev-ja-eval
```

### Fetching the routed datasets

ルーター対象は、同梱クライアント `openjev_ja_eval.py` が `manifest.json` に固定した revision で配布元から直接取得し、
[jev-ja-lab](https://github.com/hachiko85/jev-ja-lab) が読む `datasets/` 配置で保存します。

```python
from huggingface_hub import hf_hub_download

client = hf_hub_download("hachiko85/openjev-ja-eval", "openjev_ja_eval.py", repo_type="dataset", local_dir=".")
```

```bash
python openjev_ja_eval.py list --subset choice                                         # noul / choice / score / all
python openjev_ja_eval.py fetch --subset all --datasets-root ./datasets                # 全データセットを配布元から
python openjev_ja_eval.py fetch --subset choice --datasets-root ./datasets --only jmmlu  # ルーター対象だけ
```

jev-ja-lab 本体では同じ機能を `jev-ja-lab-datasets list|fetch` として使えます。JGPQA は `--include-gated` と自分のトークンで取得します。

## Licensing Information

収録データ(mirror)は **CC BY-SA 4.0** の集合物として配布します。MIT・Apache-2.0・CC BY・CC BY-SA 4.0 の
データセットのみを収録しており、CC BY-SA 4.0 はそれらを包含できる最も条件の厳しいライセンスです。

- **各データセットの元のライセンスと表示義務は維持されます。** 集合物のライセンスは、収録した各データセットの
  元のライセンス(データセット一覧の License 列)を置き換えるものではありません。ライセンス全文と帰属表示は
  [`LICENSES/`](LICENSES/)(`ATTRIBUTION.md`)にあります。
- **共通形式への変換を行っています。** 質問文・選択肢の組み立て、Choice の誤答選択肢の生成(GSM8K-JA・MGSM)、
  validation の `test` としての収録などです。内容は変更していません。詳細は `LICENSES/ATTRIBUTION.md`。
- **再配布する場合:** CC BY-SA 4.0 の条件(帰属表示、改変の明示、同じライセンスでの再配布)に加え、
  各データセットの表示義務(特に Apache-2.0 は NOTICE と改変の明示、MIT は著作権表示と許諾文の同梱)を守ってください。
  CC BY-SA 4.0 のデータを他のライセンスのデータと 1 つに混ぜて再配布すると、混ぜた全体に SA 条件が及びます。
- **商用利用:** 収録データは、各データセットの元のライセンスが許す範囲で商用利用できます(NC 条件のデータは収録していません)。
- **ルーター対象**(JMMLU・WRIME は CC BY-NC-ND 4.0、JGPQA は要承認、PAWS-X は独自条件、TextDetox は OpenRAIL++ の利用制限)は
  本リポジトリに含まれません。取得したデータの条件は各配布元のものが適用され、NC-ND のデータは商用利用も改変物の再配布もできません。
- 上記は各配布元のカード・リポジトリの記載に基づく整理であり、法的助言ではありません。再配布・商用利用の前に、
  各配布元の原文を必ず確認してください。
- 独自データ: `synthetic_score` は MIT です。`helpsteer2_ja` は kunishou/HelpSteer2-20k-ja(CC BY 4.0)の抽出物のため、
  原典(NVIDIA HelpSteer2)と翻訳者の帰属表示が必要です。

## Acknowledgements

評価データを公開されている各データセットの作成者・配布者の皆様に感謝します。
また、HelpSteer2 を公開した NVIDIA、日本語訳 HelpSteer2-20k-ja を公開した kunishou 氏に感謝します。

## Citation Information

利用時は、各データセットの配布元に記載された引用方法に従ってください。

- MMMLU (JA_JP): <https://huggingface.co/datasets/openai/MMMLU>
- JMMLU: <https://huggingface.co/datasets/nlp-waseda/JMMLU>
- JCommonsenseQA: <https://huggingface.co/datasets/sbintuitions/JCommonsenseQA>
- XWinograd (ja): <https://huggingface.co/datasets/Muennighoff/xwinograd>
- MGSM (ja): <https://huggingface.co/datasets/jbross-ibm-research/mgsm>
- GSM8K-JA (test 250-1319): <https://huggingface.co/datasets/SakanaAI/gsm8k-ja-test_250-1319>
- JNLI (JGLUE v1.1.0): <https://github.com/yahoojapan/JGLUE>
- WRIME ver2: <https://github.com/ids-cv/wrime>
- JaNLI: <https://huggingface.co/datasets/hpprc/janli>
- JCoLA: <https://github.com/osekilab/JCoLA>
- PAWS-X (ja): <https://huggingface.co/datasets/google-research-datasets/paws-x>
- TextDetox multilingual toxicity (ja): <https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset>
- JAD-AFC: <https://github.com/FujitsuResearch/japanese-dataset-for-automated-fact-checking>
- Synthetic Score (独自): <https://huggingface.co/datasets/hachiko85/openjev-ja-eval/tree/main/synthetic_score>
- HelpSteer2-JA benchmark-v1 (抽出サブセット): <https://huggingface.co/datasets/kunishou/HelpSteer2-20k-ja> (原典: <https://huggingface.co/datasets/nvidia/HelpSteer2>)
- JGPQA (diamond): <https://huggingface.co/datasets/llm-jp/jgpqa>
- 本リポジトリ: <https://huggingface.co/datasets/hachiko85/openjev-ja-eval> /
  <https://github.com/hachiko85/jev-ja-lab>
