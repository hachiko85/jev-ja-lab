---
license: cc-by-nc-nd-4.0
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
---

# openjev-ja-eval

## アブストラクト

[jev-ja-lab](https://github.com/hachiko85/jev-ja-lab) で日本語の判断モデルを評価するための
データセット集です。判断タスクを Noul(二値判定)・Choice(選択式)・Score(段階評価)の
3 種類に分け、既存の公開データセット 15 件をまとめています。

第三者のデータ本体は本リポジトリに含みません。`manifest.json` に配布元・revision・split・
ライセンスを記録し、取得時に各配布元(Hugging Face Hub / GitHub)から直接ダウンロードします。
本リポジトリ内で管理するのは、独自に作成した `synthetic_score` と、評価用に抽出した
HelpSteer2-JA(`helpsteer2_ja`、benchmark-v1)のみです。

## サブセット

| サブセット | 内容 | split | 含まれるデータセット |
|---|---|---|---|
| `noul` | Noul(二値判定) | `test` | `janli`, `jcola`, `paws_x_ja`, `textdetox_ja`, `jad_afc`, `jnli`, `wrime` |
| `choice` | Choice(選択式) | `test` | `mmmlu_ja`, `jmmlu`, `jcommonsenseqa`, `xwinograd_ja`, `mgsm_ja`, `gsm8k_ja`, `jnli` |
| `score` | Score(段階評価) | `test` | `wrime`, `synthetic_score`, `helpsteer2_ja_benchmark_v1` |
| `all` | All(上3つすべて) | `test` | `mmmlu_ja`, `jmmlu`, `jcommonsenseqa`, `xwinograd_ja`, `mgsm_ja`, `gsm8k_ja`, `jnli`, `janli`, `jcola`, `paws_x_ja`, `textdetox_ja`, `jad_afc`, `wrime`, `synthetic_score`, `helpsteer2_ja_benchmark_v1` |

評価の split は `test` です。配布元に test が無い、または test のラベルが非公開のものは、
下表「使用split」のとおり validation / valid、または split 区分のない全件を使います。
`load_dataset` で各サブセットを開くと、そのサブセットに含まれるデータセットの配布元一覧が
`test` split として返ります(データ本体は次節の手順で取得します)。

## データセット一覧

| 名称 | 詳細 | タスク | 配布元(リンク) | 使用split | ライセンス | 件数 |
|---|---|---|---|---|---|---:|
| `mmmlu_ja` MMMLU (JA_JP) | OpenAIが公開するMMLUの多言語版のうち日本語(JA_JP)。57科目の4択知識問題。 | choice | [huggingface.co/datasets/openai/MMMLU](https://huggingface.co/datasets/openai/MMMLU) | test | MIT | 14,042 |
| `jmmlu` JMMLU | MMLUの日本語翻訳・日本固有問題を含む4択知識問題(全科目)。 | choice | [huggingface.co/datasets/nlp-waseda/JMMLU](https://huggingface.co/datasets/nlp-waseda/JMMLU) | test | CC BY-NC-ND 4.0 | 7,536 |
| `jcommonsenseqa` JCommonsenseQA | JGLUEの日本語常識推論5択。JGLUEのtestはラベル非公開のためvalidationを評価に使用。 | choice | [huggingface.co/datasets/sbintuitions/JCommonsenseQA](https://huggingface.co/datasets/sbintuitions/JCommonsenseQA) | validation | CC BY-SA 4.0 | 1,119 |
| `xwinograd_ja` XWinograd (ja) | 多言語Winograd Schema Challengeの日本語(jp)。文脈・照応解析の2択。 | choice | [huggingface.co/datasets/Muennighoff/xwinograd](https://huggingface.co/datasets/Muennighoff/xwinograd) | test | CC BY 4.0 | 959 |
| `mgsm_ja` MGSM (ja) | GSM8Kを多言語化した数学文章題の日本語版。評価では選択式に変換して使用。 | choice | [huggingface.co/datasets/jbross-ibm-research/mgsm](https://huggingface.co/datasets/jbross-ibm-research/mgsm) | test | CC BY-SA 4.0 | 250 |
| `gsm8k_ja` GSM8K-JA (test 250-1319) | SakanaAIによるGSM8Kの日本語テスト分割(250〜1319件目)。評価では4択(mc4)と10択(mc10)に変換して使用。 | choice | [huggingface.co/datasets/SakanaAI/gsm8k-ja-test_250-1319](https://huggingface.co/datasets/SakanaAI/gsm8k-ja-test_250-1319) | test | Apache-2.0 | 1,069 |
| `jnli` JNLI (JGLUE v1.1.0) | JGLUEの日本語自然言語推論(含意・矛盾・中立の3値)。Choiceとして3択、Noulとして各ラベルの二値判定に使用。testはラベル非公開のためvalidを評価に使用。 | noul・choice | [github.com/yahoojapan/JGLUE](https://github.com/yahoojapan/JGLUE) | valid | CC BY-SA 4.0 | 2,434 |
| `wrime` WRIME ver2 | 文の書き手・読み手の感情強度アノテーション付きSNSコーパス。Scoreとして読み手の喜び・怒り・感情極性の強度、Noulとして二値化した判定に使用。再配布に制限があるため本リポジトリには保存しない。 | noul・score | [github.com/ids-cv/wrime](https://github.com/ids-cv/wrime) | test | CC BY-NC-ND 4.0 | 2,500 |
| `janli` JaNLI | 語順・言い換えなど日本語特有の現象に絞った敵対的NLI。含意判定(Noul)に使用。 | noul | [huggingface.co/datasets/hpprc/janli](https://huggingface.co/datasets/hpprc/janli) | test | CC BY-SA 4.0 | 720 |
| `jcola` JCoLA | 日本語の文法容認性判断。既知構文(in-domain)と未知構文(out-of-domain)のvalidを使用。 | noul | [github.com/osekilab/JCoLA](https://github.com/osekilab/JCoLA) | valid | CC BY-SA 4.0 | 1,550 |
| `paws_x_ja` PAWS-X (ja) | 語順を入れ替えた敵対的な言い換え判定の多言語版、日本語(ja)。2文が同じ意味かの二値判定(Noul)。 | noul | [huggingface.co/datasets/google-research-datasets/paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x) | test | other (PAWS-X の利用条件に従う) | 2,000 |
| `textdetox_ja` TextDetox multilingual toxicity (ja) | 多言語有害表現データセットの日本語部分。有害表現検出の二値判定(Noul)。日本語は言語名のsplit(ja)のみでtrain/testの区別はない。 | noul | [huggingface.co/datasets/textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset) | ja | OpenRAIL++ | 5,000 |
| `jad_afc` JAD-AFC | 富士通研究所の日本語ファクトチェック用データセット。真・偽・情報不足(NEI)の3種をそれぞれNoulの二値判定に使用。split区分なし(CSV1枚)。 | noul | [github.com/FujitsuResearch/japanese-dataset-for-automated-fact-checking](https://github.com/FujitsuResearch/japanese-dataset-for-automated-fact-checking) | (split区分なし) | Apache-2.0 | 612 |
| `synthetic_score` Synthetic Score (独自) | 緊急度・不満度・リスク・関連度の4軸を、決定的テンプレート(seed=42)で生成した5段階Score。1軸200件、計800件。 | score | [hachiko85/openjev-ja-eval/synthetic_score](https://huggingface.co/datasets/hachiko85/openjev-ja-eval/tree/main/synthetic_score)(本リポジトリ内) | test | MIT | 800 |
| `helpsteer2_ja_benchmark_v1` HelpSteer2-JA benchmark-v1 (抽出サブセット) | kunishou/HelpSteer2-20k-ja(19,958行)からprompt単位で重複なしに2,500件を抽出(seed=42、sha256順)。正確性・有用性・詳細度・複雑さ・一貫性の5軸(各0〜4)のScore。抽出手順と件数は同梱のbenchmark-v1.manifest.json参照。 | score | [hachiko85/openjev-ja-eval/helpsteer2_ja](https://huggingface.co/datasets/hachiko85/openjev-ja-eval/tree/main/helpsteer2_ja)(本リポジトリ内)。原典: [kunishou/HelpSteer2-20k-ja](https://huggingface.co/datasets/kunishou/HelpSteer2-20k-ja) | test | CC BY 4.0 | 2,500 |

再配布禁止・別途承認が必要なため対象外としたデータセット:

- `jgpqa_diamond` JGPQA (diamond) ([https://huggingface.co/datasets/llm-jp/jgpqa](https://huggingface.co/datasets/llm-jp/jgpqa), CC BY 4.0): 配布元で利用規約への同意とHugging Faceトークンが必要なため、ルーターのサブセットには含めない。--include-gated指定時のみ、利用者自身のトークンで取得を試みる。

## 使い方

```bash
pip install huggingface_hub datasets pyarrow

# 一覧(サブセット: noul / choice / score / all)
python openjev_ja_eval.py list --subset noul

# 取得: 各配布元から直接ダウンロードし、jev-ja-lab の datasets/ 配置で保存
python openjev_ja_eval.py fetch --subset all --datasets-root ./datasets
```

`openjev_ja_eval.py` は本リポジトリに含まれます。jev-ja-lab 本体では
`jev-ja-lab-datasets list|fetch` として利用できます。リポジトリが private の間は `HF_TOKEN` が
必要です。JGPQA など要承認のデータセットは `--include-gated` と自分のトークンで取得します。

## ライセンス

本リポジトリは **CC BY-NC-ND 4.0** で配布します。収録データセットが継承するライセンスの
うち最も厳しいものを採用しており、該当するのは **JMMLU と WRIME**(いずれも CC BY-NC-ND 4.0)です。

再配布・商用利用に関する注意:

- **本リポジトリにサードパーティのデータ本体は含まれません。** 取得したデータの利用条件は、
  データセット一覧に記載した各配布元のライセンスが適用されます。
- **JMMLU・WRIME は非商用・改変禁止(NC-ND)です。** これらを含むサブセット(`choice` の JMMLU、
  `noul` / `score` / `all` の WRIME)を取得したデータは、商用利用できません。取得したデータを
  改変したものを再配布することもできません。商用利用する場合は、該当データセットを除外
  (`--only` で必要なものだけ指定)し、残りの各ライセンスを個別に確認してください。
- **継承(SA)条件付き**: JCommonsenseQA・MGSM・JNLI(JGLUE)・JCoLA・JaNLI は CC BY-SA 4.0 です。
  改変物を再配布する場合は同じライセンスにする必要があります。
- **TextDetox** は OpenRAIL++ で、利用目的に関する制限が付きます。**PAWS-X** は配布元カードで
  `other` とされており、配布元の独自条件に従ってください。
- **MIT / Apache-2.0 / CC BY 4.0**(MMMLU・GSM8K-JA・JAD-AFC・XWinograd 等)は、著作権表示・
  ライセンス表示・帰属表示が必要です。
- 独自データ: `synthetic_score` は MIT です。`helpsteer2_ja` は kunishou/HelpSteer2-20k-ja
  (CC BY 4.0)の抽出物のため、原典(NVIDIA HelpSteer2)と翻訳者の帰属表示が必要です。
- 上記は各配布元のカード・リポジトリの記載に基づく整理であり、法的助言ではありません。
  再配布・商用利用の前に、各配布元の原文を必ず確認してください。

## 謝辞

評価データを公開されている各データセットの作成者・配布者の皆様に感謝します。
また、HelpSteer2 を公開した NVIDIA、日本語訳 HelpSteer2-20k-ja を公開した kunishou 氏に
感謝します。

## 引用先

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
