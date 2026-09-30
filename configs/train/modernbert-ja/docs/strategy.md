# modernbert-ja 学習戦略

対象: [sbintuitions/modernbert-ja-310m](https://huggingface.co/sbintuitions/modernbert-ja-310m)。
本ドキュメントは、既存の評価結果(零撃直接評価・laya-bertヘッド学習の2種)から見えた課題を整理し、
今後この encoder を Jev 系 Noul/Choice/Score 判定モデルとして育てる場合の学習方針・必要データ・
対象タスクの絞り込み方をまとめる。実装(`openjev_ja.train`、`configs/train/*.yaml`)とは独立した、
方針ドキュメント。

## 1. 現状の問題(ベンチマークから見えたもの)

### 1.1 零撃直接評価(`masked-lm` scorer、専用headなし)

`results/eval-bert-series/modernbert-ja-310m` の実測(標準35データセット、primitive別内訳):

| primitive | 得意 | 苦手 |
|---|---|---|
| Noul(F1) | JCoLA in/out-of-domain(0.85〜0.91)、JaNLI(0.67)、PAWS-X(0.61) | JNLI-missing-evidence(**0.000**)、WRIME-anger-binary(**0.000**) |
| Choice(accuracy) | XWinograd(0.71)、JNLI 3値(0.49)、JCommonsenseQA(0.44) | JGPQA(0.27、chance 0.25)、GSM8K-mc10(0.19、chance 0.10)、MMMLU/JMMLU(0.30台) |
| Score(正規化QWK) | WRIME-sentiment(0.60) | 残り6データセットほぼ0.50(chance水準) |

観測される傾向:

- **1文単位の文法性・言い換え・単純な意味関係(JCoLA/XWinograd/JNLI/PAWS-X)は零撃でも一定の実力がある。**
  これは modernbert-ja 自身の事前学習(MLM、4.09兆トークン)がこの種の局所的な言語現象を
  よく捉えていることを示す。
- **複数ステップの計算・院レベル知識・多肢からの絞り込み(GSM8K/JGPQA/MMLU系)はほぼ chance 水準。**
  後述のとおり、これは学習データ量の問題ではなく構造的な限界(1.2節)と考えられる。
- **順序尺度の段階評価(Score)はほぼ全滅。** 5段階評価のような「相対的な強度判定」は
  零撃のMLM readoutでは表現できていない。
- **Noulの一部で完全崩壊(F1=0)。** 特定データセットで「いいえ」または「はい」のどちらか
  一方に倒れている(該当データセットの多数派クラス側)。

### 1.2 なぜ多段推論・専門知識タスクが弱いか(構造的限界)

- modernbert-ja は **1回の forward pass で完結**するエンコーダで、生成(トークンを逐次出力)機構を
  持たない。GSM8K のような多段計算、JGPQA のような院レベル知識の想起は、本質的に「その場で
  考える」手順を必要とする種類のタスクであり、1パスのエンコーダには原理的に不向き。
- パラメータ数(315M、埋め込み抜き236M)も、思考の連鎖(CoT)や算術訓練を一切受けていない
  MLM-onlyの事前学習も、この種のタスクへの適性を裏付けない。
- **学習データを増やしても改善しない可能性が高い**ことは、`laya-bert` ヘッドを modernbert-ja に
  載せて学習した2種の実験で裏付けられている:

  | 学習データ | JGPQA (Choice) | GSM8K-mc10 (Choice) |
  |---|---:|---:|
  | 規定データ(jcommonsenseqa等、2,000件/5epoch) | 0.247 | 0.106 |
  | eikos日本語コーパス(9,000件超/3epoch) | 0.263 | 0.098 |

  データ量を4倍以上に増やしても誤差の範囲でしか動かない。frozen encoder + head という
  構成である限り、head をどう変えても encoder の表現力の天井は超えられない。

### 1.3 「はい」寄りの偏り(prior override 問題)

`laya-bert` on modernbert-ja(規定データ、frozen encoder)の Noul 内訳を見ると、
**recallがほぼ全データセットで0.83〜1.0**(常に「はい」寄りに倒れている)。
これは以下の理由による:

- modernbert-ja の事前学習目的関数は「周辺文脈から尤もらしい単語を当てる」ことのみ。
  **「プロンプト内に明示された事実・数値を、自分の統計的な事前知識より優先する」という
  訓練信号を一度も受けていない**(instruction tuning・RLHF に相当する工程が存在しない)。
- 結果として、学習に使った単一ソース(`jnli_entailment_train`)の多数派クラスに引っ張られ、
  他のタスクでもその偏りをそのまま引きずる。
- 例: 「東京は明日雨80%/晴れ20%」と明示されていても、学習データ・事前学習コーパスに
  「東京は晴れ」という共起が大量にあれば、統計的な事前分布に引っ張られて「晴れ」と
  答えやすい。これは frozen encoder + head という構成そのものの限界であり、
  head をいくら学習してもこの挙動を直接修正することはできない
  (head は encoder の出力を読み取るだけで、encoder 自体の判断基準は変わらない)。

## 2. 学習方針(本ドキュメントまでの議論を反映)

### 2.1 backbone を凍結しない

これまでの実装(`laya_bert/train.py`)は encoder を完全凍結し、head のみを学習していた。
これは「文脈内の明示情報を優先する」挙動を一切学習できない構成であり、1.3節の問題を
解決できない。今後は **backbone(modernbert-ja本体)と head を同時に学習する**
(full fine-tuning、または計算資源が厳しい場合は LoRA)。

- backbone を学習することで初めて、「事前学習の統計的な偏りより、プロンプト内の明示情報を
  優先する」という instruction-following 的な挙動を教え込める。
- head は backbone の**隠れ状態ベクトル**(MLM語彙分布ではない、softmax前の内部表現)を
  受け取り、Noul/Choice/Score それぞれのタスク形式に変換する役割。MLM出力層とheadは
  同じ隠れ状態から分岐する並列のルートであり、直列ではない。
- MLM出力層自体は新しいパイプラインでは通常使わない。ただし、学習データが少ない場合の
  破滅的忘却(言語理解力の劣化)を防ぐため、**MLM損失を補助タスクとして残す
  マルチタスク学習**も検討する価値がある。

### 2.2 学習方式: 教師あり cross-entropy を基本、RL(RLCD)は将来課題

- **教師あり cross-entropy**(正解ラベルに対する単純な誤差最小化)を第一候補とする。
  実装が単純で、decider/Hopper/Jeff など本プロジェクトで評価済みの手法もこの方式。
- laya 本家の学習方式は RLCD(Reinforcement Learning from Contrastive Decisions、
  正解/不正解の対比から学習する強化学習)であり、キャリブレーション品質(確率の較正)に
  優れるとされるが実装コストが高い。今回の再実装(`laya_bert`)は簡略化した教師ありCEに
  留めており、laya本家の強みを再現できていない。**RLCDへの移行は、教師ありCEでの
  改善を確認した後の将来課題とする。**
- 学習後は**温度スケーリング等のキャリブレーション**を必ず行う。decider/lev/hopper/jeff は
  いずれも公開チェックポイントに温度・キャリブレーションプロファイルを同梱しており、
  現在の `laya_bert/train.py` にこの工程が無いことが弱点の一つ。

### 2.3 対象タスクの絞り込み(スコープ限定)

1.2節の構造的限界を踏まえ、**modernbert-ja に「解けるはずのない」タスクを解かせようとしない**。

- **対象に含める**: 1文・1対の単純な意味・文法判定(容認性、含意・矛盾、言い換え、
  毒性・感情の二値判定)、短い記述に対する段階評価。零撃の時点で既に一定の実力がある
  領域(JCoLA/XWinograd/JNLI/PAWS-X 相当)を土台に伸ばす。
- **対象から外す(LLM系手法に任せる)**: 複数ステップの計算(GSM8K系)、院レベル専門知識
  (JGPQA)、広範な一般知識想起(MMLU/JMMLU)。これらは decider・Jev・semif 等の
  生成系・大規模モデルの守備範囲とし、modernbert-ja の評価・学習双方で
  「解けなくて当然」の前提を明示する。

## 3. 必要なデータセット

### 3.1 量の目安

単一ソース・少量(2,000件/5epoch)でも、複数ソース・中量(eikos、9,000件超/3epoch、
ただしドメイン不一致)でも、いずれも Noul で偏り崩壊が起きている。
量よりも **多様性とクラスバランス** が先に効くと考えられる。

- **primitive あたり 5,000〜20,000件を目安**とし、**5系統以上の異なるデータセット**から
  バランスよく混ぜる(単一ソースに頼らない)。
- 各クラス(Noulなら はい/いいえ、Scoreなら各段階)は **均等に近いバランス**に調整する
  (positive/negative の偏りをそのまま学習させない)。
- eikos日本語コーパス(9,000件超、単一ドメイン)がむしろ悪化させた事実から、
  **量より先にドメイン一致・多様性を優先**する。

### 3.2 ドメイン方針

- 本プロジェクトの標準評価プロファイル(35データセット)が対象とする**日本語の一般的な
  言語現象**(NLI、文法容認性、言い換え、毒性、感情)に近いドメインを中心に据える。
  eikos(金融・業務判断、英語からの機械翻訳)のような専門ドメインへの偏りは、
  今回明確に悪影響が確認されたため避ける。
- 学習に使う train split と、評価に使う validation/test split は明確に分離する
  (本プロジェクトの既存の規約どおり)。

### 3.3 具体的な候補ソース(本プロジェクト内)

primitiveごとに、以下のように複数ソースを組み合わせることを想定する
(train split が既に利用可能なもの、または新たに用意が必要なものを明示):

| primitive | 候補ソース(train split) | 備考 |
|---|---|---|
| Noul | JNLI(entailment/contradiction/neutralの3種)、JaNLI、PAWS-X、JCoLA(in/out-of-domain)、TextDetox、WRIME(joy/anger/positiveの二値化) | 現状は `jnli_entailment_train` 単体に依存。最低でもこの表の3〜5系統を混合する |
| Choice | JCommonsenseQA、JNLI(3値)、XWinograd | 複数選択肢の単純な意味・常識判定に限定し、GSM8K/JGPQA/MMLU系は学習対象に含めない |
| Score | WRIME(joy/anger/sentiment)、Synthetic Score、HelpSteer2-JA | 段階数・基準が異なる複数ソースを混ぜ、特定の基準への過適合を避ける |

- 実データが不足する場合、本プロジェクト既存の `synthetic_score` のような
  決定的テンプレート生成データで補うことも検討できるが、eikosの教訓(ドメイン不一致)を
  踏まえ、**生成データだけに頼らず実データとの比率を管理する**。
- 学習データの分量・出典・split は、評価用データセットと同様に
  `datasets_router`(`hachiko85/openjev-ja-eval`)のようなrevision固定・出典明記の
  管理下に置くことが望ましい。

## 4. タスク形式・primitiveの方針

- **リクエスト形式は本プロジェクトの既存 wire format(state + instructions + options)に
  統一する。** これにより、学習後のheadをそのまま `orchestrate` 経由の既存評価パイプラインに
  接続でき、標準35データセットでの評価と直接比較できる。
- **Noulの基準文(rubric)埋め込みは要検証。** Hopper・eikos変換で採用している
  「はい/いいえの意味を明示的にテキストへ埋め込む」手法は、prior override対策として
  有効な可能性があるが、eikosの実験ではドメイン不一致と絡んで悪化した。
  埋め込みの有無を切り分けたアブレーション実験を推奨する。
- **Score levelsは「値(value)」を明示的に持たせる。** 期待値ベースの採点
  (本プロジェクトの正規化QWK評価と、Bekko System Oneの`value`ベース設計)と整合させる。
- **評価は必ず標準ベンチマーク(35データセット)で行い、学習データ内の精度(train/val)だけで
  判断しない。** eikosの教訓どおり、学習データ内で改善して見えても、外部ベンチマークで
  悪化するケースがある。

## 5. 次の一手(優先順位)

1. Noul学習データを複数ソース・クラスバランス調整済みに差し替える(単一ソース依存の解消)。
2. backboneを凍結せず、head と同時にfull fine-tuning(または LoRA)で教師ありCE学習する
   仕組みを `openjev_ja.train` に追加する(現状の `laya-bert` architecture は
   frozen encoder 前提のため、trainable backbone 用の新しい architecture 登録が必要)。
3. 学習後にキャリブレーション(温度スケーリング)を追加する。
4. 標準ベンチマーク(35データセット)で before/after を比較し、汎化を確認する。
5. 改善が確認できた場合のみ、RLCD相当の学習方式への移行を検討する。
