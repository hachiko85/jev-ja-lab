# jev-ja-lab

日本語データセットをJevの3 Primitiveへ変換し、ローカルまたはHugging Face Hub上の判断モデルを同一条件で比較する評価基盤です。

- `Noul`: Yes / No の二値判断
- `Choice`: 複数候補から1つを選択
- `Score`: 順序付き尺度のスコアリング

現フェーズは評価基盤です。モデル学習機能は含みません。

## ドキュメント

- 評価設定・YAML・実行方法: `how_to_use.md`
- 別PCへのZIP移行: `MIGRATION.md`
- Primitive設計: `EVALUATION_PRIMITIVES.md`
- 今回の実行条件と結果: `EVAL_JEV_20260918.md`
- 実行設定: `configs/eval/primitives.20260918.yaml`

## 必要環境

- Python 3.11以上
- NVIDIA GPUを使う場合: 対応ドライバとCUDA対応PyTorch
- Hubモデル・データセットを取得する場合: インターネット接続
- gatedリポジトリを使う場合: 配布元での利用規約同意と `HF_TOKEN`

## 最短セットアップ

### uv

```bash
uv sync --extra eval --extra viz --extra orchestrate --extra dev
uv run pytest
```

### Python仮想環境

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[eval,viz,orchestrate,dev]"
pytest
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[eval,viz,orchestrate,dev]"
pytest
```

## 最短実行

GPU不要の動作確認:

```bash
uv run jev-ja-lab-eval --dataset mmmlu_ja --scorer mock --limit 5
```

YAML全体のsmoke評価:

```bash
uv run jev-ja-lab-eval-workflow \
  --config configs/eval/primitives.20260918.yaml \
  --phase smoke
```

本番評価:

```bash
uv run jev-ja-lab-eval-workflow \
  --config configs/eval/primitives.20260918.yaml \
  --phase production
```

`pip install -e ...` を使った環境では、先頭の `uv run` を外して実行できます。

## 標準評価プロファイル

標準設定は、全モデルを短時間で比較する30軸プロファイルです。

| Primitive | 軸数 | 1モデル当たり評価数 |
|---|---:|---:|
| Noul | 14 | 25,908 |
| Choice | 9 | 28,676 |
| Score | 7 | 8,300 |
| 合計 | 30 | 62,884 |

LLM-jp Toxicity属性別、HelpSteer2全件、Civil Comments全件、JSFactCheckBenchの定義とadapterも実装済みです。標準実行では処理時間または認証要件を理由に無効化しています。`tasks.<primitive>.datasets` へIDを戻すと追加評価できます。

## 今回の結果

12モデル、30軸、全Primitive coverage `3/3`、欠損0で完了しています。

| Overall上位 | Score |
|---|---:|
| Qwen3.5-4B | 65.57% |
| Qwen3-8B | 64.79% |
| Qwen3 4B Instruct 2507 | 64.76% |
| Qwen3-Swallow 8B CPT v0.2 | 63.69% |

主な成果物:

```text
results/eval-primitives-20260918/
├─ eval_noul.json
├─ eval_choice.json
├─ eval_score.json
├─ eval_summary.json
├─ radar-noul.png
├─ radar-choice.png
├─ radar-score.png
├─ radar-summary.png
└─ <model>/<dataset>/{metadata.json,predictions.jsonl,summary.json}
```

## 再現性

- `seed`、モデルrevision、データセットrevision、dtype、batch sizeを固定してください。
- `resume: true` は既存の `summary.json` を再利用します。条件を変更した評価は別の `run_name` を使ってください。
- scorerはテキスト生成を行わず、候補ラベルの次token logitを1回のforward passで比較します。
- APIキーやHub tokenはYAMLへ書かず、環境変数で渡してください。

## 検証

```bash
uv run pytest
uv run ruff check .
```

現在の確認結果: `41 passed`、Ruff成功。
