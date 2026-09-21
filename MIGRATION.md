# 別PCへの移行手順

ソースコード、Git履歴、設定、今回の評価結果をZIPで別PCへ移行し、uvまたはPython仮想環境で再実行する手順です。モデル本体とデータセット本体はZIPへ含めません。

## 1. ZIPの内容

含むもの:

- ソースコード
- `.git` とGit履歴
- YAML設定とテスト
- `results/` 配下の評価結果とチャート
- `uv.lock`
- ドキュメント

除外するもの:

- `.venv/`
- Python、pytest、Ruff cache
- `dist/`
- ローカルモデル重み
- ローカルデータセットcache
- `.env`
- `*.safetensors`、`*.bin`

## 2. ZIP作成

プロジェクトルートで実行します。

```bash
python scripts/build_portable_zip.py
```

出力:

```text
dist/jev-ja-lab-portable-20260918.zip
dist/jev-ja-lab-portable-20260918.zip.sha256
```

別名を使う場合:

```bash
python scripts/build_portable_zip.py --output dist/jev-ja-lab-portable.zip
```

## 3. 転送後の検証と展開

Linux/macOS:

```bash
sha256sum -c jev-ja-lab-portable-20260918.zip.sha256
unzip jev-ja-lab-portable-20260918.zip
cd jev-ja-lab
```

Windows PowerShell:

```powershell
$expected = (Get-Content .\jev-ja-lab-portable-20260918.zip.sha256).Split()[0]
$actual = (Get-FileHash .\jev-ja-lab-portable-20260918.zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw "SHA-256 mismatch" }
Expand-Archive .\jev-ja-lab-portable-20260918.zip -DestinationPath .
Set-Location .\jev-ja-lab
```

Git履歴確認:

```bash
git status
git log -1 --oneline
```

ZIP作成時点の未コミット変更もファイルとして含まれます。移行前に履歴を確定したい場合は、ZIP作成前にcommitしてください。

## 4. uvで環境導入

```bash
uv sync --extra eval --extra viz --extra orchestrate --extra dev
uv run pytest
uv run ruff check .
```

CUDA対応PyTorchを個別導入する必要がある環境では、先に対象GPUとドライバに合うPyTorchを導入してください。

## 5. Python仮想環境で環境導入

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[eval,viz,orchestrate,dev]"
pytest
ruff check .
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[eval,viz,orchestrate,dev]"
pytest
ruff check .
```

## 6. モデルとデータセットの用意

ZIPは重みと元datasetを含みません。Hub取得またはローカル配置を選びます。

### Hubから取得

```yaml
models:
  - id: gemma-hub
    repo_id: google/gemma-4-E4B-it
    revision: <commit-hash>
    dtype: bfloat16

datasets:
  - id: hub-example
    task: noul
    expected_items: 1000
    revision: <commit-hash>
    source:
      repo_id: organization/dataset-name
      config: ja
      split: test
      adapter: noul
      gold_field: label
      positive_values: [1]
      question_template: "判定: {text}"
```

gatedリポジトリ:

```bash
export HF_TOKEN=hf_xxx
```

### ローカル配置

```yaml
runtime:
  models_root: /path/to/models
  datasets_root: /path/to/datasets

models:
  - id: qwen-local
    path: Qwen3-8B

datasets:
  - id: local-example
    task: noul
    source:
      format: parquet
      path: example/test.parquet
      adapter: noul
      gold_field: label
      positive_values: [1]
      question_template: "判定: {text}"
```

WindowsパスをYAMLへ直接書く場合は、`C:/models` のようにスラッシュを使うとエスケープ問題を避けられます。

## 7. 移行先でsmoke実行

```bash
uv run jev-ja-lab-eval-workflow \
  --config configs/eval/primitives.20260918.yaml \
  --phase smoke
```

確認項目:

- モデルをロードできる
- dataset schemaがadapterと一致する
- GPUメモリ不足がない
- `A`〜候補数分のラベルが単一tokenになる
- `results/<smoke_run_name>/` にsummaryが生成される

## 8. 本番実行

```bash
uv run jev-ja-lab-eval-workflow \
  --config configs/eval/primitives.20260918.yaml \
  --phase production
```

移行先で新規評価する場合は、既存結果との混在を防ぐため `runtime.run_name` を変更してください。

```yaml
runtime:
  run_name: eval-new-pc
  smoke_run_name: eval-new-pc-smoke
  resume: true
```

今回の結果を参照するだけなら、ZIP内の `results/eval-primitives-20260918/` をそのまま利用できます。

## 9. オフライン移行

完全オフラインPCでは、モデルとdatasetを別媒体で転送し、ローカル方式を使います。

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Hub ID方式を初回実行する場合、これらを設定しないでください。

## 10. 再現条件

次を移行元と一致させてください。

- モデルrevision
- dataset revision
- Python、PyTorch、Transformers版
- dtype
- batch size
- seed
- prompt template
- GPU種別と実行device

厳密比較では、`uv.lock`、YAML、`metadata.json`、`run_config.*.json` を保存してください。
