---
name: init
description: jev-ja-labの開発環境を新規セットアップする、または壊れた環境を復旧するときに使う。依存関係インストール、GPU/CUDA検出、.env雛形作成、pytest動作確認を行う。
---

# init: jev-ja-lab環境セットアップ

新しいマシン/クリーンチェックアウトでこのリポジトリを動かせる状態にする。既存環境の
依存関係破損(torchがCPU版で入っている等)の復旧にも使う。

## 手順

1. **Python確認**: `python --version` で3.11以上か確認。無ければユーザーに報告して停止。

2. **依存関係インストール**:
   ```bash
   uv sync --extra eval --extra viz --extra orchestrate --extra dev
   ```
   `pip install -e ".[eval,viz,orchestrate,dev]"` でも同等(uv不使用環境)。

3. **GPU確認**: `nvidia-smi` を実行し、GPUの有無を確認する。
   - GPUがあるのに`uv sync`がCPU版torchを解決した場合(`python -c "import torch;
     print(torch.cuda.is_available())"` が`False`)は、`pyproject.toml`の
     `[tool.uv.sources]`/`[[tool.uv.index]]`(pytorch-cu128)設定を確認し、
     `uv pip install --reinstall torch` で入れ直す。
   - GPUが無い環境ではCPU版のままで良い(評価は遅くなるが動く)。

4. **`.env`確認**: `.env`が無ければ`.env.example`をコピーして作成し、埋めるべき項目を
   ユーザーに提示する(値自体はユーザーに入力してもらう、代入しない):
   - `TYPESAFE_API_KEY` / `TYPESAFE_API_URL` — Jev本家API評価(`scorer: jev`)を使う場合のみ必須
   - `HF_TOKEN` — gatedリポジトリ・レート制限緩和に使う場合
   - `OPENJEV_DATASETS_DIR` — データセットキャッシュ先を既定(`~/.cache/huggingface`)から
     変える場合のみ設定

5. **評価データの取得**(フル評価を回す場合。`--limit`付きの単発確認だけなら不要):
   ```bash
   uv run jev-ja-lab-datasets fetch --subset all --datasets-root ./datasets
   ```
   `hachiko85/openjev-ja-eval`のルーター経由で各配布元から直接取得する(データは再配布しない)。
   サブセットは`noul`/`choice`/`score`/`all`。privateの間は`HF_TOKEN`が必要。
   JGPQAは要承認のため対象外(`--include-gated`と自分のトークンでのみ試行)。

6. **動作確認**:
   ```bash
   uv run pytest
   uv run ruff check .
   ```
   既知の失敗2件(`test_model_reference_supports_local_and_hub`、
   `test_summary_chart_contains_primitive_features`、Windows/cp932ロケール依存)は
   無視してよい。それ以外の失敗はセットアップ不備の兆候として報告する。

7. **GPU不要のスモークテスト**(任意、余裕があれば):
   ```bash
   uv run jev-ja-lab-eval --dataset mmmlu_ja --scorer mock --limit 5
   ```
   `results/`配下にダミーの評価結果が出れば、パイプライン全体が動く状態。

## 完了報告のフォーマット

各ステップの成否(Python版、GPU有無、CUDA torch有無、.env状態、pytest結果)を短くまとめて
報告する。失敗があれば原因と対処案を添える。
