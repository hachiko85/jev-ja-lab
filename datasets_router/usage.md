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
