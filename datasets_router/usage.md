## How to download and use

本リポジトリが保持するのは manifest・クライアント・独自データ(`synthetic_score`、`helpsteer2_ja`)のみです。
第三者のデータセットは、`manifest.json` に固定したrevisionで各配布元から取得します。
リポジトリが private の間はトークン(環境変数 `HF_TOKEN` または `token=`)が必要です。

### Using the bundled client (recommended)

`openjev_ja_eval.py` がサブセットを解決し、[jev-ja-lab](https://github.com/hachiko85/jev-ja-lab) が読む `datasets/` 配置で保存します。

```python
from huggingface_hub import hf_hub_download

client = hf_hub_download("hachiko85/openjev-ja-eval", "openjev_ja_eval.py", repo_type="dataset", local_dir=".")
```

```bash
python openjev_ja_eval.py list --subset noul                         # subsets: noul / choice / score / all
python openjev_ja_eval.py fetch --subset all --datasets-root ./datasets
python openjev_ja_eval.py fetch --subset score --datasets-root ./datasets --only helpsteer2_ja_benchmark_v1
```

jev-ja-lab 本体では同じ機能を `jev-ja-lab-datasets list|fetch` として使えます。

### Using `huggingface_hub`

```python
import json
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "hachiko85/openjev-ja-eval"
manifest = json.load(open(hf_hub_download(REPO, "manifest.json", repo_type="dataset"), encoding="utf-8"))
entries = [e for e in manifest["datasets"] if e["id"] in manifest["subsets"]["score"]]

for e in entries:
    s = e["source"]
    if s["kind"] in ("hf_dataset", "hf_snapshot", "hf_parquet"):
        # a Hub dataset: download it at the pinned revision
        snapshot_download(s["repo_id"], repo_type="dataset", revision=s["revision"], local_dir=f"data/{e['id']}")
    elif s["kind"] == "own":
        # the project's own data is stored in this repository
        snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"{s['path']}/*"], local_dir="data")
    else:
        # a GitHub repository: clone s["url"] and check out s["revision"]
        print(e["id"], s["url"], s["revision"])
```

コマンドラインでも取得できます:

```bash
hf download hachiko85/openjev-ja-eval --repo-type dataset --include "helpsteer2_ja/*" --local-dir ./data
```

### Using `datasets`

各サブセットは `test` split を持つ config で、そのサブセットに含まれるデータセットのカタログ(配布元リポジトリ・config・split・revision・ライセンス)を返します。

```python
from datasets import load_dataset

REPO = "hachiko85/openjev-ja-eval"
catalog = load_dataset(REPO, "choice", split="test")
row = next(r for r in catalog if r["id"] == "mgsm_ja")

# the source dataset, at the revision recorded in the catalog
mgsm = load_dataset(row["source_repo"], row["source_config"], split=row["source_split"], revision=row["revision"])

# the project's own data
helpsteer2 = load_dataset(REPO, data_files={"test": "helpsteer2_ja/benchmark-v1.parquet"}, split="test")
```

`hf_parquet` と `git` の配布元(JaNLI、PAWS-X、TextDetox、JGLUE、WRIME、JCoLA、JAD-AFC)は `load_dataset` で直接読めないため、同梱クライアントを使ってください。
