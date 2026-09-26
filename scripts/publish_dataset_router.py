"""Render and publish the `hachiko85/openjev-ja-eval` router repository.

From `datasets_router/manifest.json` this builds
  - README.md            dataset card (subsets, per-dataset table, licences)
  - noul|choice|score|all/test.parquet   catalog of where each dataset of a subset comes from
    (what the Hub viewer / `load_dataset(repo, "noul", split="test")` returns)
  - openjev_ja_eval.py   the standalone copy of `openjev_ja.datasets_router`
  - manifest.json
and uploads them together with the project's own data (synthetic_score, HelpSteer2-JA
benchmark-v1) in one commit. Anything else in the repo — the earlier re-hosted copies of
third-party datasets and the old fetch scripts — is removed, because the repo is a router.

    python scripts/publish_dataset_router.py --dry-run     # show what would change
    python scripts/publish_dataset_router.py               # commit to the Hub
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from openjev_ja import datasets_router as router  # noqa: E402

MANIFEST = ROOT / "datasets_router" / "manifest.json"
README = ROOT / "datasets_router" / "README.md"
OWN_FILES = {
    "synthetic_score": ROOT / "datasets/openjev-extra/synthetic_score",
    "helpsteer2_ja": ROOT / "datasets/helpsteer2_ja",
}
# Only the extracted benchmark subset is the project's own data; the full 19,958-row
# translation (data.parquet) belongs to kunishou/HelpSteer2-20k-ja and is not re-hosted.
OWN_INCLUDE = {
    "synthetic_score": None,
    "helpsteer2_ja": {"benchmark-v1.parquet", "benchmark-v1.manifest.json", "source.json"},
}
SUBSET_LABELS = {
    "noul": "Noul(二値判定)",
    "choice": "Choice(選択式)",
    "score": "Score(段階評価)",
    "all": "All(上3つすべて)",
}


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def dataset_table(manifest: dict) -> str:
    lines = [
        "| 名称 | 詳細 | タスク | 配布元(リンク) | 使用split | ライセンス | 件数 |",
        "|---|---|---|---|---|---|---:|",
    ]
    for entry in manifest["datasets"]:
        source = entry["source"]
        split = source.get("split") or "(split区分なし)"
        if source["kind"] == "own":
            tree = f"https://huggingface.co/datasets/{manifest['repo_id']}/tree/main/{source['path']}"
            origin = f"[{manifest['repo_id']}/{source['path']}]({tree})(本リポジトリ内)"
            if "derived_from" in source:
                origin += f"。原典: [{source['derived_from']['repo_id']}]({entry['url']})"
        else:
            origin = f"[{entry['url'].split('//', 1)[1]}]({entry['url']})"
        lines.append(
            f"| `{entry['id']}` {_cell(entry['title'])} | {_cell(entry['description'])} "
            f"| {'・'.join(entry['tasks'])} | {origin} | {split} | {_cell(entry['license'])} "
            f"| {entry['rows']:,} |"
        )
    return "\n".join(lines)


def render_readme(manifest: dict) -> str:
    subsets = "\n".join(
        f"| `{name}` | {SUBSET_LABELS[name]} | `test` | "
        f"{', '.join(f'`{i}`' for i in manifest['subsets'][name])} |"
        for name in router.SUBSETS
    )
    excluded = "\n".join(
        f"- `{e['id']}` {e['title']} — [{e['url']}]({e['url']}) ({e['license']}): {e['reason']}"
        for e in manifest["excluded"]
    )
    return f"""---
license: {manifest['license']}
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

[jev-ja-lab](https://github.com/hachiko85/jev-ja-lab) の日本語判断モデル評価(Noul / Choice /
Score)で使うデータセットの**ルーター**です。

サードパーティのデータ本体をこのリポジトリに保存・改変・再配布することはしません。
このリポジトリが持つのは「どのデータセットを、どの配布元の、どのrevision・splitから取得するか」
を記した `manifest.json` と、その取得クライアント `openjev_ja_eval.py` だけです。取得時は
各配布元(Hugging Face Hub / GitHub)から直接ダウンロードします。
このプロジェクトが独自に作成したデータ(`synthetic_score`)と、評価用に抽出した
HelpSteer2-JA(`helpsteer2_ja`、benchmark-v1)のみ、このリポジトリ内で管理します。

## サブセットとsplit

| サブセット | 内容 | split | 含まれるデータセット |
|---|---|---|---|
{subsets}

`load_dataset` で各サブセットを開くと、上記データセットの**配布元一覧(カタログ)**が
`test` splitとして返ります(データ本体ではありません)。データ本体の取得は次節のクライアントで行います。
評価に使うsplitは原則 `test` です。ただし配布元にtestが無い・ラベルが非公開のものは、
下表「使用split」のとおり validation / valid や、split区分なしの全件を使います。

## 使い方

`huggingface_hub` と `datasets` が必要です(`pip install huggingface_hub datasets pyarrow`)。
プライベートの間は `HF_TOKEN` が必要です。

```bash
# 一覧(サブセット: noul / choice / score / all)
python openjev_ja_eval.py list --subset noul

# 取得: 各配布元から直接ダウンロードし、jev-ja-lab の datasets/ 配置で保存
python openjev_ja_eval.py fetch --subset all --datasets-root ./datasets
```

`openjev_ja_eval.py` はこのリポジトリに含まれます(`hf_hub_download` で取得できます)。
jev-ja-lab 本体には同じ機能が `jev-ja-lab-datasets list|fetch` として入っています。

```python
from openjev_ja import datasets_router as router

manifest = router.load_manifest()  # このリポジトリのmanifest.json
router.fetch(manifest, "noul", "./datasets")
```

取得したデータは jev-ja-lab の `configs/eval/*.yaml` がそのまま読めるディレクトリ構成に
なります(`datasets_root: ./datasets`)。

## データセット一覧

{dataset_table(manifest)}

各データセットのrevision(コミット)は `manifest.json` に固定しています。評価ごとの
gold変換・選択肢生成は jev-ja-lab 側のadapterが取得後に行い、取得元データは改変しません。

### ルーター対象外(再配布禁止・別途承認が必要なもの)

{excluded}

## ライセンス

このリポジトリは、収録データセットが継承するライセンスのうち**最も厳しいもの**である **{manifest['license_label']}**(JMMLU・WRIMEが該当)として配布します。

- サードパーティのデータ本体は本リポジトリに含まれません。取得したデータの利用条件は、
  上表の各配布元ライセンスが適用されます(特にWRIMEとJMMLUは非商用・改変禁止、JaNLI・
  JCommonsenseQA・MGSM・JGLUE(JNLI)・JCoLAは継承(SA)条件付き、TextDetoxはOpenRAIL++の
  利用制限、PAWS-Xは配布元の独自条件)。
- 本リポジトリ内の独自データ: `synthetic_score` は MIT。`helpsteer2_ja` は
  [kunishou/HelpSteer2-20k-ja](https://huggingface.co/datasets/kunishou/HelpSteer2-20k-ja)
  (CC BY 4.0、原典 [nvidia/HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) も CC BY 4.0)
  からの抽出物のため、帰属表示(原典と翻訳者の明記)が必要です。
- 上記は各配布元のカード・リポジトリの記載に基づく整理です。再配布・商用利用の可否は
  必ず各配布元の原文で確認してください。

## 独自データ

- `synthetic_score/`: 緊急度・不満度・リスク・関連度の4軸、各200件。決定的テンプレート(seed=42)
  から生成(生成コードは jev-ja-lab の `scripts/prepare_extended_datasets.py`)。
- `helpsteer2_ja/`: HelpSteer2-JA benchmark-v1。`kunishou/HelpSteer2-20k-ja`(train、revision
  `ea432e41`)の19,958行から、`prompt` の完全一致でグループ化して1グループ1行、sha256順
  (seed=42)で2,500行を抽出。5軸(correctness / helpfulness / verbosity / complexity /
  coherence、各0〜4)。件数・sha256は `helpsteer2_ja/benchmark-v1.manifest.json`。
"""


def build_files(manifest: dict, workdir: Path) -> dict[str, Path]:
    """repo path -> local file, for everything the router repo should contain."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    README.write_text(render_readme(manifest), encoding="utf-8")
    files: dict[str, Path] = {
        "README.md": README,
        "manifest.json": MANIFEST,
        "openjev_ja_eval.py": ROOT / "src/openjev_ja/datasets_router.py",
    }
    for subset in router.SUBSETS:
        rows = router.catalog_rows(manifest, subset)
        path = workdir / subset / "test.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)
        files[f"{subset}/test.parquet"] = path
    for directory, local in OWN_FILES.items():
        include = OWN_INCLUDE[directory]
        for path in sorted(local.rglob("*")):
            if path.is_file() and (include is None or path.name in include):
                files[f"{directory}/{path.relative_to(local).as_posix()}"] = path
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    repo = args.repo or manifest["repo_id"]

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    with tempfile.TemporaryDirectory() as tmp:
        files = build_files(manifest, Path(tmp))
        api = HfApi()
        existing = set(api.list_repo_files(repo, repo_type="dataset"))
        keep = set(files)
        deletions = sorted(
            path for path in existing - keep if path not in {".gitattributes"}
        )
        print(f"upload {len(files)} files, delete {len(deletions)} files in {repo}")
        for path in deletions:
            print("  delete", path)
        if args.dry_run:
            return 0
        operations = [CommitOperationAdd(path, str(local)) for path, local in files.items()]
        operations += [CommitOperationDelete(path) for path in deletions]
        commit = api.create_commit(
            repo,
            operations,
            repo_type="dataset",
            commit_message="Turn into a router: manifest + client; keep only own data",
        )
        print(commit.commit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
