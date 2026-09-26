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
        "| Name | Description | Task | Source | Split used | License | Rows |",
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
        f"- `{e['id']}` {e['title']} ([{e['url']}]({e['url']}), {e['license']}): {e['reason']}"
        for e in manifest["excluded"]
    )
    sources = "\n".join(
        f"- {e['title']}: <{e['url']}>"
        + (
            f" (原典: <https://huggingface.co/datasets/{e['source']['derived_from']['original']}>)"
            if "derived_from" in e["source"]
            else ""
        )
        for e in manifest["datasets"] + manifest["excluded"]
    )
    usage = (ROOT / "datasets_router" / "usage.md").read_text(encoding="utf-8").strip()
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

> Japanese evaluation datasets for Noul / Choice / Score decision models: a router, not a mirror

## What is this?

[jev-ja-lab](https://github.com/hachiko85/jev-ja-lab) で日本語の判断モデルを評価するための
データセット集です。判断タスクを Noul(二値判定)・Choice(選択式)・Score(段階評価)の
3 種類に分け、既存の公開データセット {len(manifest['datasets'])} 件をまとめています。

第三者のデータ本体は本リポジトリに含みません。`manifest.json` に配布元・revision・split・
ライセンスを記録し、取得時に各配布元(Hugging Face Hub / GitHub)から直接ダウンロードします。
本リポジトリ内で管理するのは、独自に作成した `synthetic_score` と、評価用に抽出した
HelpSteer2-JA(`helpsteer2_ja`、benchmark-v1)のみです。

## Subsets

| Subset | Description | Split | Datasets |
|---|---|---|---|
{subsets}

評価の split は `test` です。配布元に test が無い、または test のラベルが非公開のものは、
下表「使用split」のとおり validation / valid、または split 区分のない全件を使います。
Data Studio と `load_dataset` が返す各サブセットの `test` split は、データ本体ではなく、そのサブセットに
含まれるデータセットの配布元一覧(ルーティング表)です。中身を見るには `viewer_url` 列のリンクから各配布元の
ページを開いてください(Hugging Face 上のデータセットはその Data Studio、GitHub はリポジトリの該当 revision)。
データ本体の取得方法は次節を参照してください。

## Datasets

{dataset_table(manifest)}

Not routed (redistribution prohibited or separate approval required):

{excluded}

{usage}

## Licensing Information

本リポジトリは **{manifest['license_label']}** で配布します。収録データセットが継承するライセンスの
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

## Acknowledgements

評価データを公開されている各データセットの作成者・配布者の皆様に感謝します。
また、HelpSteer2 を公開した NVIDIA、日本語訳 HelpSteer2-20k-ja を公開した kunishou 氏に
感謝します。

## Citation Information

利用時は、各データセットの配布元に記載された引用方法に従ってください。

{sources}
- 本リポジトリ: <https://huggingface.co/datasets/{manifest['repo_id']}> /
  <https://github.com/hachiko85/jev-ja-lab>
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
