# ruff: noqa: E501
"""Render and publish the `hachiko85/openjev-ja-eval` repository.

The repository is a CC BY-SA 4.0 collection plus a router:

  - mirror  noul|choice|score|all/test.parquet   the redistributable datasets (MIT, Apache-2.0,
            CC BY, CC BY-SA) in one schema, built by scripts/build_dataset_mirror.py
  - router  router/test.parquet                  routing table for the datasets that are not
            mirrored (fetched from their origin): JMMLU, WRIME, JGPQA, PAWS-X, TextDetox
  - manifest.json, openjev_ja_eval.py (client), LICENSES/, README.md
  - the project's own data as files (synthetic_score/, helpsteer2_ja/) for the jev-ja-lab layout

    python scripts/publish_dataset_router.py --dry-run     # show what would change
    python scripts/publish_dataset_router.py               # commit to the Hub
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dataset_mirror  # noqa: E402

from openjev_ja import datasets_router as router  # noqa: E402

DIR = ROOT / "datasets_router"
MANIFEST = DIR / "manifest.json"
README = DIR / "README.md"
OWN_FILES = {
    "synthetic_score": "datasets/openjev-extra/synthetic_score",
    "helpsteer2_ja": "datasets/helpsteer2_ja",
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
        "| Name | Description | Task | Distribution | Source | Split used | License | Rows |",
        "|---|---|---|---|---|---|---|---:|",
    ]
    for entry in manifest["datasets"]:
        source = entry["source"]
        split = source.get("split") or "(none)"
        if source["kind"] == "own":
            tree = f"https://huggingface.co/datasets/{manifest['repo_id']}/tree/main/{source['path']}"
            origin = f"[{manifest['repo_id']}/{source['path']}]({tree})"
            if "derived_from" in source:
                origin += f"。原典: [{source['derived_from']['repo_id']}]({entry['url']})"
        else:
            origin = f"[{entry['url'].split('//', 1)[1]}]({entry['url']})"
        distribution = "mirror" if entry["distribution"] == "mirror" else "router"
        lines.append(
            f"| `{entry['id']}` {_cell(entry['title'])} | {_cell(entry['description'])} "
            f"| {'・'.join(entry['tasks'])} | {distribution} | {origin} | {split} "
            f"| {_cell(entry['license'])} | {entry['rows']:,} |"
        )
    return "\n".join(lines)


def render_readme(manifest: dict, counts: dict[str, dict[str, int]]) -> str:
    subsets = "\n".join(
        f"| `{name}` | {SUBSET_LABELS[name]} | `test` | {sum(counts[name].values()):,} | "
        f"{', '.join(f'`{k}`' for k in counts[name])} |"
        for name in router.SUBSETS
    )
    excluded = "\n".join(
        f"- `{e['id']}` {e['title']} ([{e['url']}]({e['url']}), {e['license']}): {e['reason']}"
        for e in manifest["excluded"]
    )
    routed = "、".join(
        f"{e['title']}({e['license']})"
        for e in manifest["datasets"]
        if e["distribution"] == "router"
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
    usage = (DIR / "usage.md").read_text(encoding="utf-8").strip()
    n_mirror = sum(1 for e in manifest["datasets"] if e["distribution"] == "mirror")
    n_router = len(manifest["datasets"]) - n_mirror
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
{len(manifest['datasets'])} 件をまとめています。

MIT・Apache-2.0・CC BY・CC BY-SA 4.0 で公開されている {n_mirror} 件は、共通の形式に揃えて本リポジトリに
**収録(ミラー)** し、集合物として **CC BY-SA 4.0** で配布します。それ以外の {n_router} 件({routed})は、再配布の
条件が合わないため本リポジトリには含めず、`manifest.json` に配布元・revision・split を記録して、取得時に
配布元から直接ダウンロードします(ルーター)。

## Subsets

| Subset | Description | Split | Rows | Datasets (mirror) |
|---|---|---|---:|---|
{subsets}
| `router` | ルーティング表(収録していないデータセットの配布元一覧) | `test` | {len(router.catalog_rows(manifest, 'router'))} | |

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

{dataset_table(manifest)}

収録していないデータセット(ルーター対象。`router` サブセットに配布元一覧があります):

{excluded}

{usage}

## Licensing Information

収録データ(mirror)は **{manifest['license_label']}** の集合物として配布します。MIT・Apache-2.0・CC BY・CC BY-SA 4.0 の
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

{sources}
- 本リポジトリ: <https://huggingface.co/datasets/{manifest['repo_id']}> /
  <https://github.com/hachiko85/jev-ja-lab>
"""


def build_files(manifest: dict, workdir: Path, datasets_root: Path) -> tuple[dict[str, Path], dict]:
    """repo path -> local file for everything the repository should contain, and row counts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = build_dataset_mirror.build_rows(datasets_root)
    rows["all"] = rows["noul"] + rows["choice"] + rows["score"]
    counts: dict[str, dict[str, int]] = {}
    files: dict[str, Path] = {}
    for subset, subset_rows in rows.items():
        counts[subset] = {}
        for row in subset_rows:
            counts[subset][row["dataset_id"]] = counts[subset].get(row["dataset_id"], 0) + 1
        path = workdir / subset / "test.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(subset_rows), path)
        files[f"{subset}/test.parquet"] = path
    path = workdir / "router" / "test.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(router.catalog_rows(manifest, "router")), path)
    files["router/test.parquet"] = path

    README.write_text(render_readme(manifest, counts), encoding="utf-8")
    files.update(
        {
            "README.md": README,
            "manifest.json": MANIFEST,
            "openjev_ja_eval.py": ROOT / "src/openjev_ja/datasets_router.py",
        }
    )
    for path in sorted((DIR / "LICENSES").iterdir()):
        files[f"LICENSES/{path.name}"] = path
    for directory, local in OWN_FILES.items():
        include = OWN_INCLUDE[directory]
        base = ROOT / local
        for path in sorted(base.rglob("*")):
            if path.is_file() and (include is None or path.name in include):
                files[f"{directory}/{path.relative_to(base).as_posix()}"] = path
    return files, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--datasets-root", default=str(ROOT / "datasets"))
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    repo = args.repo or manifest["repo_id"]

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    with tempfile.TemporaryDirectory() as tmp:
        files, counts = build_files(manifest, Path(tmp), Path(args.datasets_root))
        api = HfApi()
        existing = set(api.list_repo_files(repo, repo_type="dataset"))
        deletions = sorted(existing - set(files) - {".gitattributes"})
        print(f"upload {len(files)} files, delete {len(deletions)} files in {repo}")
        print({k: sum(v.values()) for k, v in counts.items()})
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
            commit_message="Mirror the CC BY-SA-compatible datasets; keep the rest as router",
        )
        print(commit.commit_url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
