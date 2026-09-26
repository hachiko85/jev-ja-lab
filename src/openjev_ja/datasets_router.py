"""Router for the jev-ja-lab evaluation datasets (`hachiko85/openjev-ja-eval`).

The Hub repository does not store third-party data. It publishes a `manifest.json` that
says, for every evaluation dataset, where the data really lives (Hugging Face Hub dataset,
GitHub repo at a pinned commit, or — for the project's own data — the repository itself),
which split is used, its licence, and which subsets it belongs to (`noul`, `choice`,
`score`, `all`; split `test`). This module resolves a subset from that manifest and fetches
each dataset straight from its origin into the directory layout `configs/eval/*.yaml`
expects, without re-hosting or modifying it.

Self-contained on purpose (standard library + lazily imported `huggingface_hub` /
`datasets`), so the same file is published next to the manifest as `openjev_ja_eval.py`:

    python openjev_ja_eval.py list --subset noul
    python openjev_ja_eval.py fetch --subset all --datasets-root ./datasets
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_REPO = "hachiko85/openjev-ja-eval"
SUBSETS = ("noul", "choice", "score", "all")


class RouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchResult:
    dataset_id: str
    status: str  # "fetched" | "cached" | "skipped"
    detail: str = ""


def load_manifest(
    source: str | Path | None = None, *, revision: str | None = None, token: str | None = None
) -> dict[str, Any]:
    """Manifest from a local file/directory, or from a Hub dataset repo (default repo)."""
    if source is not None and Path(source).exists():
        path = Path(source)
        path = path / "manifest.json" if path.is_dir() else path
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                str(source or DEFAULT_REPO),
                "manifest.json",
                repo_type="dataset",
                revision=revision,
                token=token,
            )
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise RouterError(f"unsupported manifest version: {manifest.get('version')!r}")
    return manifest


def select(
    manifest: dict[str, Any], subset: str = "all", *, include_gated: bool = False
) -> list[dict[str, Any]]:
    """Dataset entries of one subset, in manifest order. Gated datasets are excluded from the
    subsets by design; `include_gated` adds the ones whose tasks belong to the subset."""
    if subset not in SUBSETS:
        raise RouterError(f"unknown subset {subset!r}; choose from {', '.join(SUBSETS)}")
    by_id = {entry["id"]: entry for entry in manifest["datasets"]}
    entries = [by_id[dataset_id] for dataset_id in manifest["subsets"][subset]]
    if include_gated:
        entries += [
            entry
            for entry in manifest.get("excluded", [])
            if subset == "all" or subset in entry["tasks"]
        ]
    return entries


def viewer_url(entry: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Where the actual rows can be browsed: the source's own Data Studio page for Hub
    datasets, the repository page otherwise."""
    source = entry["source"]
    if source["kind"] in ("hf_dataset", "hf_parquet") and source.get("split"):
        config = source.get("config") or "default"
        repo, split = source["repo_id"], source["split"]
        return f"https://huggingface.co/datasets/{repo}/viewer/{config}/{split}"
    if source["kind"] == "hf_snapshot":
        revision = source["revision"] or "main"
        return f"https://huggingface.co/datasets/{source['repo_id']}/tree/{revision}"
    if source["kind"] == "own":
        return f"https://huggingface.co/datasets/{manifest['repo_id']}/tree/main/{source['path']}"
    return f"{source['url'].removesuffix('.git')}/tree/{source['revision']}"


def catalog_rows(manifest: dict[str, Any], subset: str) -> list[dict[str, Any]]:
    """Routing table rows: where each dataset of a subset comes from. `subset="router"` lists
    only the datasets that are not mirrored in the repository (they are fetched from their
    origin). `viewer_url` opens the actual rows at the source."""
    if subset == "router":
        entries = [e for e in manifest["datasets"] if e.get("distribution") == "router"]
        entries += manifest.get("excluded", [])
    else:
        entries = select(manifest, subset)
    rows = []
    for entry in entries:
        source = entry["source"]
        rows.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "distribution": entry.get("distribution", "router"),
                "tasks": ",".join(entry["tasks"]),
                "rows": entry["rows"],
                "license": entry["license"],
                "source_kind": source["kind"],
                "source_repo": source.get("repo_id") or source.get("url") or manifest["repo_id"],
                "source_config": source.get("config"),
                "source_split": source.get("split"),
                "revision": source.get("revision"),
                "viewer_url": viewer_url(entry, manifest),
                "description": entry["description"] if "description" in entry else entry["reason"],
            }
        )
    return rows


# --- fetch handlers -------------------------------------------------------------------


def _write_source_json(target: Path, payload: dict[str, Any]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "source.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _git(target: Path, *args: str) -> None:
    subprocess.run(["git", "-c", f"safe.directory={target}", "-C", str(target), *args], check=True)


def _fetch_git(entry: dict, root: Path, **_: Any) -> FetchResult:
    source = entry["source"]
    target = root / source["target"]
    if not (target / ".git").is_dir():
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", source["url"], str(target)], check=True)
    _git(target, "fetch", "--depth", "1", "origin", source["revision"])
    _git(target, "checkout", "--detach", source["revision"])
    _write_source_json(
        target,
        {
            "name": entry["id"],
            "url": source["url"],
            "revision": source["revision"],
            "format": "git",
        },
    )
    return FetchResult(entry["id"], "fetched", f"{source['url']}@{source['revision'][:8]}")


def _fetch_hf_dataset(entry: dict, root: Path, *, token: str | None, **_: Any) -> FetchResult:
    from datasets import load_dataset

    source = entry["source"]
    load_dataset(
        source["repo_id"],
        source["config"],
        split=source["split"],
        revision=source["revision"],
        cache_dir=str(root),
        token=token,
    )
    detail = f"{source['repo_id']} {source['config']}/{source['split']}"
    return FetchResult(entry["id"], "fetched", detail)


def _fetch_hf_snapshot(entry: dict, root: Path, *, token: str | None, **_: Any) -> FetchResult:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    source = entry["source"]
    try:
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            cache_dir=str(root),
            token=token,
        )
    except HfHubHTTPError as exc:
        if entry.get("access") != "gated":
            raise
        return FetchResult(
            entry["id"],
            "skipped",
            f"{source['repo_id']} needs accepting its terms on huggingface.co and a token "
            f"with access ({exc})",
        )
    return FetchResult(entry["id"], "fetched", source["repo_id"])


def _fetch_hf_parquet(entry: dict, root: Path, *, token: str | None, **_: Any) -> FetchResult:
    """Script-based Hub datasets (e.g. janli) cannot be loaded by `datasets` 4, so read the
    Hub's own parquet conversion (datasets-server) and write `<target>/data.parquet`
    (+ source.json), the file layout the configs read."""
    import urllib.parse
    import urllib.request

    source = entry["source"]
    target = root / source["target"]
    output = target / "data.parquet"
    if output.is_file():
        return FetchResult(entry["id"], "cached", str(output))
    import pyarrow.parquet as pq

    headers = {"Authorization": f"Bearer {token}"} if token else {}

    def get(url: str):
        return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600)

    query = urllib.parse.urlencode({"dataset": source["repo_id"]})
    with get(f"https://datasets-server.huggingface.co/parquet?{query}") as response:
        listing = json.loads(response.read())["parquet_files"]
    config = source.get("config") or "default"
    files = [f for f in listing if f["config"] == config and f["split"] == source["split"]]
    if not files:
        raise RouterError(f"{entry['id']}: no parquet conversion for {config}/{source['split']}")
    target.mkdir(parents=True, exist_ok=True)
    staging = target / ".data.parquet.tmp"
    writer = None
    rows = 0
    try:
        for index, item in enumerate(files):
            part = target / f".part-{index:04d}.parquet"
            with get(item["url"]) as download, part.open("wb") as handle:
                shutil.copyfileobj(download, handle)
            table = pq.read_table(part)
            writer = writer or pq.ParquetWriter(staging, table.schema)
            writer.write_table(table)
            rows += table.num_rows
            part.unlink()
    finally:
        if writer is not None:
            writer.close()
    staging.replace(output)
    _write_source_json(
        target,
        {
            "name": entry["id"],
            "repo_id": source["repo_id"],
            "config": source.get("config"),
            "split": source["split"],
            "revision": source["revision"],
            "rows": rows,
            "format": "parquet",
        },
    )
    return FetchResult(entry["id"], "fetched", f"{source['repo_id']} ({rows} rows)")


def _fetch_own(
    entry: dict, root: Path, *, token: str | None, manifest: dict[str, Any], **_: Any
) -> FetchResult:
    """The project's own data lives in the router repo itself; copy it into the layout."""
    from huggingface_hub import snapshot_download

    source = entry["source"]
    # local_dir (not the shared cache) so no symlinks are needed — Windows without
    # developer mode cannot create them.
    staging = root / ".router-own"
    snapshot_download(
        manifest["repo_id"],
        repo_type="dataset",
        allow_patterns=[f"{source['path']}/*"],
        local_dir=str(staging),
        token=token,
    )
    origin = staging / source["path"]
    if not origin.is_dir():
        raise RouterError(f"{manifest['repo_id']} has no directory {source['path']!r}")
    target = root / source["target"]
    target.mkdir(parents=True, exist_ok=True)
    for path in origin.rglob("*"):
        if path.is_file():
            destination = target / path.relative_to(origin)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    shutil.rmtree(staging, ignore_errors=True)
    return FetchResult(entry["id"], "fetched", f"{manifest['repo_id']}/{source['path']}")


HANDLERS = {
    "git": _fetch_git,
    "hf_dataset": _fetch_hf_dataset,
    "hf_snapshot": _fetch_hf_snapshot,
    "hf_parquet": _fetch_hf_parquet,
    "own": _fetch_own,
}


def fetch(
    manifest: dict[str, Any],
    subset: str,
    datasets_root: str | Path,
    *,
    token: str | None = None,
    include_gated: bool = False,
    only: list[str] | None = None,
) -> list[FetchResult]:
    root = Path(datasets_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    entries = select(manifest, subset, include_gated=include_gated)
    if only:
        entries = [entry for entry in entries if entry["id"] in set(only)]
    results = []
    for entry in entries:
        kind = entry["source"]["kind"]
        try:
            handler = HANDLERS[kind]
        except KeyError as exc:
            raise RouterError(f"{entry['id']}: unsupported source kind {kind!r}") from exc
        result = handler(entry, root, token=token, manifest=manifest)
        print(f"[{result.status}] {result.dataset_id}: {result.detail}")
        results.append(result)
    return results


# --- CLI -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest",
        help=f"local manifest.json / directory, or a Hub dataset repo (default: {DEFAULT_REPO})",
    )
    parser.add_argument("--revision", help="revision of the router repo to read the manifest from")
    parser.add_argument("--token", help="Hugging Face token (default: HF_TOKEN / cached login)")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("list", "fetch"):
        command = commands.add_parser(name)
        command.add_argument("--subset", choices=SUBSETS, default="all")
        command.add_argument(
            "--include-gated",
            action="store_true",
            help="also try datasets that need accepting terms (uses your own token)",
        )
        if name == "fetch":
            command.add_argument("--datasets-root", required=True)
            command.add_argument("--only", nargs="+", help="fetch only these dataset ids")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest, revision=args.revision, token=args.token)
    if args.command == "list":
        for entry in select(manifest, args.subset, include_gated=args.include_gated):
            source = entry["source"]
            origin = source.get("repo_id") or source.get("url") or manifest["repo_id"]
            print(
                f"{entry['id']:28s} {','.join(entry['tasks']):11s} {entry['license']:24s} "
                f"{source['kind']}:{origin}"
            )
        return 0
    results = fetch(
        manifest,
        args.subset,
        args.datasets_root,
        token=args.token,
        include_gated=args.include_gated,
        only=args.only,
    )
    return 1 if any(result.status == "skipped" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
