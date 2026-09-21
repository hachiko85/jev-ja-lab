"""Fetch every dataset used by the jev-ja-lab eval harness into one local directory.

Covers the sources that `prepare_extended_datasets.py` does not: two GitHub-hosted
corpora (JGLUE, WRIME) and six Hugging Face Hub datasets referenced directly by
`configs/eval/primitives.20260918.yaml`. Run this once against an empty
`--datasets-root` to reproduce the full `datasets/` layout without manually
cloning each source.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

GIT_SOURCES = (
    {
        "name": "JGLUE-v1.1.0",
        "url": "https://github.com/yahoojapan/JGLUE.git",
        "revision": "6791a5003e6deaa6fb2458bcb3aa1f58dc52ef33",
    },
    {
        "name": "wrime-official",
        "url": "https://github.com/ids-cv/wrime.git",
        "revision": "ac92deaff7845f31fdd46ef5d893300887006d13",
    },
)

SNAPSHOT_SOURCES = (
    {
        "repo_id": "nlp-waseda/JMMLU",
        "revision": "3637b25e444ccfdcde4d23a783cbe8e674faa01b",
    },
)

GATED_SNAPSHOT_SOURCES = (
    {
        "repo_id": "llm-jp/jgpqa",
        "revision": None,
    },
)

LOAD_DATASET_SOURCES = (
    {"repo_id": "openai/MMMLU", "config": "JA_JP", "split": "test"},
    {"repo_id": "sbintuitions/JCommonsenseQA", "config": "default", "split": "validation"},
    {"repo_id": "Muennighoff/xwinograd", "config": "jp", "split": "test"},
    {"repo_id": "jbross-ibm-research/mgsm", "config": "ja", "split": "test"},
)


def _clone_git(source: dict[str, str], root: Path) -> None:
    target = root / source["name"]
    if not (target / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--no-checkout", source["url"], str(target)], check=True
        )
    subprocess.run(
        [
            "git", "-c", f"safe.directory={target}", "-C", str(target),
            "fetch", "--depth", "1", "origin", source["revision"],
        ],
        check=True,
    )
    subprocess.run(
        [
            "git", "-c", f"safe.directory={target}", "-C", str(target),
            "checkout", "--detach", source["revision"],
        ],
        check=True,
    )
    print(f"cloned {source['name']}: {source['revision']}")


def _snapshot_download(source: dict[str, str], root: Path, *, gated: bool) -> None:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import HfHubHTTPError

    try:
        snapshot_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            cache_dir=str(root),
        )
        print(f"downloaded {source['repo_id']}")
    except HfHubHTTPError as exc:
        if not gated:
            raise
        print(
            f"skipped {source['repo_id']}: requires accepting the dataset's terms "
            f"on huggingface.co and a Hugging Face token with access ({exc})"
        )


def _load_dataset(source: dict[str, str], root: Path) -> None:
    from datasets import load_dataset

    load_dataset(
        source["repo_id"],
        source["config"],
        split=source["split"],
        cache_dir=str(root),
        trust_remote_code=False,
    )
    print(f"downloaded {source['repo_id']} ({source['config']}/{source['split']})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument(
        "--skip-openjev-extra",
        action="store_true",
        help="do not also invoke prepare_extended_datasets.py",
    )
    args = parser.parse_args()
    root = Path(args.datasets_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    for source in GIT_SOURCES:
        _clone_git(source, root)
    for source in SNAPSHOT_SOURCES:
        _snapshot_download(source, root, gated=False)
    for source in GATED_SNAPSHOT_SOURCES:
        _snapshot_download(source, root, gated=True)
    for source in LOAD_DATASET_SOURCES:
        _load_dataset(source, root)

    if not args.skip_openjev_extra:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("prepare_extended_datasets.py")),
                "--datasets-root",
                str(root),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
