"""Fetch the evaluation datasets into one local directory through the dataset router.

Thin wrapper around `openjev_ja.datasets_router` (`jev-ja-lab-datasets fetch`): the router
manifest in `hachiko85/openjev-ja-eval` says where each dataset really lives, and every
dataset is downloaded straight from its origin. Nothing is re-hosted by the router.

    python scripts/fetch_all_datasets.py --datasets-root ./datasets
    python scripts/fetch_all_datasets.py --datasets-root ./datasets --subset noul
    python scripts/fetch_all_datasets.py --datasets-root ./datasets --with-extras

`--with-extras` additionally runs `prepare_extended_datasets.py` for the datasets that are
defined and implemented but disabled in the standard profile (LLM-jp Toxicity, JSFactCheckBench,
Civil Comments). JGPQA (gated) is only attempted with `--include-gated`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openjev_ja import datasets_router as router  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", required=True)
    parser.add_argument("--subset", choices=router.SUBSETS, default="all")
    parser.add_argument("--manifest", help="local manifest.json or Hub repo id (default router)")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--with-extras", action="store_true")
    args = parser.parse_args()

    manifest = router.load_manifest(args.manifest)
    results = router.fetch(
        manifest, args.subset, args.datasets_root, include_gated=args.include_gated
    )
    if args.with_extras:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("prepare_extended_datasets.py")),
                "--datasets-root",
                str(Path(args.datasets_root).resolve()),
            ],
            check=True,
        )
    return 1 if any(result.status == "skipped" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
