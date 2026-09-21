from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

HF_SOURCES = (
    {
        "name": "janli",
        "repo_id": "hpprc/janli",
        "config": "base",
        "split": "test",
        "revision": "b91923216e61ded52ec77316cb329b47227e4955",
    },
    {
        "name": "paws_x_ja",
        "repo_id": "google-research-datasets/paws-x",
        "config": "ja",
        "split": "test",
        "revision": "4cd8187c404bda33cb1f62b49b001115862acf37",
    },
    {
        "name": "llmjp_toxicity",
        "repo_id": "p1atdev/LLM-jp-Toxicity-Dataset",
        "config": None,
        "split": "train",
        "revision": "cc461b3a088cafda1d03a64b71b1efb6b072b5ba",
    },
    {
        "name": "textdetox_ja",
        "repo_id": "textdetox/multilingual_toxicity_dataset",
        "config": None,
        "split": "ja",
        "revision": "01907546324b0330d2d8b7669648cc18823323e5",
    },
    {
        "name": "helpsteer2_ja",
        "repo_id": "kunishou/HelpSteer2-20k-ja",
        "config": None,
        "split": "train",
        "revision": "ea432e41ac690821953c38f77be9690c9c19cd9f",
    },
    {
        "name": "civil_comments",
        "repo_id": "google/civil_comments",
        "config": None,
        "split": "test",
        "revision": "f2970eb3a55777454c94069077cc8d9b5866312d",
    },
)

GATED_SOURCE = {
    "name": "jsfactcheckbench",
    "repo_id": "llm-jp/JSFactCheckBench",
    "config": "claims",
    "split": "test",
    "revision": "100f4c60379fad2828a1ee763c79354210078789",
}

GIT_SOURCES = (
    {
        "name": "jcola",
        "url": "https://github.com/osekilab/JCoLA.git",
        "revision": "736d9eef3af04bb17e4cf59adf01325973234ff9",
    },
    {
        "name": "jad-afc",
        "url": (
            "https://github.com/FujitsuResearch/"
            "japanese-dataset-for-automated-fact-checking.git"
        ),
        "revision": "c2a84f80dd64f28122a4c8d4bd2aeab558355921",
    },
)


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _download_hf(source: dict[str, Any], root: Path) -> None:
    import pyarrow.parquet as pq
    import requests

    target = root / source["name"]
    target.mkdir(parents=True, exist_ok=True)
    output = target / "data.parquet"
    if not output.is_file():
        response = requests.get(
            "https://datasets-server.huggingface.co/parquet",
            params={"dataset": source["repo_id"]},
            timeout=120,
        )
        response.raise_for_status()
        config = source["config"] or "default"
        files = [
            item
            for item in response.json()["parquet_files"]
            if item["config"] == config and item["split"] == source["split"]
        ]
        if not files:
            raise RuntimeError(f"no converted parquet files for {source['name']}")
        staging = target / ".data.parquet.tmp"
        staging.unlink(missing_ok=True)
        writer = None
        rows = 0
        try:
            for index, item in enumerate(files):
                part = target / f".part-{index:04d}.parquet"
                with requests.get(item["url"], stream=True, timeout=600) as download:
                    download.raise_for_status()
                    with part.open("wb") as handle:
                        for chunk in download.iter_content(1024 * 1024):
                            handle.write(chunk)
                table = pq.read_table(part)
                if writer is None:
                    writer = pq.ParquetWriter(staging, table.schema)
                writer.write_table(table)
                rows += table.num_rows
                part.unlink()
        finally:
            if writer is not None:
                writer.close()
        staging.replace(output)
    else:
        rows = pq.read_metadata(output).num_rows
    _write_metadata(
        target / "source.json",
        {**source, "rows": rows, "format": "parquet"},
    )
    print(f"prepared {source['name']}: {rows}")


def _prepare_git(source: dict[str, str], root: Path) -> None:
    target = root / source["name"]
    if not (target / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--no-checkout", source["url"], str(target)], check=True
        )
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={target}",
            "-C",
            str(target),
            "fetch",
            "--depth",
            "1",
            "origin",
            source["revision"],
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={target}",
            "-C",
            str(target),
            "checkout",
            "--detach",
            source["revision"],
        ],
        check=True,
    )
    _write_metadata(target / "source.json", {**source, "format": "git"})
    print(f"prepared {source['name']}: {source['revision']}")


def _prepare_synthetic(root: Path) -> None:
    target = root / "synthetic_score"
    target.mkdir(parents=True, exist_ok=True)
    dimensions = {
        "urgency": [
            "対応期限はなく、通常処理で問題ない",
            "一週間以内に確認してほしい",
            "二日以内の対応を希望する",
            "本日中に対応しないと業務へ影響する",
            "生命・安全に直結し、直ちに対応が必要である",
        ],
        "dissatisfaction": [
            "対応に満足している",
            "小さな不便はあるが概ね満足している",
            "問題があり改善を求めている",
            "強い不満があり早急な改善を要求している",
            "極めて強い不満があり契約解除を要求している",
        ],
        "risk": [
            "危険性は認められない",
            "軽微な注意が必要である",
            "無視できない危険がある",
            "重大事故につながる高い危険がある",
            "差し迫った生命の危険がある",
        ],
        "relevance": [
            "質問と回答は無関係である",
            "回答は質問へわずかに触れている",
            "回答は質問へ部分的に答えている",
            "回答は質問へほぼ適切に答えている",
            "回答は質問へ完全かつ直接的に答えている",
        ],
    }
    for dimension, levels in dimensions.items():
        rows = []
        for level, text in enumerate(levels):
            for variant in range(40):
                rows.append(
                    {
                        "id": f"{dimension}-{level}-{variant:02d}",
                        "state": f"事例{variant + 1}: {text}。",
                        "gold_score": level,
                    }
                )
        output = target / f"{dimension}.jsonl"
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    _write_metadata(
        target / "source.json",
        {
            "generator": "deterministic templates",
            "seed": 42,
            "dimensions": list(dimensions),
            "rows_per_dimension": 200,
        },
    )
    print("prepared synthetic_score: 4 x 200")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", required=True)
    args = parser.parse_args()
    root = Path(args.datasets_root).resolve() / "openjev-extra"
    root.mkdir(parents=True, exist_ok=True)
    for source in GIT_SOURCES:
        _prepare_git(source, root)
    for source in HF_SOURCES:
        _download_hf(source, root)
    try:
        _download_hf(GATED_SOURCE, root)
    except Exception:
        target = root / GATED_SOURCE["name"]
        target.mkdir(parents=True, exist_ok=True)
        _write_metadata(
            target / "source.json",
            {
                **GATED_SOURCE,
                "status": "requires_huggingface_terms_and_token",
            },
        )
        print("skipped jsfactcheckbench: requires Hugging Face terms and token")
    _prepare_synthetic(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
