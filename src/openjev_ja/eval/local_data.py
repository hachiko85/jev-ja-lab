from __future__ import annotations

import csv
import io
import json
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from openjev_ja.common import BenchmarkItem, DatasetUnavailableError
from openjev_ja.eval.datasets.adapters import convert_row


class LocalDatasetError(DatasetUnavailableError):
    """Raised when a configured local benchmark source cannot be loaded."""


def _resolve_one(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"dataset source not found: {pattern}")
    if len(matches) != 1:
        raise LocalDatasetError(f"dataset source must match one file: {pattern}")
    return matches[0]


def _arrow_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.ipc as ipc
    except ImportError as exc:
        message = "Install orchestration dependencies: pip install -e '.[orchestrate]'"
        raise RuntimeError(message) from exc
    with pa.memory_map(str(path), "r") as source:
        yield from ipc.open_stream(source).read_all().to_pylist()


def _csv_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def _tsv_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _jsonl_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _parquet_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        message = "Install orchestration dependencies: pip install -e '.[orchestrate]'"
        raise RuntimeError(message) from exc
    yield from pq.read_table(path).to_pylist()


def _hub_rows(source: Mapping[str, Any], root: Path) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        message = "Install evaluation dependencies: pip install -e '.[eval]'"
        raise RuntimeError(message) from exc
    kwargs: dict[str, Any] = {
        "split": str(source.get("split", "test")),
        "cache_dir": str(root / ".hub-cache"),
    }
    if source.get("config") is not None:
        kwargs["name"] = str(source["config"])
    if source.get("revision") is not None:
        kwargs["revision"] = str(source["revision"])
    yield from load_dataset(str(source["repo_id"]), **kwargs)


def _materialize_hub_file(source: Mapping[str, Any], root: Path) -> Path:
    """Download a single file referenced by source.hub_source into datasets_root.

    Uses the standard huggingface_hub cache layout (datasets_root becomes the
    HF cache dir) and returns the local path huggingface_hub placed it at,
    which does not have to match `source.path` (that field only describes
    where a manually pre-staged copy would live). A no-op if already cached.
    """
    hub_source = source["hub_source"]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        message = "Install evaluation dependencies: pip install -e '.[eval]'"
        raise RuntimeError(message) from exc
    try:
        return Path(
            hf_hub_download(
                repo_id=str(hub_source["repo_id"]),
                filename=str(hub_source["filename"]),
                repo_type=str(hub_source.get("repo_type", "dataset")),
                revision=hub_source.get("revision"),
                cache_dir=str(root),
            )
        )
    except OSError as exc:
        raise DatasetUnavailableError(
            f"could not fetch {hub_source['filename']!r} from "
            f"{hub_source['repo_id']!r}: {exc}"
        ) from exc


def _jmmlu_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    with zipfile.ZipFile(path) as bundle:
        members = sorted(
            member
            for member in bundle.namelist()
            if member.startswith("JMMLU/test/") and member.endswith(".csv")
        )
        for member in members:
            subject = Path(member).stem
            raw = bundle.read(member)
            for encoding in ("utf-8-sig", "cp932"):
                try:
                    text = raw.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise LocalDatasetError(f"cannot decode JMMLU member: {Path(member).name}")
            for row in csv.DictReader(io.StringIO(text)):
                yield {**row, "subject": subject}


def _jnli_noul_item(
    name: str, row: Mapping[str, Any], index: int, target: str
) -> BenchmarkItem:
    labels = ["entailment", "contradiction", "neutral"]
    raw_label = row["label"]
    label = labels[int(raw_label)] if isinstance(raw_label, int) else str(raw_label)
    prompts = {
        "entailment": "仮説は前提から支持されますか？",
        "contradiction": "仮説は前提と矛盾しますか？",
        "neutral": "前提だけでは仮説を判断する情報が不足していますか？",
    }
    if target not in prompts:
        raise LocalDatasetError(f"unsupported JNLI Noul target: {target}")
    premise = str(row["sentence1"]).strip()
    hypothesis = str(row["sentence2"]).strip()
    question = f"前提: {premise}\n仮説: {hypothesis}\n{prompts[target]}"
    return BenchmarkItem(
        f"{name}:{row.get('sentence_pair_id', index)}",
        question,
        ["いいえ", "はい"],
        int(label == target),
        {"dataset": name, "source_dataset": "jnli", "noul_target": target},
    )


def _score_item(
    name: str, row: Mapping[str, Any], index: int, source: Mapping[str, Any]
) -> BenchmarkItem:
    question_field = str(source.get("question_field", "state"))
    gold_field = str(source.get("gold_field", "gold_score"))
    criteria_field = str(source.get("criteria_field", "criteria"))
    criteria = row.get(criteria_field, source.get("criteria"))
    if not isinstance(criteria, list) or len(criteria) < 2:
        raise LocalDatasetError("score adapter requires at least two criteria")
    raw_gold = float(row[gold_field])
    thresholds = source.get("gold_thresholds")
    if thresholds is not None:
        gold = sum(raw_gold >= float(threshold) for threshold in thresholds)
    else:
        gold = int(raw_gold) + int(source.get("gold_offset", 0))
    if not 0 <= gold < len(criteria):
        raise LocalDatasetError("gold score is outside criteria")
    question_template = source.get("question_template")
    question = (
        str(question_template).format_map(row).strip()
        if question_template
        else str(row[question_field]).strip()
    )
    max_chars = int(source.get("max_chars", 0))
    if max_chars > 0:
        question = question[:max_chars]
    prompt = str(source.get("prompt", "次の状態を基準に沿って評価してください。"))
    return BenchmarkItem(
        f"{name}:{row.get('id', index)}",
        f"{prompt}\n状態: {question}",
        [str(value) for value in criteria],
        gold,
        {"dataset": name, "gold_score": gold, "criteria": criteria},
    )


def _noul_item(
    name: str, row: Mapping[str, Any], index: int, source: Mapping[str, Any]
) -> BenchmarkItem:
    gold_field = str(source["gold_field"])
    raw_gold = row[gold_field]
    if "positive_values" in source:
        positive = {str(value) for value in source["positive_values"]}
        gold = int(str(raw_gold) in positive)
    elif "positive_min" in source:
        gold = int(float(raw_gold) >= float(source["positive_min"]))
    else:
        raise LocalDatasetError("noul adapter requires positive_values or positive_min")
    template = str(source.get("question_template", "{text}"))
    question = template.format_map(row).strip()
    max_chars = int(source.get("max_chars", 0))
    if max_chars > 0:
        question = question[:max_chars]
    item_id_field = str(source.get("item_id_field", "id"))
    return BenchmarkItem(
        f"{name}:{row.get(item_id_field, index)}",
        question,
        ["いいえ", "はい"],
        gold,
        {"dataset": name, "source_label": raw_gold},
    )


def _convert_local_row(
    name: str,
    row: Mapping[str, Any],
    index: int,
    seed: int,
    source: Mapping[str, Any],
) -> BenchmarkItem:
    adapter = source.get("adapter")
    if adapter == "jnli_noul":
        return _jnli_noul_item(name, row, index, str(source["target"]))
    if adapter == "score":
        return _score_item(name, row, index, source)
    if adapter == "noul":
        return _noul_item(name, row, index, source)
    return convert_row(name, row, index, seed)


_FORMAT_LOADERS = {
    "arrow": _arrow_rows,
    "csv": _csv_rows,
    "tsv": _tsv_rows,
    "jsonl": _jsonl_rows,
    "parquet": _parquet_rows,
    "jmmlu_zip": _jmmlu_rows,
}


def _read_rows(path: Path, source_format: str) -> Iterable[Mapping[str, Any]]:
    try:
        return _FORMAT_LOADERS[source_format](path)
    except KeyError as exc:
        raise LocalDatasetError(f"unsupported dataset format: {source_format}") from exc


def _local_path_rows(source: Mapping[str, Any], root: Path) -> Iterable[Mapping[str, Any]]:
    path = _resolve_one(root, str(source["path"]))
    return _read_rows(path, str(source["format"]))


def _resolve_rows(name: str, source: Mapping[str, Any], root: Path) -> Iterable[Mapping[str, Any]]:
    """Resolve the row source for a dataset, preferring an already-staged local file.

    Precedence: a local `path` that already resolves under datasets_root (fast,
    offline, matches whatever the user has pre-placed there) beats every fetch
    mechanism below, so existing local setups keep working unchanged. Only when
    nothing is staged locally does this fetch from the declared remote source,
    which makes a dataset usable from a repo_id alone in a clean checkout with
    no pre-staged datasets/ directory.
    """
    if source.get("path") and list(root.glob(str(source["path"]))):
        return _local_path_rows(source, root)
    if source.get("repo_id"):
        return _hub_rows(source, root)
    if source.get("adapter_dataset"):
        from openjev_ja.eval.datasets.adapters import _raw_rows

        return _raw_rows(
            str(source["adapter_dataset"]), source.get("revision"), root / "adapter-cache"
        )
    if source.get("hub_source"):
        path = _materialize_hub_file(source, root)
        return _read_rows(path, str(source["format"]))
    if source.get("path"):
        return _local_path_rows(source, root)
    raise LocalDatasetError(f"dataset source not configured: {name}")


def load_local_benchmark(
    name: str,
    source: Mapping[str, Any],
    *,
    datasets_root: str | Path,
    seed: int = 42,
    limit: int | None = None,
) -> list[BenchmarkItem]:
    root = Path(datasets_root)
    rows = _resolve_rows(name, source, root)
    items: list[BenchmarkItem] = []
    where = source.get("where", {})
    where_in = source.get("where_in", {})
    where_not = source.get("where_not", {})
    for index, row in enumerate(rows):
        if any(str(row.get(key)) != str(value) for key, value in where.items()):
            continue
        if any(
            str(row.get(key)) not in {str(value) for value in values}
            for key, values in where_in.items()
        ):
            continue
        if any(str(row.get(key)) == str(value) for key, value in where_not.items()):
            continue
        items.append(_convert_local_row(name, row, index, seed, source))
        if limit is not None and len(items) >= limit:
            break
    if not items:
        raise LocalDatasetError(f"dataset is empty: {name}")
    return items
