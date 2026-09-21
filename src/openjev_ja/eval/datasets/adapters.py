from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import random
import re
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from openjev_ja.common import BenchmarkItem, DatasetUnavailableError

DATASET_NAMES = (
    "mmmlu_ja",
    "jmmlu",
    "jgpqa_diamond",
    "jcommonsenseqa",
    "xwinograd_ja",
    "mgsm_ja",
    "gsm8k_ja_mc4",
    "gsm8k_ja_mc10",
    "jnli",
)

_DATASET_REPOSITORIES = {
    "mmmlu_ja": "openai/MMMLU",
    "jmmlu": "nlp-waseda/JMMLU",
    "jgpqa_diamond": "llm-jp/jgpqa",
    "jcommonsenseqa": "sbintuitions/JCommonsenseQA",
    "xwinograd_ja": "Muennighoff/xwinograd",
    "mgsm_ja": "jbross-ibm-research/mgsm",
    "gsm8k_ja_mc4": "SakanaAI/gsm8k-ja-test_250-1319",
    "gsm8k_ja_mc10": "SakanaAI/gsm8k-ja-test_250-1319",
}

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _answer_index(value: Any, options: list[str]) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    upper = text.upper()
    if upper in _LABELS[: len(options)]:
        return _LABELS.index(upper)
    if text.isdigit():
        number = int(text)
        if 0 <= number < len(options):
            return number
        if 1 <= number <= len(options):
            return number - 1
    try:
        return options.index(text)
    except ValueError as exc:
        raise ValueError(f"cannot map answer {value!r} to options") from exc


def _id(dataset: str, row: Mapping[str, Any], index: int) -> str:
    for key in ("id", "q_id", "sentence_pair_id", "question_id"):
        if row.get(key) not in (None, ""):
            return f"{dataset}:{row[key]}"
    return f"{dataset}:{index}"


def _numeric_text(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.6f}".rstrip("0").rstrip(".")


def extract_answer_number(row: Mapping[str, Any]) -> str:
    if row.get("answer_number") is not None:
        raw = str(row["answer_number"]).replace(",", "")
    else:
        answer = str(row.get("answer", ""))
        match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer)
        values = _NUMBER_RE.findall(answer)
        raw = (match.group(1) if match else values[-1] if values else "").replace(",", "")
    if not raw:
        raise ValueError("numeric answer is missing")
    return _numeric_text(float(raw))


def numeric_options(gold: str, count: int, seed: int, item_id: str) -> tuple[list[str], int]:
    gold_value = float(gold)
    digest = hashlib.sha256(f"{seed}:{item_id}:{gold}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    deltas = [1, -1, 2, -2, 3, -3, 5, -5, 10, -10]
    candidates: set[str] = set()
    attempts = 0
    while len(candidates) < count - 1 and attempts < 1000:
        attempts += 1
        mode = attempts % 4
        if mode == 0:
            value = gold_value + rng.choice(deltas)
        elif mode == 1:
            value = gold_value * rng.choice([0.5, 2, 3, 10])
        elif mode == 2:
            value = gold_value + rng.randint(-50, 50)
        else:
            scale = max(1.0, abs(gold_value) * 0.1)
            value = gold_value + rng.choice([-1, 1]) * scale * rng.randint(1, 5)
        candidate = _numeric_text(value)
        if candidate != gold:
            candidates.add(candidate)
    if len(candidates) != count - 1:
        raise RuntimeError("failed to create enough numeric distractors")
    options = [gold, *sorted(candidates)]
    rng.shuffle(options)
    return options, options.index(gold)


def convert_row(name: str, row: Mapping[str, Any], index: int, seed: int = 42) -> BenchmarkItem:
    metadata: dict[str, Any] = {"dataset": name}
    item_id = _id(name, row, index)
    if name == "mmmlu_ja":
        options = [str(row[key]).strip() for key in "ABCD"]
        question, gold = str(row["Question"]).strip(), _answer_index(row["Answer"], options)
        metadata["subject"] = row.get("Subject")
    elif name == "jmmlu":
        options = [str(row[key]).strip() for key in "ABCD"]
        question, gold = str(row["question"]).strip(), _answer_index(row["answer"], options)
        metadata["subject"] = row.get("subject")
    elif name == "jgpqa_diamond":
        options = [str(row["Correct Answer"]).strip()] + [
            str(row[f"Incorrect Answer {i}"]).strip() for i in range(1, 4)
        ]
        question, gold = str(row["Question"]).strip(), 0
    elif name == "jcommonsenseqa":
        options = [str(row[f"choice{i}"]).strip() for i in range(5)]
        question, gold = str(row["question"]).strip(), int(row["label"])
    elif name == "xwinograd_ja":
        options = [str(row["option1"]).strip(), str(row["option2"]).strip()]
        question = str(row["sentence"]).strip()
        gold = _answer_index(row["answer"], options)
    elif name in {"mgsm_ja", "gsm8k_ja_mc4", "gsm8k_ja_mc10"}:
        question = str(row["question"]).strip()
        answer = extract_answer_number(row)
        count = 10 if name.endswith("mc10") else 4
        options, gold = numeric_options(answer, count, seed, item_id)
        metadata["answer_number"] = answer
    elif name == "jnli":
        premise = str(row["sentence1"]).strip()
        hypothesis = str(row["sentence2"]).strip()
        question = f"前提: {premise}\n仮説: {hypothesis}\n両者の関係を選んでください。"
        options = ["含意", "矛盾", "中立"]
        label = row["label"]
        names = ["entailment", "contradiction", "neutral"]
        gold = int(label) if isinstance(label, int) else names.index(str(label))
    else:
        raise KeyError(f"unknown dataset: {name}")
    return BenchmarkItem(item_id, question, options, gold, metadata)


def _load_hf(path: str, config: str | None, split: str, revision: str | None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
    kwargs: dict[str, Any] = {"split": split}
    cache_dir = os.getenv("OPENJEV_DATASETS_DIR")
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if revision:
        kwargs["revision"] = revision
    return load_dataset(path, config, **kwargs)


def _load_jmmlu(revision: str | None) -> Iterable[Mapping[str, Any]]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
    archive = hf_hub_download(
        repo_id="nlp-waseda/JMMLU",
        filename="JMMLU.zip",
        repo_type="dataset",
        revision=revision,
        cache_dir=os.getenv("OPENJEV_DATASETS_DIR"),
    )
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            member
            for member in bundle.namelist()
            if "/test/" in member
            and member.endswith(".csv")
            and not member.rsplit("/", 1)[-1].startswith("._")
        )
        for member in members:
            config = member.rsplit("/", 1)[-1].removesuffix(".csv")
            with bundle.open(member) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
                for row in reader:
                    yield {**row, "subject": config}


def _cached_text(cache_dir: Path | None, filename: str, fetch: Any) -> str:
    """Return fetch()'s text, cached under cache_dir on the first call.

    cache_dir is always the caller's own local datasets_root (gitignored,
    never pushed or shared) — reusing this cache is not redistribution, it
    just avoids re-downloading the same file for every model evaluated in
    the same run. Fetch functions still hit the origin every time when no
    cache_dir is given (e.g. the standalone CLI tool).
    """
    if cache_dir is None:
        return fetch()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if path.is_file():
        return path.read_text(encoding="utf-8")
    text = fetch()
    path.write_text(text, encoding="utf-8")
    return text


def _load_jnli(revision: str | None, cache_dir: Path | None = None) -> Iterable[Mapping[str, Any]]:
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
    ref = revision or "v1.1.0"

    def fetch() -> str:
        url = (
            "https://api.github.com/repos/yahoojapan/JGLUE/contents/"
            f"datasets/jnli-v1.1/valid-v1.1.json?ref={ref}"
        )
        response = httpx.get(
            url,
            headers={"Accept": "application/vnd.github+json", "User-Agent": "jev-ja-lab"},
            timeout=60.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        payload = response.json()
        return base64.b64decode(payload["content"]).decode("utf-8")

    content = _cached_text(cache_dir, f"jnli-{ref}.jsonl", fetch)
    for line in content.splitlines():
        if line.strip():
            yield json.loads(line)


def _load_wrime(revision: str | None, cache_dir: Path | None = None) -> Iterable[Mapping[str, Any]]:
    """Fetch WRIME ver.2 from its origin, cached locally under cache_dir.

    WRIME's own terms prohibit redistributing the dataset, so this never
    uploads or mirrors a copy anywhere else; caching it under the caller's
    own local datasets_root (never committed, never shared) is not
    redistribution — it is the same thing a manual `git clone` of
    https://github.com/ids-cv/wrime would leave on disk.
    """
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install evaluation dependencies: pip install -e '.[eval]'") from exc
    ref = revision or "master"

    def fetch() -> str:
        url = f"https://raw.githubusercontent.com/ids-cv/wrime/{ref}/wrime-ver2.tsv"
        response = httpx.get(
            url,
            headers={"User-Agent": "jev-ja-lab"},
            timeout=120.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        return response.text

    text = _cached_text(cache_dir, f"wrime-ver2-{ref}.tsv", fetch)
    yield from csv.DictReader(io.StringIO(text), delimiter="\t")


def _raw_rows(
    name: str, revision: str | None, cache_dir: Path | None = None
) -> Iterable[Mapping[str, Any]]:
    if name == "wrime":
        return _load_wrime(revision, cache_dir)
    if name == "jnli":
        return _load_jnli(revision, cache_dir)
    if name == "mmmlu_ja":
        return _load_hf("openai/MMMLU", "JA_JP", "test", revision)
    if name == "jmmlu":
        return _load_jmmlu(revision)
    if name == "jgpqa_diamond":
        try:
            return _load_hf("llm-jp/jgpqa", "gpqa_diamond", "train", revision)
        except Exception as exc:
            raise DatasetUnavailableError(
                "JGPQA requires accepting its terms and an authenticated Hugging Face session."
            ) from exc
    if name == "jcommonsenseqa":
        return _load_hf("sbintuitions/JCommonsenseQA", None, "validation", revision)
    if name == "xwinograd_ja":
        return _load_hf("Muennighoff/xwinograd", "jp", "test", revision)
    if name == "mgsm_ja":
        return _load_hf("jbross-ibm-research/mgsm", "ja", "test", revision)
    if name in {"gsm8k_ja_mc4", "gsm8k_ja_mc10"}:
        return _load_hf("SakanaAI/gsm8k-ja-test_250-1319", None, "test", revision)
    raise KeyError(f"unknown dataset: {name}")


def load_benchmark(
    name: str,
    *,
    limit: int | None = None,
    seed: int = 42,
    revision: str | None = None,
) -> list[BenchmarkItem]:
    if name not in DATASET_NAMES:
        raise KeyError(f"unknown dataset: {name}")
    items: list[BenchmarkItem] = []
    for index, row in enumerate(_raw_rows(name, revision)):
        items.append(convert_row(name, row, index, seed))
        if limit is not None and len(items) >= limit:
            break
    return items


def resolve_dataset_revision(name: str, requested: str | None = None) -> str | None:
    if requested:
        return requested
    if name == "jnli":
        return "v1.1.0"
    repository = _DATASET_REPOSITORIES.get(name)
    if repository is None:
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repository).sha
    except Exception:
        return None
