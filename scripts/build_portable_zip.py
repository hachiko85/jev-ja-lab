from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

EXCLUDED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "dist",
    "models",
}
EXCLUDED_NAMES = {".env"}
EXCLUDED_SUFFIXES = {".bin", ".pyc", ".safetensors", ".zip"}


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    return path.suffix.lower() not in EXCLUDED_SUFFIXES


def build_archive(root: Path, output: Path) -> tuple[int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and _included(path, root)
    )
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        for path in files:
            archive_path = Path(root.name) / path.relative_to(root)
            bundle.write(path, archive_path.as_posix())
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return len(files), digest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a portable project archive")
    parser.add_argument(
        "--output",
        default="dist/jev-ja-lab-portable-20260918.zip",
        help="Output ZIP path relative to the project root",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    count, digest = build_archive(root, output)
    print(f"archive: {output.relative_to(root)}")
    print(f"files: {count}")
    print(f"sha256: {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
