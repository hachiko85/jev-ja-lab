from __future__ import annotations

from pathlib import Path


def local_git_revision(path: str) -> str | None:
    git_dir = Path(path) / ".git"
    head_path = git_dir / "HEAD"
    if not head_path.is_file():
        return None
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head or None
    ref = head.removeprefix("ref: ")
    ref_path = git_dir / Path(ref)
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip() or None
    packed_refs = git_dir / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith(("#", "^")):
                revision, packed_ref = line.split(" ", 1)
                if packed_ref == ref:
                    return revision
    return None
