from __future__ import annotations

from openjev_ja.eval.primitive_cli import run_task_cli


def main(argv: list[str] | None = None) -> int:
    return run_task_cli("noul", argv)


if __name__ == "__main__":
    raise SystemExit(main())
