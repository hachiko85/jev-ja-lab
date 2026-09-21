from __future__ import annotations


class DatasetUnavailableError(RuntimeError):
    """Raised when a configured benchmark dataset cannot be fetched or read.

    Covers both a missing local file and a remote fetch that fails (e.g. a
    gated Hugging Face repo without an accepted-terms token). Orchestration
    treats this as a per-dataset skip rather than aborting the whole run.
    """
