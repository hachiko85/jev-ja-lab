"""Runner tests use a tiny in-process architecture (no laya, no downloaded encoder) so the
orchestration (fold looping, state-dict saving, folds=1 "production" mode) is covered without a
GPU or network access.
"""

import pytest

torch = pytest.importorskip("torch")

from openjev_ja.common import BenchmarkItem  # noqa: E402
from openjev_ja.train.architectures import ArchitectureSpec  # noqa: E402
from openjev_ja.train.runner import cross_validate, train_one_fold  # noqa: E402


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(8))


def _build(model_name, device, **_):
    return None, _TinyModel().to(device)


def _forward_logits(model, tokenizer, device, question, options, primitive):
    return model.weight[: len(options)]


ARCHITECTURE = ArchitectureSpec(
    name="tiny",
    build=_build,
    forward_logits=_forward_logits,
    trainable_parameters=lambda model: list(model.parameters()),
    state_dict=lambda model: {k: v.detach().clone() for k, v in model.state_dict().items()},
)


def _items(n: int) -> list[BenchmarkItem]:
    return [
        BenchmarkItem(f"item-{i}", f"question {i}", ["いいえ", "はい"], i % 2)
        for i in range(n)
    ]


def test_train_one_fold_runs_every_epoch_and_returns_a_state_dict():
    items = _items(12)
    result = train_one_fold(
        ARCHITECTURE, model_name="tiny", primitive="noul",
        train_items=items[:8], val_items=items[8:], epochs=3, device="cpu",
    )
    assert len(result["history"]) == 3
    assert all(record["val_metrics"] is not None for record in result["history"])
    assert set(result["state_dict"]) == {"weight"}
    assert result["val_metrics"]["count"] == 4.0


def test_cross_validate_partitions_every_item_and_reports_mean_std():
    items = _items(20)
    result = cross_validate(
        ARCHITECTURE, items, model_name="tiny", primitive="noul", folds=4, epochs=1, device="cpu",
    )
    assert len(result["fold_results"]) == 4
    assert sum(fold["val_items"] for fold in result["fold_results"]) == 20
    assert "accuracy" in result["summary"]
    assert set(result["summary"]["accuracy"]) == {"mean", "std", "values"}
    assert len(result["summary"]["accuracy"]["values"]) == 4
    # every fold's saved checkpoint is a fresh model (no held-over state)
    assert all(set(fold["state_dict"]) == {"weight"} for fold in result["fold_results"])


def test_cross_validate_with_folds_1_trains_once_on_everything_with_no_summary():
    items = _items(10)
    result = cross_validate(
        ARCHITECTURE, items, model_name="tiny", primitive="choice", folds=1, epochs=1,
        device="cpu",
    )
    assert len(result["fold_results"]) == 1
    assert result["fold_results"][0]["val_items"] == 0
    assert result["summary"] == {}
