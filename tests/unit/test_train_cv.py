import pytest

from openjev_ja.train.cv import kfold_indices, kfold_split


def test_folds_partition_every_index_exactly_once_as_validation():
    folds = kfold_indices(23, 5, seed=1)
    assert len(folds) == 5
    all_val = sorted(index for _, val in folds for index in val)
    assert all_val == list(range(23))


def test_train_and_val_within_a_fold_are_disjoint_and_cover_everything():
    for train, val in kfold_indices(17, 4, seed=7):
        assert set(train) & set(val) == set()
        assert sorted([*train, *val]) == list(range(17))


def test_deterministic_with_the_same_seed():
    assert kfold_indices(20, 4, seed=42) == kfold_indices(20, 4, seed=42)


def test_different_seeds_usually_differ():
    assert kfold_indices(20, 4, seed=1) != kfold_indices(20, 4, seed=2)


def test_rejects_too_few_folds_or_too_few_items():
    with pytest.raises(ValueError):
        kfold_indices(10, 1)
    with pytest.raises(ValueError):
        kfold_indices(2, 5)


def test_kfold_split_applies_indices_to_the_given_items():
    items = list("abcdefghij")
    splits = kfold_split(items, 5, seed=3)
    assert len(splits) == 5
    for train, val in splits:
        assert set(train) | set(val) == set(items)
        assert not set(train) & set(val)
