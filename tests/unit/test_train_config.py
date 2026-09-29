import pytest

from openjev_ja.train.config import TrainingConfigError, plan_training


def _config(**overrides):
    base = {
        "version": 1,
        "runtime": {"folds": 5, "epochs": 3, "lr": 1e-3, "device": "cpu"},
        "corpus": {"format": "eikos", "lang": "Japanese", "paths": ["a.jsonl"]},
        "models": [
            {
                "id": "m-noul", "architecture": "laya-bert", "repo_id": "some/bert",
                "revision": "abc123", "primitive": "noul", "head_layers": 2,
            },
        ],
    }
    base.update(overrides)
    return base


def test_plan_builds_one_job_per_model_with_runtime_defaults():
    jobs = plan_training(_config())
    assert len(jobs) == 1
    job = jobs[0]
    assert job.model_id == "m-noul"
    assert job.architecture == "laya-bert"
    assert job.model_name == "some/bert"
    assert job.revision == "abc123"
    assert job.primitive == "noul"
    assert job.corpus_paths == ["a.jsonl"]
    assert job.lang == "Japanese"
    assert job.folds == 5 and job.epochs == 3 and job.device == "cpu"
    assert job.build_kwargs == {"head_layers": 2}


def test_per_model_overrides_win_over_runtime_defaults():
    config = _config()
    config["models"][0]["folds"] = 3
    config["models"][0]["device"] = "cuda:0"
    job = plan_training(config)[0]
    assert job.folds == 3
    assert job.device == "cuda:0"


def test_per_model_corpus_overrides_merge_with_the_default_corpus():
    config = _config()
    config["models"][0]["corpus"] = {"paths": ["b.jsonl"], "lang": None}
    job = plan_training(config)[0]
    assert job.corpus_paths == ["b.jsonl"]
    assert job.lang is None
    assert job.corpus_format == "eikos"  # inherited from the default corpus block


def test_missing_models_or_required_fields_are_rejected():
    with pytest.raises(TrainingConfigError):
        plan_training(_config(models=[]))
    bad = _config()
    del bad["models"][0]["primitive"]
    with pytest.raises(TrainingConfigError):
        plan_training(bad)


def test_model_without_any_corpus_paths_is_rejected():
    config = _config(corpus={})
    with pytest.raises(TrainingConfigError):
        plan_training(config)
