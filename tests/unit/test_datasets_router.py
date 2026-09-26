import re
from pathlib import Path

import pytest
import yaml

from openjev_ja import datasets_router as router

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "datasets_router" / "manifest.json"


@pytest.fixture(scope="module")
def manifest():
    return router.load_manifest(MANIFEST_PATH)


def test_subsets_only_reference_known_datasets(manifest):
    known = {entry["id"] for entry in manifest["datasets"]}
    for subset in router.SUBSETS:
        assert set(manifest["subsets"][subset]) <= known
    assert set(manifest["subsets"]["all"]) == known


def test_task_subsets_match_dataset_tasks(manifest):
    for entry in manifest["datasets"]:
        for task in ("noul", "choice", "score"):
            listed = entry["id"] in manifest["subsets"][task]
            assert listed == (task in entry["tasks"]), entry["id"]


def test_every_entry_documents_origin_split_and_license(manifest):
    for entry in manifest["datasets"] + manifest["excluded"]:
        assert entry["license"] and entry["url"].startswith("https://"), entry["id"]
        assert entry["source"]["kind"] in router.HANDLERS, entry["id"]
        assert "split" in entry["source"], entry["id"]
    for entry in manifest["datasets"]:
        source = entry["source"]
        if source["kind"] in ("git", "hf_dataset", "hf_snapshot", "hf_parquet"):
            assert re.fullmatch(r"[0-9a-f]{40}", source["revision"]), entry["id"]


def test_gated_datasets_are_not_routed_unless_asked(manifest):
    assert all(entry["access"] == "open" for entry in router.select(manifest, "all"))
    routed = router.select(manifest, "choice", include_gated=True)
    assert "jgpqa_diamond" in {entry["id"] for entry in routed}
    assert "jgpqa_diamond" not in {entry["id"] for entry in router.select(manifest, "choice")}


MIRRORABLE = {"MIT", "Apache-2.0", "CC BY 4.0", "CC BY-SA 4.0"}


def test_only_licenses_that_fit_cc_by_sa_are_mirrored(manifest):
    mirrored = [e for e in manifest["datasets"] if e["distribution"] == "mirror"]
    routed = {e["id"] for e in manifest["datasets"] if e["distribution"] == "router"}
    assert {e["license"] for e in mirrored} <= MIRRORABLE
    assert routed == {"jmmlu", "wrime", "paws_x_ja", "textdetox_ja"}
    assert all(e["distribution"] == "router" for e in manifest["excluded"])


def test_repo_license_is_cc_by_sa_because_nothing_more_restrictive_is_mirrored(manifest):
    assert manifest["license"] == "cc-by-sa-4.0"
    mirrored = [e for e in manifest["datasets"] if e["distribution"] == "mirror"]
    mirrored_licenses = {e["license"] for e in mirrored}
    assert "CC BY-SA 4.0" in mirrored_licenses
    assert not any("NC" in license_ or "ND" in license_ for license_ in mirrored_licenses)


def test_own_data_is_stored_in_the_repo(manifest):
    own = {e["id"] for e in manifest["datasets"] if e["source"]["kind"] == "own"}
    assert own == {"synthetic_score", "helpsteer2_ja_benchmark_v1"}
    assert all(e["distribution"] == "mirror" for e in manifest["datasets"] if e["id"] in own)


def test_profile_datasets_exist_in_the_eval_configs(manifest):
    config_ids = set()
    for name in ("bert-series.yaml", "helpsteer-series.yaml"):
        config = yaml.safe_load((ROOT / "configs/eval" / name).read_text(encoding="utf-8"))
        config_ids |= {dataset["id"] for dataset in config["datasets"]}
    for entry in manifest["datasets"] + manifest["excluded"]:
        assert set(entry["profile_datasets"]) <= config_ids, entry["id"]


def test_select_rejects_unknown_subset(manifest):
    with pytest.raises(router.RouterError):
        router.select(manifest, "everything")


def test_catalog_rows_describe_every_dataset_of_the_subset(manifest):
    rows = router.catalog_rows(manifest, "score")
    assert [row["id"] for row in rows] == manifest["subsets"]["score"]
    assert all(row["license"] and row["source_kind"] and row["viewer_url"] for row in rows)


def test_router_catalog_lists_only_the_datasets_that_are_not_mirrored(manifest):
    rows = router.catalog_rows(manifest, "router")
    assert {row["id"] for row in rows} == {
        "jmmlu", "wrime", "paws_x_ja", "textdetox_ja", "jgpqa_diamond",
    }
    assert all(row["distribution"] == "router" for row in rows)


def test_unsupported_manifest_version_is_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"version": 2}', encoding="utf-8")
    with pytest.raises(router.RouterError):
        router.load_manifest(path)
