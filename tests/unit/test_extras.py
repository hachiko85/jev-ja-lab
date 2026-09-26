import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# scorer name in eval/orchestrate.py -> the pip extra that installs what it needs.
# A new library (scorer) must be added here AND get an extra in pyproject.toml.
SCORER_EXTRAS = {
    "jev": "jev",
    "embedding": "embedding",
    "laya": "laya",
    "jevlike": "jevlike",
    "laya-bert": "laya-bert",
    "nli-cross-encoder": "openjev",
    "hopper": "hopper",
    "decider": "decider",
    "clm": "clm",
    "semif-logit": "semif",
    "qwen-direct": "semif-ja",
    "masked-lm": "bert",
}
# extras that are not a library
BUILDING_BLOCKS = {"eval", "viz", "orchestrate", "all", "dev"}


def _extras() -> dict[str, list[str]]:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def test_every_scorer_has_an_install_extra():
    source = (ROOT / "src/openjev_ja/eval/orchestrate.py").read_text(encoding="utf-8")
    scorers = set(re.findall(r'scorer_name (?:==|in \(None, "auto"\) or ==)? ?"([\w-]+)"', source))
    scorers |= set(re.findall(r'elif scorer_name == "([\w-]+)"', source))
    assert scorers, "no scorers found in orchestrate.py"
    missing = scorers - set(SCORER_EXTRAS)
    assert not missing, f"scorers without an extra mapping (add one in pyproject.toml): {missing}"
    extras = _extras()
    for scorer, extra in SCORER_EXTRAS.items():
        assert extra in extras, f"{scorer}: extra {extra!r} missing from pyproject.toml"


def test_library_extras_are_self_sufficient_and_listed_in_all():
    extras = _extras()
    libraries = set(SCORER_EXTRAS.values())
    assert set(extras) == libraries | BUILDING_BLOCKS
    listed = set(re.search(r"\[(.*)\]", extras["all"][0]).group(1).split(","))
    assert listed == libraries, f"extra 'all' must list exactly the libraries: {libraries ^ listed}"
    for library in libraries:
        joined = " ".join(extras[library])
        assert "jev-ja-lab[" in joined, f"{library} does not pull in the shared runtime"
