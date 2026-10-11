import pytest

from openjev_ja.methods.d1.gguf import DEFAULT_SERVER, RELEASES, D1GgufScorer


def test_releases_are_pinned_to_a_full_hex_revision():
    for repo_id, revision, filename in RELEASES.values():
        assert repo_id.startswith("LiquidAI/d1-") and repo_id.endswith("-GGUF") and filename.endswith(".gguf")
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def test_scorer_validates_arguments_before_loading_anything():
    assert "b11541" in DEFAULT_SERVER
    with pytest.raises(ValueError, match="primitive"):
        D1GgufScorer("q8_0", primitive="rank")
    with pytest.raises(ValueError, match="variant"):
        D1GgufScorer("q1", primitive="choice")
    with pytest.raises(RuntimeError, match="llama-server not found"):
        D1GgufScorer("q8_0", primitive="choice", server_path="does/not/exist.exe")
