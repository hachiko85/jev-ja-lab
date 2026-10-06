import pytest

from openjev_ja.methods.clef.scorer import PRIMITIVES, QUANTIZATIONS, RELEASES, ClefScorer


def test_releases_are_pinned_to_a_full_hex_revision():
    for repo_id, revision in RELEASES.values():
        assert "/" in repo_id
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def test_supported_primitives_and_quantizations():
    assert PRIMITIVES == ("noul", "choice", "score")
    assert QUANTIZATIONS == ("none", "8bit", "4bit")


def test_unsupported_primitive_is_rejected_before_loading_anything():
    with pytest.raises(ValueError, match="primitive"):
        ClefScorer("flash", primitive="rank")


def test_unknown_variant_and_quantization_are_rejected_before_loading_anything():
    with pytest.raises(ValueError, match="variant"):
        ClefScorer("huge", primitive="choice")
    with pytest.raises(ValueError, match="quantization"):
        ClefScorer("flash", primitive="choice", quantization="2bit")


def test_gguf_releases_are_pinned_and_scorer_validates_arguments():
    from openjev_ja.methods.clef.gguf import RELEASES as GGUF_RELEASES
    from openjev_ja.methods.clef.gguf import ClefGgufScorer

    for repo_id, revision, filename in GGUF_RELEASES.values():
        assert "/" in repo_id and filename.endswith(".gguf")
        assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    with pytest.raises(ValueError, match="primitive"):
        ClefGgufScorer("q4_k_m", primitive="rank")
    with pytest.raises(ValueError, match="variant"):
        ClefGgufScorer("q1", primitive="choice")
    with pytest.raises(RuntimeError, match="llama-server not found"):
        ClefGgufScorer("q4_k_m", primitive="choice", server_path="does/not/exist.exe")
