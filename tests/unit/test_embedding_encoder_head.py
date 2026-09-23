from openjev_ja.methods.embedding.encoder_head import _token_index_for_char, build_candidate_prompt


def test_build_candidate_prompt_labels_span_their_own_line() -> None:
    text, spans = build_candidate_prompt("質問文", ["甲", "乙", "丙"])
    labels = ["A", "B", "C"]
    for label, (start, end) in zip(labels, spans, strict=True):
        assert text[start:end] == label


def test_token_index_for_char_finds_covering_token() -> None:
    offsets = [(0, 0), (0, 1), (1, 3), (3, 4), (0, 0)]
    assert _token_index_for_char(offsets, 0) == 1
    assert _token_index_for_char(offsets, 2) == 2
    assert _token_index_for_char(offsets, 3) == 3


def test_token_index_for_char_skips_empty_special_token_offsets() -> None:
    offsets = [(0, 0), (0, 2), (2, 5)]
    assert _token_index_for_char(offsets, 0) == 1
