"""Self-history n-gram draft combiner (port of llama.cpp's common_ngram_simple_draft)."""

from freetoken.scheduler.ngram_draft import ngram_draft


def test_finds_the_continuation_after_a_repeat():
    # The final [7,8,9] also occurred earlier, followed by [1,2,3].
    tokens = [0, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ngram_draft(tokens, 3, 3) == [1, 2, 3]
    assert ngram_draft(tokens, 3, 2) == [1, 2]


def test_no_match_returns_empty():
    assert ngram_draft([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 3, 3) == []


def test_match_at_position_zero_is_ignored():
    # llama.cpp ignores position 0; the final pattern only matches there.
    tokens = [7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert ngram_draft(tokens, 3, 3) == []


def test_too_short_history_is_a_noop():
    assert ngram_draft([1, 2, 3], 3, 3) == []
    assert ngram_draft([], 3, 3) == []