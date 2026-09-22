"""Self-speculative n-gram draft (port of llama.cpp's ``common_ngram_simple_draft``).

Looks up the last ``n`` tokens (ending at the pending token) earlier in the request's own
history; if found, the following up to ``m`` tokens become the draft. Captures repetitions
(code, quotes, structured output) at zero model cost -- combined with the MTP draft chain
by overriding it when a match fires.
"""

from __future__ import annotations

from typing import List, Sequence


def ngram_draft(tokens: Sequence[int], ngram_size: int, draft_len: int) -> List[int]:
    """Draft tokens following the most recent earlier occurrence of the last ``ngram_size``
    tokens of ``tokens``. ``tokens`` includes the pending token as its last element.
    Returns up to ``draft_len`` tokens, or [] when there is no match."""
    cur = len(tokens)
    if ngram_size <= 0 or draft_len <= 0 or cur <= ngram_size + draft_len:
        return []
    pattern = list(tokens[cur - ngram_size:])
    # Search backwards, skipping the current match at the end.
    for start in range(cur - ngram_size - 1, 0, -1):
        if list(tokens[start:start + ngram_size]) == pattern:
            copy_max = min(draft_len, cur - (start + ngram_size))
            if copy_max < 1:
                continue
            return [int(t) for t in tokens[start + ngram_size:start + ngram_size + copy_max]]
    return []


__all__ = ["ngram_draft"]