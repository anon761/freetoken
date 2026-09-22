"""DetokenizeManager incremental decoding, including a multi-msg batch for one uid.

A speculative round delivers several ``DetokenizeMsg`` for the same uid in ONE batch
(accepted drafts + the bonus); the final one may be a dropped EOS. The manager must
decode them incrementally with no text duplication and emit nothing for the EOS.
"""
from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager


class _FakeTokenizer:
    eos_token_id = 1
    _MAP = {11111: " Paris", 16: ".", 1: "<eos>", 455: " The", 4087: " final"}

    def decode(self, ids):
        return "".join(self._MAP.get(int(i), f"<{i}>") for i in ids)

    def batch_decode(self, batch):
        return [self.decode(ids) for ids in batch]


def _m(tok: int, finished: bool = False) -> DetokenizeMsg:
    return DetokenizeMsg(
        uid=0, next_token=tok, finished=finished,
        finish_reason=("stop" if finished else None), matched_stop=None, stop_strs=None,
    )


def test_multi_msg_round_no_duplication_and_dropped_eos():
    dm = DetokenizeManager(_FakeTokenizer(), frozenset({1}))
    # prefill publishes the pending token in its own batch
    assert dm.detokenize([_m(11111)]) == [" Paris"]
    # the round publishes d1, d2 and the bonus EOS in ONE batch
    assert dm.detokenize([_m(16), _m(455), _m(1, True)]) == [".", " The", ""]


def test_multi_msg_single_batch_accumulates_once():
    dm = DetokenizeManager(_FakeTokenizer(), frozenset({1}))
    assert dm.detokenize([_m(11111), _m(16), _m(455)]) == [" Paris", ".", " The"]


def test_dropped_eos_yields_empty_not_repeat():
    dm = DetokenizeManager(_FakeTokenizer(), frozenset({1}))
    dm.detokenize([_m(16)])
    assert dm.detokenize([_m(1, True)]) == [""]
