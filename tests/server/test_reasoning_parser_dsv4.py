"""The deepseekv32 reasoning parser (server/reasoning_parser.py), whose DSML tool-call markup
makes it the only family that ends reasoning on something other than ``</think>``. The other
families are covered in test_reasoning_parser_all_models.py."""
from __future__ import annotations

import pytest

from freetoken.server.reasoning_parser import (
    DSML_TOKEN,
    ReasoningParser,
    SUPPORTED_REASONING_PARSERS,
    strip_special_tokens,
    BOS_TOKEN,
    EOS_TOKEN,
)

TC_OPEN = f"<{DSML_TOKEN}tool_calls>"
TC_CLOSE = f"</{DSML_TOKEN}tool_calls>"
TOOL_BLOCK = (
    f"{TC_OPEN}\n"
    f'<{DSML_TOKEN}invoke name="get_weather">\n'
    f'<{DSML_TOKEN}parameter name="city" string="true">Paris</{DSML_TOKEN}parameter>\n'
    f"</{DSML_TOKEN}invoke>\n"
    f"{TC_CLOSE}"
)


def _stream(parser: ReasoningParser, chunks):
    """Feed chunks through the streaming API (incl. the end-of-stream flush);
    return (reasoning, content)."""
    reasoning, content = "", ""
    for chunk in chunks:
        r, c = parser.parse_stream_chunk(chunk)
        reasoning += r
        content += c
    fr, fc = parser.flush()
    reasoning += fr
    content += fc
    return reasoning, content


# ----------------------------------------------------------------- non-stream
def test_thinking_with_end_token():
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = parser.parse_non_stream("I should answer.</think>The answer is 42.")
    assert reasoning == "I should answer."
    assert content == "The answer is 42."


def test_thinking_with_tool_block_after_end_token():
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    text = f"Let me check the weather.</think>Sure!\n\n{TOOL_BLOCK}"
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "Let me check the weather."
    # Content keeps the tool block for the function-call parser.
    assert content.startswith("Sure!")
    assert TC_OPEN in content


def test_missing_end_token_with_tool_block():
    # dsv4 sometimes skips </think> and jumps straight to the DSML block.
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    text = f"I will call the tool.\n\n{TOOL_BLOCK}"
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == "I will call the tool."
    assert content.startswith(TC_OPEN)
    assert TC_CLOSE in content


def test_chat_mode_no_reasoning():
    # force_reasoning=False (chat mode); output is pure content.
    parser = ReasoningParser("deepseekv32", force_reasoning=False)
    reasoning, content = parser.parse_non_stream("Just a direct answer.")
    assert reasoning == ""
    assert content == "Just a direct answer."


def test_chat_mode_with_tool_block_is_content_not_reasoning():
    # In chat mode, text before a tool block is real content, not reasoning.
    parser = ReasoningParser("deepseekv32", force_reasoning=False)
    text = f"Calling now.\n\n{TOOL_BLOCK}"
    reasoning, content = parser.parse_non_stream(text)
    assert reasoning == ""
    assert content.startswith("Calling now.")
    assert TC_OPEN in content


def test_truncated_reasoning_no_end_token():
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = parser.parse_non_stream("still thinking and never finished")
    assert reasoning == "still thinking and never finished"
    assert content == ""


def test_explicit_think_start_in_chat_mode():
    # Defensive: an explicit <think> turns reasoning on even when not forced.
    parser = ReasoningParser("deepseekv32", force_reasoning=False)
    reasoning, content = parser.parse_non_stream("<think>hmm</think>done")
    assert reasoning == "hmm"
    assert content == "done"


# --------------------------------------------------------------------- stream
def test_stream_thinking_with_end_token():
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(parser, ["I should ", "answer.", "</think>", "The ", "answer."])
    assert reasoning == "I should answer."
    assert content == "The answer."


def test_stream_end_token_split_across_chunks():
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(parser, ["reason", "</th", "ink>", "done"])
    assert reasoning == "reason"
    assert content == "done"


def test_stream_missing_end_token_tool_interrupt():
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(parser, ["I will call ", "the tool", TC_OPEN, "rest"])
    assert reasoning == "I will call the tool"
    assert content.startswith(TC_OPEN)
    assert content.endswith("rest")


def test_stream_chat_mode_streams_content():
    parser = ReasoningParser("deepseekv32", force_reasoning=False)
    reasoning, content = _stream(parser, ["Hello ", "world"])
    assert reasoning == ""
    assert content == "Hello world"


def test_stream_dsml_literal_in_reasoning_then_end_token():
    # The model quotes the ｜DSML｜ marker inside reasoning and closes </think>
    # only later: streaming must NOT end reasoning at the literal (it defers to
    # </think>), matching the non-streaming path.
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        ["I should emit a ", TC_OPEN, " block, but first more thought.", "</think>", "Answer."],
    )
    assert reasoning == f"I should emit a {TC_OPEN} block, but first more thought."
    assert content == "Answer."


def test_stream_flush_recovers_trailing_partial_content():
    # A trailing '<' (prefix of a tracked token) must be flushed, not dropped.
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(parser, ["thinking", "</think>", "value is x ", "<"])
    assert reasoning == "thinking"
    assert content == "value is x <"


def test_stream_flush_recovers_truncated_reasoning():
    # Generation stops mid-marker (e.g. max_tokens) -> residue stays reasoning.
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(parser, ["reasoning text", "</th"])
    assert reasoning == "reasoning text</th"
    assert content == ""


def test_stream_skip_end_token_tool_block_flushed_to_content():
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(
        parser, ["reason", "\n\n", TC_OPEN, "body", f"</{DSML_TOKEN}tool_calls>"]
    )
    assert reasoning.strip() == "reason"
    assert content.startswith(TC_OPEN)
    assert content.endswith(f"</{DSML_TOKEN}tool_calls>")


def test_stream_skip_end_token_split_leading_marker_char():
    # Realistic tokenization glues the marker's leading '<' to preceding text
    # (".<"), then emits "｜DSML｜tool_calls>" separately. The held-partial logic
    # must reassemble the marker so the tool block goes to content, not reasoning.
    parser = ReasoningParser("deepseekv32", force_reasoning=True)
    reasoning, content = _stream(
        parser,
        ["I will look it up", ".<", f"{DSML_TOKEN}tool_calls>", "\nbody", f"</{DSML_TOKEN}tool_calls>"],
    )
    assert reasoning == "I will look it up."
    assert content.startswith(TC_OPEN)
    assert content.endswith(f"</{DSML_TOKEN}tool_calls>")


def test_stream_matches_non_stream_under_split_leading_char():
    # Same full text, two deliveries: as one blob (non-stream) and with the
    # marker's leading '<' glued to the preceding char. Results must agree.
    full = f"thinking it through.{TC_OPEN}\n{'x'}\n</{DSML_TOKEN}tool_calls>"
    ns_reasoning, ns_content = ReasoningParser("deepseekv32", force_reasoning=True).parse_non_stream(full)
    s_reasoning, s_content = _stream(
        ReasoningParser("deepseekv32", force_reasoning=True),
        ["thinking it through", ".<", full[len("thinking it through.<"):]],
    )
    assert s_reasoning == ns_reasoning
    assert s_content == ns_content


# --------------------------------------------------------------- misc / utils
def test_unknown_parser_raises():
    with pytest.raises(ValueError):
        ReasoningParser("nope")


def test_registry_lists_deepseekv32():
    assert "deepseekv32" in SUPPORTED_REASONING_PARSERS


def test_strip_special_tokens():
    text = f"{BOS_TOKEN}Answer.{EOS_TOKEN}"
    assert strip_special_tokens(text, [BOS_TOKEN, EOS_TOKEN]) == "Answer."
    assert strip_special_tokens("clean", []) == "clean"
    assert strip_special_tokens("keep", ["", BOS_TOKEN]) == "keep"
    assert strip_special_tokens("", [BOS_TOKEN]) == ""


# ------------------------------------------- parse-side wiring (always-think templates)
def _profile_state(toggleable, default_state, reasoning_parser="deepseekv32"):
    """Fake state exposing a frontend tokenizer whose probe says whether the
    chat template reacts to the thinking toggle and what a bare render does."""
    import asyncio
    from types import SimpleNamespace

    from freetoken.core import SamplingParams
    from freetoken.server.generation import GenSpec
    from freetoken.tokenizer.effort import EffortProfile, ThinkingProfile

    class _Manager:
        def thinking_profile(self):
            return ThinkingProfile(
                efforts=EffortProfile(supported=frozenset(), default=None, consumes_effort=False),
                toggleable=toggleable,
                has_adaptive=False,
                default_state=default_state,
            )

    state = SimpleNamespace(
        config=SimpleNamespace(reasoning_parser=reasoning_parser, tool_call_parser="llama3"),
        frontend_tokenizer=lambda: _Manager(),
    )
    spec = GenSpec(messages=[], sampling_params=SamplingParams())
    return asyncio, spec, state


def test_always_think_template_forces_reasoning_in_chat_mode():
    # DeepSeek-V4.1-Flash serves through a Jinja chat template that unconditionally
    # opens ersive (thinking-only checkpoint; the probe sees no toggle reaction and
    # an already-thinking bare render). Generation then starts inside the reasoning
    # block in EVERY mode — the mode-based attribution (the native dsv4 encoder's
    # contract) would mislabel the whole chain-of-thought as content.
    from freetoken.server.generation import _make_reasoning_parser_async

    asyncio, spec, state = _profile_state(toggleable=False, default_state="on")
    parser = asyncio.run(_make_reasoning_parser_async(spec, state))
    assert parser is not None and parser.detector.force_reasoning is True
    # The split itself: markerless reasoning text (the template opened the block)
    # must surface as reasoning, and text after the closer as content.
    reasoning, content = parser.parse_non_stream("thinking hard.The answer.")
    assert reasoning == "thinking hard."
    assert content == "The answer."


def test_toggleable_profile_keeps_mode_contract():
    # Native dsv4 encoder checkpoints react to the toggle: a bare (chat-mode)
    # render contains no ersive and the model opens the block itself, so leading
    # text stays content unless the request resolves to thinking mode.
    from freetoken.server.generation import _make_reasoning_parser_async

    asyncio, spec, state = _profile_state(toggleable=True, default_state="off")
    parser = asyncio.run(_make_reasoning_parser_async(spec, state))
    assert parser is not None and parser.detector.force_reasoning is False


def test_profile_probe_failure_falls_back_to_mode_contract():
    from freetoken.server.generation import _make_reasoning_parser_async

    asyncio, spec, state = _profile_state(toggleable=False, default_state="on")

    def _boom():
        raise RuntimeError("tokenizer unavailable")

    state.frontend_tokenizer = _boom
    parser = asyncio.run(_make_reasoning_parser_async(spec, state))
    assert parser is not None and parser.detector.force_reasoning is False
