"""The streaming client: where the time goes, what counts as a token, and how it fails."""
import json
import unittest

from scripts.lib.model import ModelConfig
from scripts.lib.model import ChatResponse
from scripts.lib.stream import StreamTiming, as_count, parse_stream, stream_chat
from tests.stub_sse import SseStub


def event(content: str = "", reasoning: str = "", finish: str = "") -> bytes:
    delta: dict = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    choice: dict = {"index": 0, "delta": delta}
    if finish:
        choice["finish_reason"] = finish
    return b"data: " + json.dumps({"choices": [choice]}).encode()


def usage_event(completion: int, prompt: int = 50) -> bytes:
    body = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
    return b"data: " + json.dumps(body).encode()


class FakeClock:
    """A clock that reads the next scripted time on every call."""

    def __init__(self, *times: float) -> None:
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0)


class ParseStreamTest(unittest.TestCase):
    def test_time_to_first_token_and_decode_window_are_split(self):
        lines = [event("a"), event("b"), event("c"), usage_event(3), b"data: [DONE]"]
        # started=0; token arrivals at 2.0, 2.5, 3.0; usage event at 3.0; end at 3.1
        clock = FakeClock(2.0, 2.5, 3.0, 3.0, 3.1)
        timing = parse_stream(lines, clock=clock, started=0.0)
        self.assertAlmostEqual(timing.ttft_s, 2.0)
        self.assertAlmostEqual(timing.decode_s, 1.0)
        self.assertEqual(timing.response.completion_tokens, 3)
        self.assertAlmostEqual(timing.decode_tps, 2.0)  # (3 - 1) tokens over 1.0 s

    def test_the_first_token_opens_the_window_instead_of_counting_inside_it(self):
        timing = StreamTiming(response=parse_stream(
            [event("a"), usage_event(101)], clock=FakeClock(1.0, 1.0, 1.0), started=0.0).response,
            ttft_s=1.0, decode_s=10.0)
        self.assertAlmostEqual(timing.decode_tps, 10.0)

    def test_a_server_without_usage_gets_a_chunk_count_and_says_so(self):
        timing = parse_stream([event("a"), event("b")], clock=FakeClock(1.0, 2.0, 2.0), started=0.0)
        self.assertEqual(timing.response.completion_tokens, 2)
        self.assertFalse(timing.usage_reported)

    def test_reported_usage_wins_over_the_chunk_count(self):
        timing = parse_stream([event("ab"), usage_event(40)], clock=FakeClock(1.0, 1.0, 1.0),
                              started=0.0)
        self.assertEqual(timing.response.completion_tokens, 40)
        self.assertTrue(timing.usage_reported)

    def test_content_reasoning_and_finish_reason_are_folded_in(self):
        timing = parse_stream(
            [event(reasoning="think "), event("out"), event(finish="length")],
            clock=FakeClock(1.0, 2.0, 2.0, 2.0), started=0.0)
        self.assertEqual(timing.response.content, "out")
        self.assertEqual(timing.response.reasoning, "think")
        self.assertEqual(timing.response.finish_reason, "length")

    def test_blank_comment_and_broken_lines_are_skipped(self):
        lines = [b"", b": keep-alive", b"data: {not json", b"data: [1, 2]", event("a")]
        timing = parse_stream(lines, clock=FakeClock(1.0, 1.0), started=0.0)
        self.assertEqual(timing.chunks, 1)

    def test_an_empty_stream_has_no_window_and_no_rate(self):
        timing = parse_stream([], clock=FakeClock(5.0), started=0.0)
        self.assertEqual((timing.chunks, timing.decode_s, timing.decode_tps), (0, 0.0, 0.0))


class AsCountTest(unittest.TestCase):
    def test_numbers_and_numeric_strings_become_counts(self):
        self.assertEqual((as_count(7), as_count(7.9), as_count("12")), (7, 7, 12))

    def test_anything_else_is_zero(self):
        for junk in (None, "n/a", [], {}, True, -5):
            self.assertEqual(as_count(junk), 0, junk)

    def test_a_server_that_sends_junk_for_prompt_tokens_reads_as_zero(self):
        payload = {"choices": [], "usage": {"prompt_tokens": "lots", "completion_tokens": 3}}
        timing = StreamTiming(response=ChatResponse(status=200, payload=payload))
        self.assertEqual(timing.prompt_tokens, 0)


class StreamChatTest(unittest.TestCase):
    def config(self, stub: SseStub, limit: int = 40) -> ModelConfig:
        return ModelConfig(name="m", endpoint=stub.endpoint, token_limit=limit, timeout_s=10)

    def test_a_live_stream_reports_prefill_apart_from_decode(self):
        with SseStub(prefill_s_per_ktok=0.0, token_s=0.005) as stub:
            timing = stream_chat(self.config(stub), [{"role": "user", "content": "x" * 400}],
                                 max_tokens=40, temperature=0.0)
        self.assertTrue(timing.response.ok)
        self.assertEqual(timing.response.completion_tokens, 40)
        self.assertEqual(timing.prompt_tokens, 100)
        self.assertGreater(timing.decode_s, 0.15)
        self.assertTrue(0 < timing.decode_tps < 250)

    def test_the_request_asks_for_a_stream_with_usage(self):
        with SseStub() as stub:
            stream_chat(self.config(stub), [{"role": "user", "content": "hi"}], max_tokens=4)
        self.assertTrue(stub.seen[0]["stream"])
        self.assertEqual(stub.seen[0]["stream_options"], {"include_usage": True})

    def test_a_prefill_delay_lands_in_time_to_first_token_only(self):
        with SseStub(prefill_s_per_ktok=0.5, token_s=0.0) as stub:
            timing = stream_chat(self.config(stub), [{"role": "user", "content": "x" * 4000}],
                                 max_tokens=8)
        self.assertGreater(timing.ttft_s, 0.4)
        self.assertLess(timing.decode_s, timing.ttft_s)

    def test_an_http_error_comes_back_as_a_failed_response(self):
        with SseStub(status=500) as stub:
            timing = stream_chat(self.config(stub), [{"role": "user", "content": "hi"}],
                                 max_tokens=4)
        self.assertFalse(timing.response.ok)
        self.assertIn("HTTP 500", timing.response.error or "")
        self.assertEqual(timing.decode_tps, 0.0)

    def test_a_dead_endpoint_comes_back_as_a_failed_response(self):
        cfg = ModelConfig(name="m", endpoint="http://127.0.0.1:1/v1", timeout_s=2)
        timing = stream_chat(cfg, [{"role": "user", "content": "hi"}], max_tokens=4)
        self.assertFalse(timing.response.ok)
        self.assertTrue(timing.response.error)


if __name__ == "__main__":
    unittest.main()
