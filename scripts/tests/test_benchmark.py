import io
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import consume_sse


class StreamingProtocol(unittest.TestCase):
    def test_uses_model_usage_and_times_empty_deltas(self):
        response = io.BytesIO(b'data: {"choices":[{"text":"","finish_reason":null}]}\n\n'
                              b'data: {"choices":[{"text":"hello","finish_reason":null}]}\n\n'
                              b'data: {"choices":[{"text":"","finish_reason":"length"}],"usage":{"completion_tokens":7}}\n\n'
                              b'data: [DONE]\n\n')
        clock = iter([1., 2.])
        result = consume_sse(response, 0., lambda: next(clock))
        self.assertEqual(result['tokens'], 7)
        self.assertEqual(result['ttft'], 1.)
        self.assertEqual(result['intervals'], [1.])

    def test_incomplete_or_error_stream_never_counts_as_success(self):
        for payload in [b'data: [DONE]\n', b'', b'data: {"error":{"message":"cancelled"}}\n']:
            with self.assertRaises(RuntimeError):
                consume_sse(io.BytesIO(payload), 0.)


if __name__ == '__main__':
    unittest.main()
