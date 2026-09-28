import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sustained_load import run, summarize


class SustainedLoad(unittest.TestCase):
    def args(self, **overrides):
        values = dict(host='127.0.0.1', port=8000, arrival_rate=100, duration=.06,
                      concurrency=1, long_repeat=2, short_output=1, long_output=3,
                      sample_interval=.01, server_pid=None, timeout=1, warmup=0)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_fixed_arrivals_account_for_client_saturation_and_metrics(self):
        def slow(*args, **kwargs):
            time.sleep(.08)
            return dict(ttft=.01, intervals=[.01], tokens=2, latency=.08, finish='length')
        report = run(self.args(), request=slow, metrics=lambda *a: {'requests_waiting': 1})
        summary = report['summary']
        self.assertEqual(summary['scheduled'], 6)
        self.assertGreater(summary['successful'], 0)
        self.assertGreater(summary['client_dropped'], 0)
        self.assertEqual(summary['scheduled'], summary['successful'] + summary['client_dropped'])
        self.assertTrue(report['samples'])
        for row in report['requests']:
            if row['status'] == 'success':
                self.assertGreaterEqual(row['scheduled_latency'], row['latency'])
                self.assertGreaterEqual(row['scheduled_ttft'], row['ttft'])

    def test_errors_are_not_counted_as_success_and_probe_failures_are_visible(self):
        def fail(*args, **kwargs):
            raise RuntimeError('HTTP 503')
        report = run(self.args(arrival_rate=10, duration=.01), request=fail, metrics=fail)
        summary = report['summary']
        self.assertEqual(summary['successful'], 0)
        self.assertEqual(summary['server_or_transport_errors'], 1)
        self.assertEqual(summary['failure_fraction'], 1)
        self.assertEqual(summary['scheduled_ttft_ms']['count'], 0)
        self.assertTrue(all('metrics_error' in s for s in report['samples']))

    def test_percentiles_include_count_and_scheduled_delays(self):
        rows = [dict(status='success', tokens=1, latency=i, scheduled_latency=i + 2,
                     ttft=i, scheduled_ttft=i + 2, intervals=[], dispatch_lag=2)
                for i in range(1, 101)]
        summary = summarize(rows, 120, 100)
        self.assertEqual(summary['latency_ms'], dict(count=100, p50=50000, p95=95000, p99=99000))
        self.assertEqual(summary['scheduled_latency_ms']['p99'], 101000)
        self.assertAlmostEqual(summary['tokens_per_second_including_drain'], 100/120)


if __name__ == '__main__':
    unittest.main()
