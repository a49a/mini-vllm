import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_kv import distribution, trial_order


class CompareTests(unittest.TestCase):
    def test_trace_pairs_alternate_and_cover_each_mode(self):
        rows = list(trial_order(['paged', 'prefix'], 3, False, True))
        self.assertEqual(len(rows), 12)
        for i in range(0, len(rows), 2):
            self.assertEqual(rows[i][:2], rows[i + 1][:2])
            self.assertNotEqual(rows[i][2], rows[i + 1][2])
        self.assertEqual(rows[:4], [('paged', 0, False), ('paged', 0, True),
                                   ('prefix', 0, True), ('prefix', 0, False)])
        self.assertEqual(list(trial_order(['paged'], 2, True, False)),
                         [('paged', 0, True), ('paged', 1, True)])

    def test_nearest_rank_distributions_and_empty_samples(self):
        self.assertEqual(distribution(range(1, 101)), dict(count=100, p50=50, p95=95, p99=99))
        self.assertEqual(distribution([]), dict(count=0, p50=None, p95=None, p99=None))
        self.assertEqual(distribution([9])['p99'], 9)
