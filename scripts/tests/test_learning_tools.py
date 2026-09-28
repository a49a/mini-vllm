import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from trace_replay import render
from workload_matrix import percentile

class LearningTools(unittest.TestCase):
    def test_trace_cannot_inject_script_through_request_id(self):
        html=render([{'schema_version':1,'event':'queued','request_id':'</script><script>alert(1)</script>'}])
        self.assertNotIn('</script><script>alert(1)',html)
        self.assertIn('\\u003c/script>',html)
        self.assertIn('type="range"',html)
    def test_unknown_schema_is_rejected(self):
        with self.assertRaises(ValueError): render([{'schema_version':99}])
    def test_percentiles_interpolate_and_handle_empty_samples(self):
        self.assertEqual(percentile([0,100],95),95)
        self.assertEqual(percentile([12],99),12)
        self.assertIsNone(percentile([],99))
