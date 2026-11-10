"""Reject misleading missing-data and scope interpretations without a GPU."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('summary', ROOT / 'scripts/summarize_representative_profiles.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class ProfileSummary(unittest.TestCase):
    def parse(self, content, metrics):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'ncu.csv'
            path.write_text(content)
            return summary.parse_ncu(path, metrics)

    def test_wide_keeps_zero_distinct_from_missing(self):
        rows = self.parse('==PROF== preamble\n"ID","Kernel Name","a.sum","b.sum"\n'
                          '"","","byte","sector"\n"0","kernel","0","n/a"\n',
                          ['a.sum', 'b.sum', 'c.sum'])
        self.assertEqual([r['value'] for r in rows], [0, None, None])
        self.assertEqual(rows[0]['raw_unit'], 'byte')

    def test_long_preserves_each_kernel_and_unavailable_metric(self):
        rows = self.parse('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                          '"0","a","x.sum","byte","1,024"\n'
                          '"1","b","x.sum","byte","N/A"\n', ['x.sum', 'y.sum'])
        values = {(r['kernel_id'], r['metric']): r['value'] for r in rows}
        self.assertEqual(values[('0', 'x.sum')], 1024)
        self.assertIsNone(values[('1', 'x.sum')])
        self.assertIsNone(values[('0', 'y.sum')])

    def test_repair_cannot_replace_another_campaign(self):
        repair = {'campaign_metadata': {'capture_repair': {'parent_campaign_sha256': 'other'}}}
        with patch.object(summary, 'summarize', return_value=repair):
            with self.assertRaisesRegex(ValueError, 'different parent'):
                summary.apply_nsys_repair({'campaign_sha256': 'original'}, Path('repair'), Path('out'))

    def test_repair_cannot_change_measured_binary(self):
        repair = {'campaign_metadata': {'gpu_uuid': 'gpu', 'file_hashes': {'binary': 'changed'},
                  'capture_repair': {'parent_campaign_sha256': 'original'}}}
        original = {'campaign_sha256': 'original', 'campaign_metadata': {
                    'gpu_uuid': 'gpu', 'file_hashes': {'binary': 'frozen'}}}
        with patch.object(summary, 'summarize', return_value=repair):
            with self.assertRaisesRegex(ValueError, 'changed measured inputs'):
                summary.apply_nsys_repair(original, Path('repair'), Path('out'))


if __name__ == '__main__':
    unittest.main()
