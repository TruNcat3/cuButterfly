import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_interrupted_async_handoff as audit
import run_semantic_campaign as runner


class OrphanEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.row = {'id': 'case', 'descriptor': {'semantics': {'operator': 0}},
                    'status': 'runner-error', 'error': 'MemoryError', 'baselines': []}
        runner.write_case_checkpoint(self.root, self.row)
        self.cases = {'case': {'id': 'case', 'descriptor': self.row['descriptor']}}

    def test_orphan_is_retained_but_never_adopted(self):
        records = audit.orphan_evidence(self.root, set(), self.cases)
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]['adopted'])
        self.assertEqual(records[0]['status'], 'runner-error')

    def test_indexed_row_is_not_an_orphan(self):
        self.assertEqual(audit.orphan_evidence(self.root, {'case'}, self.cases), [])

    def test_changed_orphan_checksum_rejected(self):
        path = self.root / 'rows/case.json'
        envelope = json.loads(path.read_text())
        envelope['row']['status'] = 'pass'
        path.write_text(json.dumps(envelope))
        with self.assertRaisesRegex(ValueError, 'orphan checksum'):
            audit.orphan_evidence(self.root, set(), self.cases)

    def test_changed_orphan_descriptor_rejected(self):
        self.cases['case']['descriptor'] = {'semantics': {'operator': 1}}
        with self.assertRaisesRegex(ValueError, 'orphan checksum/descriptor'):
            audit.orphan_evidence(self.root, set(), self.cases)

    def test_commit_after_compiler_failure_rejected(self):
        path = self.root / 'queue.log'
        path.write_text('could not start specialization compiler\n' +
                        json.dumps({'status': 'pass', 'completed': 1, 'case': 'case'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'adoption unsafe'):
            audit.check_compiler_failure_boundary(path)

    def test_compiler_failure_without_success_is_safe(self):
        path = self.root / 'queue.log'
        path.write_text(json.dumps({'status': 'pass', 'completed': 1}) + '\n' +
                        'could not start specialization compiler\nMemoryError\n')
        self.assertTrue(audit.check_compiler_failure_boundary(path)['adoption_safe'])


if __name__ == '__main__':
    unittest.main()
