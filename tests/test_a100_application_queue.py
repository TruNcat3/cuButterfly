"""Regression for an inherited exhaustive budget in the finite application lane."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('a100_application_queue',
    ROOT / 'results/a100_closure_20261102/applications/run_queue.py')
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class FiniteSelectionEnvironment(unittest.TestCase):
    def test_inherited_exhaustive_budget_cannot_change_historical_protocol(self):
        with patch.dict(queue.os.environ, {'CUBUTTERFLY_SEARCH_BUDGET': '0',
                                         'OMP_NUM_THREADS': '64', 'MKL_NUM_THREADS': '64'}):
            result = queue.selection_environment()
        self.assertEqual(result['CUBUTTERFLY_SEARCH_BUDGET'], '4')
        self.assertEqual(result['OMP_NUM_THREADS'], '8')
        self.assertEqual(result['MKL_NUM_THREADS'], '8')
        self.assertEqual(result['CUBUTTERFLY_REGISTRY'], str(queue.REGISTRY))

    def test_preflight_rejects_changed_selection_protocol(self):
        with patch.object(queue, 'JOBS', []), patch.object(queue, 'frozen_sources', return_value={}):
            evidence = {'sources': {}, 'jobs': [], 'selection_environment': queue.selection_environment()}
            queue.verify_preflight(evidence)
            evidence['selection_environment']['CUBUTTERFLY_SEARCH_BUDGET'] = '0'
            with self.assertRaisesRegex(RuntimeError, 'budget/thread'):
                queue.verify_preflight(evidence)


if __name__ == '__main__':
    unittest.main()
