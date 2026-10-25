"""CPU-only integrity, gap accounting and accuracy qualification tests."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('semantic_summary', ROOT / 'scripts/summarize_semantic_campaign.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def row(case_id='a', storage=6, status='accuracy-review-required'):
    metrics = {'elements': 4, 'max_abs': .1, 'max_relative_nonzero_reference': .2,
               'l2_relative': .05, 'error_distributions': {'absolute': {
                   'elements': 4, 'quantiles': {'0': 0, '0.5': .03, '1': .1}}}}
    if storage in (8, 9):
        metrics = {'elements': 4, 'exact': True, 'mismatches': 0}
    return {'id': case_id, 'status': status, 'category': 'native-contract',
            'populations': ['conformance', 'performance'],
            'descriptor': {'semantics': {'operator': 1 if storage == 9 else 0,
                           'storage': storage, 'compute': 1, 'fft_type': 0}},
            'correctness': metrics, 'timing': {'median_ms': 2, 'samples_ms': [1, 2, 3]},
            'baselines': [{'name': 'reference', 'status': status, 'category': 'adapter',
                           'correctness': copy.deepcopy(metrics), 'timing': {'median_ms': 4}}]}


def checkpoint(directory, rows):
    directory = Path(directory)
    (directory / 'rows').mkdir(exist_ok=True)
    indices = []
    for value in rows:
        checksum = summary.digest(value)
        relative = 'rows/' + value['id'] + '.json'
        (directory / relative).write_text(json.dumps({'sha256': checksum, 'row': value}))
        indices.append({'id': value['id'], 'status': value['status'], 'path': relative, 'sha256': checksum})
    report = {'schema': 'cubutterfly-semantic-results-v1', 'rows': indices,
              'rows_sha256': summary.digest(indices), 'identity': {'population': 'performance'},
              'scope': {'bounded': True}, 'status': 'tranche-complete',
              'remaining_case_ids': ['pending'], 'invalid_request_count': 3}
    path = directory / 'results.json'
    path.write_text(json.dumps(report))
    return path, report


class SemanticSummaryTests(unittest.TestCase):
    def test_gaps_errors_pending_and_population_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            value = row()
            value['baselines'].extend([{'name': 'missing', 'status': 'missing-baseline'},
                                      {'name': 'broken', 'status': 'baseline-error'},
                                      {'name': 'unrelated', 'status': 'not-applicable'}])
            gap = row('gap', status='missing-lowering')
            gap.pop('correctness')
            gap.pop('timing')
            gap['baselines'] = []
            path, _ = checkpoint(directory, [value, gap])
            result = summary.summarize(path)
            self.assertEqual(result['remaining'], 1)
            self.assertEqual(result['remaining_case_ids'], ['pending'])
            self.assertEqual(result['invalid_count'], 3)
            self.assertEqual(result['scope'], {'bounded': True})
            self.assertEqual(result['population'], 'performance')
            self.assertEqual(result['status_counts'], {'accuracy-review-required': 1, 'missing-lowering': 1})
            self.assertEqual(result['baseline_counts']['missing'], 1)
            self.assertEqual(result['baseline_counts']['not_applicable'], 1)
            self.assertEqual(result['baseline_counts']['total'], 4)
            self.assertEqual(result['baseline_counts']['other'], 0)
            self.assertEqual(result['baseline_counts']['errors'], 1)
            self.assertEqual(result['baseline_counts']['awaiting_accuracy_review'], 1)
            self.assertEqual(result['baseline_counts']['exact_pass'], 0)

    def test_float_pairs_are_only_diagnostic_even_with_legacy_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = checkpoint(directory, [row(status='pass')])
            result = summary.summarize(path)
            pair = result['pairs'][0]
            self.assertEqual(pair['paired_count'], 1)
            self.assertEqual(pair['actual_timing_paired_count'], 1)
            self.assertEqual(pair['qualified_count'], 0)
            self.assertEqual(pair['accuracy_status'], 'accuracy-review-required')
            self.assertEqual(pair['observed_baseline_over_cubutterfly_ratio_range']['min'], 2)
            self.assertEqual(pair['cubutterfly_category'], 'native-contract')
            self.assertEqual(pair['baseline_category'], 'adapter')
            self.assertEqual(result['status_counts'], {'accuracy-review-required': 1})
            self.assertEqual(result['recorded_status_counts'], {'pass': 1})
            self.assertFalse(any(word in json.dumps(result) for word in ('speedup', 'geomean', 'superiority')))
            self.assertEqual(result['cases'][0]['backends'][0]['correctness']['error_distributions']['absolute']['elements'], 4)

    def test_integer_pairs_require_two_exact_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            good = row('good', 9, 'pass')
            bad = row('bad', 9, 'pass')
            bad['baselines'][0]['correctness']['exact'] = False
            failed = row('failed', 9, 'pass')
            failed['baselines'][0]['status'] = 'correctness-failure'
            path, _ = checkpoint(directory, [good, bad, failed])
            pair = summary.summarize(path)['pairs'][0]
            self.assertEqual(pair['actual_timing_paired_count'], 3)
            self.assertEqual(pair['paired_count'], 1)
            self.assertEqual(pair['qualified_count'], 1)

    def test_corruption_of_index_envelope_and_status_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, report = checkpoint(directory, [row()])
            report['rows'][0]['status'] = 'pass'
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'rows checksum'):
                summary.summarize(path)
            report['rows_sha256'] = summary.digest(report['rows'])
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'ID/status'):
                summary.summarize(path)
            path, _ = checkpoint(directory, [row()])
            envelope_path = Path(directory) / 'rows/a.json'
            envelope = json.loads(envelope_path.read_text())
            envelope['row']['correctness']['max_abs'] = 99
            envelope_path.write_text(json.dumps(envelope))
            with self.assertRaisesRegex(ValueError, 'row checkpoint checksum'):
                summary.summarize(path)

    def test_artifact_is_referenced_and_checksum_verified_and_cli_works(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = Path(directory) / 'errors.npz'
            artifact_path.write_bytes(b'lossless error artifact fixture')
            value = row()
            artifact = {'path': str(artifact_path), 'sha256': summary.file_digest(artifact_path)}
            value['error_artifact'] = artifact
            path, _ = checkpoint(directory, [value])
            output = Path(directory) / 'summary.json'
            self.assertEqual(summary.main(['--results', str(path), '--output', str(output)]), 0)
            result = json.loads(output.read_text())
            self.assertEqual(result['cases'][0]['backends'][0]['error_artifact'], artifact)
            artifact_path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'error artifact'):
                summary.summarize(path)


if __name__ == '__main__':
    unittest.main()
