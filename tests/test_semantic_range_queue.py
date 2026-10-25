"""CPU-only queue provenance, historical reuse and failure-isolation checks."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_semantic_range_queue as q
from semantic_campaign_manifest import MANIFEST_SCHEMA

GPU = 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02'


class RangeQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.library = self.root / 'library.so'
        self.library.write_bytes(b'CPU fixture')
        for name in ('registry', 'rates', 'model'):
            (self.root / (name + '.json')).write_text('{}')
        calibration = {'measurement_identity': {'compile_mode': 'research',
                       'library_sha256': q.runner.sha256_file(self.library)},
                       'registry_path': str(self.root / 'registry.json'),
                       'hardware_rates_source': str(self.root / 'rates.json'),
                       'cost_model': str(self.root / 'model.json')}
        self.calibration = self.root / 'calibration.json'
        self.calibration.write_text(json.dumps(calibration))
        self.args = SimpleNamespace(controller=self.root / 'controller.json', library=self.library,
                                    calibration_manifest=self.calibration, gpu_uuid=GPU,
                                    output=self.root / 'queue', vkfft_library=None,
                                    gpuntt_library=None, existing_control=[], resume=False)
        self.args.controller.write_text('{}')

    def row(self, number=0, operator=0, storage=4):
        n = 8 << number
        s = {'schema': 'cubutterfly-semantic-v1', 'operator': operator, 'storage': storage,
             'compute': 0, 'shape': [n], 'batch': 1, 'inverse': 0, 'normalize_inverse': False,
             'placement': 0, 'fft_type': 0, 'fft_output_scale': 1.0, 'length_mode': 0,
             'input_strides': [1], 'output_strides': [1], 'input_batch_stride': n,
             'output_batch_stride': n, 'input_orders': [0], 'output_orders': [0],
             'ntt_input_layout': 0, 'ntt_output_layout': 0, 'modulus': None,
             'word_bits': None, 'matrices': [[]]}
        identifier = q.runner.digest(s)
        directory = self.root / identifier
        directory.mkdir()
        artifact = directory / 'original.npz'
        np.savez(artifact, actual=np.array([np.inf]), reference=np.array([1]))
        row = {'id': identifier, 'descriptor': {'schema': 'cubutterfly-descriptor-v1',
               'semantics': s, 'selection': {'algorithm_policy': 0, 'execution_mode': 0,
                                            'explicit_algorithm': ''}},
               'mapping': {'contract': s, 'kind': 'butterfly'}, 'status': 'correctness-failure',
               'populations': ['fixture'], 'baselines': [],
               'error_artifact': {'path': str(artifact), 'sha256': q.runner.sha256_file(artifact)}}
        index = q.runner.write_case_checkpoint(directory, row)
        result = directory / 'results.json'
        result.write_text(json.dumps({'schema': 'cubutterfly-semantic-results-v1',
            'rows': [index], 'rows_sha256': q.runner.digest([index]),
            'identity': {'library_sha256': q.runner.sha256_file(self.library), 'gpu_uuid': GPU,
                         'timing': False}, 'status': 'stopped-correctness-failure'}))
        item = q.campaign.quarantine_entry(row, result)
        return item, row

    def control_report(self, item, row, output, status='accuracy-review-required'):
        args = SimpleNamespace(row=Path(item['row']['path']), library=self.library,
                               output=output, gpu_uuid=GPU, calibration_manifest=self.calibration,
                               vkfft_library=None, gpuntt_library=None)
        with patch.dict(os.environ):
            report, source, expected, _ = q.control.prepare(args)
        report['status'] = status
        report['error_artifact'] = q.runner.retain_error_artifact(output / 'errors', report['id'],
                                                               'cubutterfly', source, expected)
        q.control.save_report(output, report)
        return output / 'results.json'

    def launch(self, rows, status='accuracy-review-required', returncode=None, baseline=None):
        calls = []
        def launch(command, check):
            calls.append(command)
            row_path = command[command.index('--row') + 1]
            item, row = next(pair for pair in rows if pair[0]['row']['path'] == row_path)
            output = Path(command[command.index('--output') + 1])
            path = self.control_report(item, row, output, status)
            if baseline:
                report = q.campaign.read(path)
                report['baselines'] = [baseline]
                q.control.save_report(output, report)
            return SimpleNamespace(returncode=(returncode if returncode is not None else
                                               1 if status == 'correctness-failure' else 0))
        return launch, calls

    def context(self, rows):
        return patch.object(q, 'checked_controller', return_value=(rows, []))

    def test_reuse_historical_identity_and_resume_skips(self):
        pair = self.row()
        path = self.control_report(*pair, self.root / 'old')
        report = q.campaign.read(path)
        report['identity']['runner_sha256'] = 'historical-source'
        report['identity']['profile_script_sha256'] = 'historical-profile-source'
        q.control.save_report(path.parent, report)
        self.args.existing_control = [path]
        with self.context([pair]):
            self.assertEqual(q.run_queue(self.args, lambda *a, **k: self.fail('must reuse')), 0)
            self.args.resume = True
            self.assertEqual(q.run_queue(self.args, lambda *a, **k: self.fail('must skip')), 0)
        state = q.campaign.load_state(self.args.output / 'state.json')
        self.assertEqual(state['historical_control_references'], 1)
        self.assertEqual(state['new_controls'], 0)
        self.assertEqual(state['cases'][0]['measurement_identity']['runner_sha256'], 'historical-source')
        self.assertEqual(state['status'], 'complete-with-unresolved-accuracy')

    def test_correctness_failure_continues_and_completed_resume_skips(self):
        rows = [self.row(), self.row(1, operator=2, storage=0)]
        launch, calls = self.launch(rows, 'correctness-failure')
        with self.context(rows):
            self.assertEqual(q.run_queue(self.args, launch), 0)
            self.args.resume = True
            self.assertEqual(q.run_queue(self.args, launch), 0)
        self.assertEqual(len(calls), 2)
        state = q.campaign.load_state(self.args.output / 'state.json')
        self.assertEqual(state['case_counts'], {'correctness-failure': 2})

    def test_baseline_errors_stop_and_cannot_retry(self):
        for baseline in ({'status': 'baseline-error'},
                         {'status': 'missing-baseline', 'support_status': 'external-binding-error'}):
            with self.subTest(baseline=baseline):
                rows = [self.row(2 + len(list(self.root.glob('*/rows'))))]
                self.args.output = self.root / ('queue-' + baseline['status'])
                self.args.resume = False
                launch, calls = self.launch(rows, baseline=baseline)
                with self.context(rows):
                    self.assertEqual(q.run_queue(self.args, launch), 1)
                    self.args.resume = True
                    with self.assertRaisesRegex(ValueError, 'no automatic retry'):
                        q.run_queue(self.args, launch)
                self.assertEqual(len(calls), 1)

    def test_unknown_exit_stops(self):
        rows = [self.row(), self.row(1)]
        launch, calls = self.launch(rows, returncode=-15)
        with self.context(rows):
            self.assertEqual(q.run_queue(self.args, launch), 1)
        self.assertEqual(len(calls), 1)

    def test_unsupported_scope_is_not_a_pass(self):
        pair = self.row(storage=5)
        with self.context([pair]):
            self.assertEqual(q.run_queue(self.args, lambda *a, **k: self.fail('unsupported')), 0)
        state = q.campaign.load_state(self.args.output / 'state.json')
        self.assertEqual(state['case_counts'], {'not-applicable-control': 1})

    def test_unfinished_controller_refused(self):
        q.campaign.save_state(self.args.controller, {'status': 'running'})
        with self.assertRaisesRegex(ValueError, 'must finish'):
            q.run_queue(self.args, lambda *a, **k: self.fail('premature GPU launch'))

    def test_invalid_scale_and_noncanonical_fwht_are_out_of_scope(self):
        _, row = self.row(operator=2, storage=0)
        row['descriptor']['semantics']['fft_output_scale'] = .5
        self.assertIn('canonical', q.scope_reason(row))
        row['descriptor']['semantics']['fft_output_scale'] = float('nan')
        self.assertIn('absolute output scale', q.scope_reason(row))

    def test_profile_mismatch_and_artifact_corruption_rejected(self):
        item, row = self.row()
        path = self.control_report(item, row, self.root / 'old')
        original = q.campaign.read(path)
        report = copy.deepcopy(original)
        report['input_profile']['factor'] = .5
        q.control.save_report(path.parent, report)
        with self.assertRaisesRegex(ValueError, 'profile algorithm'):
            q.checked_control(path, row, self.args, historical=True)
        q.control.save_report(path.parent, original)
        Path(original['source_artifact']['path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'artifact checksum'):
            q.checked_control(path, row, self.args, historical=True)

    def test_controller_full_prior_quarantine_integrity(self):
        item, row = self.row()
        parent = {'schema': MANIFEST_SCHEMA,
                  'cases': [{'id': row['id'], 'descriptor': row['descriptor']}], 'metrics': {}}
        parent['manifest_sha256'] = q.campaign.manifest_hash(parent)
        parent_path = self.root / 'manifest.json'
        parent_path.write_text(json.dumps(parent))
        _, _, files = q.measurement_rows(item['results']['path'], self.args)
        state = {'status': 'complete-with-unresolved-evidence', 'identity': {
            'phase': 'conformance', 'gpu_uuid': GPU,
            'inputs': [q.campaign.evidence(p) for p in (parent_path, self.library, self.calibration)],
            'prior_evidence': files}, 'shards': [], 'quarantine': [item], 'processed_case_ids': []}
        q.campaign.save_state(self.args.controller, state)
        quarantine, _ = q.checked_controller(self.args)
        self.assertEqual(quarantine, [(item, row)])
        Path(row['error_artifact']['path']).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            q.checked_controller(self.args)


if __name__ == '__main__':
    unittest.main()
