"""CPU-only evidence and stop-policy tests for independent campaign shards."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('continuation', ROOT / 'scripts/continue_semantic_campaign.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)
runner = controller.runner
UUID = 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02'


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        library = self.root / 'library.so'
        library.write_bytes(b'library')
        self.cases = []
        for value in range(5):
            semantics = {'value': value}
            self.cases.append({'id': runner.digest(semantics),
                'descriptor': {'semantics': semantics, 'selection': {'algorithm_policy': 0}},
                'populations': ['population'], 'capabilities': {'status': 'eligible'},
                'requested_variants': [{'population': 'population', 'value': value}]})
        self.parent = {'schema': 'cubutterfly-semantic-campaign-v1',
            'fingerprints': {'library': controller.evidence(library), 'sources': []},
            'cases': self.cases, 'invalid_requests': [{'invalid': True}] * 3,
            'metrics': {'unique_cases': 5, 'invalid_variants': 3}}
        self.parent['manifest_sha256'] = controller.manifest_hash(self.parent)
        manifest = self.root / 'manifest.json'
        controller.save(manifest, self.parent)
        self.args = SimpleNamespace(manifest=manifest, library=library,
            output=self.root / 'new', prior_results=[], gpu_uuid=UUID, phase='conformance',
            calibration_manifest=None, vkfft_library=None, gpuntt_library=None)
        self.calls = []

    def report(self, path, manifest, statuses, timing=False, status=None):
        indices = []
        for case, row_status in zip(manifest['cases'], statuses):
            artifact = path.parent / 'errors' / (case['id'] + '.npz')
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_bytes(b'lossless fake artifact')
            row = {'id': case['id'], 'descriptor': copy.deepcopy(case['descriptor']),
                   'status': row_status, 'error_artifact': controller.evidence(artifact)}
            indices.append(runner.write_case_checkpoint(path.parent, row))
        identity = {'manifest_sha256': runner.digest(manifest),
            'library_sha256': runner.sha256_file(self.args.library), 'gpu_uuid': UUID,
            'timing': timing, 'population': None, 'runner_sha256': runner.sha256_file(runner.__file__),
            'protocol': dict(runner.PROTOCOL, float_verdict='error-report'),
            'error_directory': str(path.parent / 'errors')}
        report = {'schema': 'cubutterfly-semantic-results-v1', 'identity': identity,
                  'rows': indices, 'rows_sha256': runner.digest(indices),
                  'remaining_case_ids': [case['id'] for case in manifest['cases'][len(indices):]],
                  'status': status or ('stopped-correctness-failure' if 'correctness-failure' in statuses
                                     else 'complete-with-gaps-and-pending-accuracy')}
        controller.save(path, report)
        return report

    def prior(self, statuses):
        path = self.root / ('prior-%d' % len(self.args.prior_results)) / 'results.json'
        self.report(path, self.parent, statuses)
        self.args.prior_results.append(path)
        return path

    def launcher(self, failure=None, runtime=False):
        def launch(command, check):
            self.assertFalse(check)
            self.calls.append(command)
            manifest = controller.read(command[command.index('--manifest') + 1])
            output = Path(command[command.index('--output') + 1])
            statuses = []
            for case in manifest['cases']:
                if runtime:
                    statuses.append('runtime-error')
                    break
                if case['id'] == failure:
                    statuses.append('correctness-failure')
                    break
                statuses.append('accuracy-review-required')
            self.report(output / 'results.json', manifest, statuses,
                        timing='--timing' in command,
                        status='stopped-runtime-error' if runtime else None)
            return SimpleNamespace(returncode=1 if failure in [case['id'] for case in manifest['cases']] or runtime else 0)
        return launch

    def test_prior_untouched_completed_not_rerun_and_exact_failure_quarantined(self):
        prior = self.prior(['accuracy-review-required', 'correctness-failure'])
        original = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.assertEqual(len(self.calls), 1)
        shard = controller.read(self.calls[0][self.calls[0].index('--manifest') + 1])
        self.assertEqual(shard['cases'], self.cases[2:])
        self.assertEqual(shard['metrics']['unique_cases'], 3)
        self.assertEqual(shard['metrics']['requested_variants'], 3)
        self.assertEqual(shard['metrics']['invalid_variants'], 0)
        self.assertEqual(shard['invalid_requests'], [])
        provenance = shard['continuation_provenance']
        self.assertEqual(provenance['parent_metrics'], self.parent['metrics'])
        self.assertEqual(provenance['invalid_requests']['count'], 3)
        self.assertEqual(provenance['quarantine'][0]['id'], self.cases[1]['id'])
        self.assertEqual(provenance['quarantine'][0]['results'], controller.evidence(prior))
        for path, content in original.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertNotIn('--resume', self.calls[0])
        self.assertEqual(self.calls[0][self.calls[0].index('--float-verdict') + 1], 'error-report')
        self.assertEqual(self.calls[0][self.calls[0].index('--max-cases') + 1], '3')

    def test_new_correctness_failure_advances_only_unprocessed_cases(self):
        failed = self.cases[2]['id']
        self.assertEqual(controller.continue_campaign(self.args, self.launcher(failed)), 0)
        self.assertEqual(len(self.calls), 2)
        second = controller.read(self.calls[1][self.calls[1].index('--manifest') + 1])
        self.assertEqual(second['cases'], self.cases[3:])
        state = controller.load_state(self.args.output / 'conformance/controller.json')
        self.assertEqual(state['excluded_quarantine_case_ids'], [failed])
        self.assertTrue(state['quarantine'][0]['artifacts'])
        self.assertEqual(len(state['quarantine']), 1)
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.assertEqual(len(self.calls), 2)
        resumed = controller.load_state(self.args.output / 'conformance/controller.json')
        self.assertEqual(len(resumed['quarantine']), 1)

    def test_runtime_failure_never_retries_or_launches_another_shard(self):
        self.assertEqual(controller.continue_campaign(self.args, self.launcher(runtime=True)), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 1)
        self.assertEqual(len(self.calls), 1)

    def test_changed_parent_prior_row_and_artifact_rejected(self):
        prior = self.prior(['pass'])
        row = prior.parent / 'rows' / (self.cases[0]['id'] + '.json')
        original_row = row.read_bytes()
        row.write_bytes(original_row + b' ')
        # Content envelope remains valid, but resume freezes exact file evidence.
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        row.write_bytes(original_row)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            controller.continue_campaign(self.args, self.launcher())
        row.write_bytes(original_row + b' ')
        artifact = prior.parent / 'errors' / (self.cases[0]['id'] + '.npz')
        artifact.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            controller.continue_campaign(self.args, self.launcher())
        changed = copy.deepcopy(self.parent)
        changed['cases'][0]['descriptor']['semantics']['value'] = 99
        controller.save(self.args.manifest, changed)
        with self.assertRaisesRegex(ValueError, 'parent manifest checksum'):
            controller.continue_campaign(self.args, self.launcher())

    def test_foreign_index_and_corrupt_envelope_rejected(self):
        prior = self.prior(['pass'])
        report = controller.read(prior)
        report['rows'][0]['id'] = 'foreign'
        report['rows_sha256'] = runner.digest(report['rows'])
        controller.save(prior, report)
        with self.assertRaisesRegex(ValueError, 'foreign case'):
            controller.continue_campaign(self.args, self.launcher())
        self.report(prior, self.parent, ['pass'])
        row_path = prior.parent / 'rows' / (self.cases[0]['id'] + '.json')
        envelope = controller.read(row_path)
        envelope['row']['status'] = 'accuracy-review-required'
        controller.save(row_path, envelope)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            controller.continue_campaign(self.args, self.launcher())

    def test_comparison_times_completed_conformance_but_excludes_quarantine(self):
        self.prior(['pass', 'correctness-failure'])
        calibration = self.root / 'calibration.json'
        controller.save(calibration, {})
        self.args.calibration_manifest = calibration
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.args.phase = 'comparison'
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        command = self.calls[-1]
        manifest = controller.read(command[command.index('--manifest') + 1])
        self.assertEqual(manifest['cases'], [self.cases[0], *self.cases[2:]])
        self.assertIn('--timing', command)
        self.assertEqual(manifest['continuation_provenance']['omitted_processed_case_ids'], [])

    def validated_timing(self):
        self.args.phase = 'validated-timing'
        self.args.calibration_manifest = self.root / 'calibration.json'
        controller.save(self.args.calibration_manifest, {})

    def test_validated_timing_launches_once_without_conformance_and_resumes(self):
        self.validated_timing()
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.assertEqual(len(self.calls), 1)
        self.assertIn('--timing', self.calls[0])
        self.assertIn('--calibration-manifest', self.calls[0])
        self.assertFalse((self.args.output / 'conformance').exists())
        state = controller.load_state(self.args.output / 'validated-timing/controller.json')
        self.assertEqual(state['identity']['phase'], 'validated-timing')
        self.assertEqual(state['identity']['conformance_evidence'], [])
        self.assertEqual(state['measurement_identity']['timing'], True)
        self.assertEqual(state['processed_case_ids'], sorted(case['id'] for case in self.cases))
        report = controller.read(Path(state['shards'][0]['output']) / 'results.json')
        self.assertEqual({row['status'] for row in report['rows']}, {'accuracy-review-required'})
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.assertEqual(len(self.calls), 1)
        self.args.phase = 'comparison'
        with self.assertRaises(FileNotFoundError):
            controller.continue_campaign(self.args, self.launcher())
        self.assertEqual(len(self.calls), 1)

    def test_validated_timing_requires_calibration_and_rejects_prior_results(self):
        self.args.phase = 'validated-timing'
        with self.assertRaisesRegex(ValueError, 'requires --calibration-manifest'):
            controller.continue_campaign(self.args, self.launcher())
        self.validated_timing()
        self.prior(['pass'])
        with self.assertRaisesRegex(ValueError, 'forbids --prior-results'):
            controller.continue_campaign(self.args, self.launcher())
        self.assertEqual(self.calls, [])
        self.assertFalse(self.args.output.exists())

    def test_validated_timing_isolates_correctness_failure_without_repeating_cases(self):
        self.validated_timing()
        failed = self.cases[2]['id']
        self.assertEqual(controller.continue_campaign(self.args, self.launcher(failed)), 0)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(all('--timing' in command for command in self.calls))
        second = controller.read(self.calls[1][self.calls[1].index('--manifest') + 1])
        self.assertEqual(second['cases'], self.cases[3:])
        state = controller.load_state(self.args.output / 'validated-timing/controller.json')
        self.assertEqual(state['excluded_quarantine_case_ids'], [failed])
        self.assertEqual(len(state['quarantine']), 1)
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.assertEqual(len(self.calls), 2)

    def test_validated_timing_runtime_error_stops_without_retry(self):
        self.validated_timing()
        self.assertEqual(controller.continue_campaign(self.args, self.launcher(runtime=True)), 1)
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 1)
        self.assertEqual(len(self.calls), 1)

    def test_validated_timing_rejects_untimed_measurement(self):
        self.validated_timing()
        def launch(command, check):
            self.calls.append(command)
            manifest = controller.read(command[command.index('--manifest') + 1])
            output = Path(command[command.index('--output') + 1])
            self.report(output / 'results.json', manifest, ['pass'] * len(manifest['cases']))
            return SimpleNamespace(returncode=0)
        with self.assertRaisesRegex(ValueError, 'phase identity mismatch'):
            controller.continue_campaign(self.args, launch)
        self.assertEqual(len(self.calls), 1)

    def test_validated_timing_baseline_error_blocks_correctness_isolation(self):
        self.validated_timing()
        self.assert_baseline_error_stops_investigation()

    def test_validated_timing_cli_checks_before_creating_output(self):
        argv = ['--manifest', str(self.args.manifest), '--library', str(self.args.library),
                '--output', str(self.args.output), '--gpu-uuid', UUID,
                '--phase', 'validated-timing']
        with self.assertRaises(SystemExit) as missing:
            controller.main(argv)
        self.assertEqual(missing.exception.code, 2)
        self.validated_timing()
        with self.assertRaises(SystemExit) as prior:
            controller.main(argv + ['--calibration-manifest', str(self.args.calibration_manifest),
                                    '--prior-results', str(self.root / 'old-results.json')])
        self.assertEqual(prior.exception.code, 2)
        self.assertFalse(self.args.output.exists())

    def test_changed_shard_manifest_and_results_rejected_on_resume(self):
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        path = Path(self.calls[0][self.calls[0].index('--output') + 1]) / 'results.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            controller.continue_campaign(self.args, self.launcher())

    def test_runner_exit_without_checkpoint_stops_and_resume_does_not_retry(self):
        def launch(command, check):
            self.calls.append(command)
            return SimpleNamespace(returncode=2)
        self.assertEqual(controller.continue_campaign(self.args, launch), 2)
        with self.assertRaisesRegex(ValueError, 'no result'):
            controller.continue_campaign(self.args, launch)
        self.assertEqual(len(self.calls), 1)

    def test_fingerprint_cache_reuses_hash_and_detects_file_change(self):
        path = self.root / 'cached.npz'
        path.write_bytes(b'original')
        with patch.object(runner, 'sha256_file', wraps=runner.sha256_file) as hashed, \
                patch.object(controller.time, 'time_ns', return_value=path.stat().st_ctime_ns + 10_000_000_000):
            first = controller.evidence(path)
            self.assertEqual(controller.evidence(path), first)
            self.assertEqual(hashed.call_count, 1)
            path.write_bytes(b'modified with a different size')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                controller.verify_evidence(first)
            self.assertEqual(hashed.call_count, 2)

    def test_recent_same_size_evidence_is_rehashed(self):
        path = self.root / 'recent.npz'
        path.write_bytes(b'original')
        with patch.object(runner, 'sha256_file', wraps=runner.sha256_file) as hashed:
            first = controller.evidence(path)
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                controller.verify_evidence(first)
            self.assertEqual(hashed.call_count, 2)

    def test_nonzero_exit_with_complete_checkpoint_is_not_continued(self):
        successful = self.launcher()
        def launch(command, check):
            successful(command, check)
            return SimpleNamespace(returncode=1)
        self.assertEqual(controller.continue_campaign(self.args, launch), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(controller.continue_campaign(self.args, launch), 1)
        self.assertEqual(len(self.calls), 1)

    def test_many_shards_checked_once_and_resume_checks_full_chain(self):
        def launch(command, check):
            self.calls.append(command)
            manifest = controller.read(command[command.index('--manifest') + 1])
            output = Path(command[command.index('--output') + 1])
            self.report(output / 'results.json', manifest, ['correctness-failure'])
            return SimpleNamespace(returncode=1)
        with patch.object(controller, 'checked_report', wraps=controller.checked_report) as checked:
            self.assertEqual(controller.continue_campaign(self.args, launch), 0)
            self.assertEqual(len(self.calls), 5)
            self.assertEqual(checked.call_count, 5)
        with patch.object(controller, 'checked_report', wraps=controller.checked_report) as checked:
            self.assertEqual(controller.continue_campaign(self.args, launch), 0)
            self.assertEqual(checked.call_count, 5)
            self.assertEqual(len(self.calls), 5)

    def test_closed_evidence_change_and_missing_detected_in_same_process(self):
        for remove in (False, True):
            with self.subTest(remove=remove):
                self.args.output = self.root / ('tamper-%s' % remove)
                count = 0
                successful = self.launcher(self.cases[0]['id'])
                def launch(command, check):
                    nonlocal count
                    result = successful(command, check)
                    count += 1
                    if count == 2:
                        state = controller.load_state(self.args.output / 'conformance/controller.json')
                        old = Path(state['shards'][0]['output']) / 'results.json'
                        if remove:
                            old.unlink()
                        else:
                            old.write_text(old.read_text() + ' ')
                    return result
                with self.assertRaisesRegex(ValueError, 'evidence changed or missing'):
                    controller.continue_campaign(self.args, launch)
                self.assertEqual(count, 2)

    def old_controller_state(self):
        self.args.verify_only = True
        controller.continue_campaign(self.args, self.launcher())
        old_source = self.root / 'old-controller.py'
        old_source.write_text('# previous controller source\n')
        state_path = self.args.output / 'conformance/controller.json'
        state = controller.load_state(state_path)
        for item in state['identity']['inputs']:
            if item['path'] == str(Path(controller.__file__).resolve()):
                item['sha256'] = runner.sha256_file(old_source)
        controller.save_state(state_path, state)
        self.args.upgrade_controller_from = old_source
        return state_path

    def test_source_only_upgrade_preserves_envelope_and_records_hashes(self):
        state_path = self.old_controller_state()
        previous = state_path.read_bytes()
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.assertEqual(self.calls, [])
        directories = list(state_path.parent.glob('controller-upgrade-*'))
        self.assertEqual(len(directories), 1)
        self.assertEqual((directories[0] / 'controller.before.json').read_bytes(), previous)
        record = controller.read(directories[0] / 'migration.json')
        self.assertEqual(record['old_state_sha256'], json.loads(previous)['sha256'])
        self.assertEqual(record['new_source_sha256'], runner.sha256_file(controller.__file__))
        self.assertEqual(record['new_state_sha256'], controller.read(state_path)['sha256'])

    def test_upgrade_rejects_other_identity_change_and_wrong_old_source(self):
        state_path = self.old_controller_state()
        old = state_path.read_bytes()
        self.args.gpu_uuid = 'GPU-wrong'
        with self.assertRaisesRegex(ValueError, 'only controller source'):
            controller.continue_campaign(self.args, self.launcher())
        self.assertEqual(state_path.read_bytes(), old)
        self.args.gpu_uuid = UUID
        self.args.upgrade_controller_from.write_text('# mismatched source\n')
        with self.assertRaisesRegex(ValueError, 'old source hash mismatch'):
            controller.continue_campaign(self.args, self.launcher())
        self.assertEqual(state_path.read_bytes(), old)
        self.assertFalse(list(state_path.parent.glob('controller-upgrade-*')))

    def test_upgrade_checks_entire_chain_before_adopting_identity(self):
        self.assertEqual(controller.continue_campaign(self.args, self.launcher(self.cases[1]['id'])), 0)
        state_path = self.old_controller_state()
        old = state_path.read_bytes()
        state = controller.load_state(state_path)
        path = Path(state['shards'][-1]['output']) / 'results.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            controller.continue_campaign(self.args, self.launcher())
        self.assertEqual(state_path.read_bytes(), old)
        self.assertFalse(list(state_path.parent.glob('controller-upgrade-*')))

    def test_upgrade_ingests_known_correctness_stop_without_rerunning_cases(self):
        successful = self.launcher(self.cases[1]['id'])
        class Interrupted(Exception):
            pass
        def launch(command, check):
            result = successful(command, check)
            state_path = self.args.output / 'conformance/controller.json'
            state = controller.load_state(state_path)
            state['shards'][-1]['returncode'] = result.returncode
            controller.save_state(state_path, state)
            raise Interrupted()
        with self.assertRaises(Interrupted):
            controller.continue_campaign(self.args, launch)
        old_source = self.root / 'old-controller.py'
        old_source.write_text('# previous controller\n')
        self.args.upgrade_controller_from = old_source
        self.args.verify_only = True
        state_path = self.args.output / 'conformance/controller.json'
        state = controller.load_state(state_path)
        for item in state['identity']['inputs']:
            if item['path'] == str(Path(controller.__file__).resolve()):
                item['sha256'] = runner.sha256_file(old_source)
        controller.save_state(state_path, state)
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        migrated = controller.load_state(state_path)
        self.assertEqual(migrated['processed_case_ids'], [self.cases[0]['id']])
        self.assertEqual([item['id'] for item in migrated['quarantine']], [self.cases[1]['id']])
        self.assertEqual(len(self.calls), 1)

    def test_correctness_stop_with_baseline_error_stops_investigation(self):
        self.assert_baseline_error_stops_investigation()

    def assert_baseline_error_stops_investigation(self):
        successful = self.launcher(self.cases[1]['id'])
        def launch(command, check):
            result = successful(command, check)
            path = Path(command[command.index('--output') + 1]) / 'results.json'
            report = controller.read(path)
            row_path = path.parent / report['rows'][0]['path']
            row = controller.read(row_path)['row']
            row['baselines'] = [{'status': 'baseline-error'}]
            report['rows'][0] = runner.write_case_checkpoint(path.parent, row)
            report['rows_sha256'] = runner.digest(report['rows'])
            controller.save(path, report)
            return result

        self.assertEqual(controller.continue_campaign(self.args, launch), 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(controller.load_state(self.args.output / self.args.phase / 'controller.json')['status'],
                         'stopped-investigation-required')

    def test_comparison_rejects_changed_calibration_baseline_and_environment(self):
        calibration = self.root / 'calibration.json'
        controller.save(calibration, {})
        self.args.calibration_manifest = calibration
        controller.continue_campaign(self.args, self.launcher())
        self.args.phase = 'comparison'
        changed = self.root / 'other-calibration.json'
        controller.save(changed, {})
        self.args.calibration_manifest = changed
        with self.assertRaisesRegex(ValueError, 'execution identity differs'):
            controller.continue_campaign(self.args, self.launcher())
        self.args.calibration_manifest = calibration
        self.args.vkfft_library = self.args.library
        with self.assertRaisesRegex(ValueError, 'execution identity differs'):
            controller.continue_campaign(self.args, self.launcher())
        self.args.vkfft_library = None
        with patch.dict(controller.os.environ, {'CUBUTTERFLY_TEST_CHANGED': '1'}):
            with self.assertRaisesRegex(ValueError, 'execution identity differs'):
                controller.continue_campaign(self.args, self.launcher())
        self.assertEqual(len(self.calls), 1)

    def test_verify_only_never_launches_and_unknown_exit_never_retries(self):
        self.args.verify_only = True
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 0)
        self.assertEqual(self.calls, [])
        self.args.verify_only = False
        controller.continue_campaign(self.args, self.launcher())
        state_path = self.args.output / 'conformance/controller.json'
        state = controller.load_state(state_path)
        state['shards'][0].pop('returncode')
        controller.save_state(state_path, state)
        self.assertEqual(controller.continue_campaign(self.args, self.launcher()), 1)
        self.assertEqual(len(self.calls), 1)

    def test_missing_baseline_with_external_binding_error_is_fatal(self):
        successful = self.launcher()
        def launch(command, check):
            result = successful(command, check)
            path = Path(command[command.index('--output') + 1]) / 'results.json'
            report = controller.read(path)
            row = controller.read(path.parent / report['rows'][0]['path'])['row']
            row['baselines'] = [{'status': 'missing-baseline',
                                 'support_status': 'external-binding-error'}]
            report['rows'][0] = runner.write_case_checkpoint(path.parent, row)
            report['rows_sha256'] = runner.digest(report['rows'])
            controller.save(path, report)
            return result
        self.assertEqual(controller.continue_campaign(self.args, launch), 1)
        self.assertEqual(len(self.calls), 1)
        state = controller.load_state(self.args.output / 'conformance/controller.json')
        self.assertEqual(state['status'], 'stopped-investigation-required')


if __name__ == '__main__':
    unittest.main()
