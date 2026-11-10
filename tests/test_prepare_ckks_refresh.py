import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_ckks_refresh', ROOT / 'scripts/prepare_ckks_refresh.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class CKKSRefreshTests(unittest.TestCase):
    def test_rejects_gpu_visible_cpu_preparation(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'CPU preparation'):
                prepare.prepare({})

    def test_rejects_sdb_and_symlink(self):
        with self.assertRaises(ValueError):
            prepare.safe_path('/sdb/asset')
        with tempfile.TemporaryDirectory() as temporary:
            link = Path(temporary) / 'asset'
            link.symlink_to('/sdb/asset')
            with self.assertRaises(ValueError):
                prepare.safe_path(link)

    def test_actual_population_and_dependency_contract(self):
        config = prepare.read_json(ROOT / 'configs/experiments/ckks_refresh_20261025.json')
        if not (ROOT / config['requests']).exists():
            self.skipTest('CKKS request population is an optional external workload bundle')
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            config['output'] = temporary
            with patch.object(prepare.subprocess, 'run', side_effect=AssertionError('CPU manifest must not run a child')):
                result = prepare.prepare(config)
        self.assertFalse(result['offline_mapping_ready'])
        self.assertEqual(result['auto_configurations'], 36)
        self.assertEqual(result['distinct_rns_requests'], 118)
        self.assertEqual(result['bootstrap_application_batches'], [1, 8])
        followups = result['normalized_followups']
        self.assertEqual(len(followups), 23)
        self.assertTrue(all(item['argv'][0] == 'scripts/prepare_ckks_refresh.py' for item in followups))
        self.assertTrue(all(not dependency.startswith('build_') for item in followups for dependency in item['depends_on']))
        queue = {entry['id']: entry for entry in result['queue']}
        self.assertEqual(len(queue), 23)
        self.assertIn('replay_bootstrap_118', queue['prepare_bootstrap_v1_small_b1_b8']['depends_on'])
        self.assertIn('populate_method1_n14', queue['prepare_method1_n14']['depends_on'])
        for name, entry in queue.items():
            self.assertTrue(entry['gpu_required'])
            if name.startswith('measure_'):
                self.assertEqual(entry['env']['CUBUTTERFLY_RNS_CACHE_POLICY'], 'require')
                self.assertIn('--after-preparation', entry['argv'])
            if name.startswith('measure_method'):
                argv = entry['argv']
                for option, expected in (('--schedule-rounds', '7'), ('--schedule-replays', '5'),
                                         ('--repetitions', '20'), ('--graph-replays', '10')):
                    self.assertEqual(argv[argv.index(option) + 1], expected)
                self.assertIn('--thermal-guard', argv)

    def test_rns_replay_rejects_failed_missing_or_non_hit_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'replay.json'
            module = Mock()
            module._load_requests.return_value = [(str(index), 'in_place', {}) for index in range(118)]
            runner = Mock()
            runner.file_identity.return_value = {'sha256': 'result'}
            rows = [{'id': str(index), 'status': 'ok', 'cache_hit': True} for index in range(118)]
            plan = {'libcuntt': {'path': '/tmp/libcuntt.so'}}
            job = {'id': 'replay_bootstrap_118', 'argv': ['python', 'prepare', '--manifest', '/tmp/manifest',
                    '--policy', 'require', '--output', str(output)]}
            data = {'results': rows, 'policy': 'require', 'library': '/tmp/libcuntt.so',
                    'repetitions': 5, 'available_request_count': 118, 'limit': None}
            with patch.object(prepare, 'load_module', side_effect=lambda name, path: module if 'requests' in name else runner):
                output.write_text(json.dumps(data))
                prepare.validate_job(job, plan, 'GPU-test')
                for bad_rows in (rows[:-1], rows + [rows[0]], [dict(row, cache_hit=False) for row in rows],
                                 [dict(row, status='error') for row in rows]):
                    output.write_text(json.dumps(dict(data, results=bad_rows)))
                    with self.assertRaises(ValueError):
                        prepare.validate_job(job, plan, 'GPU-test')

    def test_run_job_input_change_rejected_before_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            job = {'id': 'test', 'argv': ['should-never-run'], 'receipt': str(output / 'receipt.json')}
            plan = {'queue': [job], 'static_inputs': [{'path': str(output / 'missing'), 'sha256': 'expected'}]}
            prepare.write_json(output / 'gpu_queue.json', plan)
            with patch.object(prepare.subprocess, 'run', side_effect=AssertionError('must not launch GPU')):
                code = prepare.run_job(output, 'test', 'GPU-00000000-0000-0000-0000-000000000003')
            self.assertNotEqual(code, 0)
            self.assertEqual(prepare.read_json(output / 'receipt.json')['status'], 'failed')

    def test_interrupted_owned_attempt_archives_all_partial_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            csv_path = output / 'case.csv'
            export = output / 'requests.ndjson'
            job = {'id': 'measure_case', 'argv': ['python', 'runner.py', '--output', str(csv_path)],
                   'env': {'CUBUTTERFLY_RNS_REQUEST_EXPORT': str(export)},
                   'receipt': str(output / 'job_receipts/measure_case.json')}
            binding = {'job_id': job['id'], 'gpu_uuid': 'GPU-test', 'static_inputs_sha256': 'build1'}
            attempt_path, first = prepare.begin_attempt(output, job, binding)
            for path in prepare.job_artifacts(output, job):
                if path.name.endswith('_extracted_requests'):
                    path.mkdir()
                    (path / 'partial.json').write_text('partial')
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text('partial samples')
            first['owner_pid'] = 2147483647
            prepare.write_json(attempt_path, first)
            _, second = prepare.begin_attempt(output, job, binding)
            self.assertNotEqual(first['attempt_id'], second['attempt_id'])
            self.assertTrue(all(not path.exists() for path in prepare.job_artifacts(output, job)))
            archive = output / 'attempt_archives' / job['id'] / first['attempt_id']
            self.assertTrue((archive / 'attempt.json').exists())
            self.assertEqual(len(list(archive.iterdir())), len(prepare.job_artifacts(output, job)) + 1)
            csv_path.write_text('new complete run')
            self.assertTrue(any(path.read_text() == 'partial samples' for path in archive.iterdir() if path.is_file()))

    def test_resume_refuses_unknown_changed_binding_and_live_owner(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / 'result.csv'
            job = {'id': 'case', 'argv': ['python', 'runner.py', '--output', str(path)],
                   'receipt': str(output / 'receipt.json')}
            binding = {'gpu_uuid': 'GPU-one', 'build': 'build1'}
            path.write_text('unknown')
            with self.assertRaises(FileExistsError):
                prepare.begin_attempt(output, job, binding)
            path.unlink()
            attempt_path, attempt = prepare.begin_attempt(output, job, binding)
            path.write_text('owned')
            with self.assertRaises(ValueError):
                prepare.begin_attempt(output, job, binding)
            attempt['owner_pid'] = 2147483647
            prepare.write_json(attempt_path, attempt)
            with self.assertRaises(FileExistsError):
                prepare.begin_attempt(output, job, dict(binding, gpu_uuid='GPU-two'))
            self.assertEqual(path.read_text(), 'owned')

    def test_environment_whitelist_excludes_rns_and_unrelated_values(self):
        self.assertIn('HOME', prepare.ENV_ALLOWLIST)
        for name in ('CUBUTTERFLY_JIT_PYTHON', 'CUBUTTERFLY_TEMPLATE_ROOT', 'CUBUTTERFLY_MATHDX_ROOT',
                     'CUBUTTERFLY_NVCC', 'CUBUTTERFLY_JIT_CACHE', 'CUBUTTERFLY_REGISTRY'):
            self.assertIn(name, prepare.ENV_ALLOWLIST)
        self.assertNotIn('CUBUTTERFLY_RNS_CACHE', prepare.ENV_ALLOWLIST)
        self.assertNotIn('CUBUTTERFLY_RNS_REQUEST_EXPORT', prepare.ENV_ALLOWLIST)


if __name__ == '__main__':
    unittest.main()
