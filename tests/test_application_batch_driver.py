"""CPU-only cohort identities and guarded cache promotion contracts."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('application_batch_driver', ROOT/'experiments/tools/run_application_batch_20261012.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class ApplicationBatchTests(unittest.TestCase):
    def calibration(self):
        return dict(registry_path='/tmp/registry.json', measurement_identity=dict(jit_runtime=dict(
            ld_library_path='/tmp/runtime', config=dict(root='/tmp/templates', cache='/tmp/jit', mathdx='/tmp/mathdx', nvcc='/tmp/nvcc'))))

    def test_effective_budget_and_environment_fingerprints(self):
        with patch.dict(os.environ, {'CUBUTTERFLY_XD_PROMOTION_MARGIN': '0.02',
                                     'CUBUTTERFLY_PLAN_CACHE': '/old/cache'}, clear=True):
            env, identity = driver.effective_environment(self.calibration(), 'GPU-test', 8*1024**3)
            self.assertEqual(env['CUBUTTERFLY_XD_TUNE_MAX_BYTES'], '8589934592')
            self.assertNotIn('CUBUTTERFLY_PLAN_CACHE', env)
            _, different = driver.effective_environment(self.calibration(), 'GPU-test', 512*1024**2)
            self.assertNotEqual(driver.fingerprint(identity), driver.fingerprint(different))
            self.assertEqual(identity['CUBUTTERFLY_XD_PROMOTION_MARGIN'], '0.02')
            self.assertEqual(driver.fingerprint(identity), driver.fingerprint(dict(reversed(list(identity.items())))))

    def test_input_artifact_mismatch_is_closed_and_manifest_is_fingerprinted(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(driver, 'OUT', Path(directory)):
            prepared = Path(directory)/'fft_filter'
            prepared.mkdir()
            artifact = prepared/'input.npz'
            artifact.write_bytes(b'original')
            manifest = prepared/'manifest.json'
            manifest.write_text(json.dumps(dict(contracts=[dict(artifact='input.npz', sha256=driver.digest(artifact))])))
            first = driver.input_identity('fft_filter')
            manifest.write_text(manifest.read_text()+'\n')
            self.assertNotEqual(first['manifest_sha256'], driver.input_identity('fft_filter')['manifest_sha256'])
            artifact.write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'artifact hash mismatch'):
                driver.input_identity('fft_filter')

    def test_failed_tuning_does_not_touch_seed_or_produce_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root/'seed.plans'
            seed.write_bytes(b'verified\n')
            attempt = root/'attempt'
            attempt.mkdir()
            def fail(command, **kwargs):
                Path(kwargs['env']['CUBUTTERFLY_PLAN_CACHE']).write_bytes(b'failed timing\n')
                return SimpleNamespace(returncode=2)
            with patch.dict(os.environ, {'XD_REPORT_DIR': str(attempt)}), patch.object(driver.subprocess, 'run', side_effect=fail):
                self.assertEqual(driver.cache_child(['--cache-seed', str(seed), '--policy', 'measure', '--', 'mock']), 2)
            self.assertEqual(seed.read_bytes(), b'verified\n')
            self.assertFalse((attempt/'cache_receipt.json').exists())
            record = dict(report_dir=str(root), policy='measure')
            (root/'status.json').write_text(json.dumps(dict(state='failed')))
            with self.assertRaisesRegex(RuntimeError, 'successful release'):
                driver.accept_release(record, 'spec', seed)
            self.assertFalse((root/'promoted.plans').exists())

    def test_default_mutation_fails_and_never_changes_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root/'seed.plans'
            seed.write_bytes(b'frozen\n')
            attempt = root/'attempt'
            attempt.mkdir()
            def mutate(command, **kwargs):
                Path(kwargs['env']['CUBUTTERFLY_PLAN_CACHE']).write_bytes(b'unqualified\n')
                return SimpleNamespace(returncode=0)
            with patch.dict(os.environ, {'XD_REPORT_DIR': str(attempt)}), patch.object(driver.subprocess, 'run', side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, 'changed its frozen cache'):
                    driver.cache_child(['--cache-seed', str(seed), '--policy', 'default', '--', 'mock'])
            self.assertEqual(seed.read_bytes(), b'frozen\n')
            self.assertFalse((attempt/'cache_receipt.json').exists())

    def test_successful_release_promotion_and_strict_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            release = report/'release'
            release.mkdir()
            cache = release/'plan_cache.plans'
            cache.write_bytes(b'qualified\n')
            driver.save(release/'cache_receipt.json', dict(policy='measure', seed_sha256=None, cache_sha256=driver.digest(cache)))
            driver.save(report/'status.json', dict(state='complete', release=str(release)))
            (release/'result.json').write_text('{"result":"original"}')
            record = dict(id='tune', policy='measure', report_dir=str(report), command=['mock'])
            promoted = driver.accept_release(record, 'fixed-spec', None)
            self.assertEqual(promoted.read_bytes(), b'qualified\n')
            self.assertEqual(driver.reuse_release(record, 'fixed-spec', None), promoted)
            with self.assertRaisesRegex(RuntimeError, 'immutable cohort'):
                driver.reuse_release(record, 'different-spec', None)
            (release/'result.json').write_text('{"result":"changed"}')
            with self.assertRaisesRegex(RuntimeError, 'immutable cohort'):
                driver.reuse_release(record, 'fixed-spec', None)

    def test_fft_jobs_keep_original_prepared_input_directory(self):
        for _, _, command in driver.jobs('fft_filter'):
            self.assertEqual(command[command.index('--output')+1], str(driver.OUT/'fft_filter'))


if __name__ == '__main__':
    unittest.main()
