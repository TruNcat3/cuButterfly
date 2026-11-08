"""CPU-only prepared profile contract checks; no CUDA process is launched."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('factor_graph', ROOT / 'scripts/run_factor_graph_experiment.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ProfileContract(unittest.TestCase):
    def run_profile(self, actual_backend='cufft', case='selected', mutate_binary=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {name: root / name for name in ('binary', 'libcuntt.so', 'hardware.json', 'reference.bin')}
            for name, path in files.items():
                path.write_text(name)
            prepared = root / 'suite.json'
            suite = dict(cases=[dict(id='selected', mapping=dict(backend='cufft'))],
                         reference_file=str(files['reference.bin']))
            prepared.write_text(json.dumps(suite))
            manifest = dict(complete=True, prepared_sha256=runner.sha(prepared),
                            binary_sha256=runner.sha(files['binary']),
                            library_sha256=runner.sha(files['libcuntt.so']),
                            hardware_model=str(files['hardware.json']),
                            hardware_sha256=runner.sha(files['hardware.json']),
                            reference_sha256=runner.sha(files['reference.bin']), cache=directory,
                            modules=[dict(id='selected', provider='cufft')])
            prepared.with_suffix('.manifest.json').write_text(json.dumps(manifest))
            if mutate_binary:
                files['binary'].write_text('changed')
            observed = dict(status='ok', profile_case='selected', logical_calls=1,
                            normal_timing_accepted=False, nonfinite=0, kernel_nodes=2,
                            actual_mapping=dict(backend=actual_backend))
            completed = runner.subprocess.CompletedProcess([], 0, json.dumps(observed),
                'cuButterfly hardware resources: static-hardware-model:' + str(files['hardware.json']))
            argv = ['runner', 'profile', '--case', case, '--binary', str(files['binary']),
                    '--prepared', str(prepared), '--output', str(root / 'output')]
            with patch.object(sys, 'argv', argv), patch.object(runner.subprocess, 'run', return_value=completed) as run, \
                    patch.object(runner.subprocess, 'check_output', return_value='mock gpu'), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                runner.main()
                self.assertEqual(json.loads(output.getvalue()), observed)
                command = run.call_args[0][0]
                self.assertEqual(command[1:], ['--profile-suite', str(prepared), '--case', case])
                env = run.call_args[1]['env']
                self.assertEqual(env['CUBUTTERFLY_FACTOR_GRAPH_PREPARED'], str(prepared.with_suffix('.manifest.json')))
                self.assertEqual(env['CUBUTTERFLY_JIT_PYTHON'], str(Path(runner.__file__).resolve()))

    def test_profile_prepared_only_command(self):
        self.run_profile()

    def test_mapping_mismatch_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
            self.run_profile(actual_backend='factor-streamed')

    def test_changed_binary_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'identity changed'):
            self.run_profile(mutate_binary=True)

    def test_unknown_case_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_profile(case='unknown')


if __name__ == '__main__':
    unittest.main()
