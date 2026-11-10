"""CPU-only reproduction of the shared-JIT calibration identity boundary."""
import ast
import copy
import os
from pathlib import Path
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
import jit_calibration_identity as jit
import semantic_compile_broker as broker


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        root = self.directory / 'templates'
        mathdx = self.directory / 'mathdx'
        toolkit = self.directory / 'cuda'
        for relative, content in {
            'templates/include/api.h': 'ABI',
            'templates/src/tile.cuh': 'template',
            'templates/scripts/compile_module.py': 'command = ["--shared", "--cudart=shared"]\n',
            'templates/scripts/resident_mapping.py': 'mapping',
            'templates/scripts/cufftdx_capabilities.py': 'provider parser',
            'mathdx/include/cufftdx.hpp': 'mathdx',
            'cuda/bin/nvcc': 'compiler',
            'cuda/lib64/libcudart.so': 'shared runtime',
            'cuda/lib64/libcudart_static.a': 'static runtime',
        }.items():
            path = self.directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.config = dict(root=str(root), mathdx=str(mathdx), nvcc=str(toolkit / 'bin/nvcc'),
                           sm=80, cache=str(self.directory / 'cache'))
        self.environment = patch.dict(os.environ, {'LD_LIBRARY_PATH': ''}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        identity = dict(config=self.config, sources_sha256=broker._sources(self.config),
                        compiler_version='fake nvcc', compiler_stat=broker._stat(self.config['nvcc']))
        self.document = dict(schema=broker.SCHEMA, **identity, id=broker._id(identity))
        self.broker_directory = self.directory / 'broker'
        self.broker_directory.mkdir()
        broker._atomic(self.broker_directory / 'config.json', self.document)

    def opt_in(self):
        os.environ['CUBUTTERFLY_CALIBRATION_BROKER'] = str(self.broker_directory)
        for variable, field in (('CUBUTTERFLY_TEMPLATE_ROOT', 'root'),
                                ('CUBUTTERFLY_MATHDX_ROOT', 'mathdx'),
                                ('CUBUTTERFLY_NVCC', 'nvcc'),
                                ('CUBUTTERFLY_JIT_CACHE', 'cache')):
            os.environ[variable] = self.config[field]

    def manifest(self):
        return {'measurement_identity': {'library_sha256': 'unchanged-library',
                                         'jit_runtime': jit.runtime_identity(self.document)}}

    def test_old_library_only_calibration_rejected_for_shared(self):
        with self.assertRaisesRegex(ValueError, 'runtime-bound calibration'):
            jit.validate_timing({'measurement_identity': {'library_sha256': 'unchanged-library'}},
                                self.document)

    def test_matching_calibration_accepted(self):
        jit.validate_timing(self.manifest(), self.document)

    def test_generator_and_template_changes_invalidate_broker(self):
        for relative in ('scripts/compile_module.py', 'src/tile.cuh', 'include/api.h'):
            path = Path(self.config['root']) / relative
            original = path.read_text()
            path.write_text(original + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'broker inputs changed'):
                jit.validate_timing({}, self.document)
            path.write_text(original)

    def test_mathdx_change_invalidates_broker(self):
        (Path(self.config['mathdx']) / 'include/cufftdx.hpp').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'broker inputs changed'):
            jit.runtime_identity(self.document)

    def test_runtime_file_change_rejects_same_library_and_broker(self):
        manifest = self.manifest()
        (self.directory / 'cuda/lib64/libcudart.so').write_text('replacement runtime')
        with self.assertRaisesRegex(ValueError, 'runtime identity mismatch'):
            jit.validate_timing(manifest, self.document)

    def test_new_broker_and_cache_reject_old_bound_calibration(self):
        manifest = self.manifest()
        new = copy.deepcopy(self.document)
        new['config']['cache'] += '-different'
        new['id'] = 'new-broker'
        with self.assertRaisesRegex(ValueError, 'runtime identity mismatch'):
            jit.validate_timing(manifest, new)

    def test_opt_in_and_environment_consistency(self):
        self.assertIsNone(jit.calibration_identity())
        self.opt_in()
        self.assertEqual(jit.calibration_identity(), jit.runtime_identity(self.document))
        os.environ['CUBUTTERFLY_TEMPLATE_ROOT'] += '-different'
        with self.assertRaisesRegex(ValueError, 'environment differs'):
            jit.calibration_identity()

    def test_stage_checkpoint_records_runtime_and_static_probe_hash(self):
        import stage_service_calibration as stage
        self.opt_in()
        binary = self.directory / 'stage_probe'
        binary.write_bytes(b'unchanged-stage-probe')
        identity = stage._cache_identity(binary, {}, {}, 'points')
        self.assertEqual(identity['jit_runtime'], jit.calibration_identity())
        self.assertIn('binary_sha256', identity)

    def test_measurement_manifest_identity_uses_same_binding(self):
        source = (ROOT / 'scripts/calibrate_local_hardware.py').read_text()
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'measurement_identity')
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None
        import hashlib
        import types
        namespace = dict(hashlib=hashlib, os=os, pathlib=pathlib,
                         effective_compile_mode=lambda path: 'research')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<measurement_identity>', 'exec'), namespace)
        self.opt_in()
        for name in ('cubutterfly_bench', 'cuntt_bench'):
            (self.directory / name).write_bytes(b'unchanged-binary')
        identity = namespace['measurement_identity'](types.SimpleNamespace(build_dir=self.directory),
                                                      types.SimpleNamespace(digest=lambda: 'same-library'))
        self.assertEqual(identity['jit_runtime'], jit.calibration_identity())

    def test_untimed_diagnostic_skips_new_guard_before_cuda(self):
        source = (ROOT / 'scripts/run_async_semantic_campaign.py').read_text()
        main = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        guard = next(n for n in main.body if isinstance(n, ast.If)
                     and isinstance(n.test, ast.Attribute) and n.test.attr == 'timing'
                     and any(isinstance(x, ast.ImportFrom) and x.module == 'jit_calibration_identity'
                             for x in ast.walk(n)))
        import types
        namespace = dict(args=types.SimpleNamespace(timing=False), config=self.document)
        exec(compile(ast.Module(body=[guard], type_ignores=[]), '<timing_guard>', 'exec'), namespace)
        self.assertLess(source.index('validate_timing('), source.index('    import torch\n'))

    def test_static_historical_calibration_still_accepted(self):
        source = Path(self.config['root']) / 'scripts/compile_module.py'
        source.write_text('command = ["--shared"]\n')
        self.document['sources_sha256'] = broker._sources(self.config)
        jit.validate_timing({}, self.document)

    def test_composition_checkpoint_records_same_binding(self):
        source = (ROOT / 'scripts/calibrate_composition.py').read_text()
        function = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == 'run_calibration')
        start = next(i for i, n in enumerate(function.body)
                     if isinstance(n, ast.ImportFrom) and n.module == 'jit_calibration_identity')
        statements = function.body[start:start + 3]
        self.opt_in()
        namespace = {'inputs': {}}
        exec(compile(ast.Module(body=statements, type_ignores=[]), '<composition_identity>', 'exec'), namespace)
        self.assertEqual(namespace['inputs']['jit_runtime'], jit.calibration_identity())


if __name__ == '__main__':
    unittest.main()
