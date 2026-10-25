"""CPU-only checks for the install manifest and isolated installed tools."""
import argparse
import ast
import importlib
import importlib.util
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def installed_scripts():
    cmake = (ROOT / "CMakeLists.txt").read_text()
    files = []
    for block in re.findall(r"install\((?:FILES|PROGRAMS)\s+(.*?)\)", cmake, re.S):
        if 'DESTINATION "${CMAKE_INSTALL_BINDIR}"' in block:
            files.extend(re.findall(r"scripts/([\w]+\.py)", block))
    return files


def isolated_checks(bin_dir, templates):
    sys.path.insert(0, str(bin_dir.resolve()))
    for name in ("calibration_space", "evolutionary_search", "stage_service_calibration",
                 "calibrate_composition", "research_compile_requests", "jit_calibration_identity",
                 "stage_schedule_native"):
        module = importlib.import_module(name)
        if pathlib.Path(module.__file__).parent != bin_dir.resolve():
            raise AssertionError("import escaped installation: " + name)
    native = importlib.import_module('stage_schedule_native')
    if not pathlib.Path(native.__file__).with_suffix('.cpp').is_file():
        raise AssertionError('native schedule simulator source missing from installation')
    # This dependency is lazy during normal calibration; --help alone misses it.
    identity = importlib.import_module("jit_calibration_identity")
    import os
    os.environ.pop("CUBUTTERFLY_CALIBRATION_BROKER", None)
    if identity.calibration_identity() is not None:
        raise AssertionError("unbound calibration unexpectedly has a broker identity")
    spec = importlib.util.spec_from_file_location("installed_compiler", templates / "scripts/compile_module.py")
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    for operator, precision in (("fft", "fp32"), ("ntt", "word64"), ("fwht", "fp32"),
                                ("subset-zeta", "uint32"), ("superset-zeta", "uint64"),
                                ("structured-2x2", "fp64")):
        source = compiler.render_shared({"backend": "shared-iterative", "operator": operator,
            "precision": precision, "logN": 8, "stage_partition": [4, 4], "threads": 96,
            "local_stage_partitions": [[2, 2], [3, 1]], "exchange_chunks": [0, 0],
            "writer_aligned": True, "batch_tile": 3}, 80, "installation-smoke")
        if "shared_resident_kernel" not in source:
            raise AssertionError("installed generator lost resident lowering")
        pending = list(re.findall(r'#include\s*[<"]([^>"\n]+)[>"]', source))
        visited = set()
        while pending:
            include = pending.pop()
            if include in visited:
                continue
            visited.add(include)
            local = next((path for path in (templates / "src" / include, templates / "include" / include)
                          if path.is_file()), None)
            if local:
                pending.extend(re.findall(r'#include\s*[<"]([^>"\n]+)[>"]', local.read_text()))
            elif include.startswith(("cuntt/", "cubutterfly/")) or include.endswith(".cuh"):
                raise AssertionError("missing installed template dependency: " + include)
    print("installed calibration imports and six resident templates passed")


class InstallToolDependencies(unittest.TestCase):
    def test_install_manifest_covers_local_imports_including_lazy_paths(self):
        installed = set(installed_scripts())
        self.assertTrue(installed)
        for name in sorted(installed):
            tree = ast.parse((ROOT / "scripts" / name).read_text())
            for node in ast.walk(tree):
                modules = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
                for module in modules:
                    dependency = module.split(".")[0] + ".py"
                    if (ROOT / "scripts" / dependency).is_file():
                        self.assertIn(dependency, installed, name + " imports uninstalled " + dependency)

    def test_smoke_checks_in_isolated_staged_install(self):
        import shutil
        with tempfile.TemporaryDirectory(prefix="cubutterfly-install-import-") as directory:
            stage = pathlib.Path(directory)
            bin_dir = stage / "bin"
            templates = stage / "templates"
            bin_dir.mkdir()
            for name in installed_scripts():
                shutil.copy2(ROOT / "scripts" / name, bin_dir / name)
            shutil.copy2(ROOT / 'scripts/stage_schedule_native.cpp', bin_dir / 'stage_schedule_native.cpp')
            for name in ("include", "src", "scripts"):
                (templates / name).mkdir(parents=True)
            shutil.copytree(ROOT / "include", templates / "include", dirs_exist_ok=True)
            cmake = (ROOT / "CMakeLists.txt").read_text()
            for block in re.findall(r"install\((?:FILES|PROGRAMS)\s+(.*?)\)", cmake, re.S):
                for directory_name in ("src", "scripts"):
                    if 'templates/' + directory_name + '"' in block:
                        for relative in re.findall(r"(?:src|scripts)/[\w]+\.(?:cuh|py)", block):
                            shutil.copy2(ROOT / relative, templates / directory_name / pathlib.Path(relative).name)
            result = subprocess.run([sys.executable, "-I", str(pathlib.Path(__file__).resolve()),
                "--bin", str(bin_dir), "--templates", str(templates)], cwd=stage, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--bin" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--bin", required=True, type=pathlib.Path)
        parser.add_argument("--templates", required=True, type=pathlib.Path)
        args = parser.parse_args()
        isolated_checks(args.bin, args.templates)
    else:
        unittest.main()
