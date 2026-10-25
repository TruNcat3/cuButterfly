import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from benchmark.runner_registry import resolve


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_current_comprehensive_manifest",
    ROOT / "scripts/prepare_current_comprehensive_manifest.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CurrentComprehensiveManifestTests(unittest.TestCase):
    def test_current_paths_replace_registry_and_legacy_binaries(self):
        for registry in ("runners", "binaries"):
            with self.subTest(registry=registry), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                names = ("butterfly", "ntt", "fft-library")
                paths = [root / name for name in names]
                for path in (*paths, root / "libcuntt.so"):
                    path.write_bytes(b"current-build")
                document = {
                    "schema_version": 1,
                    "cases": [{"id": "unchanged", "runner": "ntt", "args": []}],
                    registry: {
                        name: ({"kind": "custom-kind", "executable": "old/" + name,
                                "metadata": "preserved"} if registry == "runners"
                               else "old/" + name)
                        for name in names
                    },
                }
                base = root / "base.json"
                base.write_text(json.dumps(document))
                result = MODULE.prepare(base, *paths)
                self.assertEqual(result["cases"], document["cases"])
                for name, path in zip(names, paths):
                    self.assertEqual(resolve(result, name, root)["path"], path)
                    self.assertEqual(result["binaries"][name], str(path))
                    if registry == "runners":
                        self.assertEqual(result["runners"][name]["kind"], "custom-kind")
                        self.assertEqual(result["runners"][name]["metadata"], "preserved")


if __name__ == "__main__":
    unittest.main()
