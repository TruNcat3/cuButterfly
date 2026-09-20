import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "research_source_package", ROOT / "scripts/package_research_source.py")
packager = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packager)


class ResearchSourcePackageTests(unittest.TestCase):
    def make_tree(self, root):
        for directory in packager.WHITELIST_DIRS:
            (root / directory).mkdir(parents=True, exist_ok=True)
        (root / "src/runtime.cu").write_text("runtime")
        (root / "include/runtime.hpp").write_text("header")
        (root / "examples/untracked_example.cpp").write_text("int main() { return 0; }\n")
        (root / "scripts/untracked_runner.py").write_text("print('runner')\n")
        (root / "config/research_campaign").mkdir(parents=True, exist_ok=True)
        (root / "config/research_campaign/composed.json").write_text('{"workloads": []}\n')
        (root / "benchmarks/crypto_ntt").mkdir(parents=True, exist_ok=True)
        (root / "benchmarks/crypto_ntt/CMakeLists.txt").write_text("project(test)\n")
        (root / "tests/untracked_test.py").write_text("assert True\n")
        for filename in packager.WHITELIST_FILES:
            if filename in {"LICENSE.txt", "README"}:
                continue
            (root / filename).write_text(filename + "\n")

    def test_untracked_whitelist_manifest_and_extract_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            self.make_tree(root)
            for relative, content in {
                "results/old/selector.csv": "old",
                "build/generated.bin": "build",
                "external/VkFFT/source.hpp": "third party",
                "paper/historical.pdf": "pdf",
                ".git/config": "credential-ish",
                "cache/ignored.txt": "cache",
            }.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            (root / "results/v100_scaling_full_summary.csv").write_text("operator,correct\nfft,1\n")
            # Historical PDFs are excluded even under an otherwise allowed
            # documentation tree; large raster assets are excluded as well.
            (root / "docs/history.pdf").write_bytes(b"pdf")
            (root / "docs/large.png").write_bytes(b"x" * (packager.MAX_IMAGE_BYTES + 1))
            (root / "scripts/.env").write_text("TOKEN=secret\n")
            output = Path(directory) / "cuButterfly-v09-source.tar.gz"
            manifest = packager.create_package(root, output)
            self.assertEqual(manifest["file_count"], len(manifest["files"]))
            names = set()
            with tarfile.open(output, "r:gz") as archive:
                names = set(archive.getnames())
                self.assertIn("cuButterfly/scripts/untracked_runner.py", names)
                self.assertIn("cuButterfly/examples/untracked_example.cpp", names)
                self.assertIn("cuButterfly/config/research_campaign/composed.json", names)
                self.assertIn("cuButterfly/results/v100_scaling_full_summary.csv", names)
                self.assertNotIn("cuButterfly/results/old/selector.csv", names)
                self.assertNotIn("cuButterfly/build/generated.bin", names)
                self.assertNotIn("cuButterfly/external/VkFFT/source.hpp", names)
                self.assertNotIn("cuButterfly/paper/historical.pdf", names)
                self.assertNotIn("cuButterfly/docs/history.pdf", names)
                self.assertNotIn("cuButterfly/docs/large.png", names)
                self.assertNotIn("cuButterfly/scripts/.env", names)
                extract = Path(directory) / "extract"
                archive.extractall(extract)
            manifest_path = Path(str(output) + ".manifest.json")
            checksum_path = Path(str(output) + ".sha256")
            saved = json.loads(manifest_path.read_text())
            self.assertEqual(saved["archive"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(saved["file_count"], len(saved["files"]))
            legacy = next(item for item in saved["files"]
                          if item["path"] == "results/v100_scaling_full_summary.csv")
            self.assertEqual(legacy["role"], "legacy-build-input-only")
            self.assertEqual(legacy["claim_scope"], "not target calibration or performance claim")
            self.assertEqual(saved["policy"]["legacy_build_inputs"][0]["path"],
                             "results/v100_scaling_full_summary.csv")
            self.assertEqual(checksum_path.read_text().split()[0], saved["archive"]["sha256"])
            for entry in saved["files"]:
                relative = entry["path"]
                self.assertFalse(Path(relative).is_absolute())
                self.assertNotIn("..", Path(relative).parts)
                extracted = Path(directory) / "extract" / "cuButterfly" / relative
                self.assertTrue(extracted.exists() or extracted.is_symlink())
                if entry["type"] == "file":
                    self.assertEqual(hashlib.sha256(extracted.read_bytes()).hexdigest(), entry["sha256"])

    def test_existing_archive_is_not_overwritten_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            self.make_tree(root)
            output = Path(directory) / "source.tar.gz"
            packager.create_package(root, output)
            original = output.read_bytes()
            with self.assertRaisesRegex(packager.PackageError, "refusing to overwrite"):
                packager.create_package(root, output)
            self.assertEqual(output.read_bytes(), original)
            packager.create_package(root, output, force=True)

    def test_external_symlink_is_rejected_but_internal_file_link_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "repo"
            root.mkdir()
            self.make_tree(root)
            target = base / "outside.hpp"
            target.write_text("outside")
            (root / "include/escape.hpp").symlink_to(target)
            with self.assertRaisesRegex(packager.PackageError, "absolute|escapes repository"):
                packager.create_package(root, base / "bad.tar.gz")
            (root / "include/escape.hpp").unlink()
            (root / "src/shared.hpp").write_text("inside")
            (root / "include/internal.hpp").symlink_to(Path("../src/shared.hpp"))
            output = base / "good.tar.gz"
            manifest = packager.create_package(root, output)
            entry = next(item for item in manifest["files"] if item["path"] == "include/internal.hpp")
            self.assertEqual(entry["type"], "symlink")
            self.assertEqual(entry["target"], "../src/shared.hpp")

    def test_cli_reports_success_and_package_contains_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            root.mkdir()
            self.make_tree(root)
            output = Path(directory) / "source.tar.gz"
            self.assertEqual(packager.main(["--root", str(root), "--output", str(output)]), 0)
            with tarfile.open(output, "r:gz") as archive:
                self.assertIn("cuButterfly/scripts/untracked_runner.py", archive.getnames())


if __name__ == "__main__":
    unittest.main()
