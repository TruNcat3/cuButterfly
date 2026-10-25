"""Repository seeds must never replace a user's installed hardware profile."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import static_hardware_profile


def test_cmake_seed_install_preserves_local_files_and_destdir(tmp_path):
    source = tmp_path / "source"
    tag = "test-sm80-100B"
    seeds = source / "config" / "hardware" / tag
    seeds.mkdir(parents=True)
    (seeds / "hardware_model.json").write_text('{"origin":"bundled"}')
    (seeds / "hardware_profile.json").write_text('{"rate":100}')
    script = tmp_path / "install.cmake"
    script.write_text((ROOT / "cmake/install_hardware_profiles.cmake.in").read_text()
                      .replace("@CMAKE_CURRENT_SOURCE_DIR@", str(source))
                      .replace("@CMAKE_INSTALL_DATADIR@", "share"))
    staging = tmp_path / "staging"
    installed = staging / "opt/cubutterfly/share/cuButterfly/hardware" / tag
    installed.mkdir(parents=True)
    local = installed / "hardware_model.json"
    local.write_text('{"origin":"local-refreshed"}')
    env = dict(os.environ, DESTDIR=str(staging))
    command = ["cmake", "-DCMAKE_INSTALL_PREFIX=/opt/cubutterfly", "-P", str(script)]
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert json.loads(local.read_text()) == {"origin": "local-refreshed"}
    assert json.loads((installed / "hardware_profile.json").read_text()) == {"rate": 100}
    (installed / "hardware_profile.json").write_text('{"rate":200}')
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    assert json.loads((installed / "hardware_profile.json").read_text()) == {"rate": 200}


def test_bundled_profiles_are_exact_hardware_seeds_not_complete_cost_models():
    root = ROOT / "config/hardware"
    profiles = list(root.glob("*/hardware_model.json"))
    assert profiles
    for path in profiles:
        document = static_hardware_profile.load(path)
        expected = static_hardware_profile.identity(document)
        assert path.parent.name == static_hardware_profile._tag(expected)
        selected = static_hardware_profile.select(expected, bundled_root=root)
        assert selected["action"] == "reuse"
        assert selected["selected"]["hardware_complete"] is True
        assert selected["selected"]["layers"]["hardware_profile"]["present"] is True
        assert selected["selected"]["layers"]["cost_model"]["validated"] is False
        # These historical observations are available locally but not required
        # in a source distribution. If present, verify evidence provenance.
        import hashlib
        for source, digest in document["provenance"]["source_sha256"].items():
            if (ROOT / source).is_file():
                assert hashlib.sha256((ROOT / source).read_bytes()).hexdigest() == digest
