from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cufftdx_capabilities import load_database, render_header


@pytest.fixture
def tmp_path(tmpdir):
    return Path(str(tmpdir))


def record(size=1024, direction="forward", ept=16, fpb=1, sm=800):
    return f"""template<> struct block_fft_record<{size}, double, fft_type::c2c,
    fft_direction::{direction}, {sm}> {{
      static constexpr bool defined = true;
      using blobs = type_list<block_fft_implementation<{ept}, {ept}, 64, {fpb}, 8192, 549>>;
    }};"""


def installation(tmp_path):
    database = tmp_path / "include/cufftdx/include/database"
    (database / "detail").mkdir(parents=True)
    (database / "detail/block_fft.hpp").write_text("""
      template<unsigned int Size, class Precision, fft_type Type, fft_direction Direction>
      struct block_fft_record<Size, Precision, Type, Direction, 860>:
        public block_fft_record<Size, Precision, Type, Direction, 800> {};
    """)
    (database / "records/800").mkdir(parents=True)
    for suffix, direction in (("fwd", "forward"), ("inv", "inverse")):
        (database / f"records/800/database_fp64_{suffix}.hpp.inc").write_text(record(direction=direction))
    return database


def test_direction_precision_size_ept_and_alias(tmp_path):
    installation(tmp_path)
    db = load_database(tmp_path)
    assert not db.diagnostics
    assert db.check_both(800, "fp64", 1024, 16) == "supported"
    assert db.check_both(860, 64, 1024, 16) == "supported"
    assert db.check_both(800, "fp64", 1024, 32) == "unsupported"
    assert db.check_both(800, "fp64", 4096, 16) == "unsupported"
    assert db.check_both(800, "fp32", 1024, 16) == "unknown"
    assert db.check_both(810, "fp64", 1024, 16) == "unknown"
    assert db.check(800, "fp64", 1024, 16, "invalid") == "unknown"


def test_direction_missing_or_unsupported(tmp_path):
    root = installation(tmp_path)
    inverse = root / "records/800/database_fp64_inv.hpp.inc"
    inverse.unlink()
    assert load_database(tmp_path).check_both(800, "fp64", 1024, 16) == "unknown"
    inverse.write_text(record(direction="inverse", ept=8))
    db = load_database(tmp_path)
    assert db.check(800, "fp64", 1024, 16, "forward") == "supported"
    assert db.check_both(800, "fp64", 1024, 16) == "unsupported"


def test_include_forwarding_and_fpb_is_only_a_suggestion(tmp_path):
    root = installation(tmp_path)
    forward = root / "records/800/database_fp64_fwd.hpp.inc"
    (forward.parent / "shared.inc").write_text(record(fpb=7))
    forward.write_text('#include "shared.inc"\n')
    db = load_database(tmp_path)
    assert db.check_both(800, "fp64", 1024, 16) == "supported"
    assert len(db.sources) == 4
    assert db.metadata()["status"] == "parsed"


@pytest.mark.parametrize("include", [False, True])
def test_record_sm_mismatch_does_not_create_specialization(tmp_path, include):
    root = installation(tmp_path)
    forward = root / "records/800/database_fp64_fwd.hpp.inc"
    if include:
        (forward.parent / "different_sm.inc").write_text(record(sm=700))
        forward.write_text('#include "different_sm.inc"\n')
    else:
        forward.write_text(record(sm=700))
    db = load_database(tmp_path)
    assert db.check(800, "fp64", 1024, 16, "forward") == "unknown"
    assert db.check(800, "fp64", 1024, 32, "forward") == "unknown"
    assert db.check(700, "fp64", 1024, 16, "forward") == "unknown"
    assert "SM differs" in db.diagnostics[0]


def test_fp16_c2c_uses_explicit_ept_without_halving(tmp_path):
    root = installation(tmp_path)
    for suffix, direction in (("fwd", "forward"), ("inv", "inverse")):
        text = record(direction=direction, ept=8, fpb=2).replace("double", "half")
        (root / f"records/800/database_fp16_{suffix}.hpp.inc").write_text(text)
    db = load_database(tmp_path)
    assert db.check_both(800, "fp16", 1024, 8) == "supported"
    assert db.check_both(800, "fp16", 1024, 16) == "unsupported"


def test_render_binds_standard_fnv1a64_and_rejects_changed_source(tmp_path):
    root = installation(tmp_path)
    db = load_database(tmp_path)
    header = render_header(db)
    # Independently fixed standard FNV-1a64 test vector: ASCII "hello".
    sample = tmp_path / "sample.inc"
    sample.write_bytes(b"hello")
    import hashlib
    db.sources[str(sample.resolve())] = hashlib.sha256(b"hello").hexdigest()
    assert "11831194018420276491ULL" in render_header(db)
    assert "std::array<CufftDxProviderSource, 3>" in header
    target = root / "records/800/database_fp64_fwd.hpp.inc"
    target.write_text(target.read_text() + "\n// upgraded in place\n")
    with pytest.raises(ValueError, match="source changed after parsing"):
        render_header(db)


@pytest.mark.parametrize("bad", ["new_format();", "#if CUSTOM\n#endif", "", ' #include "missing.inc"'])
def test_unrecognized_file_is_unknown_and_auditable(tmp_path, bad):
    root = installation(tmp_path)
    (root / "records/800/database_fp64_fwd.hpp.inc").write_text(bad)
    db = load_database(tmp_path)
    assert db.check(800, "fp64", 1024, 32, "forward") == "unknown"
    assert db.check_both(800, "fp64", 1024, 16) == "unknown"
    assert db.metadata()["status"] == "partial"
    assert db.diagnostics


def test_generated_header_matches_python_and_compiles_on_cpu(tmp_path):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    installation(tmp_path)
    db = load_database(tmp_path)
    header = tmp_path / "capabilities.hpp"
    header.write_text(render_header(db))
    assert db.metadata()["fingerprint"] in header.read_text()
    assert "inline constexpr const char* cufftdx_provider_fingerprint" in header.read_text()
    source = tmp_path / "check.cpp"
    source.write_text('''#include "capabilities.hpp"
using namespace cuntt::detail;
int main() {
 return cufftdx_provider_support_both(860,64,1024,16) != CufftDxSupport::Supported ||
        cufftdx_provider_support_both(800,64,1024,32) != CufftDxSupport::Unsupported ||
        cufftdx_provider_support_both(810,64,1024,16) != CufftDxSupport::Unknown;
}
''')
    executable = tmp_path / "check"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    source.write_text('''#define CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_X_TRANSPOSITION
#include "capabilities.hpp"
int main() {
 return cuntt::detail::cufftdx_provider_support_both(800,64,1024,16) !=
        cuntt::detail::CufftDxSupport::Unknown;
}
''')
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)


def test_installed_mathdx_fp64_regressions():
    root = Path(__file__).resolve().parents[1] / "external/mathdx/nvidia/mathdx"
    if not root.is_dir():
        pytest.skip("MathDx installation unavailable")
    db = load_database(root)
    assert not db.diagnostics
    for size in (1024, 4096):
        assert db.check_both(800, "fp64", size, 32) == "unsupported"
        assert db.check_both(800, "fp64", size, 16) == "supported"
