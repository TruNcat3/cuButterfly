"""Read default, non-folded cuFFTDx C2C block records without CUDA.

Only complete, recognized record files establish negative capability evidence.
The database's FFTsPerBlock field is a suggestion and is deliberately ignored.
This queries provider implementation presence, not launch geometry legality.
Default TRPOption=0 accepts any record implementation with matching EPT. For
non-folded C2C, effective EPT equals user EPT, including FP16. FP16 still requires
an even requested FPB and packs two FFTs per block-dimension Y coordinate; those
request geometry rules are outside this EPT-only capability query.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

SUPPORTED = "supported"
UNSUPPORTED = "unsupported"
UNKNOWN = "unknown"
_PRECISIONS = {"fp16": "half", "fp32": "float", "fp64": "double"}
_RECORD = re.compile(
    r"template\s*<\s*>\s*struct\s+block_fft_record\s*<\s*(\d+)\s*,\s*"
    r"(half|float|double)\s*,\s*fft_type::c2c\s*,\s*fft_direction::"
    r"(forward|inverse)\s*,\s*(\d+)\s*>\s*\{(.*?)\}\s*;", re.S)
_BLOB = re.compile(r"block_fft_implementation\s*<\s*(\d+)\s*,\s*(\d+)\s*,\s*"
                   r"(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>")
_ALIAS = re.compile(
    r"struct\s+block_fft_record\s*<\s*Size\s*,\s*Precision\s*,\s*Type\s*,\s*Direction\s*,\s*(\d+)\s*>"
    r"\s*:\s*public\s+block_fft_record\s*<\s*Size\s*,\s*Precision\s*,\s*Type\s*,\s*Direction\s*,\s*(\d+)\s*>\s*\{\s*\}")


def _strip_comments(text):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)


def _expand_includes(path, sources, stack=()):
    path = path.resolve()
    if path in stack:
        raise ValueError("cyclic cuFFTDx record include")
    raw = path.read_bytes()
    sources[str(path)] = hashlib.sha256(raw).hexdigest()
    text = _strip_comments(raw.decode())
    def expand(match):
        return _expand_includes(path.parent / match.group(1), sources, (*stack, path))
    text = re.sub(r'^\s*#\s*include\s*"([^"\n]+)"\s*$', expand, text, flags=re.M)
    # Conditional contents require preprocessing; do not infer an absence.
    if "#" in text:
        raise ValueError("unrecognized preprocessor directive in cuFFTDx records")
    return text


@dataclass
class Database:
    records: dict[tuple[int, str, str], dict[int, frozenset[int]]] = field(default_factory=dict)
    aliases: dict[int, int] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)
    sources: dict[str, str] = field(default_factory=dict)

    def metadata(self):
        sources = dict(sorted(self.sources.items()))
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        return dict(schema_version=1, fingerprint=fingerprint, sources=sources,
                    parsed_record_files=len(self.records), diagnostics=self.diagnostics,
                    status="partial" if self.diagnostics or not self.records else "parsed",
                    contract="non-folded C2C, default TRPOption=0, explicit user EPT; no FPB/geometry validation")

    def _resolve(self, sm):
        seen = set()
        while sm in self.aliases:
            if sm in seen:
                return None
            seen.add(sm)
            sm = self.aliases[sm]
        return sm

    def check(self, sm, precision, size, ept, direction):
        """Query default C2C implementation presence; sm uses MathDx units (800)."""
        precision = {16: "fp16", 32: "fp32", 64: "fp64"}.get(precision, precision)
        if precision not in _PRECISIONS or direction not in ("forward", "inverse"):
            return UNKNOWN
        records = self.records.get((self._resolve(sm), precision, direction))
        if records is None:
            return UNKNOWN
        return SUPPORTED if ept in records.get(size, ()) else UNSUPPORTED

    def check_both(self, sm, precision, size, ept):
        states = {self.check(sm, precision, size, ept, direction)
                  for direction in ("forward", "inverse")}
        if UNSUPPORTED in states:
            return UNSUPPORTED
        return UNKNOWN if UNKNOWN in states else SUPPORTED


def load_database(mathdx_root):
    """Parse MathDx installation root (the directory containing include/)."""
    root = Path(mathdx_root) / "include/cufftdx/include/database"
    db = Database()
    try:
        alias_path = (root / "detail/block_fft.hpp").resolve()
        raw = alias_path.read_bytes()
        db.sources[str(alias_path)] = hashlib.sha256(raw).hexdigest()
        aliases = _strip_comments(raw.decode())
        db.aliases = {int(source): int(target) for source, target in _ALIAS.findall(aliases)}
    except (OSError, UnicodeError) as error:
        db.diagnostics.append(str(error))
    for path in sorted((root / "records").glob("[0-9]*/database_fp*_*.hpp.inc")):
        name = re.fullmatch(r"database_(fp16|fp32|fp64)_(fwd|inv)\.hpp\.inc", path.name)
        if not name:
            continue
        precision = name[1]
        direction = {"fwd": "forward", "inv": "inverse"}[name[2]]
        sm = int(path.parent.name)
        try:
            text = _expand_includes(path, db.sources)
            records = {}
            matches = list(_RECORD.finditer(text))
            if not matches or _RECORD.sub("", text).strip():
                raise ValueError("unrecognized cuFFTDx record syntax")
            for match in matches:
                size, real, record_direction, record_sm, body = match.groups()
                # A textual include does not create a specialization for a
                # different SM. Only explicit inheritance in block_fft.hpp
                # establishes the architecture aliases resolved by Database.
                if (real != _PRECISIONS[precision] or record_direction != direction or
                        int(record_sm) != sm):
                    raise ValueError("record precision/direction/SM differs from file")
                body_match = re.fullmatch(
                    r"\s*static\s+constexpr\s+bool\s+defined\s*=\s*true\s*;\s*"
                    r"using\s+blobs\s*=\s*type_list\s*<(.*)>\s*;\s*", body, re.S)
                if not body_match:
                    raise ValueError("unrecognized cuFFTDx record body")
                blobs = body_match[1]
                implementations = _BLOB.findall(blobs)
                if not implementations or _BLOB.sub("", blobs).replace(",", "").strip():
                    raise ValueError("unrecognized cuFFTDx implementation")
                if int(size) in records:
                    raise ValueError("duplicate cuFFTDx size record")
                records[int(size)] = frozenset(int(blob[0]) for blob in implementations)
            db.records[(sm, precision, direction)] = records
        except (OSError, UnicodeError, ValueError) as error:
            db.diagnostics.append(f"{path}: {error}")
    return db


def render_header(db):
    """Render CPU-only C++ queries from exactly the Python capability data."""
    bindings = []
    for path, expected_sha in sorted(db.sources.items()):
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha:
            raise ValueError(f"cuFFTDx provider source changed after parsing: {path}")
        fnv = 14695981039346656037
        for byte in raw:
            fnv = ((fnv ^ byte) * 1099511628211) & ((1 << 64) - 1)
        bindings.append((path, fnv))
    lines = ["// Generated by scripts/cufftdx_capabilities.py; do not edit.",
             f"// Database fingerprint: {db.metadata()['fingerprint']}",
             "#pragma once", "#include <array>", "namespace cuntt::detail {",
             f"inline constexpr const char* cufftdx_provider_fingerprint = \"{db.metadata()['fingerprint']}\";",
             "struct CufftDxProviderSource { const char* path; unsigned long long fnv1a64; };",
             f"inline constexpr std::array<CufftDxProviderSource, {len(bindings)}> cufftdx_provider_sources = {{{{"]
    lines += [f"    {{{json.dumps(path)}, {fnv}ULL}}," for path, fnv in bindings]
    lines += ["}};", "enum class CufftDxSupport { Supported, Unsupported, Unknown };",
             "inline CufftDxSupport cufftdx_provider_support(unsigned sm, unsigned precision_bits,",
             "    unsigned size, unsigned ept, bool inverse) noexcept {",
             "#if defined(CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_X_TRANSPOSITION) || defined(CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_XY_TRANSPOSITION)",
             "    return CufftDxSupport::Unknown;",
             "#endif"]
    for source in sorted(db.aliases):
        target = db._resolve(source)
        if target is not None:
            lines.append(f"    if (sm == {source}) sm = {target};")
    for (sm, precision, direction), records in sorted(db.records.items()):
        bits = precision[2:]
        inv = "true" if direction == "inverse" else "false"
        lines.append(f"    if (sm == {sm} && precision_bits == {bits} && inverse == {inv}) {{")
        lines.append("        switch (size) {")
        for size, epts in sorted(records.items()):
            condition = " || ".join(f"ept == {ept}" for ept in sorted(epts))
            lines.append(f"        case {size}: return ({condition}) ? CufftDxSupport::Supported : CufftDxSupport::Unsupported;")
        lines += ["        default: return CufftDxSupport::Unsupported;", "        }", "    }"]
    lines += ["    return CufftDxSupport::Unknown;", "}",
              "inline CufftDxSupport cufftdx_provider_support_both(unsigned sm, unsigned precision_bits,",
              "    unsigned size, unsigned ept) noexcept {",
              "    const auto forward = cufftdx_provider_support(sm, precision_bits, size, ept, false);",
              "    const auto inverse = cufftdx_provider_support(sm, precision_bits, size, ept, true);",
              "    if (forward == CufftDxSupport::Unsupported || inverse == CufftDxSupport::Unsupported)",
              "        return CufftDxSupport::Unsupported;",
              "    if (forward == CufftDxSupport::Unknown || inverse == CufftDxSupport::Unknown)",
              "        return CufftDxSupport::Unknown;",
              "    return CufftDxSupport::Supported;", "}", "}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mathdx", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", type=Path)
    args = parser.parse_args()
    database = load_database(args.mathdx)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_header(database))
    if args.metadata_output:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(database.metadata(), indent=2) + "\n")
    for diagnostic in database.diagnostics:
        print(f"cuFFTDx capability warning: {diagnostic}")


if __name__ == "__main__":
    main()
