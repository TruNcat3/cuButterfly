#!/usr/bin/env python3
"""Package the current research source tree for a second GPU host.

``git archive`` is deliberately not used: this repository contains important
untracked research kernels, scripts and workload matrices.  The package is
bounded by an explicit source whitelist and records a relative-path SHA256
manifest beside the archive.  It never downloads, commits, or overwrites an
existing archive unless ``--force`` is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "cubutterfly-research-source-package-v1"
ARCHIVE_PREFIX = "cuButterfly/"
WHITELIST_DIRS = (
    "src", "include", "apps", "examples", "scripts", "config", "configs", "benchmarks",
    "tests", "cmake", "patches", "docs",
)
WHITELIST_FILES = (
    "CMakeLists.txt", "LICENSE", "LICENSE.txt", "README", "README.md", "AGENTS.md",
    "CITATION.cff", "CHANGELOG.md",
)
LEGACY_BUILD_FILES = ("results/v100_scaling_full_summary.csv",)
PRUNED_DIRS = {".git", ".gitmodules", ".cache", "__pycache__", ".pytest_cache",
               "build", "results", "external", "paper"}
SECRET_NAMES = {
    ".env", ".env.local", ".env.production", ".env.development", "credentials",
    "credentials.json", "kubeconfig", "id_rsa", "id_ed25519", "id_ecdsa",
    "private_key", "private.key", "secret", "secrets.json", "token", "password",
}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".zip", ".7z", ".whl")


class PackageError(RuntimeError):
    """The source tree cannot be packaged safely."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_secret(path: Path) -> bool:
    name = path.name.lower()
    return name in SECRET_NAMES or name.startswith(".env.") or name.endswith((".pem", ".key"))


def _is_excluded_asset(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return True
    if suffix in IMAGE_SUFFIXES:
        try:
            return path.stat().st_size > MAX_IMAGE_BYTES
        except OSError as error:
            raise PackageError(f"cannot inspect asset {path}") from error
    return path.name.lower().endswith(ARCHIVE_SUFFIXES)


def _link_target(path: Path) -> str:
    return os.readlink(str(path))


def _validate_symlink(path: Path, root: Path, allowed_roots: tuple[Path, ...]) -> None:
    link_target = _link_target(path)
    if Path(link_target).is_absolute():
        raise PackageError(f"absolute source symlink is not portable: {path}")
    target = path.resolve(strict=False)
    if not _inside(target, root):
        raise PackageError(f"source symlink escapes repository: {path} -> {link_target}")
    if not any(_inside(target, allowed) or target == allowed.resolve() for allowed in allowed_roots):
        raise PackageError(f"source symlink targets an un-packaged dependency: {path} -> {link_target}")
    if target.is_dir():
        # Directory links are not traversed by this package walk.  Rejecting
        # them avoids a tar member whose contents depend on traversal order.
        raise PackageError(f"directory source symlink is not portable: {path}")


def _candidate_files(root: Path, output_paths: set[Path]) -> list[tuple[Path, str]]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise PackageError(f"source root does not exist: {root}")
    allowed_roots = tuple((root / name).resolve() for name in WHITELIST_DIRS if (root / name).exists())
    candidates: list[tuple[Path, str]] = []

    def consider(path: Path) -> None:
        resolved = path.resolve(strict=False)
        if path in output_paths or resolved in output_paths:
            return
        relative = path.relative_to(root).as_posix()
        if not relative or relative.startswith("../") or "/../" in relative:
            raise PackageError(f"invalid source path: {relative}")
        if _is_secret(path):
            return
        if _is_excluded_asset(path):
            return
        if path.is_symlink():
            _validate_symlink(path, root, allowed_roots)
        if not path.is_file() and not path.is_symlink():
            return
        candidates.append((path, relative))

    for name in WHITELIST_FILES:
        path = root / name
        if path.exists() or path.is_symlink():
            consider(path)
    # CMake's default selector generation consumes this one tracked legacy
    # table.  It is build input only; the package must not import any other
    # historical result or calibration artifact.
    for relative in LEGACY_BUILD_FILES:
        path = root / relative
        if path.exists() or path.is_symlink():
            consider(path)
    for name in WHITELIST_DIRS:
        base = root / name
        if not base.exists() and not base.is_symlink():
            continue
        if base.is_symlink():
            _validate_symlink(base, root, allowed_roots)
            consider(base)
            continue
        for directory, directories, files in os.walk(base, topdown=True, followlinks=False):
            directory_path = Path(directory)
            directories[:] = sorted(directories)
            files = sorted(files)
            # A symlinked directory is moved into files so it is validated and
            # represented once, without following it during traversal.
            for child in list(directories):
                child_path = directory_path / child
                if child in PRUNED_DIRS:
                    directories.remove(child)
                elif child_path.is_symlink():
                    directories.remove(child)
                    consider(child_path)
            for filename in files:
                consider(directory_path / filename)
    by_relative = {relative: path for path, relative in candidates}
    return [(by_relative[relative], relative) for relative in sorted(by_relative)]


def _manifest_entry(path: Path, relative: str) -> dict[str, Any]:
    if path.is_symlink():
        target = _link_target(path)
        target_path = path.resolve(strict=False)
        digest = sha256(target_path) if target_path.is_file() else hashlib.sha256(target.encode()).hexdigest()
        entry = {"path": relative, "sha256": digest, "size": 0, "type": "symlink", "target": target}
    else:
        entry = {"path": relative, "sha256": sha256(path), "size": path.stat().st_size, "type": "file"}
    if relative in LEGACY_BUILD_FILES:
        entry.update({"role": "legacy-build-input-only",
                      "claim_scope": "not target calibration or performance claim"})
    return entry


def _sidecar_paths(output: Path) -> tuple[Path, Path]:
    manifest = Path(str(output) + ".manifest.json")
    checksum = Path(str(output) + ".sha256")
    return manifest, checksum


def create_package(root: Path, output: Path, *, force: bool = False) -> dict[str, Any]:
    """Create an archive and sidecars, returning the written manifest."""
    root = Path(root).resolve()
    output = Path(output).expanduser().resolve()
    manifest_path, checksum_path = _sidecar_paths(output)
    existing = [path for path in (output, manifest_path, checksum_path) if path.exists()]
    if existing and not force:
        raise PackageError("refusing to overwrite existing package sidecar/archive: "
                           + ", ".join(str(path.name) for path in existing))
    output.parent.mkdir(parents=True, exist_ok=True)
    output_paths = {path.resolve() for path in (output, manifest_path, checksum_path)}
    candidates = _candidate_files(root, output_paths)
    if not candidates:
        raise PackageError("source whitelist produced no files")
    entries = [_manifest_entry(path, relative) for path, relative in candidates]
    with tarfile.open(output, mode="w:gz") as archive:
        for path, relative in candidates:
            archive.add(path, arcname=ARCHIVE_PREFIX + relative, recursive=False)
    archive_digest = sha256(output)
    manifest = {
        "schema": SCHEMA,
        "archive": {"name": output.name, "sha256": archive_digest},
        "source_root": root.name,
        "member_prefix": ARCHIVE_PREFIX.rstrip("/"),
        "files": entries,
        "file_count": len(entries),
        "policy": {
            "whitelist_directories": list(WHITELIST_DIRS),
            "excluded_trees": sorted(PRUNED_DIRS | {"paper", "external", "results"}),
            "excluded_historical_pdfs": True,
            "max_packaged_image_bytes": MAX_IMAGE_BYTES,
            "legacy_build_inputs": [
                {"path": relative, "role": "legacy-build-input-only",
                 "claim_scope": "not target calibration or performance claim"}
                for relative in LEGACY_BUILD_FILES
            ],
            "git_archive_used": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_path.write_text(f"{archive_digest}  {output.name}\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root (default: repository containing this script)")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination .tar.gz; existing files are never overwritten by default")
    parser.add_argument("--force", action="store_true", help="allow replacing the explicit output and sidecars")
    args = parser.parse_args(argv)
    try:
        manifest = create_package(args.root, args.output, force=args.force)
    except (OSError, PackageError, ValueError) as error:
        print(f"package_research_source: {error}", file=os.sys.stderr)
        return 2
    manifest_path, _ = _sidecar_paths(Path(args.output).expanduser().resolve())
    print(json.dumps({"archive": manifest["archive"], "manifest": str(manifest_path),
                      "file_count": manifest["file_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
