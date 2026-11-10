#!/usr/bin/env python3
"""Restore pinned HEonGPU runtime dependencies without sudo or GPU execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    ('libntl43', '11.4.3-1build1', 'pool/universe/n/ntl/libntl43_11.4.3-1build1_amd64.deb',
     '403fab73e0ce12d8263f243322d11c404fe13c93877d2c7b3d97e00065619d67'),
    ('libgf2x3', '1.3.0-1', 'pool/universe/g/gf2x/libgf2x3_1.3.0-1_amd64.deb',
     'ff81ea14795442bba57cab1c5f8cfbca6e23aa44a27ff17bf8e952bfe4587ac1'),
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/e13_assets/heongpu-runtime-dependencies')
    parser.add_argument('--binary', type=Path, default=ROOT / 'results/full_current_20261025/ckks/consumer_builds/bootstrap_v1_small/cubutterfly_heongpu_ckks')
    args = parser.parse_args()
    output = args.output.resolve()
    for path in (output, args.binary.resolve()):
        if path == Path('/sdb') or Path('/sdb') in path.parents:
            parser.error('/sdb is outside this project\'s permitted storage paths')
    archives, prefix = output / 'archives', output / 'root'
    archives.mkdir(parents=True, exist_ok=True)
    prefix.mkdir(parents=True, exist_ok=True)
    records = []
    for name, version, relative, expected in PACKAGES:
        target = archives / Path(relative).name
        url = 'https://archive.ubuntu.com/ubuntu/' + relative
        if not target.exists():
            partial = target.with_suffix('.deb.partial')
            with urllib.request.urlopen(url, timeout=120) as source, partial.open('wb') as destination:
                while block := source.read(1024 * 1024):
                    destination.write(block)
            partial.replace(target)
        actual = sha256(target)
        if actual != expected:
            raise RuntimeError(f'package checksum mismatch: {target}')
        metadata = subprocess.check_output(['dpkg-deb', '-f', str(target), 'Package', 'Version', 'Architecture'], text=True)
        if f'Package: {name}\n' not in metadata or f'Version: {version}\n' not in metadata or 'Architecture: amd64\n' not in metadata:
            raise RuntimeError(f'package metadata mismatch: {metadata}')
        subprocess.run(['dpkg-deb', '-x', str(target), str(prefix)], check=True)
        records.append(dict(package=name, version=version, architecture='amd64', source=url,
                            archive=str(target), sha256=actual))
    directory = prefix / 'usr/lib/x86_64-linux-gnu'
    environment = dict(os.environ)
    environment['LD_LIBRARY_PATH'] = str(directory) + (':' + environment['LD_LIBRARY_PATH'] if environment.get('LD_LIBRARY_PATH') else '')
    result = subprocess.run(['ldd', str(args.binary.resolve())], env=environment, capture_output=True, text=True)
    (output / 'ldd.log').write_text(result.stdout + result.stderr)
    resolved = result.returncode == 0 and 'not found' not in result.stdout + result.stderr
    libraries = {str(path.resolve()): sha256(path.resolve()) for path in directory.glob('*.so.*') if path.is_file()}
    receipt = dict(schema='heongpu-persistent-runtime-dependencies-v1', status='ready' if resolved else 'unresolved',
                   packages=records, libraries=libraries, binary=str(args.binary.resolve()), binary_sha256=sha256(args.binary),
                   ld_library_path_prepend=str(directory), verification_environment_ld_library_path=environment['LD_LIBRARY_PATH'],
                   ldd_returncode=result.returncode, all_dependencies_resolved=resolved, gpu_executed=False,
                   historical_version_match='Original records identify libntl.so.43 and libgf2x.so.3.0.0, but no original package version/hash receipt was located; ABI matches, exact old package identity unproven.')
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0 if resolved else 1


if __name__ == '__main__':
    raise SystemExit(main())
