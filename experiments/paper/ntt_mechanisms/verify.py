#!/usr/bin/env python3
"""Check frozen numerical export and sanitized input integrity (CPU only)."""
import argparse,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output-dir',type=Path,default=ROOT/'output')
a=p.parse_args()
for item in json.loads((ROOT/'provenance.json').read_text())['inputs']:
 path=ROOT/item['bundled']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['bundled_sanitized_sha256'],path
for item in json.loads((ROOT/'gpu/source-provenance.json').read_text()):
 path=ROOT/'gpu'/item['path']
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['bundled_sha256'],path
with (ROOT/'data/expected_rows.csv').open(newline='') as f: expected=list(csv.DictReader(f))
with (a.output_dir/'ntt_mechanism_profiles_20261118.csv').open(newline='') as f: actual=list(csv.DictReader(f))
assert actual==expected, 'Numerical rows differ from frozen source export'
assert len(actual)==82,len(actual)
for path in ROOT.rglob('*'):
 if path.is_file() and path.suffix in {'.json','.csv','.py','.md','.svg'}:
  text=path.read_text()
  assert ('/'+'home'+'/') not in text and ('/'+'sdb'+'/') not in text,path
print(f'PASS: {len(actual)} numerical rows match source exactly; bundled hashes and local-path anonymity checked')
