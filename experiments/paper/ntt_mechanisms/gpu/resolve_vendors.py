#!/usr/bin/env python3
"""Resolve pinned arithmetic sources from supplied archives; never download."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
LOCK=json.loads((ROOT/'vendor-lock.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(dest):
 for name,digest in {**{k:v for k,v in LOCK['sha256'].items() if not k.endswith('.tar.gz')},**LOCK['header_sha256']}.items():
  p=dest/name
  if not p.is_file() or sha(p)!=digest:raise ValueError('missing or changed pinned source/license: '+name)
 print('Pinned arithmetic headers and licenses verified')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--archives',type=Path,help='directory containing GPU-NTT.tar.gz and PhantomFHE.tar.gz')
 p.add_argument('--output',type=Path,default=ROOT/'vendors')
 p.add_argument('--verify',action='store_true',help='verify an existing source tree without extraction')
 a=p.parse_args();dest=a.output.resolve()
 if a.verify:verify(dest);return
 if not a.archives:p.error('--archives is required unless --verify')
 if dest.exists():raise ValueError('use a fresh output directory (or --verify)')
 # Check both archive hashes before writing either tree.
 for name in LOCK['sources']:
  archive=a.archives/(name+'.tar.gz')
  if sha(archive)!=LOCK['sha256'][name+'.tar.gz']:raise ValueError('archive hash mismatch: '+name)
 dest.mkdir(parents=True)
 for name in LOCK['sources']:
  with tarfile.open(a.archives/(name+'.tar.gz')) as tar:
   for m in tar.getmembers():
    parts=Path(m.name).parts
    if len(parts)<2:continue
    rel=Path(*parts[1:])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('unsafe archive path')
    target=dest/name/rel
    if m.isdir():target.mkdir(parents=True,exist_ok=True)
    elif m.isfile():
     target.parent.mkdir(parents=True,exist_ok=True)
     with tar.extractfile(m) as inp,target.open('wb') as out:shutil.copyfileobj(inp,out)
    else:raise ValueError('unsupported archive member: '+m.name)
 verify(dest)
if __name__=='__main__':main()
