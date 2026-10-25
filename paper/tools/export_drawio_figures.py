#!/usr/bin/env python3
"""Export editable paper figures with the real draw.io desktop renderer.

This reads .drawio sources; it never regenerates them, so edits made in the
editor survive export. A private Xvfb display supports headless Linux without
using the research GPUs. Set DRAWIO/XVFB or pass their paths when not installed.
"""
import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import subprocess
import tempfile
import urllib.parse
import xml.etree.ElementTree as ET
import zlib

FIGURES = ('overview', 'organization_tradeoffs', 'roofline', 'hierarchy',
           'semantic_transport', 'bank_layout', 'pipeline', 'system_architecture')
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / 'figures' / 'drawio'
CACHE = Path.home() / '.cache/cuButterfly/tools/drawio-32.4.1/root'


def executable(explicit, name, fallback):
    candidate = explicit or shutil.which(name) or str(fallback)
    path = Path(candidate)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f'{name} is unavailable: {candidate}; see paper/figures/drawio/README.md')
    return path.resolve()


def fingerprint(path):
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def read_document(path):
    root = ET.parse(path).getroot()
    for page in root.findall('diagram'):
        if page.find('mxGraphModel') is None and page.text:
            text = urllib.parse.unquote(zlib.decompress(base64.b64decode(page.text), -15).decode())
            page.text = None
            page.append(ET.fromstring(text))
    return root


def inspect_source(path):
    root = read_document(path)
    page = root.find('diagram')
    model = page.find('mxGraphModel') if page is not None else None
    if model is None: raise ValueError(f'{path}: expected uncompressed editable diagram')
    cells = model.findall('root/mxCell')
    ids = [cell.get('id') for cell in cells]
    if len(ids) != len(set(ids)): raise ValueError(f'{path}: duplicate cell IDs')
    if any('image=' in cell.get('style', '') for cell in cells):
        raise ValueError(f'{path}: embedded pictures are not native editable construction')
    return page, dict(native_cells=len(cells), page_width=model.get('pageWidth'),
                      page_height=model.get('pageHeight'))


def export(names, drawio, xvfb, display):
    selected = [(name, SOURCES / (name+'.drawio')) for name in names]
    source_metadata = {name: inspect_source(path)[1] for name,path in selected}
    output = SOURCES / 'exports'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema='cubutterfly-drawio-exports-v1', kind='analytical-schematics',
                  renderer=fingerprint(drawio), figures={})
    with tempfile.TemporaryDirectory(prefix='cubutterfly-drawio-') as temporary:
        env = dict(os.environ)
        env.pop('ELECTRON_RUN_AS_NODE', None)
        server = None
        try:
            if not display:
                read_fd, write_fd = os.pipe()
                server = subprocess.Popen([str(xvfb), '-displayfd', str(write_fd),
                    '-screen', '0', '1600x1200x24', '-nolisten', 'tcp'],
                    pass_fds=(write_fd,), stdout=subprocess.DEVNULL,
                    stderr=open(Path(temporary)/'xvfb.log', 'w'))
                os.close(write_fd)
                try:
                    if not select.select([read_fd], [], [], 15)[0]:
                        raise RuntimeError('private Xvfb did not publish a display')
                    number = os.read(read_fd, 64).decode().strip()
                    if not number.isdigit(): raise RuntimeError('Xvfb failed to start')
                    display = ':'+number
                finally:
                    os.close(read_fd)
            env['DISPLAY'] = display
            base = [str(drawio), '--no-sandbox', '--disable-gpu', '--disable-update',
                    '--user-data-dir='+str(Path(temporary)/'profile')]
            version = subprocess.run(base+['--version'], env=env, text=True,
                                     capture_output=True, check=True, timeout=30)
            report['renderer_version'] = version.stdout.strip()
            for name,path in selected:
                record = dict(source=fingerprint(path), **source_metadata[name], exports={})
                for fmt in ('pdf','svg','png'):
                    destination = output / (name+'.'+fmt)
                    staged = Path(temporary) / (name+'.'+fmt)
                    command = base+['--export','--theme','light','--timeout','40',
                                    '--format',fmt,'--output',str(staged)]
                    if fmt == 'pdf': command += ['--crop']
                    if fmt == 'svg': command += ['--embed-diagram','--embed-svg-fonts','false']
                    if fmt == 'png': command += ['--scale','2']
                    result = subprocess.run(command+[str(path)],env=env,text=True,
                                            capture_output=True,timeout=55)
                    if result.returncode or not staged.is_file() or staged.stat().st_size==0:
                        raise RuntimeError(f'{name}.{fmt}: {result.stdout}\n{result.stderr}')
                    shutil.copyfile(staged,destination)
                    record['exports'][fmt] = fingerprint(destination)
                report['figures'][name] = record
                print(f'Exported {name}: PDF / SVG / PNG',flush=True)
        finally:
            if server is not None:
                server.terminate()
                try: server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()
    # Refresh the editing convenience bundle even after a single-figure export.
    # Individual documents remain the canonical sources.
    bundle = ET.Element('mxfile',host='app.diagrams.net',type='device',version='32.4.1')
    for name in FIGURES:
        bundle.append(copy.deepcopy(inspect_source(SOURCES/(name+'.drawio'))[0]))
    ET.indent(bundle,space='  ')
    ET.ElementTree(bundle).write(SOURCES/'cubutterfly-methods.drawio',encoding='utf-8',xml_declaration=True)
    report['bundle'] = fingerprint(SOURCES/'cubutterfly-methods.drawio')
    report_path=output/'manifest.json'
    if report_path.exists() and tuple(names)!=FIGURES:
        previous=json.loads(report_path.read_text())
        previous['figures'].update(report['figures'])
        previous.update({k:v for k,v in report.items() if k!='figures'})
        report=previous
    report_path.write_text(json.dumps(report,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('names',nargs='*',help='optional figure stems; default exports all eight')
    parser.add_argument('--import-bundle',type=Path,
                        help='first adopt edited named pages from an eight-page draw.io file')
    parser.add_argument('--drawio',default=os.environ.get('DRAWIO'))
    parser.add_argument('--xvfb',default=os.environ.get('XVFB'))
    parser.add_argument('--display',help='use a provided display instead of a private Xvfb')
    args=parser.parse_args()
    if set(args.names)-set(FIGURES): parser.error('unknown figure stem')
    if args.import_bundle:
        document=read_document(args.import_bundle)
        pages={page.get('name'):page for page in document.findall('diagram')}
        if set(pages)!=set(FIGURES) or len(document.findall('diagram'))!=len(FIGURES):
            parser.error('bundle must have one page named for each of the eight figure stems')
        for name in FIGURES:
            root=ET.Element('mxfile',host='app.diagrams.net',type='device',version='32.4.1')
            root.append(copy.deepcopy(pages[name]))
            ET.indent(root,space='  ')
            ET.ElementTree(root).write(SOURCES/(name+'.drawio'),encoding='utf-8',xml_declaration=True)
    drawio=executable(args.drawio,'drawio',CACHE/'opt/drawio/drawio')
    xvfb=None if args.display else executable(args.xvfb,'Xvfb',CACHE/'usr/bin/Xvfb')
    export(args.names or FIGURES,drawio,xvfb,args.display)


if __name__=='__main__': main()
