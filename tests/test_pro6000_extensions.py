"""Frozen extension evidence must keep exact cache and measured profile choices."""
import copy,gzip,hashlib,importlib.util,json,shutil
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pro_extensions',ROOT/'experiments/tools/report_pro6000_extensions.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

def test_complete_frozen_cohorts_retain_all_timings_and_gaps():
 manifest,data=report.verify(report.DEFAULT)
 assert len(data['fft17']['records'])==17 and len(data['mma9']['records'])==9
 pairs=[b for r in data['fft17']['records'] for b in r['row']['baselines']]
 assert sum(report.timing(b) is not None for b in pairs)==47
 assert sum(report.timing(b) is None for b in pairs)==4
 assert len(manifest['campaigns']['mma9']['public_records'])==68
 csv_data,text=report.render(report.DEFAULT)
 assert 'FP64 N=4096 loses' in text and 'unsupported' in text

def fixture(tmp_path):
 target=tmp_path/'snapshot';shutil.copytree(report.DEFAULT,target)
 return target,json.loads((target/'manifest.json').read_text())

def replace_record(root,record,value):
 raw=json.dumps(value,sort_keys=True).encode();compressed=gzip.compress(raw,mtime=0)
 (root/record['path']).write_bytes(compressed)
 record['sha256']=hashlib.sha256(compressed).hexdigest()
 record['uncompressed_sha256']=hashlib.sha256(raw).hexdigest()

def test_refreshing_row_checksums_cannot_change_calibrated_mapping(tmp_path):
 root,manifest=fixture(tmp_path);record=manifest['campaigns']['fft17']['records'][0]
 envelope=report.unpack(root,record);envelope['row']['mapping']['tile_threads']+=32
 envelope['row_sha256']=report.digest(envelope['row']);replace_record(root,record,envelope)
 (root/'manifest.json').write_text(json.dumps(manifest))
 with pytest.raises(AssertionError):report.verify(root)

def test_refreshing_public_checksum_cannot_change_profile_geometry(tmp_path):
 root,manifest=fixture(tmp_path)
 record=next(r for r in manifest['campaigns']['mma9']['public_records'] if report.unpack(root,r)['selected_kind']=='mma')
 value=report.unpack(root,record);value['mapping']['local_core']['groups_per_block']*=2
 replace_record(root,record,value);(root/'manifest.json').write_text(json.dumps(manifest))
 with pytest.raises(AssertionError):report.verify(root)

def test_corrupted_gzip_bytes_are_rejected(tmp_path):
 root,manifest=fixture(tmp_path);record=manifest['campaigns']['fft17']['records'][0]
 path=root/record['path'];path.write_bytes(path.read_bytes()[:-1])
 with pytest.raises(AssertionError):report.verify(root)
