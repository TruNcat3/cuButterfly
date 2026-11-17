"""Scientific scope and integrity checks for the frozen target measurements."""
import copy,csv,importlib.util,io,json,shutil
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixed_samples',ROOT/'experiments/tools/report_pro6000_fixed_samples.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

def test_frozen_selection_keeps_target_losses_and_integer_failures():
 rows,_,_,_,_,selected=report.verify(report.DEFAULT)
 assert len(selected)==4
 files,_=report.render(report.DEFAULT)
 pairs=list(csv.DictReader(io.StringIO(files['baselines.csv'])))
 assert len(pairs)==150
 failures=[p for p in pairs if p['status']=='correctness-failure']
 assert len(failures)==2 and all(p['baseline_over_cubutterfly']=='' for p in failures)
 assert any(p['case_id'] in selected and p['baseline']=='vkfft' and float(p['baseline_over_cubutterfly'])<1 for p in pairs)
 assert all(r['correctness']['exact'] for r in rows if r['id'] in {p['case_id'] for p in failures})

def test_source_selection_does_not_read_target_timings_or_unpublished_axes():
 rows,_,requests,pairs,_,selected=report.verify(report.DEFAULT)
 row=copy.deepcopy(next(r for r in rows if r['id']==selected[0]));source=requests[row['id']]
 assert report.source_match(row,source)
 row['timing']['median_ms']=1e99
 assert report.source_match(row,source) and report.source_wins(pairs,row['id'])
 # A summary match deliberately makes no claim about unreported source axes.
 row['mapping']['an_unpublished_axis']='unknown'
 assert report.source_match(row,source)
 row['descriptor']['semantics']['inverse']=1-row['descriptor']['semantics']['inverse']
 assert not report.source_match(row,source)

@pytest.mark.parametrize('bad',[float('nan'),float('inf'),0,-1,True])
def test_invalid_sample_is_not_a_performance_result(bad):
 rows,*_=report.verify(report.DEFAULT);row=copy.deepcopy(rows[0]);row['timing']['samples_ms'][0]=bad
 with pytest.raises(AssertionError):report.median(row)

def test_changed_bytes_fail_integrity_check(tmp_path):
 destination=tmp_path/'snapshot';shutil.copytree(report.DEFAULT,destination)
 with (destination/'measurements.json').open('a') as handle:handle.write(' ')
 with pytest.raises(AssertionError,match='measurements.json'):report.verify(destination)

def test_checksum_refresh_cannot_turn_explicit_data_into_auto_evidence(tmp_path):
 destination=tmp_path/'snapshot';shutil.copytree(report.DEFAULT,destination)
 path=destination/'measurements.json';data=report.load(path)
 row=data['rows'][0];row['plan_descriptor']['selection']['algorithm_policy']=0
 data['row_sha256'][row['id']]=report.digest(row);path.write_text(json.dumps(data))
 manifest=report.load(destination/'manifest.json');manifest['files']['measurements.json']=report.sha(path)
 (destination/'manifest.json').write_text(json.dumps(manifest))
 with pytest.raises(AssertionError):report.verify(destination)
