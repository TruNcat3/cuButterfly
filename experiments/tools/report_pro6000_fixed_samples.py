#!/usr/bin/env python3
"""Verify and render the frozen PRO6000 fixed-mapping comparison, CPU only."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,math,statistics,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'experiments/snapshots/pro6000_fixed_mapping_20261117'
BASELINES=('cufft','vkfft','daofht','gpu-ntt','gpu-fft','hadacore')
GOOD={'pass','accuracy-review-required'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def median(row):
 timing=row.get('timing')
 if timing is None:return None
 samples=timing['samples_ms'];value=timing['median_ms']
 assert len(samples)==7 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in samples)
 assert math.isclose(statistics.median(samples),value,rel_tol=1e-10)
 assert timing['protocol']=='20 complete calls per CUDA graph; 7 replays; 10 warmup'
 return value*1000
def source_match(row,request):
 return row['descriptor']['semantics']==request['semantics'] and all(
  row['mapping'].get(k)==v for k,v in request['mapping_summary'].items())
def source_wins(pairs,case_id):
 return [p for p in pairs if p['case_id']==case_id and p.get('speedup') is not None and p['speedup']>1 and p['status'] in GOOD]
def label(row):
 s=row['descriptor']['semantics'];seed=row['fixed_seed']['semantics']
 return f"{seed['operator']} {seed['precision']} N={math.prod(s['shape'])} B={s['batch']} {'inverse' if s['inverse'] else 'forward'}"
def verify(directory,require_array_payloads=True):
 directory=Path(directory);manifest=load(directory/'manifest.json')
 provenance=load(directory/'provenance.json')
 array_files={a['path']:a for a in provenance['bundled_error_arrays']}
 for relative,checksum in manifest['files'].items():
  path=Path(relative);assert not path.is_absolute() and '..' not in path.parts
  if relative in array_files and not require_array_payloads:
   assert checksum==array_files[relative]['sha256']==array_files[relative]['original_sha256']
   continue
  assert sha(directory/path)==checksum,relative
 source=ROOT/provenance['a100_snapshot']['path']
 assert sha(source)==provenance['a100_snapshot']['sha256']
 a100=load(source);requests={r['id']:r for r in a100['operators']['requests'] if r['cohort']=='main66'}
 assert len(requests)==66
 assert a100['coverage']['campaign']['operator_track_complete'] is True and a100['coverage']['campaign']['remaining']==0
 pairs=[p for p in a100['operators']['pairs'] if p['cohort']=='main66']
 assert a100['identities']['main66']==provenance['a100_identity']
 data=load(directory/'measurements.json');rows=data['rows'];cases=load(directory/'cases.json')
 catalog=load(ROOT/'config/research_campaign/mapping_seeds.json')
 assert sha(ROOT/'config/research_campaign/mapping_seeds.json')==cases['catalog_sha256']
 assert sha(directory/'population.json')==cases['manifest_sha256']
 prepared={t['case']['id']:t for t in cases['tasks']}
 declared={c['id']:c for c in load(directory/'population.json')['cases']}
 assert len(rows)==25 and len(prepared)==25 and len(declared)==66
 assert len({r['id'] for r in rows})==25
 assert set(prepared)|{g['id'] for g in cases['gaps']}==set(declared)
 assert not set(prepared)&{g['id'] for g in cases['gaps']}
 selected=[]
 for row in rows:
  task=prepared[row['id']];assert row['descriptor']==task['case']['descriptor']==declared[row['id']]['descriptor']
  assert digest(row)==data['row_sha256'][row['id']]
  assert row['guards']=={'before':True,'after':True}
  assert row['gpu_uuid']==provenance['target']['gpu_uuid']
  assert row['library_sha256']==provenance['target']['library_sha256']
  assert row['source_catalog_sha256']==cases['catalog_sha256']
  assert row['fixed_seed']==task['seed']
  assert task['seed']==catalog['seeds'][task['seed_index']]
  assert row['plan_descriptor']['selection']['algorithm_policy']==2
  assert row['plan_descriptor']['semantics']==row['descriptor']['semantics']
  assert all(row['mapping'].get(k)==v for k,v in row['fixed_seed']['mapping'].items())
  assert median(row) is not None
  assert 'correctness' in row and 'mapping_replay_correctness' in row
  if row['descriptor']['semantics']['storage'] in (8,9):
   assert all(r['exact'] is True and r['mismatches']==0 for r in (row['correctness'],row['mapping_replay_correctness']))
  for baseline in row['baselines']:
   value=median(baseline)
   if baseline['status'] not in GOOD:assert value is None
  if source_match(row,requests[row['id']]) and source_wins(pairs,row['id']):selected.append(row['id'])
 assert sorted(selected)==manifest['selected_case_ids'] and len(selected)==4
 for artifact in provenance['bundled_error_arrays']:
  if require_array_payloads:
   assert sha(directory/artifact['path'])==artifact['sha256']==artifact['original_sha256']
  else:
   assert manifest['files'][artifact['path']]==artifact['sha256']==artifact['original_sha256']
  row=next(r for r in rows if r['id']==artifact['case_id'])
  implementation=row if artifact['implementation']=='cubutterfly' else next(b for b in row['baselines'] if b['name']==artifact['implementation'])
  assert implementation['error_artifact']['sha256']==artifact['sha256']
 return rows,cases,requests,pairs,provenance,selected
def csv_text(rows):
 output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows);return output.getvalue()
def verify_arrays(directory):
 """Optional NumPy audit of the bundled full outputs and CPU inputs/oracles."""
 import numpy as np
 sys.path.insert(0,str(ROOT/'scripts'))
 import run_semantic_campaign as campaign
 rows,_,_,_,provenance,_=verify(directory);by_id={r['id']:r for r in rows}
 for artifact in provenance['bundled_error_arrays']:
  row=by_id[artifact['case_id']];s=row['descriptor']['semantics']
  implementation=row if artifact['implementation']=='cubutterfly' else next(b for b in row['baselines'] if b['name']==artifact['implementation'])
  with np.load(Path(directory)/artifact['path'],allow_pickle=False) as arrays:
   actual,expected=arrays['actual'],arrays['reference']
   if artifact['implementation']!='cubutterfly' and implementation['status']=='correctness-failure':
    assert actual.dtype==expected.dtype==np.dtype('uint64')
    assert np.count_nonzero(actual!=expected)>0
   else:assert campaign.check_full(actual,expected,s['storage'],'error-report',True)==implementation['correctness']
 for identifier,records in provenance['selected_input_and_reference_hashes'].items():
  s=by_id[identifier]['descriptor']['semantics'];source=campaign.make_source(s)
  expected=campaign.reference(s,source,1048576,16777216)
  for name,value in [('source',source),('expected',expected)]:
   assert str(value.dtype)==records[name]['dtype'] and list(value.shape)==records[name]['shape']
   assert hashlib.sha256(value.tobytes()).hexdigest()==records[name]['sha256'],(identifier,name)
 print('Verified six lossless error arrays and regenerated four exact CPU inputs/references; no CUDA')
def render(directory):
 rows,cases,requests,pairs,provenance,selected=verify(directory)
 records=[];comparison=[];main=[];mapping_lines=[];appendix=[];failures=[];coverage=[]
 for operator,name in enumerate(('FFT','NTT','FWHT','subset-zeta','superset-zeta','structured-2×2')):
  source_cases=[r for r in requests.values() if r['semantics']['operator']==operator]
  target_cases=[r for r in rows if r['descriptor']['semantics']['operator']==operator]
  baselines={}
  for pair in pairs:
   if pair['semantics']['operator']==operator and pair.get('speedup') is not None:
    baselines[pair['baseline']]=baselines.get(pair['baseline'],0)+1
  baseline_text=', '.join(f'{name}: {count}' for name,count in sorted(baselines.items())) or 'no selected external comparator; CPU oracle'
  coverage.append(f"| {name} | {len(source_cases)} | {len(target_cases)} | {sum(r['id'] in selected for r in target_cases)} | {baseline_text} |")
 for row in sorted(rows,key=lambda r:(r['descriptor']['semantics']['operator'],r['descriptor']['semantics']['storage'],math.prod(r['descriptor']['semantics']['shape']),r['descriptor']['semantics']['batch'],r['descriptor']['semantics']['inverse'])):
  own=median(row);a100=requests[row['id']];match=source_match(row,a100)
  is_selected=row['id'] in selected
  records.append(dict(case_id=row['id'],workload=label(row),selected_a100_nominal_win_sample=is_selected,
   published_a100_mapping_axes_match=match,a100_nominal_win_baselines=','.join(p['baseline'] for p in source_wins(pairs,row['id'])),
   cubutterfly_us=own,status=row['status'],integer_exact=row['correctness'].get('exact'),
   max_abs=row['correctness'].get('max_abs'),l2_relative=row['correctness'].get('l2_relative'),
   mapping=json.dumps(row['mapping'],sort_keys=True,separators=(',',':'))))
  for name in BASELINES:
   baseline=next((b for b in row['baselines'] if b['name']==name),None)
   status=baseline['status'] if baseline else 'not-reached-after-earlier-failure'
   value=median(baseline) if baseline else None
   ratio=value/own if value is not None and status in GOOD else None
   source=next((p for p in pairs if p['case_id']==row['id'] and p['baseline']==name),None)
   comparison.append(dict(case_id=row['id'],workload=label(row),baseline=name,status=status,
    cubutterfly_us=own,baseline_us=value,baseline_over_cubutterfly=ratio,
    a100_baseline_over_cubutterfly=source['speedup'] if source else None,
    max_abs=baseline.get('correctness',{}).get('max_abs') if baseline else None,
    l2_relative=baseline.get('correctness',{}).get('l2_relative') if baseline else None,
    reason=baseline.get('error',baseline.get('reason','')) if baseline else 'Earlier GPU-NTT failure stopped this case before optional plugins'))
   if status=='correctness-failure':failures.append(f"- `{row['id']}` {name}: {baseline['error']}. cuButterfly and explicit replay are integer-exact; this failed comparator has no timing ratio.")
   if is_selected and (value is not None or status=='correctness-failure'):
    ar=f"{source['speedup']:.3f}×" if source and source.get('speedup') is not None else 'not in source'
    main.append(f"| {label(row)} | {name} | {ar} | {own:.3f} | {value:.3f} | {ratio:.3f}× |" if value is not None else f"| {label(row)} | {name} | {ar} | {own:.3f} | integer failure | excluded |")
  appendix.append(f"| {label(row)} | {own:.3f} | {row['status']} | {'yes' if is_selected else 'no'} |")
  if is_selected:
   m=row['mapping'];mapping_lines.append(f"| `{row['id'][:12]}` | {m['backend']} | `{m.get('stage_partition',[])}` | {m.get('fft_core',m.get('hierarchical_core','unknown'))} | {m.get('tile_threads',m.get('threads_per_block','unknown'))} | {m.get('local_exchange',m.get('dataflow_layout','unknown'))} |")
 text=['# PRO6000 fixed-mapping samples at A100 nominal-win workloads','',
  '**Status: measured target timings and derived comparisons; preliminary subset, not full acceptance.**','',
  'The four highlighted requests were chosen from the published A100 80GB main66 snapshot: cuButterfly was nominally faster than at least one external comparator, the complete semantics match, and every published mapping-summary field matches the PRO6000 execution. Selection uses A100 evidence, not PRO6000 speedups. All 25 measured fixed-seed requests remain in the appendix; the other 41 Full66 requests have no exact seed match and remain explicit gaps.','',
  'The A100 frozen operator cohort is complete at 66/66 requests across all six operator categories, with the original comparator gaps and floating error-report status retained. This target publication also covers all six categories, at a smaller sampling density. Completing operator execution does not imply an external comparator exists for every request.','',
  '| Operator | A100 frozen requests | PRO fixed-seed timings | Highlighted source samples | A100 measured external pairs |',
  '|---|---:|---:|---:|---|',*coverage,'',
  '“Nominal win” means a source baseline/cuButterfly ratio greater than 1, without a statistical significance claim. The A100 snapshot publishes mapping summaries, not the complete original mappings. Matching those fields does **not** prove bit-for-bit transfer of a full A100 search winner. The repository seed catalog is a candidate portfolio, not a certified winner list. No A100 timings, binary, calibration or registry were imported into target selection.','',
  f"A100 source: {provenance['a100_identity']['gpu_name']}, UUID `{provenance['a100_identity']['gpu_uuid']}`, {provenance['a100_identity']['gpu_total_memory']} bytes, library SHA256 `{provenance['a100_identity']['library_sha256']}`. Source snapshot: [current_refresh](../snapshots/current_refresh/snapshot.json).",'',
  f"Target: {provenance['target']['gpu_name']}, UUID `{provenance['target']['gpu_uuid']}`, {provenance['target']['memory_bytes']} bytes, SM120; driver {provenance['target']['driver_version']}, CUDA 12.8. Library SHA256 `{provenance['target']['library_sha256']}`. Measured source commit `{provenance['measured_source_commit']}` with the archived build compatibility patch. The PR base is a newer documentation/workflow commit; it does not relabel the measured build.",'',
  'Each target request uses an explicit fixed mapping (policy 2), CPU-prepared full inputs and references, complete public execution, unchanged precision/shape/batch/direction/normalization, mapping replay, padding/input checks, and exclusive GPU guards before and after measurement. The controller also monitored foreign contexts during the attempt. No automatic search was performed. Each implementation has 10 warmup calls and 7 CUDA graph replays of 20 complete calls; the table uses sample medians. These are one-process steady-state samples, not seven independent processes or alternating paired trials. Planning, compilation, allocation and CPU checking are excluded; required order conversion and normalization are included.','',
  'Floating-point results retain full error distributions and require numerical review; a timing ratio is not numerical acceptance. Integer cuButterfly and replay results are exact. Failed baselines are excluded from ratios. Higher baseline/cuButterfly is faster for cuButterfly. A100 and PRO6000 ratios describe separate builds and devices; no pooled cross-platform geometric mean is reported.','',
  '| Request | Comparator | A100 comparator/cuButterfly | PRO cuButterfly µs | PRO comparator µs | PRO comparator/cuButterfly |',
  '|---|---|---:|---:|---:|---:|',*main,'',
  'The four source nominal-win workloads include both PRO6000 wins and losses. In particular, the FP32 FFT samples are slower than VkFFT, and the NTT comparator fails. GPU-FFT was not present in the cited A100 cohort; its new target timings are additional external comparisons, not evidence about an A100 win.','',
  'Selected actual design fields (complete mappings, nested execution groups, batch/element strides, orders and moduli remain in the JSON/CSV):','',
  '| Case prefix | Backend | Stage partition | Core | Threads | Local exchange/layout |',
  '|---|---|---|---|---:|---|',*mapping_lines,'',
  'Stage partition is a serialized mapping axis; empty NTT partitions are not interpreted as zero execution stages. Other inactive/default mapping fields are not inferred as measured hardware occupancy or pipeline behavior.','',
  '**Negative results**','',*failures,'',
  '**All measured fixed-seed requests**','',
  '| Request | cuButterfly µs | Raw case status | Highlighted source sample |',
  '|---|---:|---|---|',*appendix,'',
  'The two raw case statuses `correctness-failure` originate in GPU-NTT after cuButterfly and mapping replay passed; they are preserved as failures. The frozen data also retain unsupported/missing baselines and plugins not reached after a preceding failure. Zeta and structured transforms have no selected external comparator. HadaCore supports a different FP16 cohort, so its absence here is not a completed HadaCore comparison.','',
  'Frozen artifacts: [provenance](../snapshots/pro6000_fixed_mapping_20261117/provenance.json), [raw rows](../snapshots/pro6000_fixed_mapping_20261117/measurements.json), [case/seed/gap table](../snapshots/pro6000_fixed_mapping_20261117/cases.json), [all timings and mappings](../snapshots/pro6000_fixed_mapping_20261117/performance.csv), [all comparator statuses](../snapshots/pro6000_fixed_mapping_20261117/baselines.csv). Lossless arrays for the four highlighted cuButterfly results and both failed GPU-NTT results are bundled and hashed. Other floating comparator error arrays and build binaries remain on the measurement host; their original paths, hashes and full error distributions are preserved in the raw rows.','',
  'Verify frozen evidence and regenerate tables without CUDA:','',
  '```bash','python3 experiments/tools/report_pro6000_fixed_samples.py --check',
  '# Optional: requires NumPy, verifies full outputs and regenerates CPU inputs/oracles',
  'python3 experiments/tools/report_pro6000_fixed_samples.py --check --arrays',
  'python3 experiments/tools/report_pro6000_fixed_samples.py --write','```','',
  'The snapshot includes the original bounded measurement-controller source and build command/patch for reproduction on the recorded host. Its original absolute paths describe the measurement environment; they are not public-download dependencies. To remeasure elsewhere, rebuild for the target, prepare the frozen cases with the repository semantic CPU pipeline, use each recorded fixed seed with explicit policy 2, preserve the resolved mapping on replay, and run the same complete-call timing and baseline adapters on an exclusive non-default CUDA stream. Calibration and comprehensive public AUTO comparisons remain separate experiments.','',
  'This subset is deliberately biased toward source-platform nominal wins; it cannot establish overall competitiveness, global optimality, the complete A100 winner portfolio, all 66 target cases, or completion of issue #9. FFT17, HadaCore/public AUTO, MMA9 and the main acceptance gates retain their own status.']
 return {'performance.csv':csv_text(records),'baselines.csv':csv_text(comparison)},'\n'.join(text)+'\n'
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,default=DEFAULT);parser.add_argument('--write',action='store_true');parser.add_argument('--check',action='store_true');parser.add_argument('--arrays',action='store_true',help='Also verify lossless arrays and regenerate CPU inputs/references (NumPy required)');args=parser.parse_args()
 files,report=render(args.snapshot);path=ROOT/'experiments/reports/PRO6000_FIXED_MAPPING_SAMPLES_20261117.md'
 for name,content in files.items():
  target=args.snapshot/name
  if args.write:target.write_text(content)
  else:assert target.read_text()==content,'Derived table differs: '+name
 if args.write:path.write_text(report)
 else:assert path.read_text()==report,'Derived report differs'
 if args.arrays:verify_arrays(args.snapshot)
 print('Verified 25 measured mappings, 41 exact-seed gaps, 4 source nominal-win samples and all comparator statuses; CPU only')
if __name__=='__main__':main()
