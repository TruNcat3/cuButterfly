#!/usr/bin/env python3
"""Verify and render the frozen PRO6000 FFT17 and MMA9 cohorts; no CUDA."""
from __future__ import annotations
import argparse,copy,csv,gzip,hashlib,io,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'experiments/snapshots/pro6000_extensions_20261117'
REPORT=ROOT/'experiments/reports/PRO6000_EXTENSIONS_20261117.md'
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(value):return hashlib.sha256(value).hexdigest()
def unpack(root,record):
 relative=Path(record['path']);assert not relative.is_absolute() and '..' not in relative.parts
 raw=(root/relative).read_bytes();assert sha(raw)==record['sha256'],str(relative)
 raw=gzip.decompress(raw);assert sha(raw)==record['uncompressed_sha256'],str(relative)
 return json.loads(raw) if '.json' in relative.name else raw
def timing(row):
 value=row.get('timing')
 if value is None:return None
 samples=value['samples_ms']
 assert len(samples)==7 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in samples)
 assert statistics.median(samples)==value['median_ms']
 assert value['protocol']=='20 complete calls per CUDA graph; 7 replays; 10 warmup'
 return value['median_ms']*1000
def native_mapping(value):
 value=copy.deepcopy(value);value.pop('arithmetic_policy',None)
 if isinstance(value.get('contract'),dict):value['contract'].pop('arithmetic_policy',None)
 return value
def verify(root,require_array_payloads=True):
 root=Path(root);manifest=json.loads((root/'manifest.json').read_text());data={}
 assert manifest['schema']=='cubutterfly-pro6000-extensions-publication-v1'
 for name,count in [('fft17',17),('mma9',9)]:
  cohort=manifest['campaigns'][name];preparation=unpack(root,cohort['preparation'])
  config=preparation['config'];assert digest(config)==preparation['config_sha256']
  assert config['runtime_bindings']==cohort['source_runtime_bindings']
  records=[unpack(root,record) for record in cohort['records']]
  assert len(records)==count and len({r['row']['id'] for r in records})==count
  assert {r['row']['id'] for r in records}=={c['id'] for c in config['cases']}
  assert len({r['identity_sha256'] for r in records})==1
  cases={c['id']:c for c in config['cases']}
  prepared={c['id']:c for c in preparation['cases']}
  for envelope in records:
   row=envelope['row'];case=cases[row['id']];s=case['descriptor']['semantics']
   assert envelope['row_sha256']==digest(row) and envelope['identity_sha256']==digest(envelope['identity'])
   assert envelope['guards']=={'before':True,'after':True}
   assert envelope['identity']['runtime_bindings']==config['runtime_bindings']
   assert envelope['identity']['gpu_uuid'].lower().removeprefix('gpu-')==config['gpu_uuid'].lower().removeprefix('gpu-')
   assert envelope['identity']['libraries']['main']['sha256']==manifest['library_sha256']
   assert envelope['prepared_sha256']==prepared[row['id']]['sha256']
   assert row['descriptor']['semantics']==s==row['plan_descriptor']['semantics']
   assert row['plan_descriptor']['selection']['algorithm_policy']==0
   assert row['status']=='accuracy-review-required' and timing(row) is not None
   assert [b['name'] for b in row['baselines']]==case['baselines']
   for item in [row,*row['baselines']]:
    if timing(item) is not None:
     assert item['status']=='accuracy-review-required'
     assert item.get('numeric_diagnostics',{}).get('actual_nonfinite_elements',0)==0
  data[name]=dict(preparation=preparation,records=records)
 fft=manifest['campaigns']['fft17'];calibrated={r['case']:r for r in (unpack(root,c) for c in fft['calibration_records'])}
 assert len(calibrated)==17
 cache=unpack(root,fft['cache']);audit=unpack(root,fft['completion'])
 assert sha(cache)==audit['cache_sha256'] and audit['cases']==17 and audit['timed_pairs']==47 and len(audit['gaps'])==4
 for envelope in data['fft17']['records']:
  row=envelope['row'];calibration=calibrated[row['id']]
  assert calibration['cache_gate'] and calibration['guards']=={'before':True,'after':True}
  assert calibration['mapping']==row['mapping']
  assert calibration['measured_descriptor']['selection']['algorithm_policy']==1
  assert calibration['replay_descriptor']['selection']['algorithm_policy']==0
  assert calibration['measured_descriptor']['semantics']==row['descriptor']['semantics']
  assert (calibration.get('replay_selection_source_id')==1 or
          'cache' in calibration['replay_reason'].lower())
  assert envelope['identity']['runtime_bindings'][data['fft17']['preparation']['config']['environment']['CUBUTTERFLY_PLAN_CACHE']]==sha(cache)
  for baseline in row['baselines']:
   if baseline['name']=='gpu-fft' and len(row['descriptor']['semantics']['shape'])>1:
    assert timing(baseline) is None and baseline['status']=='missing-baseline'
   else:assert timing(baseline) is not None
 mma=manifest['campaigns']['mma9'];profile=unpack(root,mma['profile']);public=[unpack(root,r) for r in mma['public_records']]
 calibration={record['original']:unpack(root,record) for record in mma['calibration_records']}
 assert len(profile['native_rows'])==68 and len(profile['rows'])==204 and len(public)==68 and len(calibration)==272
 assert profile['gpu_uuid']==data['mma9']['preparation']['config']['gpu_uuid']
 native_keys={(r['log_n'],r['batch']) for r in profile['native_rows']};assert len(native_keys)==68
 assert len({r['case'] for r in public})==68
 for log,batch in native_keys:
  candidates=[r for r in profile['rows'] if (r['log_n'],r['batch'])==(log,batch)]
  assert len(candidates)==3 and {r['groups_per_block'] for r in candidates}=={1,2,4}
 for group in (profile['native_rows'],profile['rows']):
  for item in group:
   evidence=calibration[item['evidence_path']]
   assert next(r for r in mma['calibration_records'] if r['original']==item['evidence_path'])['uncompressed_sha256']==item['evidence_sha256']
   assert evidence['guards']=={'before':True,'after':True}
   assert evidence['gpu_uuid']==profile['gpu_uuid']
   assert evidence['timing']['median_ms']==item['median_ms'] and evidence['timing']['protocol']==item['timing_protocol']
   assert timing(evidence) is not None
   assert evidence['input_immutable'] and evidence['buffer_guards'] and evidence['graph_reproduced']
   if 'exact_upstream_match' in evidence:assert evidence['exact_upstream_match']
 kinds={'native':0,'mma':0}
 for row in public:
  assert row['status']=='validated-public-selection' and row['guards']=={'before':True,'after':True}
  assert row['gpu_uuid']==profile['gpu_uuid'] and row['library_sha256']==manifest['library_sha256']
  assert row['profile_sha256']==mma['profile']['uncompressed_sha256']
  assert row['finite'] and row['exact_selected_match'] and timing(row) is not None
  case=row['case'];n,b=map(int,case[1:].split('xB'));log=n.bit_length()-1
  native=next(r for r in profile['native_rows'] if r['log_n']==log and r['batch']==b)
  candidates=[('native',native)]+[('mma',r) for r in profile['rows'] if r['log_n']==log and r['batch']==b and r['timing_protocol']==native['timing_protocol']]
  kind,winner=min(candidates,key=lambda entry:entry[1]['median_ms']);assert row['selected_kind']==kind
  assert row['selected_evidence_sha256']==winner['evidence_sha256']
  assert row['payload_sha256']==calibration[winner['evidence_path']]['payload_sha256']
  if kind=='native':assert native_mapping(row['mapping'])==native_mapping(native['mapping'])
  else:
   core=row['mapping']['local_core'];assert core['name']=='hadacore-local-tile' and core['chunks']==winner['chunks'] and core['groups_per_block']==winner['groups_per_block']
   assert row['exact_upstream_match']
  kinds[kind]+=1
 assert kinds=={'native':1,'mma':67}
 for envelope in data['mma9']['records']:
  row=envelope['row'];s=row['descriptor']['semantics'];assert s['arithmetic_policy']==1
  assert 'same-protocol measured native' in row['selection_reason']
  assert row['mapping']['local_core']['profile_sha256']==mma['profile']['uncompressed_sha256']
  assert timing(row['baselines'][0]) is not None
 for artifact in manifest['array_samples']:
  relative=Path(artifact['path']);assert not relative.is_absolute() and '..' not in relative.parts
  assert artifact['sha256']==artifact['original_sha256']
  if require_array_payloads:assert sha((root/relative).read_bytes())==artifact['sha256']
  row=next(r['row'] for r in data[artifact['campaign']]['records'] if r['row']['id']==artifact['case'])
  implementation=row if artifact['implementation']=='cubutterfly' else next(b for b in row['baselines'] if b['name']==artifact['implementation'])
  assert implementation['error_artifact']['sha256']==artifact['sha256']
 return manifest,data
def csv_text(rows):
 stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows);return stream.getvalue()
def render(root):
 manifest,data=verify(root);csv_rows=[];tables={};identities={}
 for name,cohort in data.items():
  table=[];identities[name]=cohort['records'][0]['identity']
  for envelope in cohort['records']:
   row=envelope['row'];s=row['descriptor']['semantics'];own=timing(row);values=[]
   for baseline in row['baselines']:
    value=timing(baseline);ratio=value/own if value is not None else None
    values.append(f'{value:.3f} / {ratio:.3f}×' if value is not None else 'unsupported')
    csv_rows.append(dict(cohort=name,case=row['id'],shape='x'.join(map(str,s['shape'])),batch=s['batch'],storage=s['storage'],compute=s['compute'],inverse=s['inverse'],normalize_inverse=s['normalize_inverse'],arithmetic_policy=s.get('arithmetic_policy',0),baseline=baseline['name'],own_us=own,baseline_us=value,baseline_over_own=ratio,own_status=row['status'],baseline_status=baseline['status'],own_l2_relative=row['correctness'].get('l2_relative'),baseline_l2_relative=baseline.get('correctness',{}).get('l2_relative'),reason=baseline.get('reason',''),mapping=json.dumps(row['mapping'],sort_keys=True,separators=(',',':'))))
   if name=='fft17':table.append(f"| {'FP64' if s['storage']==7 else 'FP32'} | {'×'.join(map(str,s['shape']))} | {s['batch']} | {'inverse / 1/N' if s['inverse'] else 'forward'} | {own:.3f} | {' | '.join(values)} |")
   else:
    core=row['mapping']['local_core'];table.append(f"| {s['shape'][0]} | {s['batch']} | {own:.3f} | {values[0]} | {core['chunks']} | {core['groups_per_block']} | {row['correctness']['l2_relative']:.6g} |")
  tables[name]=table
 text=['# PRO6000 FFT17 and ALLOW_FP16_MMA9 complete-call comparisons','',
  '**Status: measured finite extension cohorts, with floating error review and declared provider gaps.** All original FFT17 and MMA9 requests were retained. These are separate from the [25 fixed-seed samples](PRO6000_FIXED_MAPPING_SAMPLES_20261117.md), the main Full66 acceptance and global model validation.','',
  'The measured build is source commit `'+manifest['measured_source_commit']+'` with the archived [compatibility patch](../snapshots/pro6000_fixed_mapping_20261117/provenance/build-compatibility.patch.gz). Main library SHA256: `'+manifest['library_sha256']+'`. CUDA 12.8, driver 580.159.04; NVIDIA RTX PRO 6000 Blackwell Server Edition, 101,973,950,464 bytes per device (SM120). The native SM120 HadaCore provider is a separate build from the preserved SM80/PTX illegal-instruction attempt.','',
  'FFT17 uses GPU UUID `'+identities['fft17']['gpu_uuid']+'`; MMA9 uses `'+identities['mma9']['gpu_uuid']+'`. Each cohort records its own hardware, libraries, provider/profile hashes and source fingerprints; timings are not pooled across cards or with A100.','',
  'Protocol: complete public calls, out-of-place, same full shape/batch/precision/direction/normalization and CPU-prepared input/reference per comparison. Each implementation has 10 warmup calls, then 7 CUDA Graph replays of 20 complete calls; values are sample medians in µs. Required copies, permutations and scales are included; setup, compilation, planning, allocation and CPU checks excluded. Implementations are collected sequentially, not alternating independent process trials. Both outer and per-case guards reject foreign CUDA contexts.','',
  'Floats remain `accuracy-review-required`, with complete error distributions; no threshold is introduced or relaxed. FP16 MMA and native stage-rounded FWHT have different reduction/rounding contracts. Ratios are comparator/cuButterfly; above 1 is nominally faster for cuButterfly. Near-1 differences are not statistical evidence. Selection/calibration costs are excluded from steady-state execution and are not amortization claims.','',
  '## FFT17','',
  'All 17 requests use public DEFAULT replay from same-GPU exact-request MEASURE calibration (candidate budget 16), with complete mapping and semantic equality checked. This avoids using the cost model for unknown requests: its warning/unvalidated status was retained. No hardware-rate-prior attempt is admitted. There are 51 comparator records: 47 measured pairs and four explicit GPU-FFT multi-axis gaps.','',
  '| Precision | Shape | Batch | Direction | cuButterfly µs | cuFFT µs / ratio | VkFFT µs / ratio | GPU-FFT µs / ratio |','|---|---|---:|---|---:|---:|---:|---:|',*tables['fft17'],'',
  'The FP32 N=4096 FFT samples lose to VkFFT. FP64 N=4096 loses to GPU-FFT while beating cuFFT/VkFFT. Every negative region is retained; GPU-FFT multi-axis gaps are not flattened or counted as wins.','',
  '## HadaCore local validation and MMA9','',
  'The same-GPU paired profile contains 68 STRICT native endpoints and 204 independently validated MMA tile organizations (groups=1/2/4). All 68 public AUTO requests reproduce the profile-selected candidate: one native and 67 MMA. Public selection is checked against the minimum same-protocol measured candidate, with native retained on exact ties. Every raw calibration/public record is bundled; full output arrays remain on the measurement host except for the bundled small-request samples.','',
  'All nine original FP16 FWHT requests use explicit ALLOW_FP16_MMA and public DEFAULT selection from this local profile. ALLOW permits either native or MMA; these nine selected the integrated HadaCore tile provider. The external comparator is the complete original HadaCore transform plus its required output scale. Public-policy validation ignores only the intended STRICT→ALLOW policy change, including its duplicated contract field; physical mapping and all other semantics remain checked.','',
  '| N | Batch | cuButterfly µs | HadaCore µs / ratio | Chunks | Groups/CTA | cuButterfly relative L2 |','|---:|---:|---:|---:|---:|---:|---:|',*tables['mma9'],'',
  'These external complete-call comparisons do not isolate an organization mechanism. The profile retains all groups=1/2/4 candidates and the chosen actual mapping. Error distributions and provider/compiler identities are preserved; an ALLOW result is not a STRICT result.','',
  '## Frozen evidence and reproduction','',
  'The [snapshot manifest](../snapshots/pro6000_extensions_20261117/manifest.json) binds original raw envelopes, CPU preparation records, exact cache contents, 17 calibration/replay receipts, 272 native/tile records, 68 public AUTO records, six lossless smallest-request implementation arrays, and completion audits. Gzip preserves the source bytes and both compressed/uncompressed SHA256 values. Other lossless outputs and compiled binaries remain local; their original paths/hashes and full error distributions are retained. [All 60 comparator records and actual mappings](../snapshots/pro6000_extensions_20261117/comparisons.csv) are available as CSV.','',
  '```bash','python3 experiments/tools/report_pro6000_extensions.py --check','python3 experiments/tools/report_pro6000_extensions.py --write','```','',
  'The CPU verifier checks envelope identities, explicit cache provenance, full semantic/mapping replay, every median/protocol, native/MMA candidate choice, guards and array hashes, and deterministically regenerates this report. Original runtime paths are provenance, not downloadable dependencies. A fresh target run needs the recorded target-compatible libraries/profiles and the published comparator workflows; the snapshots do not replace rebuilding or recalibration.','',
  'This closes the finite FFT17/MMA9 extension measurements in issue #9. It does not claim complete Full66 acceptance, validated predictions for unseen workloads, all numerical acceptance, an A100 winner transfer, or same-installed-build application timing. The main benchmark’s original GPU-NTT correctness failures remain separate negative results.']
 return csv_text(csv_rows),'\n'.join(text)+'\n'
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,default=DEFAULT);parser.add_argument('--write',action='store_true');parser.add_argument('--check',action='store_true');args=parser.parse_args()
 csv_data,report=render(args.snapshot);csv_path=args.snapshot/'comparisons.csv'
 if args.write:csv_path.write_text(csv_data);REPORT.write_text(report)
 else:assert csv_path.read_text()==csv_data and REPORT.read_text()==report,'Derived report or CSV differs'
 print('Verified FFT17: 17 requests/47 timings/4 gaps; MMA9: 9 pairs; 272 calibration and 68 public selection records; CPU only')
if __name__=='__main__':main()
