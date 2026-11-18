#!/usr/bin/env python3
"""Verify the finite cyclic NTT8 and three-process local CPU follow-up."""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import statistics

import report_pro6000_figure9 as prior

ROOT=prior.ROOT
SNAP=ROOT/'experiments/snapshots/pro6000_figure9_followup_20261118'
REPORT=ROOT/'experiments/reports/PRO6000_FIGURE9_FOLLOWUP_20261118.md'
COHORT='pro6000-cyclic-ntt8'


def unpack(record):
    relative=Path(record['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    compressed=(SNAP/relative).read_bytes();assert prior.sha(compressed)==record['sha256']
    raw=gzip.decompress(compressed);assert prior.sha(raw)==record['uncompressed_sha256']
    return raw


def load(record):return json.loads(unpack(record))


def verify():
    manifest=json.loads((SNAP/'manifest.json').read_text())
    assert manifest['schema']=='pro6000-figure9-followup-publication-v1'
    def visit(value):
        if isinstance(value,dict):
            if {'path','sha256','uncompressed_sha256'}<=value.keys():unpack(value)
            else:
                for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    visit(manifest)
    source=manifest['cyclic_ntt8'];inputs=load(source['input_hashes'])
    campaign=load(source['manifest']);prepared=load(source['preparation'])
    records=[load(v) for v in source['rows']]
    assert len(inputs)==len(records)==len(campaign['cases'])==8
    cases={c['id']:c for c in campaign['cases']}
    assert set(cases)==set(inputs)=={r['row']['id'] for r in records}
    assert {prior.digest(r['descriptor']['semantics']) for r in campaign['cases']}==set(cases)
    assert {(v['semantics']['word_bits'],v['semantics']['batch'],v['semantics']['inverse']) for v in inputs.values()}=={
        (bits,batch,direction) for bits in (32,64) for batch in (1,16) for direction in (0,1)}
    a100=json.loads((ROOT/'experiments/snapshots/current_refresh/snapshot.json').read_text())['cyclic']
    a100_rows={r['id']:r for r in a100['requests']}
    output_audit=load(source['output_array_audit'])
    for envelope in records:
        row=envelope['row'];identifier=row['id'];s=row['descriptor']['semantics'];item=inputs[identifier]
        assert prior.digest(row)==envelope['row_sha256']
        assert prior.digest(envelope['identity'])==envelope['identity_sha256']
        assert envelope['guards']==dict(before=True,after=True)
        assert s==cases[identifier]['descriptor']['semantics']==item['semantics']==a100_rows[identifier]['semantics']
        assert s['modulus']==(1073479681 if s['word_bits']==32 else 582090251837636609)
        assert s['shape']==[65536] and s['batch'] in (1,16) and s['placement']==0
        assert s['input_orders']==s['output_orders']==[0] and s['normalize_inverse']==bool(s['inverse'])
        assert row['plan_descriptor']['selection']['algorithm_policy']==1
        assert row['correctness']['exact'] and row['mapping_replay_correctness']['exact'] and prior.timing(row)
        assert prior.backend_status(row)=='pass'
        assert envelope['identity']['gpu_uuid']=='GPU-a257f4db-8c62-f26b-9c08-a4045615a140'
        assert envelope['identity']['library_sha256']=='3b0668f2530a75d1401dcbf5decea720f8e47db0c1357aafdf1680fa1bba6adb'
        assert envelope['identity']['preparation_sha256']==source['preparation']['uncompressed_sha256']
        assert item['support']['supported'] and pow(item['root'],65536,s['modulus'])==1
        assert pow(item['root'],32768,s['modulus'])!=1
        assert item['inverse_root']==pow(item['root'],-1,s['modulus'])
        assert output_audit[identifier]['cubutterfly']['exact']
        assert output_audit[identifier]['cubutterfly']['reference_sha256']==item['reference_sha256']
        baseline,=row['baselines'];assert baseline['name']=='gpu-ntt'
        if s['word_bits']==64 and not s['inverse']:
            assert baseline['status']=='correctness-failure' and not baseline.get('timing')
            assert output_audit[identifier]['gpu-ntt']['mismatches']>0
        else:
            assert baseline['status']=='pass' and baseline['correctness']['exact'] and prior.timing(baseline)
    completion=load(source['completion']);assert completion['requests']==8 and completion['valid_pairs']==6
    repros=[load(v) for v in source['minimal_repros']];assert len(repros)==2
    for r in repros:
        assert not r['exact'] and r['mismatches']>0 and r['input_preserved']
        assert r['input_sha256']==inputs[r['case_id']]['input_sha256']
        assert r['guards']==dict(before=True,after=True)
    processes=[load(v) for v in manifest['cpu_confirmation']]
    assert len(processes)==3 and len({p['identity']['pid'] for p in processes})==3
    assert {p['identity']['process_number'] for p in processes}=={1,2,3}
    _,old_inputs,_,old_cpu,old_mma,_,_=prior.verify()
    expected_ids=set(old_inputs)|set(inputs)
    # old_inputs contains the original 66 plus nine explicitly adapted MMA9 requests.
    assert len(expected_ids)==83
    old_refs={}
    for r in old_cpu['rows']:
        if r['semantics']['operator']!=0 or r['backend']=='fftw-cpu-t8':old_refs[r['case_id']]=r
    old_refs.update({r['case_id']:r for r in old_mma['rows']})
    for process in processes:
        assert process['status']=='complete-provisional' and len(process['rows'])==83
        assert {r['case_id'] for r in process['rows']}==expected_ids
        assert not process['identity']['cpu_exclusive'] and not process['identity']['cpu_reserved']
        assert process['identity']['host']['cpu']=='Intel(R) Xeon(R) Platinum 8488C'
        assert process['identity']['protocol']==dict(warmup=10,samples=7,calls=20,fft_threads=8,other_threads=1)
        for row in process['rows']:
            identifier=row['case_id'];s=row['semantics'];item=inputs.get(identifier,old_inputs.get(identifier))
            assert s==item['semantics'] and row['input_sha256']==item['input_sha256']
            assert prior.timing(row,True) and row['status'] in prior.GOOD
            assert row['affinity']==([16] if s['storage']==0 else list(range(16,24)))
            assert not row['load_before']['reserved'] and not row['load_before']['exclusive']
            assert row['arrays']['reference_sha256']==row['reference_sha256']
            if s['storage'] in (8,9):assert row['correctness']['exact'] and row['arrays']['exact']
            if s['operator']==0:assert row['backend']=='fftw-cpu-t8' and row['metadata']['threads']==8
            if s['storage']==0:
                assert not row['metadata']['native_fp16'] and row['metadata']['all_conversion_and_scaling_timed']
            old=old_refs.get(identifier)
            assert row['old_median_ms']==(old['timing']['median_ms'] if old else None)
    return manifest,inputs,records,processes,a100,a100_rows,old_refs


def render():
    manifest,inputs,records,processes,a100,a100_rows,old_refs=verify()
    old_tables,_=prior.render();identities=dict(old_tables['identities'])
    def intern(value):
        digest=prior.digest(value);identities[digest]=value;return digest
    by_process=[{r['case_id']:r for r in p['rows']} for p in processes]
    cpu=[];samples=[]
    for identifier in sorted(by_process[0]):
        rows=[p[identifier] for p in by_process];first=rows[0];s=first['semantics']
        medians=[prior.timing(r,True) for r in rows];chosen=statistics.median(medians)
        cpu.append(dict(case_id=identifier,operator=prior.OPS[s['operator']],precision=prior.PREC[s['storage']],
            semantics=prior.js(s),backend=first['backend'],input_sha256=first['input_sha256'],
            reference_sha256=first['reference_sha256'],process_medians_us=prior.js(medians),
            median_of_medians_us=chosen,minimum_process_us=min(medians),maximum_process_us=max(medians),
            process_max_over_min=max(medians)/min(medians),old_us=first['old_median_ms']*1000 if first['old_median_ms'] else None,
            confirmed_over_old=chosen/(first['old_median_ms']*1000) if first['old_median_ms'] else None,
            status='provisional-shared-host',correctness_status=first['status'],
            occupancy_max_percent=max(max(r['load_before']['selected_core_busy_percent'].values()) for r in rows),
            cpu_metadata_sha256=intern(first['metadata'])))
        for process,row in zip(processes,rows):
            samples.append(dict(case_id=identifier,process=process['identity']['process_number'],pid=process['identity']['pid'],
                backend=row['backend'],semantics=prior.js(s),input_sha256=row['input_sha256'],
                median_us=prior.timing(row,True),samples_ms=prior.js(row['timing']['samples_ms']),
                correctness=prior.js(row['correctness']),correctness_status=row['status'],affinity=prior.js(row['affinity']),
                load_before=prior.js(row['load_before']),loadavg_end=prior.js(row['loadavg_end']),
                process_identity_sha256=intern(process['identity']),metadata_sha256=intern(row['metadata']),
                error_array_hashes=prior.js(row['arrays'])))
    denominators={r['case_id']:r for r in cpu}
    ntt_rows=[];cross=[]
    for envelope in records:
        row=envelope['row'];identifier=row['id'];s=row['descriptor']['semantics'];inp=inputs[identifier]
        common=dict(cohort=COHORT,case_id=identifier,operator='NTT',precision=prior.PREC[s['storage']],
            semantics=prior.js(s),N=65536,batch=s['batch'],inverse=s['inverse'],word_bits=s['word_bits'],
            modulus=s['modulus'],modulus_bits=inp['modulus_bits'],root=inp['root'],inverse_root=inp['inverse_root'],
            campaign_status=row['status'],own_backend_status=prior.backend_status(row),
            input_sha256=inp['input_sha256'],reference_sha256=inp['reference_sha256'],
            mapping=prior.js(row['mapping']),gpu_uuid=envelope['identity']['gpu_uuid'],
            gpu_name=envelope['identity']['gpu_name'],gpu_memory=envelope['identity']['gpu_memory'],
            library_sha256=envelope['identity']['library_sha256'],bridge_sha256=envelope['identity']['bridge_sha256'],
            source_row_sha256=envelope['row_sha256'],source_identity_sha256=intern(envelope['identity']),
            cpu_reference_us=denominators[identifier]['median_of_medians_us'],cpu_status='provisional-shared-host')
        for name,item in [('cubutterfly',row),('gpu-ntt',row['baselines'][0])]:
            status=prior.backend_status(row) if name=='cubutterfly' else item['status']
            value=prior.timing(item);valid=status=='pass' and value is not None
            ntt_rows.append(dict(common,backend=name,backend_status=status,status=status,
                reason=item.get('error','') if status!='pass' else '',latency_us=value,
                samples_ms=prior.js(item.get('timing',{}).get('samples_ms')),correctness=prior.js(item.get('correctness')),
                replay_correctness=prior.js(row['mapping_replay_correctness']) if name=='cubutterfly' else '',
                gpu_speedup=value/prior.timing(row) if name=='gpu-ntt' and valid else None,
                cpu_normalized=common['cpu_reference_us']/value if valid else None,
                backend_metadata_sha256=intern(item.get('metadata',item.get('support_metadata',{})))))
        a=a100_rows[identifier]
        baseline,=a['baselines']
        cross.append(dict(case_id=identifier,semantics=prior.js(s),exact_semantic_match=True,precision=prior.PREC[s['storage']],
            a100_own_us=a['own_us'],pro6000_own_us=prior.timing(row),a100_over_pro6000=a['own_us']/prior.timing(row),
            a100_baseline_status=baseline['status'],pro6000_baseline_status=row['baselines'][0]['status'],
            a100_source_row_sha256=a['source_row_sha256'],a100_identity_sha256=intern(a100['identities']),
            pro6000_identity_sha256=common['source_identity_sha256'],
            input_hash_scope='PRO6000 input hashed; historical A100 input hash unavailable; semantic intersection only'))
    selected=[r for r in old_tables['backends'] if r['cohort'] in ('fft17-final','mma9-original')
              or r['cohort']=='full66' and r['operator']!='FFT']
    normalized=[]
    for r in selected+ntt_rows:
        d=denominators[r['case_id']];latency=r['latency_us']
        value=d['median_of_medians_us']/latency if r['cpu_normalized'] is not None else None
        normalized.append(dict(cohort=r['cohort'],case_id=r['case_id'],operator=r['operator'],precision=r['precision'],
            semantics=r['semantics'],backend=r['backend'],backend_status=r['backend_status'],campaign_status=r['campaign_status'],
            input_sha256=r['input_sha256'],gpu_uuid=r['gpu_uuid'],library_sha256=r['library_sha256'],
            latency_us=latency,cpu_reference_us=d['median_of_medians_us'],cpu_process_medians_us=d['process_medians_us'],
            cpu_status=d['status'],cpu_normalized=value,gpu_speedup=r['gpu_speedup']))
    stats=prior.statistics_rows(ntt_rows)
    occupancy=[sum(max(r['load_before']['selected_core_busy_percent'].values())>5 for r in p['rows']) for p in processes]
    report=['# PRO6000 Figure 9 finite follow-up','',
      'Scope: [PR10 follow-up](https://github.com/TruNcat3/cuButterfly/pull/10#issuecomment-6979824753). Original GPU cohorts and raw records remain unchanged. This adds a separately named legal cyclic NTT8 population, fixes derived backend-status ownership, and confirms only the 83 requested CPU denominators in three independent processes.','',
      '## Backend status ownership','',
      'The three original Full66 and two fixed-mapping NTT campaign failures are external GPU-NTT failures. Derived request tables show campaign_status separately from own_backend_status; backend tables and coverage count each backend_status. cuButterfly is pass because its own and replay checks are exact. GPU-NTT remains correctness-failure with no speedup. The original source status, row bytes and hashes are retained. GPU medians and all 78 prior statistics are unchanged.','',
      '## Independent pro6000-cyclic-ntt8','',
      'Predeclared eight requests: N=65536, B=1 or 16, forward and inverse, word32 q=1073479681 and word64 q=582090251837636609. Packed natural input/output, out-of-place, forward unnormalized and inverse multiplied by N inverse modulo q. Full descriptions match the A100 cyclic requests; actual primitive roots, input/reference hashes and build identities are recorded. No Full66 modulus or population was edited.','',
      'GPU1 uses the already calibrated original SM120 library 3b0668f2… with public MEASURE budget 16 and exact mapping replay. The existing NTT selector counts scheduled candidates; word32 has 14 accepted and two rejected in each 16 scheduled, word64 has 16 accepted. This is a bounded search, not exhaustive coverage. All eight cuButterfly outputs and replays are exact. GPU-NTT has six exact timed pairs: four word32 and two word64 inverse. Both word64 forward results fail; they have no baseline latency or ratio.','',
      '| Width | GPU-NTT valid pairs | Geometric speedup | Range | Above / below 1 |','|---|---:|---:|---:|---:|']
    for row in stats:
        if row['metric']=='gpu_speedup':report.append(f"| {row['precision']} | {row['pairs']} | {row['geomean']:.3f}× | {row['minimum']:.3f}–{row['maximum']:.3f}× | {row['above_one']} / {row['below_one']} |")
    report+=['','The same original bridge is independently reproduced with one complete call on each failing word64 forward request: 65,534 / 65,536 mismatches at B=1, and 1,048,544 / 1,048,576 at B=16. Input preservation and exclusive GPU guards pass. Minimal reproduction commands and source are retained; no new root/order/residue diagnosis, third-party kernel repair or mechanism profiling was attempted. Required order conversion/inverse scaling is inside the timed complete call. Seven warm Graph samples × twenty calls follow ten warmups; setup is excluded.','',
      '## CPU confirmation and remaining provisional status','',
      'Three different process IDs complete 83/83 denominators each: Full66 66 (FFTW eight threads only), MMA9 nine explicit FP16-I/O/FP32-FXT adapters, and cyclic NTT8 eight HEXL records. Every process uses the same inputs, thread settings and affinities (16–23 for Full66/NTT8, CPU16 for serial MMA9). Each process retains seven wall-clock samples × twenty calls after ten warmups. The fixed summary is median of each process, then median of the three process medians; no fastest-process selection. Original timings and all new process values/ranges remain available.','',
      f'No CPU reservation or exclusivity is claimed. Pre-row 0.2-second per-core utilization probes and start/end process tick/affinity snapshots are retained. Rows with some selected core above 5% observed pre-row utilization: {occupancy[0]}, {occupancy[1]}, {occupancy[2]} of 83 in the three processes; this is a descriptive load indicator, not a numerical acceptance threshold. All confirmed CPU-normalized values remain provisional-shared-host. This does not block the valid GPU/GPU pairs.','',
      f"Across the 83 denominators, the median process max/min ratio is {statistics.median(r['process_max_over_min'] for r in cpu):.3f}, maximum {max(r['process_max_over_min'] for r in cpu):.3f}. Across the 75 existing denominators, new/old geometric ratio is {prior.geomean([r['confirmed_over_old'] for r in cpu if r['confirmed_over_old'] is not None]):.3f}. These are timing stability observations, not precision error tolerances.",'',
      'Each process has 37 exact integer and 46 floating review records. FP16 conversion/scaling/narrowing and HEXL uint32-to-uint64 storage adaptation remain inside every execute. Full numerical distributions and reference hashes are retained; no floating acceptance threshold is introduced or relaxed. No GPU Full66 or final FFT cohort was rerun for this CPU confirmation.','',
      '## Tables and clean-directory reproduction','',
      '[NTT8 every backend](../snapshots/pro6000_figure9_followup_20261118/ntt8_backends.csv), [independent NTT8 statistics](../snapshots/pro6000_figure9_followup_20261118/statistics.csv), [83 CPU denominators](../snapshots/pro6000_figure9_followup_20261118/cpu_confirmation.csv), [249 process records](../snapshots/pro6000_figure9_followup_20261118/cpu_processes.csv), [CPU-normalized plot inputs](../snapshots/pro6000_figure9_followup_20261118/normalized_backends.csv), [exact A100 intersections](../snapshots/pro6000_figure9_followup_20261118/a100_intersections.csv) and [identity registry](../snapshots/pro6000_figure9_followup_20261118/identities.json). The original Figure9 tables retain the old CPU denominators; the confirmation table is separate. Platform/build/cohort means are never pooled.','',
      'The self-contained performance-record package includes the four report verifiers, every older cohort source checkpoint, required A100 snapshot/mapping catalog, new NTT8/CPU records and derived tables. Large output-array payloads and binaries are omitted: original array hashes remain metadata, and the Figure9 verifier checks their matching source receipts without claiming to replay absent arrays. The fixed-sample standalone report keeps its stricter default payload verification in a full checkout.','',
      'After extracting the standalone package into an empty directory, run `python3 experiments/tools/report_pro6000_figure9.py --check`. Only Python standard library is needed. It checks both Figure9 and this follow-up. All compressed records have compressed/uncompressed hashes, and the package inventory covers every file.','',
      'The [confirmed main CPU-normalized figure](../../figures/pro6000_figure9_confirmed.svg), [confirmed MMA9 panel](../../figures/pro6000_figure9_mma9_confirmed.svg) and [independent NTT8 panel](../../figures/pro6000_figure9_cyclic_ntt8.svg) use the fixed three-process medians and retain provisional labels. PDF and PNG exports accompany each SVG.','',
      'Download the [self-contained performance records](https://github.com/TruNcat3/cuButterfly/releases/download/pro6000-performance-20261118/pro6000-figure9-followup-standalone-20261118.zip) and [independent SHA256](https://github.com/TruNcat3/cuButterfly/releases/download/pro6000-performance-20261118/pro6000-figure9-followup-standalone-20261118.zip.sha256). Earlier immutable bundles remain available.','',
      'Minimal failing-call reproduction on the measurement installation (after sourcing its recorded environment):','',
      '```bash',
      'python3 experiments/tools/reproduce_pro6000_ntt8_failure.py \\',
      '  --measured-root /home/wt/git/cuButterfly-issue9 \\',
      '  --preparation /home/wt/git/cuButterfly-issue9/results/pro6000-figure9-followup-20261118/cyclic-ntt8 \\',
      '  --case-id d4b0e39341b4b0f05c12a4e2f104181e16f556b019af57b3fa124d2bbb400560 \\',
      '  --output /tmp/ntt8-reproduction.json',
      '```','',
      'The B=16 failing case is `04a82eaac33704ba5fdbf53f1073fbd197a55b7b84277240ccce58946e244574`. This diagnostic executes only the original GPU-NTT call, reports exact mismatches and includes no search or substitute kernel. Both direct reproduction records and complete source/bridge/input hashes are in the snapshot; raw output arrays stay on the host.','',
      'No Figure11 mechanisms, NCU, fixed-core migration, changed Full66 population, unbounded search or all-green third-party numerical claim is added.','']
    return dict(ntt8_backends=ntt_rows,statistics=stats,cpu_confirmation=cpu,cpu_processes=samples,
        normalized_backends=normalized,a100_intersections=cross,identities=identities),'\n'.join(report)


def check(write=False):
    tables,text=render()
    outputs={SNAP/(k+'.csv'):prior.csv_text(v).encode() for k,v in tables.items() if k!='identities'}
    outputs[SNAP/'identities.json']=(json.dumps(tables['identities'],sort_keys=True,indent=2)+'\n').encode()
    outputs[SNAP/'tables.json.gz']=gzip.compress((prior.js(tables)+'\n').encode(),mtime=0)
    outputs[REPORT]=text.encode()
    for path,raw in outputs.items():
        if write:path.write_bytes(raw)
        elif path.suffix=='.gz':assert gzip.decompress(path.read_bytes())==gzip.decompress(raw),str(path)
        else:assert path.read_bytes()==raw,str(path)
    print('Verified separate cyclic NTT8 (6 pairs/2 external failures), backend ownership and CPU 3×83 process medians')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true');parser.add_argument('--check',action='store_true')
    args=parser.parse_args();check(args.write)
