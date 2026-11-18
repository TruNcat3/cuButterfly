#!/usr/bin/env python3
"""Verify figure9 cohorts and emit complete backend and CPU-normalized tables."""
from __future__ import annotations
import argparse,collections,csv,gzip,hashlib,io,json,math,statistics
from pathlib import Path
import report_pro6000_full66 as main66
import report_pro6000_extensions as extensions
import report_pro6000_fixed_samples as fixed

ROOT=Path(__file__).resolve().parents[2]
SNAP=ROOT/'experiments/snapshots/pro6000_figure9_20261118'
REPORT=ROOT/'experiments/reports/PRO6000_FIGURE9_20261118.md'
OPS=('FFT','NTT','FWHT','subset-zeta','superset-zeta','structured-2x2')
PREC={0:'FP16',2:'FP32',3:'FP64',6:'FP32',7:'FP64',8:'word32',9:'word64'}
BACKENDS=('cubutterfly','cufft','vkfft','gpu-fft','daofht','gpu-ntt','hadacore')
GOOD={'pass','accuracy-review-required'}

def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(v):return sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
def js(v):return json.dumps(v,sort_keys=True,allow_nan=False,separators=(',',':'))

def unpack(record,raw_bytes=False):
    p=Path(record['path']);assert not p.is_absolute() and '..' not in p.parts
    raw=(SNAP/p).read_bytes();assert sha(raw)==record['sha256']
    raw=gzip.decompress(raw);assert sha(raw)==record['uncompressed_sha256']
    return raw if raw_bytes else json.loads(raw)

def timing(row,cpu=False):
    t=row.get('timing')
    if not t:return None
    samples=t['samples_ms'];assert len(samples)==7
    assert all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in samples)
    assert statistics.median(samples)==t['median_ms']
    assert t['protocol']==('7 wall-clock samples x 20 complete calls; 10 warmup' if cpu else
                           '20 complete calls per CUDA graph; 7 replays; 10 warmup')
    return t['median_ms']*1000

def verify():
    manifest=json.loads((SNAP/'manifest.json').read_text())
    assert manifest['schema']=='cubutterfly-pro6000-figure9-publication-v1'
    def check_records(value):
        if isinstance(value,dict):
            if {'path','sha256','uncompressed_sha256'} <= set(value):unpack(value,raw_bytes=True)
            else:
                for item in value.values():check_records(item)
        elif isinstance(value,list):
            for item in value:check_records(item)
    check_records(manifest)
    inputs=unpack(manifest['input_hashes']);support=unpack(manifest['support'])
    cpu=unpack(manifest['cpu_full66']);mma_cpu=unpack(manifest['cpu_mma9'])
    assert len(cpu['rows'])==83 and len({(r['case_id'],r['backend']) for r in cpu['rows']})==83
    assert cpu['status']=='complete-pending-floating-review'
    assert cpu['identity']['host']['hostname']=='lab502-Super-server-RTX-pro-6000'
    assert cpu['identity']['host']['cpu']=='Intel(R) Xeon(R) Platinum 8488C'
    assert cpu['identity']['host']['affinity']==list(range(16,24))
    assert cpu['identity']['host']['cpu_exclusive'] is False
    assert len(mma_cpu['rows'])==9 and mma_cpu['status']=='complete-pending-floating-review'
    for row in cpu['rows']:
        item=inputs[row['case_id']]
        assert row['semantics']==item['semantics'] and row['input_sha256']==item['input_sha256']
        assert row['status'] in GOOD and timing(row,True)
        if row['semantics']['storage'] in (8,9):assert row['correctness']['exact'] is True
    for row in mma_cpu['rows']:
        item=inputs[row['case_id']]
        assert row['semantics']==item['semantics'] and row['input_sha256']==item['input_sha256']
        assert row['reference_sha256']==item['reference_sha256']
        assert timing(row,True) and row['status']=='accuracy-review-required'
        assert row['metadata']['native_fp16'] is False and row['metadata']['external_core_storage']=='float32'
        assert row['metadata']['all_conversion_and_scaling_timed'] is True
    _,population,original,_=main66.verify(ROOT/'experiments/snapshots/pro6000_full66_20261118')
    cohorts={'full66':[(r,i) for r,_,i in original]}
    _,ext=extensions.verify(ROOT/'experiments/snapshots/pro6000_extensions_20261117')
    ext_manifest=json.loads((ROOT/'experiments/snapshots/pro6000_extensions_20261117/manifest.json').read_text())
    assert mma_cpu['identity']['source_sha256']==ext_manifest['campaigns']['mma9']['preparation']['uncompressed_sha256']
    for name in ['fft17','mma9']:
        cohorts[name+'-original']=[(e['row'],e['identity']) for e in ext[name]['records']]
    fp=manifest['fft17_final'];preparation=unpack(fp['preparation']);config=preparation['config']
    assert digest(config)==preparation['config_sha256']
    assert config['gpu_uuid']=='GPU-1067ab39-e411-e141-1775-64426bca5701'
    expected_library='a00188ff3415760359a52eb2e812153e0189c957ce5139c9ed4963d7d15c2c0b'
    cases={c['id']:c for c in config['cases']};selections={s['case']:s for s in map(unpack,fp['selection'])}
    records=list(map(unpack,fp['rows']));assert len(records)==len(cases)==len(selections)==17
    completion=unpack(fp['completion']);assert completion['status']=='complete' and completion['paired_timings']==47
    cache=unpack(fp['cache'],raw_bytes=True)
    assert cache.startswith(b'v4|') and b'|semantics:' in cache
    # Cache file hash preserves original whitespace; use the archived byte hash.
    cache_file_sha=fp['cache']['uncompressed_sha256']
    assert completion['cache_sha256']==cache_file_sha
    for e in records:
        r=e['row'];s=r['descriptor']['semantics'];selection=selections[r['id']]
        assert e['identity_sha256']==digest(e['identity']) and e['row_sha256']==digest(r)
        assert e['guards']==selection['guards']=={'before':True,'after':True}
        assert r['status']=='accuracy-review-required' and timing(r)
        assert r['plan_descriptor']['selection']['algorithm_policy']==0
        assert s==cases[r['id']]['descriptor']['semantics']
        assert r['mapping']==selection['mapping'] and selection['selection_source_id']==1 and selection['cache_gate']
        assert e['identity']['libraries']['main']['sha256']==expected_library
        assert e['identity']['gpu_uuid'].removeprefix('GPU-')==config['gpu_uuid'].removeprefix('GPU-')
        assert e['identity']['runtime_bindings'][config['environment']['CUBUTTERFLY_PLAN_CACHE']]==cache_file_sha
        assert e['prepared_sha256']==next(v['sha256'] for v in preparation['cases'] if v['id']==r['id'])
        assert s==inputs[r['id']]['semantics']
        assert [b['name'] for b in r['baselines']]==['cufft','vkfft','gpu-fft']
        for b in r['baselines']:
            if b.get('timing'):assert b['status']=='accuracy-review-required' and timing(b)
    cohorts['fft17-final']=[(e['row'],e['identity']) for e in records]
    frows,_,_,_,provenance,_=fixed.verify(ROOT/'experiments/snapshots/pro6000_fixed_mapping_20261117',require_array_payloads=False)
    cohorts['fixed-mapping']=[(r,dict(gpu_uuid=r['gpu_uuid'],gpu_name=r['gpu_name'],gpu_memory=r['gpu_memory'],library_sha256=r['library_sha256'],provenance=provenance)) for r in frows]
    audit=unpack(manifest['ntt_adapter_audit']);assert len(audit['rows'])==13
    for row in audit['rows']:
        if row['semantics']['word_bits']==32:
            assert row['modulus_bits']==31 and row['support']['supported'] is False
        if row['recorded_baseline_status']=='correctness-failure':
            assert row['own_exact'] and row['replay_exact']
            assert row['residue_order_tests']['after_mod_q_mismatches']>0
            assert row['residue_order_tests']['after_mod_q_and_bit_reversal_mismatches']>0
    return manifest,inputs,support,cpu,mma_cpu,cohorts,audit

def cpu_reference(s,case_id,input_sha,cpu,mma_cpu):
    source=mma_cpu if s['storage']==0 else cpu
    choices=[r for r in source['rows'] if r['case_id']==case_id]
    if s['operator']==0:choices=[r for r in choices if r['backend']=='fftw-cpu-t8']
    assert len(choices)==1,(case_id,len(choices))
    r=choices[0];assert r['semantics']==s and r['input_sha256']==input_sha,'CPU/GPU semantic or input mismatch'
    return r,source['identity']

def own_valid(row):
    if row['status'] in GOOD:return True
    return (row['status']=='correctness-failure' and row['descriptor']['semantics']['operator']==1
            and row.get('correctness',{}).get('exact') is True
            and row.get('mapping_replay_correctness',{}).get('exact') is True
            and any(b['name']=='gpu-ntt' and b['status']=='correctness-failure' for b in row['baselines']))

def backend_status(row):
    """An external failure changes the campaign, not an exact own/replay verdict."""
    if row['status']=='correctness-failure' and own_valid(row):
        return 'pass'
    return row['status']

def reference_identity(cohort,info):
    """Keep native extension references distinct from historical wide diagnostics."""
    if cohort.startswith('fft17-'):
        return info['fft17_reference_sha256'],'native-precision CPU FFT reference'
    if cohort=='mma9-original':
        return info['reference_sha256'],'FP16 per-stage-rounded Sylvester reference'
    return info['reference_sha256'],'original campaign reference; wider float diagnostics retained'

def geomean(values):
    assert values and all(math.isfinite(v) and v>0 for v in values)
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))

def statistics_rows(rows):
    groups=collections.defaultdict(list)
    for r in rows:
        for metric in ['gpu_speedup','cpu_normalized']:
            if r[metric] is not None:groups[(r['cohort'],r['operator'],r['precision'],r['backend'],metric)].append(r[metric])
    return [dict(cohort=k[0],operator=k[1],precision=k[2],backend=k[3],metric=k[4],pairs=len(v),
                 geomean=geomean(v),minimum=min(v),maximum=max(v),above_one=sum(x>1 for x in v),below_one=sum(x<1 for x in v),equal_one=sum(x==1 for x in v))
            for k,v in sorted(groups.items())]

def render():
    manifest,inputs,support,cpu,mma_cpu,cohorts,audit=verify();ntt={r['case_id']:r for r in audit['rows']}
    backend_rows=[];completion=[];cross=[];identities={}
    def intern(value):
        token=digest(value);identities[token]=value;return token
    a100=json.loads((ROOT/'experiments/snapshots/current_refresh/snapshot.json').read_text())
    a100_cases={r['id']:r for r in a100['operators']['requests'] if r['cohort']=='main66'}
    for cohort,observations in cohorts.items():
        for row,identity in observations:
            s=row['descriptor']['semantics'];identifier=row['id'];op=OPS[s['operator']];precision=PREC[s['storage']]
            own=timing(row);valid=own_valid(row);input_info=inputs.get(identifier)
            assert input_info is not None
            reference_sha,reference_role=reference_identity(cohort,input_info)
            cpu_row,cpu_identity=cpu_reference(s,identifier,input_info['input_sha256'],cpu,mma_cpu)
            cpu_us=timing(cpu_row,True)
            roots=ntt.get(identifier,{}).get('roots',[])
            common=dict(cohort=cohort,case_id=identifier,operator=op,precision=precision,storage=s['storage'],compute=s['compute'],
                shape=js(s['shape']),N=math.prod(s['shape']),batch=s['batch'],inverse=s['inverse'],normalize_inverse=s['normalize_inverse'],
                input_orders=js(s['input_orders']),output_orders=js(s['output_orders']),placement=s['placement'],
                input_strides=js(s['input_strides']),output_strides=js(s['output_strides']),input_batch_stride=s['input_batch_stride'],output_batch_stride=s['output_batch_stride'],
                modulus=s.get('modulus'),modulus_bits=s['modulus'].bit_length() if s.get('modulus') else None,word_bits=s.get('word_bits'),roots=js(roots),
                semantics=js(s),campaign_status=row['status'],own_backend_status=backend_status(row),own_usable=valid,input_sha256=input_info['input_sha256'],reference_sha256=reference_sha,reference_role=reference_role,
                cubutterfly_selected_mapping=js(row.get('mapping')),gpu_uuid=identity.get('gpu_uuid'),gpu_name=identity.get('gpu_name'),gpu_memory=identity.get('gpu_memory',identity.get('gpu_total_memory')),
                library_sha256=identity.get('library_sha256',identity.get('libraries',{}).get('main',{}).get('sha256')),
                source_identity_sha256=intern(identity),source_row_sha256=digest(row),cpu_reference_backend=cpu_row['backend'],cpu_reference_us=cpu_us,
                cpu_reference_metadata_sha256=intern(cpu_row['metadata']),cpu_identity_sha256=intern(cpu_identity),cpu_reference_array_sha256=cpu_row.get('reference_array_sha256',cpu_row.get('reference_sha256')))
            completion.append(dict(common,own_us=own,own_correctness=js(row.get('correctness')),mapping_replay_correctness=js(row.get('mapping_replay_correctness'))))
            actual={b['name']:b for b in row.get('baselines',[])}
            for name in BACKENDS:
                item=row if name=='cubutterfly' else actual.get(name)
                if item is None:
                    info=support.get(identifier,{}).get(name)
                    relevant=(s['operator']==0 and name in ['cufft','vkfft','gpu-fft'] or s['operator']==1 and name=='gpu-ntt' or s['operator']==2 and name in ['daofht','hadacore'])
                    item=dict(status=('unsupported-semantic' if info and not info['supported'] else 'not-measured-in-this-cohort') if relevant else 'not-applicable',
                              reason=info.get('reason','') if info else ('No selected same-semantic GPU comparator for this operator/cohort' if relevant else 'Operator outside backend domain'))
                value=timing(item);usable=(valid if name=='cubutterfly' else item['status'] in GOOD) and value is not None
                status=backend_status(row) if name=='cubutterfly' else item['status']
                reason=('' if name=='cubutterfly' and status=='pass' else item.get('error',item.get('reason','')))
                backend_rows.append(dict(common,backend=name,status=status,backend_status=status,source_backend_status=item['status'],support_status=item.get('support_status',''),backend_usable=usable,reason=reason,
                    latency_us=value,samples_ms=js(item.get('timing',{}).get('samples_ms')),timing_protocol=item.get('timing',{}).get('protocol',''),
                    correctness=js(item.get('correctness')),backend_metadata_sha256=intern(item.get('metadata',item.get('support_metadata',{}))),
                    gpu_speedup=value/own if name!='cubutterfly' and usable and valid else None,
                    cpu_normalized=cpu_us/value if usable else None))
            prior=a100_cases.get(identifier);matched=prior is not None and prior['semantics']==s
            cross.append(dict(cohort=cohort,case_id=identifier,operator=op,precision=precision,exact_semantic_match=matched,
                a100_us=prior['own_us'] if matched else None,pro6000_us=own,a100_over_pro6000=prior['own_us']/own if matched and valid else None,
                pro6000_gpu_uuid=common['gpu_uuid'],pro6000_library_sha256=common['library_sha256'],a100_identity_sha256=intern(a100['identities']['main66']),
                a100_input_sha256=None,scope='Historical A100 median and source identity; exact semantic intersection only; A100 raw input hash unavailable in this snapshot. Separate builds/devices, not a fixed-mapping causal comparison.'))
    stats=statistics_rows(backend_rows)
    coverage=[]
    for cohort,rows in cohorts.items():
        selected=[r for r in backend_rows if r['cohort']==cohort]
        counts=collections.Counter((r['operator'],r['precision'],r['backend'],r['backend_status']) for r in selected)
        coverage += [dict(cohort=cohort,operator=k[0],precision=k[1],backend=k[2],status=k[3],backend_status=k[3],requests=v) for k,v in sorted(counts.items())]
    text=['# PRO6000 figure 9 operator comparison extension','',
      'The declared Full66 population remains FFT17, NTT13, FWHT13, subset-zeta8, superset-zeta8 and structured-2x2 seven. All 66 original requests, including negative results, are retained. The original FFT17, independent final-build FFT17, explicit ALLOW_FP16_MMA9 and fixed-mapping samples remain separate cohorts; overlapping requests never add weight to a combined mean.','',
      'The final-build FFT17 completes 17 exact-request MEASURE selections and typed DEFAULT cache replay checks on GPU0, then 47 matched complete-call pairs against cuFFT, VkFFT and GPU-FFT. Four GPU-FFT multi-axis requests remain unsupported. The measured library is a00188ff3415760359a52eb2e812153e0189c957ce5139c9ed4963d7d15c2c0b, using unchanged native SM120 CUDA objects and the documented host search fix. GPU1 calibration/profile files were bound as same-SKU priors; every selected request was actually measured on GPU0. This is not a validated static fallback or globally validated cost model.','',
      'Local CPU denominators complete 83 measurements for the exact Full66 requests: FFTW 3.3.10 at 1 and 8 threads, FXT, HEXL and the author Butterfly CPU implementation. Primary FFT normalization uses the predeclared eight-thread FFTW, not the fastest observed thread setting. Other cores use one thread. The host is Intel Xeon Platinum 8488C, affinity 16–23. Measurements are shared-host and provisional, with load averages retained; no A100-host CPU measurement is relabeled as local. HEXL word32 uses explicitly labeled uint32-I/O/uint64-core storage adaptation, not native word32.','',
      'Nine additional CPU FWHT records use the authorized FP16-I/O/FP32-arithmetic FXT adapter, timed with input widening, inverse scale when required, and final narrowing. They are not native FP16 CPU arithmetic. CPU reference error arrays and full error distributions remain separate from GPU reference roles. CPU/GPU normalization and GPU/GPU ratios are never pooled.','',
      'GPU timing uses ten warmups, seven CUDA Graph samples of twenty complete calls. Required copies, ordering, layout and normalization remain timed; compilation/search/planning/allocation and initialization transfers are outside. CPU timing uses the corresponding seven wall-clock samples of twenty full calls after ten warmups. Floating outputs remain accuracy-review-required, without new or relaxed thresholds. Integer outputs require exact equality.','',
      '## Final-build FFT17 GPU/GPU results','',
      '| Precision | Comparator | Valid pairs | Geometric mean | Range | Above / below 1 |','|---|---|---:|---:|---:|---:|']
    for r in stats:
        if r['cohort']=='fft17-final' and r['metric']=='gpu_speedup':
            text.append(f"| {r['precision']} | {r['backend']} | {r['pairs']} | {r['geomean']:.3f}× | {r['minimum']:.3f}–{r['maximum']:.3f}× | {r['above_one']} / {r['below_one']} |")
    text += ['', 'Each request has equal weight in each operator/precision/comparator/cohort geometric mean. Nominal above-one counts are observations, not independent-process significance tests. All losing points are retained. The complete original-cohort statistics remain in the linked CSV; the new two-point diagnostic was not substituted into the old population.','',
      '## Support and NTT negative results','',
      'Full66 FWHT FP32/FP64 requests do not match the current FP16-only HadaCore adapter; the gaps are explicit. Dao FHT has ten legal FP32 matches; the length limit and missing FP64 arithmetic explain the other three cells. MMA9 retains nine HadaCore pairs; Dao native-FP32 arithmetic cannot substitute for the requested native-FP16/MMA contract. Zeta and structured operators have native CPU baselines but no selected high-performance external GPU comparator.','',
      'The four main word32 NTT requests use q=2013265921 (31 bits), outside the pinned GPU-NTT Data32 q<2^30 range. No modulus or storage width was changed. The five rank-one word64 requests use a 60-bit modulus inside the Data64 q<2^61 range; two inverse pairs pass and three forward cases remain failed. Four multidimensional word64 requests are outside the bridge rank-one support. NTT word64 is never labeled FP64.','',
      'The canonical primitive root and inverse root are recorded per axis. The pinned bridge returns the canonical root. Applying mod-q reduction or bit reversal to the three original wrong outputs does not repair them. Existing hash-verified stage evidence shows correct first launches and divergence after the second; the isolated synchronized prefix diagnostic is exact. This does not identify or repair the internal kernel defect. The three external failures retain their original error states and no speedup; cuButterfly and its mapping replay are independently exact.','',
      'Derived tables distinguish original campaign_status from each backend_status. For the three Full66 and two fixed-mapping external GPU-NTT failures, cuButterfly is pass because its own and mapping-replay integer checks are exact. GPU-NTT remains correctness-failure with no ratio. Original source rows and their hashes are unchanged; source_backend_status retains their original aggregate label. Coverage counts use backend_status.','',
      'The [finite follow-up](PRO6000_FIGURE9_FOLLOWUP_20261118.md) adds a separately predeclared legal cyclic NTT8 population and three independent CPU confirmation processes. Original cohort timings, CPU records and GPU/GPU statistics above remain unchanged.','',
      '## Auditable tables','',
      '[Every request and full semantics](../snapshots/pro6000_figure9_20261118/requests.csv), [every GPU backend cell](../snapshots/pro6000_figure9_20261118/backends.csv), [local CPU records](../snapshots/pro6000_figure9_20261118/cpu.csv), [per-cohort statistics](../snapshots/pro6000_figure9_20261118/statistics.csv), [support counts](../snapshots/pro6000_figure9_20261118/coverage.csv), [A100 exact semantic intersections](../snapshots/pro6000_figure9_20261118/a100_intersections.csv), [CPU/GPU/library identity registry](../snapshots/pro6000_figure9_20261118/identities.json), and [source/checksum manifest](../snapshots/pro6000_figure9_20261118/manifest.json). Identity/metadata SHA columns resolve to complete records in identities.json, avoiding duplicated source inventories in every CSV cell.','',
      'A100 and PRO6000 retain their own build/device/cohort identities. The cross-platform table admits only exact semantic matches and keeps each PRO6000 cohort separate. The historical A100 snapshot does not include per-request input hashes here; those cells are explicitly unavailable, not invented. No pooled platform mean or claim of identical physical mappings is made.','',
      'The [CPU-normalized main figure](../../figures/pro6000_figure9.svg) and [separate FP16-adapted MMA9 panel](../../figures/pro6000_figure9_mma9.svg) use the same verified tables. FFT panels use only the final-build FFT17 cohort on GPU0; all other main panels use only Full66 on GPU1. Each box retains its own valid request count. CPU shared-host timings and floating numerical review make these provisional paper inputs. The [plot source](../tools/plot_pro6000_figure9.py) also exports PDF and PNG.','',
      'Download the [performance-record bundle](https://github.com/TruNcat3/cuButterfly/releases/download/pro6000-performance-20261118/pro6000-figure9-20261118.zip) and [SHA256](https://github.com/TruNcat3/cuButterfly/releases/download/pro6000-performance-20261118/pro6000-figure9-20261118.zip.sha256). It contains tables, compressed original records and a JSON representation of all derived tables; full output arrays and libraries are not included.','',
      'Full lossless arrays and binaries remain local. The portable snapshot retains original JSON/CSV evidence, numerical error distributions, input/reference hashes, source identities, complete timing samples and library/calibration bindings. This completes the requested finite figure9 evidence collection with retained gaps/failures; it is not all-green numerical acceptance, a full new-build Full66 rerun, figure11 mechanism evidence or global model validation.','',
      '```bash','python3 experiments/tools/report_pro6000_figure9.py --check','```','']
    cpu_rows=[dict(cpu_cohort=label,case_id=r['case_id'],backend=r['backend'],semantics=js(r['semantics']),latency_us=timing(r,True),samples_ms=js(r['timing']['samples_ms']),
                   status=r['status'],input_sha256=r['input_sha256'],reference_sha256=r.get('reference_array_sha256',r.get('reference_sha256')),correctness=js(r['correctness']),metadata_sha256=intern(r['metadata']),host_identity_sha256=intern(data['identity']))
              for label,data in [('full66-local',cpu),('mma9-fp16-adapted',mma_cpu)] for r in data['rows']]
    return dict(requests=completion,backends=backend_rows,cpu=cpu_rows,statistics=stats,coverage=coverage,a100_intersections=cross,identities=identities), '\n'.join(text)

def csv_text(rows):
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows);return f.getvalue()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args()
    tables,text=render();outputs={SNAP/(k+'.csv'):csv_text(v) for k,v in tables.items() if k!='identities'}
    outputs[SNAP/'identities.json']=json.dumps(tables['identities'],indent=2,sort_keys=True,allow_nan=False)+'\n';outputs[REPORT]=text
    outputs[SNAP/'tables.json.gz']=gzip.compress((json.dumps(tables,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode(),mtime=0)
    for path,value in outputs.items():
        raw=value if isinstance(value,bytes) else value.encode()
        if a.write:path.write_bytes(raw)
        elif path.suffix=='.gz':
            # Python/zlib versions can emit different valid gzip headers/streams.
            assert gzip.decompress(path.read_bytes())==gzip.decompress(raw),'Derived figure9 JSON differs: '+str(path)
        else:assert path.read_bytes()==raw,'Derived figure9 output differs: '+str(path)
    print('Verified figure9 complete cohorts, exact CPU input joins, timings, gaps, failures and independent denominators')
    if (ROOT/'experiments/snapshots/pro6000_figure9_followup_20261118/manifest.json').exists() and not a.write:
        import report_pro6000_figure9_followup
        report_pro6000_figure9_followup.check()

if __name__=='__main__':main()
