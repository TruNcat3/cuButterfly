"""CPU-only extraction; run from any working directory. No GPU imports."""
import csv, hashlib, json, pathlib, statistics
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(__file__).resolve().parent
S=ROOT/'results/a100_fft_l_scale_selection_20261104'
B=ROOT/'results/a100_fft_l_boundary_20261104'
C=ROOT/'results/a100_fft_l_completion_20261104'
provenance=[]
def read(p):
    data=p.read_bytes(); provenance.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest()}); return json.loads(data)
def writecsv(name, rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def compact(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
s=read(S/'summary.json'); invalid=read(S/'invalid-probes.json'); b=read(B/'summary.json'); status=read(C/'campaign-status.json')
scale=[]
for i,r in enumerate(s['rows']):
    p=S/'runs'/r['case']/r['variant']/'result.json'; raw=read(p)
    assert r==raw, f'scale summary/raw mismatch: {p}'
    t=r.get('graph_timing',{});d=r.get('direct_timing',{}); m=r.get('mapping',{})
    if t: assert t['median_ms']==statistics.median(t['samples_ms'])
    scale.append(dict(case=r['case'],precision={4:'fp16',5:'bf16'}[r['semantics']['storage']],N=r['semantics']['shape'][0],B=r['semantics']['batch'],variant=r['variant'],status=r['status'],graph_us=t.get('median_ms','')*1000 if t else '',direct_us=d.get('median_ms','')*1000 if d else '',native_l2=r.get('native_cpu_error',{}).get('l2_relative_finite',''),mathematical_l2=r.get('mathematical_cpu_error',{}).get('l2_relative_finite',''),all_outputs_finite=r.get('all_outputs_finite',''),library_sha256=r['library_sha256'],source=str(p.relative_to(ROOT)),summary_pointer=f'/rows/{i}',mapping_json=compact(m),reason=r.get('reason','')))
writecsv('scale_full.csv',scale)
boundary=[]
for i,r in enumerate(b['rows']):
    p=B/r['path']; raw=read(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r['result_sha256']
    t=raw['ordinary_graph_timing']; assert abs(t['median_ms']*1000-r['latency_us'])<1e-9
    assert abs(statistics.median(r['samples_us'])-r['latency_us'])<1e-9
    boundary.append(dict(phase=r['phase'],case=r['case'],precision={4:'fp16',5:'bf16'}[r['semantics']['storage']],N=r['semantics']['shape'][0],B=r['semantics']['batch'],gpu=r['gpu']['name'],gpu_uuid=r['gpu']['uuid'],latency_us=r['latency_us'],C=r['control'].get('C',''),G=r['control'].get('G',''),T=r['control'].get('T',''),old_c1_speedup=r.get('old_c1_speedup',''),old_reference_speedup=r.get('old_reference_speedup',''),vendor_speedup_storage_only=r.get('vendor_speedup_storage_only',''),native_l2=r['cpu_error']['l2_relative_finite'],old_bitwise=(r.get('old_library_bitwise') or {}).get('equal',''),library_sha256=r['library_sha256'],source=str(p.relative_to(ROOT)),summary_pointer=f'/rows/{i}'))
writecsv('boundary_full.csv',boundary)
# All final confirmations, including the small-batch endpoint; no winner-only filtering.
confirmed=[]
for r in b['final_confirmed']:
    old=next(x for x in b['rows'] if x['path']==r['old_c1_reference'])
    vendor=next(x for x in b['rows'] if x['phase']=='vendor' and x['case']==r['case'])
    confirmed.append(dict(case=r['case'],N=r['semantics']['shape'][0],B=r['semantics']['batch'],old_c1_us=old['latency_us'],confirmed_us=r['latency_us'],cufft_us=vendor['latency_us'],organization_speedup=r['old_c1_speedup'],cufft_over_confirmed=r['vendor_speedup_storage_only'],confirmed_source=str((B/r['path']).relative_to(ROOT))))
writecsv('boundary_confirmed.csv',confirmed)
lines=[r'\begin{tabular}{lrrrrrr}',r'\hline',r'Precision & $N$ & $B$ & Control & Confirmed & cuFFT & Org. gain \\',r'\hline']
for r in sorted(confirmed,key=lambda x:x['case']):
    lines.append(f"{r['case'].split('-')[0].upper()} & {r['N']} & {r['B']} & {r['old_c1_us']:.2f} & {r['confirmed_us']:.2f} & {r['cufft_us']:.2f} & {r['organization_speedup']:.2f}$\\times$ \\\\")
lines += [r'\hline',r'\end{tabular}', '% All six final-confirmed N4096 cases; A100 80GB; latencies in microseconds.', '% Control=old C1/G1/T256; confirmed uses C1/G1 for B4, C8/G32 for B64/B256.', '% Fixed-core organization comparison; cuFFT storage/semantics match only.', '% Not public AUTO selections; no accuracy acceptance threshold.']
(OUT/'representative.tex').write_text('\n'.join(lines)+'\n')
whole=read(B/'whole40/summary.json')
for r in whole:
    raw=read(ROOT/r['source'])
    assert abs(raw['ordinary_graph_timing']['median_ms']*1000-r['latency_us'])<1e-9
writecsv('whole40.csv',[dict(phase=r['phase'],latency_us=r['latency_us'],gpu=r['gpu']['name'],gpu_uuid=r['gpu']['uuid'],library_sha256=r['library_sha256'],source=r['source']) for r in whole])
for variant in ['old','new']: read(B/f'ncu40-{variant}/summary.json')
profile=read(B/'profile_summary.json');(OUT/'profile_summary.json').write_text(json.dumps(profile,indent=2)+'\n')
meta={'schema':'fft-l-paper-evidence-v1','scale_rows':len(scale),'scale_requests':len(set(r['case'] for r in scale)),'boundary_rows':len(boundary),'audit':'Every scale summary row equals its raw result; every boundary result SHA256 and timing median checked.','selection':'All six final-confirmed N4096 boundary cases, including B4 endpoint and negative result; complete scale and boundary matrices retained.','completion_status_snapshot':status,'excluded':invalid,'source_files':provenance,'scale_contracts':[{k:r.get(k) for k in ['semantics','precision_contract','gpu','gpu_uuid','library_sha256','vendor_sha256','runner_sha256','prepared_sha256','reference_population','direct_timing','graph_timing']} for r in s['rows']],'boundary_claim_limits':b['claim_limits'],'boundary_strict_vendor_arithmetic_match':b['strict_vendor_arithmetic_match']}
(OUT/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
print('Audited',len(scale),'scale rows;',len(boundary),'boundary rows')
print(json.dumps(confirmed,indent=2))
