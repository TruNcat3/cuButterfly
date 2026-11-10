"""CPU-only final presentation; refuse incomplete three-process measurements."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'results/fwht_return_contract_20261107'
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('bench',ROOT/'scripts/benchmark_fwht_return_contract.py')
bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=HERE)
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/snapshots/fwht_return_contract_20261107.json')
    args=parser.parse_args(argv)
    global_data_root=args.root.resolve()
    result=bench.summarize(global_data_root)
    result['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['contract_interpretation']={
        'primary_comparison':'Within each interface, compare Dao/cuButterfly with the same result lifetime.',
        'cross_interface_limit':'Native-return retains twenty independently allocated captured outputs; supplied-output repeatedly writes one caller buffer. Working set and cache behavior therefore differ. Cross-interface differences cannot be attributed entirely to the Dao copy.',
        'event_boundary':'GPU complete transform and interface operations. Host allocation, planning, compilation, capture and checking excluded.',
        'graph_lifetime':'Each native-return call owns a distinct result inside the twenty-call capture. Replaying the graph reuses those private addresses; it does not allocate fresh user outputs across replays.',
        'working_set_boundary':'Reported result-buffer bytes only. Input, plan workspace, backend scratch and allocator pool overhead are excluded; Dao supplied-output has backend temporary allocation of unmeasured captured live footprint.',
        'historical_limit':'Current775a independent cohort, not a rewrite or direct copy-cost subtraction from historical main66 1.474x.'}
    result['reference_continuity']=json.loads((global_data_root/'reference_continuity.json').read_text())
    result['dao_inventory_recheck']=json.loads((global_data_root/'dao_inventory_recheck.json').read_text())
    stats=[]
    for case in result['config']['cases']:
        s=case['descriptor']['semantics'];size=4*s['batch']*math.prod(s['shape'])
        for interface in result['config']['contracts']:
            row=dict(case_id=case['id'],semantics=s,interface=interface,result_bytes_per_call=size,
                caller_result_buffers=20 if interface=='native-return' else 1,
                caller_result_live_bytes=size*(20 if interface=='native-return' else 1),
                backend_temporary_live_bytes='unknown; not included in caller-result count')
            for backend in ('cubutterfly','dao'):
                values=[r['timing']['median_ms'] for r in result['records'] if r['case_id']==case['id'] and r['interface']==interface and r['backend']==backend]
                assert len(values)==3
                row[backend]=dict(process_median_ms=values,range_ms=[min(values),max(values)],median_of_process_medians_ms=statistics.median(values))
            values=[p['speedup'] for p in result['pairs'] if p['case_id']==case['id'] and p['interface']==interface]
            row['paired_process_ratios']=values;row['ratio_range']=[min(values),max(values)];stats.append(row)
    result['absolute_latency_and_result_working_set']=stats
    bench.write(global_data_root/'summary.json',result)
    lines=['# Main66 FWHT return-interface follow-up','',
        'A100 80GB PCIe GPU1; current775a library; ten unchanged FP32 requests; three clean independent processes. Each process measures both interfaces and both backends, with alternating backend/interface order. Ten warmups, twenty calls per Graph, seven CUDA-event replays. Floating-point errors are report-only, with finite outputs required.','',
        '**Interpretation:** compare paired backends within each interface. Native-return retains twenty independent result buffers, whereas supplied-output reuses one caller buffer. Different result working sets and cache behavior prevent interpreting the whole cross-interface gap as copy cost. Graph replays reuse their captured addresses. No historical main66 result is overwritten.','',
        '| Interface | Process | Pairs | GM | Faster |','|---|---:|---:|---:|---:|']
    for g in result['groups']:
        for t in g['trials']:lines.append(f"| {g['interface']} | {t['trial']} | {t['pairs']} | {t['gm']:.6f} | {t['wins']} |")
    lines+=['','Ranges below are across three independent processes, not confidence intervals. Absolute latency ranges use per-process event medians; ratio ranges use matched process ratios. Result bytes exclude input/workspace/scratch/allocator overhead.','',
        '| Shape / B / direction | Interface | cuButterfly us range | Dao us range | Ratio range | Result bytes retained, each backend |',
        '|---|---|---:|---:|---:|---:|']
    for r in stats:
        s=r['semantics'];own=r['cubutterfly']['range_ms'];base=r['dao']['range_ms'];rat=r['ratio_range'];label='x'.join(map(str,s['shape']))+f" / {s['batch']} / "+('inverse normalized' if s['inverse'] else 'forward')
        lines.append(f"| {label} | {r['interface']} | {own[0]*1000:.3f}--{own[1]*1000:.3f} | {base[0]*1000:.3f}--{base[1]*1000:.3f} | {rat[0]:.6f}--{rat[1]:.6f} | {r['caller_result_live_bytes']} |")
    lines+=['','All ten CPU references match the original main66 error-artifact references in logical order. The JSON retains all 120 measurements, 840 event samples, full error distributions and lossless-array hashes, actual cuButterfly mappings, Dao artifact inventory, three clean-release hashes, semantics, preparation identity and the separate-interface pairwise ratios. Native-return direct checks verify different output addresses, old-result preservation after changed-input calls, and exact restored output; twenty captured results have distinct addresses and match the direct result after every replay.','',
        'Files: `summary.json` (self-contained report), `process1/release/report.json` through `process3/release/report.json` (raw process reports), `prepared.json` and ten prepared NPZ inputs/references, per-process `errors/*.npz`, and `reference_continuity.json`.']
    (global_data_root/'summary.md').write_text('\n'.join(lines)+'\n')
    result['preparation']=json.loads((global_data_root/'prepared.json').read_text())
    result['guard_records']={str(trial):json.loads((global_data_root/('process'+str(trial))/'status.json').read_text()) for trial in (1,2,3)}
    for row in result['records']:
        row['error_summary']=row['correctness']
    bench.write(args.output,result)
    print(json.dumps(result['groups'],indent=2))


if __name__=='__main__':main()
