#!/usr/bin/env python3
"""CPU-only summary of validated local-core timings and independent NCU evidence."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    return dict(count=len(values), geometric_mean=math.exp(sum(map(math.log, values))/len(values)),
                minimum=min(values), maximum=max(values), wins=sum(x>1 for x in values),
                ties=sum(x==1 for x in values), losses=sum(x<1 for x in values))


def summarize(root, trace, output):
    profile=json.loads((root/'profile.json').read_text())
    prepare=json.loads((root/'prepare.json').read_text())
    grouped={}; resources=[]
    for entry in profile['rows']:
        path=Path(entry['evidence_path'])
        if sha(path)!=entry['evidence_sha256']:
            raise ValueError('evidence hash changed: '+str(path))
        row=json.loads(path.read_text())
        required=('exact_upstream_match','exact_repeat','input_immutable','buffer_guards','graph_reproduced','upstream_graph_reproduced','finite')
        if row['status']!='validated-exact-upstream' or not all(row[k] is True for k in required) or row['guards']!={'before':True,'after':True}:
            raise ValueError('unvalidated row '+str(path))
        for key in ('gpu_uuid','library_sha256'):
            if row[key]!=profile[key]: raise ValueError('identity mismatch '+key)
        if row['contract']!=profile['arithmetic_contract']: raise ValueError('contract mismatch')
        if row['timing']['protocol']!=row['upstream_timing']['protocol']:
            raise ValueError('timing protocol mismatch')
        for timing in ('timing','upstream_timing'):
            if not math.isfinite(row[timing]['median_ms']) or row[timing]['median_ms']<=0:
                raise ValueError('invalid timing')
        key=(entry['log_n'],entry['batch'])
        if entry['groups_per_block'] in grouped.setdefault(key,{}): raise ValueError('duplicate geometry')
        grouped[key][entry['groups_per_block']]=(row,path)
        resources.append(dict(case=row['case'],groups_per_block=entry['groups_per_block'],**row['resources']))
    pairs=[]
    for (log_n,batch),variants in sorted(grouped.items()):
        if set(variants)!={1,2,4}: raise ValueError('incomplete geometry matrix')
        fixed,path=variants[1]
        best_group=min(variants,key=lambda g:variants[g][0]['timing']['median_ms'])
        best,best_path=variants[best_group]
        m=fixed['mapping']
        pairs.append(dict(N=2**log_n,B=batch,local_groups1_us=fixed['timing']['median_ms']*1000,
            upstream_us=fixed['upstream_timing']['median_ms']*1000,
            fixed_speedup=fixed['upstream_timing']['median_ms']/fixed['timing']['median_ms'],
            oracle_groups=best_group,oracle_local_us=best['timing']['median_ms']*1000,
            oracle_paired_upstream_us=best['upstream_timing']['median_ms']*1000,
            oracle_paired_speedup=best['upstream_timing']['median_ms']/best['timing']['median_ms'],
            stage_groups=1,tile_length=2**log_n,threads=m['threads'],chunks_per_warp=m['chunks_per_warp'],
            shared_bytes=m['shared_bytes'],blocks=m['blocks'],prefetch_depth=m['prefetch_depth'],
            mapping='packed-natural; one scratch group; FP16 MMA; FP32 scale fused at output',
            fixed_evidence=str(path),oracle_evidence=str(best_path)))
    ncu=[]
    metrics=['gpu__time_duration.sum','launch__registers_per_thread','launch__shared_mem_per_block',
        'sm__warps_active.avg.pct_of_peak_sustained_active',
        'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
        'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
        'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
        'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum']
    for path in sorted((trace/'profiler').rglob('*.ncu-rep')):
        text=subprocess.check_output(['ncu','--import',str(path),'--page','raw','--csv'],text=True)
        rows=csv.DictReader(io.StringIO(text)); units=next(rows)
        for row in rows:
            ncu.append(dict(case=path.parent.parent.name,backend=path.parent.name,
                kernel=row['Kernel Name'],**{k:row[k] for k in metrics},
                units=json.dumps({k:units[k] for k in metrics}),source=str(path),sha256=sha(path)))
    nsys=[dict(path=str(p),sha256=sha(p)) for p in sorted((trace/'profiler').rglob('*.nsys-rep'))]
    summary=dict(schema='hadacore-local-summary-v1',scope='explicit validated local core; not public AUTO',
        gpu_identity=profile['gpu_identity'],library_sha256=profile['library_sha256'],
        arithmetic_contract=profile['arithmetic_contract'],requested_cases=len(prepare['cases']),
        supported_cases=len(grouped),unsupported_cases=[{'case':r['case'],'reason':r['reason']} for r in prepare['cases'] if r['status']=='unsupported'],
        geometries=len(resources),exact_upstream_geometries=len(resources),resource_records=len(resources),
        fixed_groups1=stats([r['fixed_speedup'] for r in pairs]),
        oracle_not_AUTO=stats([r['oracle_paired_speedup'] for r in pairs]),
        oracle_note='post-hoc minimum local latency over groups 1/2/4; paired baseline from that trial; not independent selection validation',
        ncu_kernel_records=len(ncu),nsys_reports=nsys,
        source_profile=str(root/'profile.json'),source_profile_sha256=sha(root/'profile.json'),
        cpu_reference='radix2-stage-FP16 diagnostic, different contract; no new acceptance threshold',
        register_range=[min(r['registers_per_thread'] for r in resources),max(r['registers_per_thread'] for r in resources)],
        local_memory_bytes_range=[min(r['local_bytes_per_thread'] for r in resources),max(r['local_bytes_per_thread'] for r in resources)])
    output.mkdir(parents=True,exist_ok=True)
    for filename,rows in [('paired.csv',pairs),('resources.csv',resources),('ncu.csv',ncu)]:
        if not rows:continue
        with (output/filename).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(summarize(a.input,a.trace,a.output),indent=2))
