#!/usr/bin/env python3
"""Freeze independent boundary controls, or redraw from the frozen JSON on CPU.

Ratios are baseline / intervention latency (above one is faster). Ranges are
min/max across three independently launched processes, never confidence intervals.
Only the diagnostic panels use a declared subset; all raw files are frozen.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
STEM = 'boundary_io_control_20261109'
UUID = {'global': 'GPU-2971e500-d3f6-d7e6-e5f2-cb67a06fff02',
        'bank': 'GPU-0eae5f07-33de-321b-70a9-ffeae49309e6'}
GLOBAL_REQUESTS = [('fft',16,16),('fft',20,4),('fwht',16,16),('fwht',20,4)]
BANK_REQUESTS = [(20,4),(20,16)]
MODES = ['strided','materialized','producer']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    result = float(str(value).replace(',', ''))
    require(math.isfinite(result), 'missing/nonfinite metric: '+str(value))
    return result


def wide_ncu(text, required_metrics, expected):
    """NCU raw wide page: skip units and all nonnumeric-ID lines explicitly."""
    rows = list(csv.reader(io.StringIO(text)))
    headers = [i for i,r in enumerate(rows) if r[:2] == ['ID','Process ID']]
    require(len(headers)==1, 'expected one NCU wide table')
    header = rows[headers[0]]
    for metric in required_metrics:
        require(metric in header, 'missing NCU column: '+metric)
    kernels = []
    for row in rows[headers[0]+1:]:
        if not row or not row[0].isdigit():
            continue
        require(len(row)==len(header), 'malformed NCU numeric row')
        raw = dict(zip(header,row))
        kernels.append({'id':int(raw['ID']), 'kernel_name':raw['Kernel Name'],
                        'metrics':{m:number(raw[m]) for m in required_metrics},
                        'actual_launch':{key:raw[key] for key in ['Block Size','Grid Size','launch__block_size','launch__grid_size','launch__block_dim_x','launch__block_dim_y','launch__block_dim_z','launch__grid_dim_x','launch__grid_dim_y','launch__grid_dim_z']},
                        'raw_columns':raw})
    require(len(kernels)==expected, f'expected {expected} kernels, got {len(kernels)}')
    require(len({k['id'] for k in kernels})==expected, 'duplicate NCU kernel IDs')
    return kernels


def summarize(values):
    require(len(values)==3, 'three process ratios required')
    return {'process_ratios':values,'median':statistics.median(values),
            'min':min(values),'max':max(values)}


def freeze(input_root):
    frozen = {'schema':'boundary-io-control-frozen-v1','input_root':str(input_root.resolve()),
              'raw_inputs':{},'identities':{},'global_timing':[], 'bank_timing':[],
              'profiles':{},'ratios':[],
              'interpretation':{'ratio':'baseline median_ms / intervention median_ms within process; above one is faster',
                'range':'min/max across three process ratios; not a confidence interval',
                'global_baseline':'strided; shared writer-aligned core fixed',
                'bank_baseline':'linear; only prefix shared staging bank permutation changes',
                'profile_subset':'panel c: FWHT N16 B16, all global modes; panel d: prefix only, both bank requests/modes; all kernels retained',
                'profile_timing':'NCU duration is diagnostic and never used for timing ratios',
                'execution_groups':'logical plan descriptors; generic grid_ctas is not an observed physical launch count; use NCU actual_launch for physical grid/block',
                'figure_size_inches':[7.1,3.5], 'font_size_pt':7.5, 'upper_point_panel_ylim':[0.94,1.30], 'lower_bar_panel_baseline':0}}
    def read(path):
        content=path.read_bytes()
        frozen['raw_inputs'][str(path.relative_to(input_root))]={'source_path':str(path.resolve()),
            'sha256':hashlib.sha256(content).hexdigest(),'text':content.decode()}
        return content.decode()
    def load(path):return json.loads(read(path))
    for group, expected_calls in [('global_timing',12),('global_profile',6),('bank_timing',6),('bank_profile',4)]:
        directory=input_root/group;release=directory/'release'
        status=load(directory/'status.json');complete=load(release/'complete.json');identity=load(release/'identity.json')
        expected_uuid=UUID[group.split('_')[0]]
        require(status['state']=='complete' and status['gpu_uuid']==expected_uuid, group+': guard incomplete/wrong GPU')
        require(Path(status['release']).resolve()==release.resolve(), group+': guard release mismatch')
        require(identity['gpu_uuid']==expected_uuid, group+': identity GPU mismatch')
        require(complete['status']=='complete' and complete['calls']==expected_calls, group+': incomplete process matrix')
        require(len(identity['calls'])==expected_calls and all(c['returncode']==0 for c in identity['calls']),group+': failed/missing calls')
        require(len({c['label'] for c in identity['calls']})==expected_calls, group+': duplicate labels')
        frozen['identities'][group]=identity
        for path in sorted(release.iterdir()):
            if path.is_file() and str(path.relative_to(input_root)) not in frozen['raw_inputs']:
                read(path)
    preparation=load(input_root/'prepared/boundary_bank_control_prepare.json')
    require(preparation['returncode']==0 and '-ffp-contract=off' in preparation['command'], 'bank preparation failed or CPU FP contraction policy missing')
    require(all(check['passed'] for check in preparation['cpu_only_cli_checks']), 'bank static CLI checks failed')
    bank_identity=frozen['identities']['bank_timing']
    for basename,key in [('boundary_bank_control.cpp','source_sha256'),('boundary_bank_control','binary_sha256'),('libcuntt.so','library_sha256')]:
        hashes=[value for path,value in preparation['sha256'].items() if Path(path).name==basename]
        require(hashes==[bank_identity[key]], 'bank preparation identity mismatch: '+key)
    frozen['bank_preparation']=preparation
    for family in ['global','bank']:
        a,b=[frozen['identities'][family+'_'+phase] for phase in ['timing','profile']]
        for key in ['binary_sha256','library_sha256','source_sha256']:
            require(a[key]==b[key],family+': timing/profile build mismatch: '+key)
    require(len({x['library_sha256'] for x in frozen['identities'].values()})==1,'library identity differs')
    def text(group,name):return frozen['raw_inputs'][group+'/release/'+name]['text']
    gt=frozen['identities']['global_timing']['config']['global_boundary']
    require(gt['processes']==3 and gt['repeat']==20 and gt['trials']==7,'global protocol changed')
    require(gt==frozen['identities']['global_profile']['config']['global_boundary'],'global protocol mismatch')
    rows=list(csv.DictReader(io.StringIO(text('global_timing','rows.csv'))))
    frozen['global_timing']=rows
    for process in range(3):
        for op,log,batch in GLOBAL_REQUESTS:
            label=f'p{process}-{op}-n{log}-b{batch}'
            selected=[r for r in rows if r['run']==label]
            require({r['mode'] for r in selected}==set(MODES)|({'cufft'} if op=='fft' else set()),label+': missing modes')
            for r in selected:
                require(int(r['verified_batches'])==batch and int(r['nonfinite'])==0,label+': invalid full-batch check')
                require(r['operator']==op and r['precision']=='fp32' and int(r['logN'])==log and int(r['batch'])==batch,label+': request mismatch')
                require(int(r['split'])==log//2 and int(r['threads'])==256 and int(r['tiles_per_cta'])==1 and int(r['inverse'])==0,label+': mapping mismatch')
                trials=[number(v) for v in r['trials_ms'].split(';')]
                require(len(trials)==7 and number(r['median_ms'])>0,label+': missing trials')
                require(math.isclose(statistics.median(trials),number(r['median_ms']),rel_tol=1e-8),label+': median mismatch')
                if r['mode']!='cufft':require(number(r['max_abs_same_core'])==0,label+': core output difference')
    require(len(rows)==42,'unexpected global timing rows')
    for op,log,batch in GLOBAL_REQUESTS:
        for mode in MODES[1:]:
            ratios=[]
            for p in range(3):
                pair={r['mode']:number(r['median_ms']) for r in rows if r['run']==f'p{p}-{op}-n{log}-b{batch}'}
                ratios.append(pair['strided']/pair[mode])
            frozen['ratios'].append({'study':'global','operator':op,'logN':log,'batch':batch,'mode':mode,'baseline':'strided',**summarize(ratios)})
    for process in range(3):
        for log,batch in BANK_REQUESTS:
            label=f'p{process}-n{log}-b{batch}';report=json.loads(text('bank_timing',label+'.json'))
            validate_bank(report,log,batch,False)
            frozen['bank_timing'].append({'run':label,**report})
    for log,batch in BANK_REQUESTS:
        ratios=[]
        for report in frozen['bank_timing']:
            if report['batch']==batch:
                pair={c['mode']:number(c['median_ms']) for c in report['cases']}
                ratios.append(pair['linear']/pair['xor'])
        frozen['ratios'].append({'study':'bank','operator':'fft','logN':log,'batch':batch,'mode':'xor','baseline':'linear',**summarize(ratios)})
    for op in ['fft','fwht']:
        for mode in MODES:
            label=f'profile-{op}-{mode}';count=3 if mode=='materialized' else 2
            metrics=frozen['identities']['global_profile']['config']['profile_metrics']
            kernels=wide_ncu(text('global_profile',label+'.csv'),metrics,count)
            require('shared_iterative_kernel' in kernels[0]['kernel_name'] and 'shared_iterative_kernel' in kernels[-1]['kernel_name'],label+': stage order mismatch')
            if count==3:require('boundary_transpose_kernel' in kernels[1]['kernel_name'],label+': transpose missing')
            correctness=list(csv.DictReader(io.StringIO(text('global_profile',label+'.stdout'))))
            require(len(correctness)==1 and int(correctness[0]['nonfinite'])==0 and int(correctness[0]['verified_batches'])==16 and number(correctness[0]['max_abs_same_core'])==0,label+': correctness invalid')
            frozen['profiles'][label]={'study':'global','operator':op,'logN':16,'batch':16,'mode':mode,'kernels':kernels,'correctness':correctness}
    for log,batch in BANK_REQUESTS:
        for mode in ['linear','xor']:
            label=f'profile-n{log}-b{batch}-{mode}'
            report=json.loads(text('bank_profile',label+'.json'));validate_bank(report,log,batch,True)
            require(report['only']==mode and report['profile_complete_calls']==1,label+': invalid profiler scope')
            metrics=frozen['identities']['bank_profile']['config']['profile_metrics']
            kernels=wide_ncu(text('bank_profile',label+'.csv'),metrics,3)
            require('cufftdx_online_first_kernel' in kernels[0]['kernel_name'],label+': prefix missing')
            require('cufftdx_online_direct_second_kernel' in kernels[1]['kernel_name'],label+': suffix missing')
            require('transpose' in kernels[2]['kernel_name'].lower(),label+': transpose missing')
            frozen['profiles'][label]={'study':'bank','logN':log,'batch':batch,'mode':mode,'kernels':kernels,'correctness':report}
    for obj in frozen['profiles'].values():
        for index,kernel in enumerate(obj['kernels']):
            kernel['role']='prefix' if index==0 else ('suffix' if (obj['study']=='bank' and index==1) or (obj['study']=='global' and index==len(obj['kernels'])-1) else 'transpose')
    frozen['diagnostics']=diagnostics(frozen)
    return frozen


def validate_bank(report,log,batch,profile):
    require(report['logN']==log and report['batch']==batch and report['precision']=='fp32' and not report['inverse'],'bank request mismatch')
    require(report['strict_control_valid'] and report['byte_identical_outputs'],'bank strict control failed')
    require(report['profile']==profile and len(report['cases'])==2,'bank profile/mode mismatch')
    mappings=[]
    for case in report['cases']:
        require(case['cpu_error']['nonfinite']==0 and case['linear_output_error']['nonfinite']==0 and case['linear_output_error']['max_abs']==0 and case['verified_batches']==batch,'bank full-batch exactness failure')
        mapping=dict(case['mapping_json']);layout=mapping.pop('shared_layout');require(layout==('linear' if case['mode']=='linear' else 'xor-swizzle'),'bank layout mismatch')
        require(mapping['stage_partition']==[8,12] and mapping['fft_core']=='cufftdx-block' and mapping['prefix_threads']==256 and mapping['suffix_threads']==256 and mapping['prefix_ept']==16 and mapping['suffix_ept']==16 and mapping['cross_twiddle']=='recurrence' and mapping['direct_boundary']=='tiled-transpose' and not mapping['stage_overlap'],'bank mapping changed')
        require(len(case['execution_groups'])==2,'bank execution groups missing')
        if not profile:
            require(len(case['trials_ms'])==7 and case['graph_nodes']==60,'bank trial/graph count mismatch')
            require(math.isclose(statistics.median(case['trials_ms']),case['median_ms'],rel_tol=1e-10),'bank median mismatch')
        mappings.append(mapping)
    require(mappings[0]==mappings[1],'undeclared bank mapping difference')


def diagnostics(frozen):
    out=[]
    for mode in MODES:
        kernels=frozen['profiles']['profile-fwht-'+mode]['kernels']
        for role,op,kernel in [('prefix','st',kernels[0]),('suffix','ld',kernels[-1])]:
            m=kernel['metrics'];sectors=m[f'l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum'];requests=m[f'l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum']
            require(requests>0,'zero diagnostic requests')
            out.append({'study':'global','operator':'fwht','logN':16,'batch':16,'mode':mode,'role':role,'operation':op,'metric':'sectors_per_request','value':sectors/requests})
    for log,batch in BANK_REQUESTS:
        for mode in ['linear','xor']:
            kernel=frozen['profiles'][f'profile-n{log}-b{batch}-{mode}']['kernels'][0]
            for op in ['ld','st']:
                out.append({'study':'bank','operator':'fft','logN':log,'batch':batch,'mode':mode,'role':'prefix','operation':op,'metric':'shared_bank_conflicts_per_kernel','value':kernel['metrics'][f'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum']})
    return out


def export_csv(frozen,path):
    fields=['study','operator','logN','batch','mode','baseline','kind','process','value','median','min','max','role','operation','metric']
    with path.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in frozen['ratios']:
            base={k:v for k,v in row.items() if k!='process_ratios'}
            writer.writerow(dict(base,kind='speedup_summary'))
            for process,value in enumerate(row['process_ratios']):
                writer.writerow({**{k:v for k,v in base.items() if k not in ['median','min','max']},'kind':'speedup_process','process':process,'value':value})
        for row in frozen['diagnostics']:writer.writerow(dict(row,kind='ncu_diagnostic'))


def render(frozen,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    with plt.rc_context({'font.size':7.5,'axes.titlesize':7.5,'axes.labelsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'}):
        fig,axes=plt.subplots(2,2,figsize=(7.1,3.5),layout='constrained')
        colors=['#0072B2','#D55E00'];markers=['o','s']
        for ax,study,requests in [(axes[0,0],'global',GLOBAL_REQUESTS),(axes[0,1],'bank',[('fft',20,4),('fft',20,16)])]:
            modes=['materialized','producer'] if study=='global' else ['xor']
            for j,mode in enumerate(modes):
                for x,(op,log,batch) in enumerate(requests):
                    r=next(r for r in frozen['ratios'] if r['study']==study and r['operator']==op and r['logN']==log and r['batch']==batch and r['mode']==mode)
                    center=x+(j-(len(modes)-1)/2)*.25
                    ax.errorbar(center,r['median'],yerr=[[r['median']-r['min']],[r['max']-r['median']]],color=colors[j],marker=markers[j],ms=4,capsize=3,label=mode if x==0 else None)
                    ax.scatter([center-.08,center+.04,center+.08],r['process_ratios'],marker=markers[j],s=11,facecolors='white',edgecolors=colors[j],linewidths=.7,zorder=3)
                    ax.annotate(f"{r['median']:.3f}",(center,r['median']),xytext=((-5 if j==0 else 5) if study=='global' else 0,(8 if j==0 else 18) if study=='global' else 4),textcoords='offset points',ha='center',va='bottom',color=colors[j],fontsize=7.5)
            ax.axhline(1,color='.4',linestyle='--',lw=.7);ax.set_xticks(range(len(requests)),[f'{op.upper()}\n$2^{{{log}}}$, B{batch}' for op,log,batch in requests]);ax.set_ylabel('Speedup');ax.set_ylim(.94,1.30);ax.set_yticks([.95,1.00,1.10,1.20,1.30]);ax.legend(loc='upper left' if study=='global' else 'center left',frameon=False,ncol=len(modes));ax.grid(axis='y',alpha=.2)
        axes[0,0].set_title('(a) Global boundary; baseline: strided',loc='left')
        axes[0,1].set_title('(b) Prefix bank map; baseline: linear',loc='left')
        ax=axes[1,0]
        for j,(role,operation,label) in enumerate([('prefix','st','prefix store'),('suffix','ld','suffix load')]):
            values=[next(r['value'] for r in frozen['diagnostics'] if r['study']=='global' and r['mode']==mode and r['role']==role) for mode in MODES]
            ax.bar(np.arange(3)+(j-.5)*.32,values,.32,color=colors[j],hatch='' if j==0 else '///',label=label,edgecolor='black',linewidth=.4)
        ax.set_xticks(range(3),['strided','materialized','producer']);ax.set_ylabel('Sectors / request');ax.set_ylim(0,max(r['value'] for r in frozen['diagnostics'] if r['study']=='global')*1.38);ax.set_title('(c) Boundary traffic: FWHT $2^{16}$, B16',loc='left');ax.legend(frameon=False,ncol=2,loc='upper right')
        ax=axes[1,1];categories=[(b,m) for b in [4,16] for m in ['linear','xor']]
        for j,op in enumerate(['ld','st']):
            values=[next(r['value'] for r in frozen['diagnostics'] if r['study']=='bank' and r['batch']==b and r['mode']==mode and r['operation']==op) for b,mode in categories]
            ax.bar(np.arange(4)+(j-.5)*.32,values,.32,color=colors[j],hatch='' if j==0 else '///',label='load' if op=='ld' else 'store',edgecolor='black',linewidth=.4)
        ax.set_xticks(range(4),[f'{mode}\nB{b}' for b,mode in categories]);ax.set_ylabel('Shared conflicts / kernel');ax.set_ylim(0,max(r['value'] for r in frozen['diagnostics'] if r['study']=='bank')*1.38);ax.set_title('(d) Prefix only: FFT $2^{20}$',loc='left');ax.legend(frameon=False,ncol=2,loc='upper right');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        for ax in axes.flat:ax.spines[['top','right']].set_visible(False)
        fig.supxlabel('Speedup = reference/variant. Upper scale: 0.94–1.30. Open: 3 processes; filled: median; range: min/max (not CI).', fontsize=7.5)
        # Keep exact physical page size. The footnote is encoded in the JSON/CSV;
        # titles and labels make the different controls and diagnostic subsets explicit.
        for suffix in ['pdf','svg','png']:fig.savefig(path.with_suffix('.'+suffix),dpi=300)
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,help='explicit live results root; required unless --frozen-only')
    parser.add_argument('--frozen-only',action='store_true',help='redraw frozen JSON without accessing results')
    args=parser.parse_args();json_path=ROOT/'experiments/snapshots'/ (STEM+'.json')
    require(args.frozen_only != (args.input_root is not None),'choose --input-root or --frozen-only')
    if args.frozen_only:
        frozen=json.loads(json_path.read_text())
        require(frozen['schema']=='boundary-io-control-frozen-v1','unexpected frozen schema')
        for key,raw in frozen['raw_inputs'].items():require(hashlib.sha256(raw['text'].encode()).hexdigest()==raw['sha256'],'frozen input hash mismatch: '+key)
    else:
        frozen=freeze(args.input_root)
        frozen['exporter_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        json_path.parent.mkdir(parents=True,exist_ok=True);json_path.write_text(json.dumps(frozen,indent=2,allow_nan=False)+'\n')
    export_csv(frozen,json_path.with_suffix('.csv'))
    figure_path=ROOT/'experiments/figures'/STEM;figure_path.parent.mkdir(parents=True,exist_ok=True);render(frozen,figure_path)
    print('Validated and exported',len(frozen['ratios']),'paired summaries;',len(frozen['profiles']),'complete-call profiles')


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,FileNotFoundError) as error:
        print('Export refused:',error,file=sys.stderr);sys.exit(1)
