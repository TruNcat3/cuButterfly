"""Read-only CPU audit plus finite, unexecuted NCU command preparation."""
import hashlib
import json
import math
import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper/evidence'
sources={}
def src(path,field='/'):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    name=str(p.relative_to(ROOT))
    sources[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
    return dict(path=name,field=field,**sources[name])
def load(path):
    src(path);return json.loads((ROOT/path).read_text())

xd=[]
for mode in ['measure','default']:
    path=f'results/paper_targeted_20261012/xd_application_breakdown_v1/{mode}_guard/release/result.json'
    d=load(path)
    for i,c in enumerate(d['direct']):
        plans=[]
        for p in c['plans']:
            plans.append({k:p[k] for k in ['direction','selection_source','algorithm','mapping','descriptor']})
        xd.append(dict(mode=mode,shape=c['shape'],B=c['batch'],plans=plans,methods=c['methods'],source=src(path,f'/direct/{i}')))
filters=load('results/paper_targeted_20261012/xd_application_breakdown_v1/default_guard/release/result.json')['filters']
xd_code_rows=[]
for case in [c for c in xd if c['mode']=='default']:
    physical={p['direction']:p['mapping']['physical'] for p in case['plans']}
    f,i=physical['forward'],physical['inverse']
    xd_code_rows.append(dict(shape=case['shape'],batch=case['B'],
        forward_axis_order=f['axis_order'],inverse_axis_order=i['axis_order'],
        forward_cores=[a['core'] for a in f['axes']],inverse_cores=[a['core'] for a in i['axes']],
        forward_ffts_per_block=[a['ffts_per_block'] for a in f['axes']],
        inverse_ffts_per_block=[a['ffts_per_block'] for a in i['axes']],
        actual_packed_axes={'forward':[a['packed'] for a in f['axes']], 'inverse':[a['packed'] for a in i['axes']]},
        output_permutation={'forward':f['output_permutation'],'inverse':i['output_permutation']},
        forward_output_scale=f['output_scale'],inverse_output_scale=i['output_scale'],
        inverse_normalize=i['normalize_inverse'],copy_elision_source_eligible=bool(i['axes']),
        inverse_postscale_needed=(i['output_scale']*(1.0/math.prod(case['shape']) if i['normalize_inverse'] else 1.0)!=1.0),
        source=case['source']))
xd_code_audit=dict(evidence_level='current-source inference plus recorded selected-plan metadata; no modified binary or GPU experiment',
    source=src('src/xd_real_fft.cu','399-406 compact; 462-470 real/complex plans; 537-550 workspace; 690-695 same-stride C2C plan; 745-775 real execution; 778-872 C2C out-of-place/scaling/layout'),
    kernel_sources=[src(p,'const input / separate output / final-write scaling') for p in
        ['src/xd_axis_fft.cu','src/xd_pair_fft.cu','src/xd_register_fft.cu','src/xd_local_fft.cu','src/xd_cooperative_pair_fft.cu']],
    recorded_default_cases=xd_code_rows,
    existing_coverage=dict(compact='Both real directions retain last-axis n/2+1 spectrum; no full-spectrum expansion. C2R scratch is compact-sized.',
        scale='R2C and complex-only C2C pass scale to the last fused complex core; vendor-last/no-axis paths retain standalone complex scaling. Real inverse currently uses standalone real scaling only when effective scale != 1.',
        direction='Independent plans already exist. Default 128^3 B4 chooses forward [0,1] and inverse [1,0]; packed_axes capability is not an executed permutation: all eight cases have per-axis packed=false and empty output permutation.',
        real_core='Last real axis is cuFFT R2C/D2Z versus C2R/Z2D; different vendor real kernels are not isolated arithmetic-core comparisons.'),
    minimal_suggestions=[dict(id='inverse_first_axis_out_of_place',status='source-feasible, unmeasured',
        edit='At 762-765, if axis_plans_.empty() retain compact D2D copy; otherwise execute_c2c_axes(const_cast<Complex*>(input_complex), scratch, CUFFT_INVERSE, true). Keep compact scratch allocation.',
        applicability='All valid real rank>=2 plans with at least one complex axis, including all eight default diagnostics; rank1/no-axis remains copy fallback. Real FFT batch tiling is not enabled by constructor (145-146).',
        justification='Unpacked fused kernels read const input and write separate output. Vendor C2C uses identical input/output stride/distance in its plan (690-695), so first OOP only changes base addresses; prefix offsets apply symmetrically (851-854). Packed first axis already permutes to owned packed storage (798-805), subsequent ping-pong never targets caller input; last FFT/final restore deliver scratch (810-811,869-871). No stride-remapping plan is required.',
        expected_change='Remove one compact read+write D2D operation, not complex FFT work or C2R vendor work. Workspace scratch remains needed because C2R may overwrite input. Speedup and copy contribution unquantified.',
        validation='Isolated new build: full-output error plus exact caller-input preservation; rank1 fallback; strided vendor with multiple prefixes; fused local/register/axis/pair; packed first/mixed axes and restore parity; fp32/fp64; graph and eager. Do not add a float acceptance threshold or assert bit-exact across vendor in-place/OOP algorithms.'),
      dict(id='inverse_fused_pre_scale',status='linear-equivalent suggestion; changed floating rounding, unmeasured',
        edit='Compute effective inverse scale before complex axes; if last complex plan is fused, pass it as final_scale and omit real post-scale; otherwise retain real post-scale.',
        applicability='Only real inverses with nonunit effective scale and a final fused complex axis. No benefit for the current eight norm=forward inverse plans: output_scale=1 and normalize_inverse=false.',
        justification='Scalar multiplication commutes mathematically with remaining linear C2R, and fused core final-write scale already exists (813-844). Vendor complex-last/no-axis cannot absorb this scale through the current interface.',
        validation='Report full-output errors and scale placement; not bit-exact. Earlier scaling changes rounding and possible underflow/overflow behavior; test supported normalization/custom-scale contracts and preserve fallback. Separate this experiment from copy elision.')])
filter_contrasts=[]
for c in filters:
    t={m['method']:m['timing']['median_cuda_event_ms']*1000 for m in c['methods']}
    filter_contrasts.append(dict(id=c['contract']['id'],median_us=t,
        matched_normalization_vs_original_extra_us=t['torch_cufft_same_padding_matched_normalization']-t['torch_cufft_same_padding_original_v2_diagnostic'],
        cubutterfly_vs_matched_torch_extra_us=t['cubutterfly']-t['torch_cufft_same_padding_matched_normalization'],
        caveat='within this single-process eager diagnostic only; differing normalization implementation cost, not isolated core cost'))
padding_example=load('results/paper_targeted_20261012/application_batch/v2/fft_filter/process1_guard/release/image512_filter31_batch4.process1.json')

fno=[]
for b in ['cubutterfly','torch']:
    path=f'results/paper_targeted_20261012/application_diagnostics/fno_v1/release/fno_{b}.trace.json'
    trace=load(path)['traceEvents'];kernels=[x for x in trace if x.get('cat')=='kernel' and x.get('ph')=='X']
    names=Counter();groups=Counter()
    for k in kernels:
        name=k['name'];dur=k['dur'];names[name]+=dur
        group=('fft' if any(t in name for t in ['xd_local_fft','regular_fft','vector_fft','fft_kernel']) else
               'convolution' if any(t in name.lower() for t in ['cudnn','scudnn']) else
               'gemm' if 'gemm' in name.lower() else
               'copy_or_layout' if any(t in name.lower() for t in ['copy','roll','catarray']) else 'other')
        groups[group]+=dur
    fno.append(dict(backend=b,kernel_count=len(kernels),kernel_duration_sum_us=sum(k['dur'] for k in kernels),
        category_duration_us=dict(groups),kernel_name_duration_us=dict(names),
        excluded_categories=sorted({x.get('cat','') for x in trace if x.get('cat')!='kernel'}),
        caveat='Profiled sum of kernel durations, not complete-call latency, wall time, critical-path sum or annotation device_total.',source=src(path,'traceEvents: cat=kernel and ph=X')))
fno_e2e=load('results/paper_targeted_20261012/application_diagnostics/fno_v1/release/fno_end_to_end.json')

controls=[];commands=[];ptx=[]
metric_names=['gpu__time_duration.sum','dram__bytes_read.sum','dram__bytes_write.sum',
 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
 'smsp__inst_executed.sum','smsp__sass_thread_inst_executed_op_integer_pred_on.sum',
 'smsp__warp_issue_stalled_barrier_per_warp_active.pct','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
 'sm__warps_active.avg.pct_of_peak_sustained_active','launch__registers_per_thread','launch__shared_mem_per_block']
for name,directory in [('fwht','medium_large_mechanism/fwht'),('fft','medium_large_mechanism/fft'),
 ('ntt','medium_large_mechanism_harnessfix/ntt'),('fwht-residency','medium_large_residency_v2')]:
    base=f'results/paper_targeted_20261012/{directory}'
    path=base+'/guard/release/timing.json';d=load(path);launch=load(base+'/launch.json')
    case=d['cases'][0];c=case['comparisons'][0]
    controls.append(dict(name=name,workload=case['workload'],baseline_us=c['baseline_median_kernel_ms']*1000,
        treatment_us=c['treatment_median_kernel_ms']*1000,
        baseline_over_treatment=c['baseline_median_kernel_ms']/c['treatment_median_kernel_ms'],
        actual_changed_axes=c['actual_changed_axes'],baseline=c['baseline'],treatment=c['treatment'],
        source=src(path,'/cases/0/comparisons/0'),scope='fixed shared-iterative radix2 mappings; not public automatic-library selection',protocol=d['protocol']))
    for v in case['variants']:
        cmd=list(v.get('timed_command',v['command']))
        for flag,value in [('--warmup','100'),('--repeat','1')]:cmd[cmd.index(flag)+1]=value
        label=name+'-'+v['id']
        ncu=['/usr/sbin/ncu','--target-processes','all','--replay-mode','kernel','--cache-control','none',
             '--clock-control','none','--kernel-name-base','demangled','--kernel-name','regex:shared_(iterative|resident)_kernel',
             '--launch-skip','200','--launch-count','2','--metrics',','.join(metric_names),'--page','raw','--csv',
             '--log-file',str(ROOT/'results/performance_bottleneck_profile_20261012'/f'{label}.ncu.csv')]+cmd
        commands.append(dict(label=label,argv=ncu,environment=launch['environment'],
            source=src(path,'/cases/0/variants: command/mapping'),gpu_uuid=launch['gpu_uuid'],
            binary_current_identity=src(cmd[0],'profile executable'),
            binary_expected_sha256=launch['binaries_sha256'].get(Path(cmd[0]).name),
            binary_hash_matches_recorded=(src(cmd[0])['sha256']==launch['binaries_sha256'].get(Path(cmd[0]).name)),
            expected_kernel_groups=2,matching_warmup_kernel_launches=200,
            status='prepared, never executed; validate captured group kernels against expected two groups before interpreting',
            eligibility='only after this UUID is idle/exclusive; preflight same frozen identity/correctness; do not touch GPU1 queue'))
    for dep in launch.get('ready_dependencies',[]):
        p=Path(dep['path']);src(p,'compiled module')
        resources=subprocess.run(['cuobjdump','--dump-resource-usage',str(p)],capture_output=True,text=True,check=True).stdout
        code=subprocess.run(['cuobjdump','--dump-ptx',str(p)],capture_output=True,text=True,check=True).stdout
        ops=Counter(re.findall(r'^\s*(?:@!?%\w+\s+)?([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)\s',code,re.M))
        ptx.append(dict(study=name,mapping=dep['mapping'],module=src(p,'embedded PTX/resources'),
            resource_dump=resources,static_ptx_opcode_counts=dict(ops),
            caveat='Counts are static textual instruction counts across kernels; not executed instructions, cycles, occupancy or stall attribution. Dynamic shared is omitted by cuobjdump SHARED:0.'))

claims=[
 dict(id='direction_selection',level='source-confirmed interface + measured plans',conclusion='Forward/inverse plans already created independently with separate semantics and direction keys. Same winning organization is not evidence of tied selection.',
      source=src('results/e13_assets/cubutterfly_real_fft.py','__init__ _forward/_inverse; _make_plan; semantic direction/output scale')),
 dict(id='short_fft_core',level='unquantified',conclusion='Slow short xD transforms cannot be causally assigned to weak arithmetic cores from complete-call timings alone. Launch, scratch/layout, native real kernels, Python dispatch and normalization remain possible contributors.'),
 dict(id='normalization',level='source-confirmed mismatch in diagnostic protocol',conclusion='Original v2 Torch same-padding diagnostic uses default backward normalization; cuButterfly uses norm=forward with product *= volume. Same convolution outputs but different placement/cost of scaling. New breakdown adds matched-normalization control.',source=src('paper/tools/run_xd_application_breakdown_20261012.py','filter loop: same versus original_same')),
 dict(id='padding_label',level='source-confirmed metadata risk',conclusion='breakdown cupy_padding uses scipy.fft.next_fast_len(real=True), although runtime competitor is cupyx.scipy.signal.fftconvolve; original filter runner correctly records cp_fft.next_fast_len. Do not assume metadata equals executed CuPy padding without vendor trace.',source=src('paper/tools/run_xd_application_breakdown_20261012.py','filters record cupy_padding')),
 dict(id='fno_kernel_sum',level='measured profiler diagnostic',conclusion='Use cat=kernel trace events only; annotation/device totals and instrumented intervals are not additive kernel totals. GPU kernel work is near parity in this trace and does not establish complete application speedup.'),
 dict(id='layout_loss',level='normal-timing measured; cause unquantified',conclusion='Writer-aligned/hermes-xor loses at this fixed N2^20/B4 for all three tested arithmetic families. Need same-point counters to separate bank conflicts, integer address instructions, barriers and occupancy effects. This says nothing about automatic-library winner selection.'),
 dict(id='resident_gain',level='normal-timing measured; source mechanism inference',conclusion='FWHT resident2 improves this fixed point about 3.5%; two-stage register leaves reduce stage-publication frequency in source. Actual bank/occupancy/address/stall contribution is not measured at this point.'),
]
for p in ['src/shared_iterative.cuh','src/shared_resident.cuh','include/cuntt/iterative_layout.hpp','src/butterfly.cu','src/ntt.cu',
 'paper/experiments/e13/run_fft_filter_end_to_end.py']:src(p,'relevant kernel/normalization/timing implementation; current source not asserted historical identity')
params=Path('/proc/driver/nvidia/params').read_text();admin=re.search(r'RmProfilingAdminOnly:\s*(\d+)',params)
data=dict(schema='performance-bottleneck-audit-20261012-v1',human_verification='not recorded',submission_ready=False,
 claims=claims,xd=xd,xd_code_audit=xd_code_audit,filter_breakdown=filters,filter_contrasts=filter_contrasts,
 padding_example=dict(scipy_reference=padding_example['contract']['scipy_reference_fast_padding'],
     actual_cupy_record=padding_example['cupy_native_fft_padding'],
     source=src('results/paper_targeted_20261012/application_batch/v2/fft_filter/process1_guard/release/image512_filter31_batch4.process1.json','/contract/scipy_reference_fast_padding; /cupy_native_fft_padding')),
 fno_kernel_audit=fno,fno_end_to_end=fno_e2e,controls=controls,
 ptx_static_audit=ptx,ncu_prepared=commands,ncu_capability=dict(version=subprocess.run(['ncu','--version'],capture_output=True,text=True).stdout,
 profiling_admin_only=int(admin.group(1)) if admin else None,gpu_run=False,permission_probe='read-only driver params; actual metric availability/permission not runtime-tested'),sources=sources)
(OUT/'PERFORMANCE_BOTTLENECK_AUDIT_20261012.json').write_text(json.dumps(data,indent=2)+'\n')
lines=['# Performance bottleneck audit (2026-10-12)','',
 'CPU-only evidence audit. No GPU profiling was launched. Raw identities, source fields, per-kernel trace categories, static PTX/resource dumps and eight unexecuted NCU commands are retained in the JSON ledger. Recompute with `CUDA_VISIBLE_DEVICES="" python3 paper/evidence/audit_performance_bottleneck_20261012.py`.','',
 '## Corrected conclusions','', '| Claim | Evidence level | Supported conclusion |','| --- | --- | --- |']
for c in claims:lines.append(f"| {c['id']} | {c['level']} | {c['conclusion']} |")
lines+=['','## Matched fixed-point timing','','| Study | Baseline us | Treatment us | Baseline / treatment | Changed axes |','| --- | ---: | ---: | ---: | --- |']
for c in controls:lines.append(f"| {c['name']} | {c['baseline_us']:.3f} | {c['treatment_us']:.3f} | {c['baseline_over_treatment']:.6f} | {c['actual_changed_axes']} |")
lines+=['','All four comparisons use N=2^20, B=4, [10,10] physical groups, 256 threads and three ordinary timing trials. Labels such as `median_treatment_over_baseline` in existing reports actually contain baseline/treatment; the audit recomputes the direction from absolute latencies. These are fixed mappings, not automatic-library performance. FWHT residency v2 uses linear shared layout in both cells; historical helper labels must not be mistaken for a layout change.','',
 '## FNO kernel-only trace','','| Backend | Kernel count | Kernel duration sum us | FFT kernel sum us |','| --- | ---: | ---: | ---: |']
for f in fno:lines.append(f"| {f['backend']} | {f['kernel_count']} | {f['kernel_duration_sum_us']:.3f} | {f['category_duration_us'].get('fft',0):.3f} |")
lines+=['','These are sums of profiler kernel durations, not critical-path or normal end-to-end times. `gpu_user_annotation`, `user_annotation`, copies and runtime events are excluded. Kernel category grouping is name-based and all exact names remain in JSON. Annotation `application/fno_*` totals and CUDA interval timings cannot be summed with child kernels.','',
 '## Normalization and padding controls','',
 '| Filter | Matched-normalization Torch extra over original diagnostic us | cuButterfly extra over matched Torch us |',
 '| --- | ---: | ---: |']
for c in filter_contrasts:lines.append(f"| {c['id']} | {c['matched_normalization_vs_original_extra_us']:.3f} | {c['cubutterfly_vs_matched_torch_extra_us']:.3f} |")
lines+=['','The additional matched-normalization control shows that some earlier same-padding gap belongs to normalization execution differences. These are paired single-process diagnostics, not an isolated arithmetic-core factorial. An existing v2 N512 filter31 B4 record has scipy reference padding [576,576] but actual CuPy metadata [560,560], demonstrating that the two next_fast_len implementations must not be interchanged. The breakdown metadata uses SciPy for the field labelled cupy_padding; actual execution padding remains to be traced for its four cases.','',
 '## Finite NCU Preparation','',
 'Nsight Compute 2024.1 is installed; the read-only driver parameter is `RmProfilingAdminOnly: 0`. This suggests non-sudo profiling is permitted but does not runtime-test counters. Eight commands profile two group kernels for each of the four already timed pairs, using the frozen launch environment and UUID. No command was executed.',
 '', 'Each prepared command retains 100 warmup calls, reduces repeat to one, filters shared_iterative/shared_resident kernel names, skips 200 matching warmup group launches and captures two group kernels. Source timing loops support the skip assumption; the eventual capture must confirm the two expected group names and mapping identity. Counters are kernel-only and memcpy is excluded. Profiler latency does not replace the ordinary timing row.','',
 'Keep bank conflicts, address/integer instructions, barriers/scoreboard, register/shared resources and occupancy per kernel. Sum byte/event counts across the two kernels where meaningful; do not average ratios or infer complete-call traffic. Compare same semantic point and same kernel scope only. Launches must wait for the specified UUID to be idle/exclusive and must not touch the GPU1 queue.','',
 'Static PTX and resource dumps are from hash-bound ready modules. Static address/bitwise counts may motivate hypotheses but cannot establish executed instruction costs or a bottleneck. cuobjdump reports static SHARED:0 while this template allocates dynamic shared buffers; do not call the kernel shared-memory-free.','']
lines+=['## xD real-executor code audit','',
 'This appendix audits current source, not an identity assertion about the already measured frozen binary. No runtime edit, build or GPU run was performed. The JSON binds source hashes and each selected plan to its original result field. There is no `src/xd_executor*` file in this tree: execution is implemented in `src/xd_real_fft.cu`.','',
 '| Shape | B | Forward axis order / cores | Inverse axis order / cores | Inverse post-scale? |','| --- | ---: | --- | --- | --- |']
for c in xd_code_rows:lines.append(f"| {c['shape']} | {c['batch']} | {c['forward_axis_order']} / {c['forward_cores']} | {c['inverse_axis_order']} / {c['inverse_cores']} | {c['inverse_postscale_needed']} |")
lines+=['',
 'Forward executes vendor last-axis R2C/D2Z then complex axes; inverse executes compact D2D copy, complex axes, then vendor C2R/Z2D (`src/xd_real_fft.cu:745-775`). Both preserve compact n/2+1 storage, not a full-spectrum expansion. All eight selected plans have actual per-axis `packed=false`, empty axis permutations and empty output permutation; `packed_axes=true` only enables a policy and does not imply a pack/restore pass. Forward scale is already fused when its final complex core is fused (754-758); vendor-last cases use a separate complex scale. All eight inverse plans have unit scale and no inverse normalization, so no extra real scale kernel is launched. The 128^3 B4 directions actually select different axis orders. A direction-selection gap or an inverse-normalization explanation for these eight points is unsupported.','',
 'Minimal generic optimization: keep the C2R scratch but replace its initial copy by the existing first-axis out-of-place execution when complex axes are nonempty. Unpacked fused cores have const input and separate output; strided vendor plans have equal input/output strides and distances (690-695), so changing base pointers requires no alternate stride plan. Prefix loops offset both pointers equally (851-854). Packed first axes already permute into owned storage (798-805); final FFT/restore chooses the required scratch destination (810-811,869-871). This applies to all eight current diagnostics and real rank>=2 plans; retain the copy for rank1/no-axis because vendor C2R may overwrite the input. Real batch tiling is disallowed (145-146), so there is no real tiled-tail exception. Source feasibility does not quantify copy cost or guarantee speedup.','',
 'Suggested replacement at 762-765 (not applied):','',
 '```cpp',
 'if (axis_plans_.empty()) {',
 '    check_cuda(cudaMemcpyAsync(scratch, input_complex, compact_bytes_,',
 '                               cudaMemcpyDeviceToDevice, stream),',
 '               "copy xD inverse compact input");',
 '} else {',
 '    execute_c2c_axes(const_cast<Complex*>(input_complex), scratch,',
 '                     CUFFT_INVERSE, true);',
 '}',
 '```','',
 'Validate an isolated new binary with full-output error and exact caller-input preservation across vendor strided/prefix loops, fused local/register/axis/pair, packed/mixed axes and restore parity, fp32/fp64, rank1 fallback, eager/graph. Vendor in-place versus out-of-place algorithms need not be bit-identical. No new floating threshold is defined here.','',
 'A separate optional optimization can move nonunit inverse scale to the last fused complex-axis write and skip the final real scale. This is mathematically linear-equivalent but changes floating rounding and possible underflow/overflow; report errors and placement, not bit-exactness. Preserve post-scale for vendor-last/no-axis paths. It benefits none of these eight inverse plans and must not be bundled into the copy-elision attribution.','']
(OUT/'PERFORMANCE_BOTTLENECK_AUDIT_20261012.md').write_text('\n'.join(lines))
print(json.dumps(dict(controls=controls,fno=[{k:v for k,v in f.items() if k in ['backend','kernel_count','kernel_duration_sum_us','category_duration_us']} for f in fno],ncu_commands=len(commands),modules=len(ptx),sources=len(sources)),indent=2))
