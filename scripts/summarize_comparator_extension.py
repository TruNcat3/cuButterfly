#!/usr/bin/env python3
"""Read-only audit and tables for a prepared fresh-pair comparator campaign."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

STRUCTURAL = {'pass', 'accuracy-review-required'}
PRECISION = {0:'fp16',1:'bf16',2:'fp32',3:'fp64',4:'complex-fp16',5:'complex-bf16',6:'fp32',7:'fp64'}
OPERATORS = {0:'fft',1:'ntt',2:'fwht',3:'subset-zeta',4:'superset-zeta',5:'structured'}
ARITHMETIC_POLICIES = {0:'STRICT',1:'ALLOW_FP16_MMA'}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_number(value, positive=False):
    return type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0)


def warnings(implementation, label):
    result=[]
    status=implementation.get('status','missing')
    if 'failure' in status or status in {'runner-error','runtime-error','baseline-error','resource-limit','execution-error','cuda-error','error'}:
        result.append(label+': execution/verification failure')
    l2=implementation.get('correctness',{}).get('l2_relative')
    if l2 is not None and not finite_number(l2):
        result.append(label+': nonfinite/invalid L2 diagnostic')
    elif finite_number(l2) and l2>0.1:
        result.append(label+': large relative L2 (>0.1), requires investigation')
    if implementation.get('numeric_diagnostics',{}).get('actual_nonfinite_elements',0):
        result.append(label+': nonfinite output')
    return result


def text_value(value):
    return value if isinstance(value,str) else json.dumps(value,sort_keys=True) if value is not None else ''


def selection_warning(row):
    text=(text_value(row.get('selection_scope'))+' '+text_value(row.get('selection_reason'))).lower()
    if 'hardware-rate-prior' in text or ('fingerprint' in text and ('missing' in text or 'incompatible' in text)):
        return 'uncalibrated model selection diagnostic; excluded from formal calibrated comparison'
    return ''


def scope_warning(scope):
    text=json.dumps(scope,sort_keys=True).lower()
    status=str(scope.get('status','')).lower()
    excluded=any(word in status for word in ('exploratory','stopped','excluded','invalid','uncalibrated'))
    excluded=excluded or any(word in text for word in ('exclude from','excluded from','排除','不纳入'))
    excluded=excluded or any(value is True for key,value in scope.items() if key.startswith('exclude'))
    return 'campaign SCOPE excludes calibrated comparison: '+text_value(scope) if excluded else ''


def mapping_summary(mapping):
    keys=('backend','fft_core','stage_partition','factor_partition','tile_threads',
          'factor_ept','prefix_ept','suffix_ept','compute_unit','local_exchange',
          'local_core','arithmetic_policy','prefetch_depth','groups_per_cta',
          'groups','data_tiles_per_cta')
    return json.dumps({k:mapping[k] for k in keys if k in mapping},sort_keys=True,separators=(',',':')) if mapping else ''


def summarize(campaign):
    campaign=Path(campaign)
    preparation_path=campaign/'preparation.json'
    manifest=json.loads(preparation_path.read_text())
    config=manifest['config']
    scope_path=campaign/'SCOPE.json'
    scope=json.loads(scope_path.read_text()) if scope_path.is_file() else {}
    scope_notice=scope_warning(scope)
    if digest(config)!=manifest['config_sha256']:
        raise ValueError('preparation config checksum mismatch')
    cases=config['cases']
    if len({c['id'] for c in cases})!=len(cases):
        raise ValueError('duplicate scheduled case')
    records={r['id']:r for r in manifest['cases']}
    if set(records)!={c['id'] for c in cases} or len(records)!=len(manifest['cases']):
        raise ValueError('preparation population mismatch')
    known={c['id'] for c in cases}
    if any(p.stem not in known for p in (campaign/'rows').glob('*.json')):
        raise ValueError('foreign row in campaign')
    identity=None
    rows=[]
    for case in cases:
        identifier=case['id']
        if Path(identifier).name!=identifier or identifier in ('.','..'):
            raise ValueError('unsafe case ID')
        semantics=case['descriptor']['semantics']
        record=records[identifier]
        if record['semantics_sha256']!=digest(semantics) or record['descriptor_sha256']!=digest(case['descriptor']):
            raise ValueError('preparation semantic binding mismatch')
        row={}
        token=''
        path=campaign/'rows'/(identifier+'.json')
        if path.is_file():
            envelope=json.loads(path.read_text())
            row=envelope['row']
            current=envelope['identity']
            token=digest(current)
            if envelope.get('row_sha256')!=digest(row) or envelope.get('identity_sha256')!=token:
                raise ValueError('row/identity checksum mismatch: '+identifier)
            if envelope.get('guards')!={'before':True,'after':True}:
                raise ValueError('incomplete exclusivity guards: '+identifier)
            if current.get('config_sha256')!=manifest['config_sha256'] or current.get('preparation_sha256')!=sha(preparation_path):
                raise ValueError('row preparation identity mismatch: '+identifier)
            if identity is not None and current!=identity:
                raise ValueError('mixed measurement identities')
            identity=current
            if row.get('id')!=identifier or row.get('descriptor')!=case['descriptor']:
                raise ValueError('row descriptor binding mismatch: '+identifier)
            if envelope.get('prepared_sha256')!=record.get('sha256'):
                raise ValueError('row prepared-array binding mismatch: '+identifier)
        baselines=row.get('baselines',[])
        names=[b['name'] for b in baselines]
        if len(set(names))!=len(names) or set(names)-set(case['baselines']):
            raise ValueError('unexpected/duplicate measured baseline')
        by_name={b['name']:b for b in baselines}
        for backend in case['baselines']:
            baseline=by_name.get(backend,{})
            own_status=row.get('status', record.get('gap_status','pending') if record['status']!='prepared' else 'pending')
            baseline_status=baseline.get('status','missing')
            own_ms=row.get('timing',{}).get('median_ms')
            base_ms=baseline.get('timing',{}).get('median_ms')
            l2=row.get('correctness',{}).get('l2_relative')
            base_l2=baseline.get('correctness',{}).get('l2_relative')
            notices=warnings(row,'own')+warnings(baseline,backend)
            selection_notice=selection_warning(row)
            notices.extend(notice for notice in (selection_notice,scope_notice) if notice)
            paired=own_status in STRUCTURAL and baseline_status in STRUCTURAL and finite_number(own_ms,True) and finite_number(base_ms,True)
            if paired and (not finite_number(l2) or not finite_number(base_l2)):
                notices.append('missing/invalid L2 diagnostics; accuracy review incomplete')
            rows.append(dict(case_id=identifier,identity_sha256=token,operator=OPERATORS.get(semantics['operator'],str(semantics['operator'])),
                shape='x'.join(map(str,semantics['shape'])),batch=semantics['batch'],precision=PRECISION.get(semantics['storage'],str(semantics['storage'])),
                arithmetic_policy=ARITHMETIC_POLICIES.get(semantics.get('arithmetic_policy',0),str(semantics.get('arithmetic_policy',0))),
                direction='inverse' if semantics['inverse'] else 'forward',placement='in-place' if semantics['placement'] else 'out-of-place',
                normalize_inverse=semantics['normalize_inverse'],backend=backend,own_status=own_status,baseline_status=baseline_status,
                own_reason=row.get('error',record.get('error','')),baseline_reason=baseline.get('reason',baseline.get('error','')),
                own_ms=own_ms,baseline_ms=base_ms,baseline_over_own=base_ms/own_ms if paired else None,
                own_l2_relative=l2,baseline_l2_relative=base_l2,structural_pair=paired,
                selection_scope=text_value(row.get('selection_scope')),selection_reason=text_value(row.get('selection_reason')),
                campaign_scope=text_value(scope) if scope else '',warning='; '.join(notices),own_mapping=mapping_summary(row.get('mapping'))))
    groups=defaultdict(list)
    for row in rows:
        groups[(row['operator'],row['precision'],row['arithmetic_policy'],row['backend'])].append(row)
    aggregates=[]
    for (operator,precision,arithmetic_policy,backend),members in sorted(groups.items()):
        pairs=[r for r in members if r['structural_pair']]
        flagged=[r for r in members if r['warning']]
        geometric=math.exp(sum(math.log(r['baseline_over_own']) for r in pairs)/len(pairs)) if pairs and not flagged else None
        aggregates.append(dict(operator=operator,precision=precision,arithmetic_policy=arithmetic_policy,backend=backend,
            identity_sha256=digest(identity) if identity else '',scheduled_count=len(members),paired_count=len(pairs),
            gap_or_pending_count=len(members)-len(pairs),warning_count=len(flagged),geometric_mean_baseline_over_own=geometric,
            selection_scopes='; '.join(sorted({r['selection_scope'] for r in members if r['selection_scope']})),
            selection_reasons='; '.join(sorted({r['selection_reason'] for r in members if r['selection_reason']})),
            campaign_scope=text_value(scope) if scope else '',
            interpretation=('withheld: uncalibrated model selection diagnostic / campaign exclusion; NOT formal calibrated comparison'
                if scope_notice or any('uncalibrated model selection diagnostic' in r['warning'] for r in members)
                else 'withheld: investigate errors/failures; no anomalous rows removed') if flagged else
                'structural timing pairs / error-report only; NOT accuracy acceptance' if pairs else 'no structural timing pairs'))
    return rows,aggregates,identity


def table(rows,columns):
    def cell(value):
        if value is None:
            return ''
        if isinstance(value,float):
            return f'{value:.6g}'
        return str(value).replace('|','\\|').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |',
        *['| '+' | '.join(cell(row.get(k)) for k in columns)+' |' for row in rows]])


def write_outputs(campaign,output):
    rows,groups,identity=summarize(campaign)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('comparisons.csv',rows),('groups.csv',groups)]:
        path=output/name
        temporary=path.with_suffix('.csv.tmp')
        with temporary.open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]) if data else ['case_id'])
            writer.writeheader();writer.writerows(data)
        temporary.replace(path)
    text=('CPU-only snapshot of '+str(Path(campaign).resolve())+'\n\n'
          'Floating results are error reports without an accuracy acceptance threshold. '
          'Ratios above one favor cuButterfly. Missing, unsupported and failed backends remain in the tables. '
          'Relative L2 >0.1 is an investigation warning, never an acceptance rule. '
          'A warning/failure withholds the entire group geometric mean; individual observed ratios remain visible. '
          'Partial coverage is explicit; no historical timing substitutes for a fresh pair.\n\n'
          'Arithmetic policy records the requested contract: ALLOW_FP16_MMA permits native or MMA execution; '
          'the selected local core is recorded in own_mapping.\n\n'
          'Measurement identity: '+(digest(identity) if identity else 'no measured rows')+'\n\n'+
          table(groups,['operator','precision','arithmetic_policy','backend','scheduled_count','paired_count','gap_or_pending_count','warning_count','geometric_mean_baseline_over_own','selection_scopes','selection_reasons','campaign_scope','interpretation'])+'\n\n'+
          table(rows,['case_id','shape','batch','precision','arithmetic_policy','direction','placement','normalize_inverse','backend','own_status','baseline_status','own_ms','baseline_ms','baseline_over_own','own_l2_relative','baseline_l2_relative','warning','own_reason','baseline_reason','selection_scope','selection_reason','campaign_scope','own_mapping'])+'\n')
    temporary=output/'summary.md.tmp'
    temporary.write_text(text)
    temporary.replace(output/'summary.md')
    return rows,groups


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('campaign',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    write_outputs(args.campaign,args.output)


if __name__=='__main__':
    main()
