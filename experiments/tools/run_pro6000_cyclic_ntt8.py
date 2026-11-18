#!/usr/bin/env python3
"""Finite PR10 cyclic NTT8: freeze CPU inputs, then pair on one exclusive GPU."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PREFIXES = ('8b22dfc010e8', '486ebac42647', '3cf5ad20f608', 'db05a58bca42',
            'd4b0e39341b4', '79a54263f75c', '04a82eaac337', 'ad2b5c2b51d6')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measured-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-uuid', default='GPU-a257f4db-8c62-f26b-9c08-a4045615a140')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    root, out = args.measured_root.resolve(), args.output.resolve()
    sys.path[:0] = [str(root / 'scripts'), str(root / 'python')]
    import numpy as np
    import run_semantic_campaign as campaign
    import run_comparator_extension as extension
    import semantic_campaign_manifest as manifest_api
    import semantic_gpuntt_baseline as gpuntt
    import semantic_ntt_reference as ntt

    library_path = root / 'build-pro6000-sm120/libcuntt.so'
    bridge = root / 'build-pro6000-sm120/libgpuntt_semantic_bridge.so'
    out.mkdir(parents=True, exist_ok=True)
    prepared_path = out / 'preparation.json'
    if not prepared_path.exists():
        workloads = json.loads((root / 'config/research_campaign/crypto.json').read_text())['workloads']
        chosen = [w for w in workloads if w['logN'] == 16 and w['batch'] in (1, 16)
                  and int(w['modulus']) in (1073479681, 582090251837636609)]
        assert len(chosen) == 8
        populations = []
        for w in chosen:
            bits = int(w['precision'][4:])
            populations.append(dict(name=f"pro6000-cyclic-ntt8-{bits}-{w['batch']}-{w['direction']}",
                shapes=[[65536]], batches=[w['batch']], directions=[w['direction']],
                normalizations=['inverse' if w['direction'] == 'inverse' else 'none'],
                placements=['out-of-place'], order_patterns=['natural'], layouts=['packed'],
                regimes=[dict(name='exact-cyclic', transform=dict(op=1, storage=8 if bits == 32 else 9,
                    compute=3 if bits == 32 else 4, modulus=int(w['modulus']), word_bits=bits))]))
        config = dict(schema=manifest_api.CONFIG_SCHEMA, populations=populations)
        extension.write(out / 'config.json', config)
        manifest = manifest_api.generate_manifest(config, library_path, config_path=out / 'config.json')
        assert not manifest['invalid_requests'] and len(manifest['cases']) == 8
        assert {c['id'][:12] for c in manifest['cases']} == set(PREFIXES)
        assert all(c['capabilities']['status'] == 'eligible' for c in manifest['cases'])
        extension.write(out / 'manifest.json', manifest)
        config = dict(library=str(library_path), gpu_uuid=args.gpu_uuid,
            gpuntt_library=str(bridge), environment={'CUBUTTERFLY_COMPILE_MODE': 'research'},
            cases=[dict(c, baselines=['gpu-ntt']) for c in manifest['cases']],
            prepare_reference_limit=1048576)
        preparation = extension.prepare(config, out)
        assert all(r['status'] == 'prepared' for r in preparation['cases'])
    preparation, payloads = extension.load_prepared(out)
    manifest=json.loads((out/"manifest.json").read_text())
    hashes = {}
    _, payloads = extension.load_prepared(out)
    for case in manifest['cases']:
        s, payload = case['descriptor']['semantics'], payloads[case['id']]
        support = gpuntt.support(s, bridge)
        assert support['supported'], support
        root_value = ntt.primitive_power_of_two_root(s['modulus'], 65536)
        hashes[case['id']] = dict(semantics=s,
            input_sha256=hashlib.sha256(payload['source'].tobytes()).hexdigest(),
            reference_sha256=hashlib.sha256(payload['expected'].tobytes()).hexdigest(),
            root=root_value, inverse_root=pow(root_value, -1, s['modulus']),
            inverse_scale=pow(65536, -1, s['modulus']), modulus_bits=s['modulus'].bit_length(),
            support=support)
    extension.write(out / 'input-hashes.json', hashes)
    print('Prepared exact cyclic NTT8 population and full CPU references', flush=True)
    if not args.run:
        return
    preparation, payloads = extension.load_prepared(out)
    assert preparation['config']['gpu_uuid'] == args.gpu_uuid
    calibration_path = root / 'results/pro6000-issue9/main/full/run/calibration/finalized/calibration_manifest.json'
    calibration = json.loads(calibration_path.read_text())
    bindings = campaign.bind_calibration_environment(calibration)
    os.environ['CUBUTTERFLY_HARDWARE_MODEL'] = str(root / 'results/pro6000-issue9/main/full/run/calibration/hardware/hardware_model.json')
    os.environ['CUBUTTERFLY_SEARCH_BUDGET'] = '16'
    os.environ['CUBUTTERFLY_PLAN_CACHE'] = str(out / 'plans.cache')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_uuid
    campaign.exclusive_gpu(args.gpu_uuid)
    import torch
    import cubutterfly as cb
    properties = torch.cuda.get_device_properties(0)
    assert campaign.same_gpu_uuid(args.gpu_uuid, str(properties.uuid))
    identity = dict(schema='pro6000-cyclic-ntt8-v1', gpu_uuid=args.gpu_uuid,
        gpu_name=properties.name, gpu_memory=properties.total_memory,
        compute_capability=[properties.major, properties.minor],
        library_sha256=campaign.sha256_file(library_path),
        bridge_sha256=campaign.sha256_file(bridge), source_root=str(root),
        sources=extension.source_fingerprints(preparation['config']),
        runner_sha256=campaign.sha256_file(__file__), preparation_sha256=campaign.sha256_file(prepared_path),
        calibration_manifest_sha256=campaign.sha256_file(calibration_path),calibration=bindings,
        environment={k:v for k,v in os.environ.items() if k.startswith('CUBUTTERFLY_')},
        python=sys.version,numpy=np.__version__,torch=torch.__version__,torch_cuda=torch.version.cuda,
        protocol=campaign.PROTOCOL,scope='Eight predeclared legal cyclic requests; no Full66 replacement')
    token = campaign.digest(identity)
    library, stream = cb.Library(library_path), torch.cuda.Stream()
    for case in preparation['config']['cases']:
        path = out / 'rows' / (case['id'] + '.json')
        if path.exists():
            saved=json.loads(path.read_text())
            assert saved['identity_sha256'] == token and saved['row_sha256'] == campaign.digest(saved['row'])
            continue
        campaign.exclusive_gpu(args.gpu_uuid)
        start = time.time()
        extension.write(out / 'status.json',dict(status='running',case_id=case['id'],started=start))
        with torch.cuda.stream(stream):
            row = campaign.run_case(case, library, torch, stream, timing=True,
                prepared=payloads[case['id']], selection_policy=1, baseline_names=['gpu-ntt'],
                gpuntt_library=bridge, reference_max_elements=1048576,
                ntt_max_operations=16777216, error_dir=out/'errors', float_verdict='error-report')
            stream.synchronize()
        campaign.exclusive_gpu(args.gpu_uuid)
        extension.write(path,dict(identity=identity,identity_sha256=token,row=row,
            row_sha256=campaign.digest(row),guards=dict(before=True,after=True),seconds=time.time()-start))
        print(case['id'],row['status'],row.get('error',''),flush=True)
    records=[json.loads(p.read_text()) for p in sorted((out/'rows').glob('*.json'))]
    assert len(records)==8
    extension.write(out/'status.json',dict(status='complete-with-retained-failures' if any(r['row']['status']!='pass' for r in records) else 'complete',
        requests=8,valid_pairs=sum(r['row']['status']=='pass' for r in records),identity_sha256=token))


if __name__ == '__main__':
    main()
