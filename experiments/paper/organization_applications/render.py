#!/usr/bin/env python3
"""Offline reproduction of paper figures 8, 10, and 12 from frozen inputs."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import render_cross_operator_attribution as attribution
import render_local_organization_profiles as profiles
import render_applications as applications
import export_boundary_positive as boundary_positive

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    attribution.OUTPUT = profiles.OUTPUT = out
    boundary_positive.export(out)
    attribution.main()
    profiles.main()
    data = json.loads((ROOT / 'inputs/applications.json').read_text())
    ckks = applications.strongest_ckks_pairs(data)
    assert math.isclose(applications.gm([r['speedup'] for r in ckks]), 1.1310519758, rel_tol=1e-9)
    manifest = {'schema': 'offline-organization-applications-v1', 'figures': {}}
    applications.application_plot(data, out, manifest)
    # All input rows are exported, including separately scoped cohorts and exclusions.
    shutil.copyfile(ROOT / 'inputs/applications.json', out / 'application_source_rows.json')
    shutil.copyfile(ROOT / 'inputs/application_historical_complete_call.json', out / 'application_historical_rows.json')
    profile_data = json.loads((ROOT / 'inputs/narrative_scaling_20261027.json').read_text())
    rows = [dict(operator=r['operator'], variant=r['variant'], group=r['group'], **r['metrics']) for r in profile_data['profile_records']]
    with (out / 'local_organization_profile_rows.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    manifest['input_bundle_sha256'] = {p.name: sha(p) for p in sorted((ROOT / 'inputs').iterdir())}
    manifest['source_provenance'] = json.loads((ROOT / 'source_provenance.json').read_text())
    manifest['validation'] = {'attribution_rows': 150, 'profile_rows': len(rows), 'ckks_requests': len(ckks), 'ckks_gm': applications.gm([r['speedup'] for r in ckks])}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['validation']))

if __name__ == '__main__':
    main()
