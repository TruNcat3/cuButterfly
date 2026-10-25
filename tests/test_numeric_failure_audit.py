"""Classification supplements preserve verdicts and reject corrupted evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_semantic_numeric_failures as audit
from test_async_scaled_controls import fixture


def test_finite_input_nonfinite_reference_is_not_labelled_kernel_failure(tmp_path):
    args, row = fixture(tmp_path / 'original')
    artifact = Path(row['error_artifact']['path'])
    np.savez(artifact, actual=np.array([np.nan]), reference=np.array([np.nan]))
    row['error_artifact']['sha256'] = audit.r.sha256_file(artifact)
    checksum = audit.r.digest(row)
    args.rows[0].write_text(json.dumps({'row': row, 'sha256': checksum}))
    report_path = args.rows[0].parent.parent / 'results.json'
    report = json.loads(report_path.read_text())
    report['rows'][0]['sha256'] = checksum
    report['rows_sha256'] = audit.r.digest(report['rows'])
    report_path.write_text(json.dumps(report))
    before = {p: p.read_bytes() for p in (artifact, args.rows[0], report_path)}
    result = audit.audit(args.rows, args.library)
    assert result['counts'] == {'reference-nonfinite': 1}
    assert result['rows'][0]['original_status'] == 'correctness-failure'
    assert all(p.read_bytes() == data for p, data in before.items())


def test_duplicate_or_corrupted_sources_rejected(tmp_path):
    args, row = fixture(tmp_path / 'original')
    with pytest.raises(ValueError, match='duplicate'):
        audit.audit(args.rows * 2, args.library)
    Path(row['error_artifact']['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='artifact checksum'):
        audit.audit(args.rows, args.library)
