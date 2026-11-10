"""Distinguish explicitly unbundled raw evidence from broken source links."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_repository.py'
SPEC = importlib.util.spec_from_file_location('repository_checker', SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, 'ROOT', tmp_path)
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'config').mkdir()
    for name in ('README.md', 'CONTRIBUTING.md', 'ROADMAP.md'):
        (tmp_path / name).write_text('Source release\n')
    (tmp_path / 'config/repository_local_artifacts.json').write_text(json.dumps(dict(
        schema='cubutterfly-repository-local-artifacts-v1',
        artifacts=['results/historical/report.json', 'results/archived-directory'])))
    return tmp_path


def test_registered_missing_raw_paths_are_reported_separately(repository):
    (repository / 'docs/history.md').write_text(
        '[raw](../results/historical/report.json#timing)\n'
        '[same](../results/historical/report.json)\n')
    documents, unbundled = checker.check_markdown_links()
    assert documents == 4
    assert unbundled == ['results/historical/report.json']
    raw = repository / 'results/historical/report.json'
    raw.parent.mkdir(parents=True)
    raw.write_text('{}')
    assert checker.check_markdown_links()[1] == []


def test_nested_document_dead_link_is_checked(repository):
    document = repository / 'docs/development/nested/guide.md'
    document.parent.mkdir(parents=True)
    document.write_text('[missing](../../../scripts/missing.py)\n')
    with pytest.raises(RuntimeError, match=r'docs/development/nested/guide.md: missing'):
        checker.check_markdown_links()


@pytest.mark.parametrize('kind', ['sibling', 'absolute', 'symlink'])
def test_existing_outside_targets_are_rejected_before_exists(repository, monkeypatch, kind):
    outside = repository.parent / (repository.name + '-outside.md')
    outside.write_text('Outside repository\n')
    if kind == 'sibling':
        target = '../../' + outside.name
    elif kind == 'absolute':
        target = str(outside)
    else:
        (repository / 'docs/escape.md').symlink_to(outside)
        target = 'escape.md'
    (repository / 'docs/guide.md').write_text(f'[outside]({target})\n')
    original_exists = Path.exists
    original_read_text = Path.read_text

    def guarded_exists(path):
        assert path != outside, 'must not probe an outside target for existence'
        return original_exists(path)

    def guarded_read_text(path, *args, **kwargs):
        assert path.resolve() != outside, 'must not read an outside document'
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'exists', guarded_exists)
    monkeypatch.setattr(Path, 'read_text', guarded_read_text)
    with pytest.raises(RuntimeError, match=r'docs/guide.md: outside repository:'):
        checker.check_markdown_links()


def test_normalized_internal_link_is_allowed(repository):
    (repository / 'docs/guide.md').write_text('[source](../docs/../README.md)\n')
    assert checker.check_markdown_links()[1] == []


def test_nested_document_preserves_external_and_exact_artifact_rules(repository):
    document = repository / 'docs/hardware/guide.md'
    document.parent.mkdir(parents=True)
    document.write_text(
        '[web](https://example.com/missing)\n'
        '[email](mailto:maintainer@example.com)\n'
        '[section](#local-section)\n'
        '[raw](../../results/historical/report.json#timing)\n')
    documents, unbundled = checker.check_markdown_links()
    assert documents == 4
    assert unbundled == ['results/historical/report.json']
    document.write_text('[child](../../results/archived-directory/unlisted-child.json)\n')
    with pytest.raises(RuntimeError, match='unlisted-child.json'):
        checker.check_markdown_links()


@pytest.mark.parametrize('target', ['../results/historical/typo.json',
    '../results/archived-directory/unlisted-child.json', '../scripts/missing.py',
    '../paper/generated/missing.json'])
def test_unregistered_raw_source_and_paper_links_remain_errors(repository, target):
    (repository / 'docs/broken.md').write_text(f'[missing]({target})\n')
    with pytest.raises(RuntimeError, match='missing'):
        checker.check_markdown_links()


def test_strict_mode_requires_the_whole_exact_inventory(repository):
    (repository / 'docs/history.md').write_text('[raw](../results/historical/report.json)\n')
    with pytest.raises(RuntimeError, match='missing'):
        checker.check_markdown_links(strict_local_artifacts=True)
    raw = repository / 'results/historical/report.json'
    raw.parent.mkdir(parents=True)
    raw.write_text('{}')
    # The registered directory is not linked, but strict inventory checking still requires it.
    with pytest.raises(RuntimeError, match='registered raw artifacts absent'):
        checker.check_markdown_links(strict_local_artifacts=True)
    (repository / 'results/archived-directory').mkdir()
    assert checker.check_markdown_links(strict_local_artifacts=True)[1] == []


@pytest.mark.parametrize('entry', ['results/*', 'results/history/../typo.json',
    'paper/generated/snapshot.json', '/results/raw.json', 'results/history/'])
def test_inventory_cannot_exempt_prefixes_or_nonresults_paths(repository, entry):
    manifest = repository / 'config/repository_local_artifacts.json'
    data = json.loads(manifest.read_text())
    data['artifacts'] = [entry]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='exact results/'):
        checker.check_markdown_links()
