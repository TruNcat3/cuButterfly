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
