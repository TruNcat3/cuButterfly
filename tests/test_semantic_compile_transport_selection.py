"""Explicit transport selection cannot silently inherit a different query path."""
import os
from pathlib import Path
import socket
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_async_semantic_campaign as campaign
from semantic_gpu_pipeline import check_preparation_events


def test_legacy_transport_clears_stale_service(monkeypatch):
    monkeypatch.setenv('CUBUTTERFLY_COMPILE_SOCKET', '/stale/socket')
    monkeypatch.setenv('CUBUTTERFLY_JIT_PYTHON', '/stale/client')
    assert campaign.configure_compile_transport() == {'mode': 'per-query-python'}
    assert 'CUBUTTERFLY_COMPILE_SOCKET' not in os.environ
    assert os.environ['CUBUTTERFLY_JIT_PYTHON'].endswith('/semantic_resource_compile_proxy.py')


@pytest.mark.parametrize('client,endpoint', [('client', None), (None, 'socket')])
def test_transport_requires_both_arguments(client, endpoint):
    with pytest.raises(ValueError, match='provided together'):
        campaign.configure_compile_transport(client, endpoint)


def test_service_transport_records_actual_executable_identity(tmp_path, monkeypatch):
    monkeypatch.setenv('CUBUTTERFLY_COMPILE_SOCKET', '')
    monkeypatch.setenv('CUBUTTERFLY_JIT_PYTHON', '')
    client, endpoint = tmp_path / 'client', tmp_path / 'socket'
    client.write_bytes(b'fixture executable')
    client.chmod(0o700)
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(str(endpoint))
        identity = campaign.configure_compile_transport(client, endpoint)
    assert identity == {'mode': 'persistent-service', 'client': str(client),
                        'client_sha256': campaign.old.sha256_file(client), 'socket': str(endpoint)}
    assert os.environ['CUBUTTERFLY_JIT_PYTHON'] == str(client)


def test_non_socket_or_non_executable_cannot_select_service(tmp_path):
    client, endpoint = tmp_path / 'client', tmp_path / 'socket'
    client.write_bytes(b'fixture')
    endpoint.write_bytes(b'not a socket')
    with pytest.raises(ValueError, match='executable'):
        campaign.configure_compile_transport(client, endpoint)
    client.chmod(0o700)
    with pytest.raises(ValueError, match='Unix socket'):
        campaign.configure_compile_transport(client, endpoint)


@pytest.mark.parametrize('message', ['semantic compile client: event file unwritable',
                                   'semantic compile service: malformed frame'])
def test_unrecorded_transport_failure_invalidates_whole_search(message):
    with pytest.raises(RuntimeError, match='unrecorded specialization'):
        check_preparation_events(None, message)
