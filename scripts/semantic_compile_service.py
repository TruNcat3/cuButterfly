#!/usr/bin/env python3
"""CPU-only persistent broker transport.

Run with --broker DIR --socket PATH. Build semantic_compile_client.cpp with a
host C++17 compiler and set CUBUTTERFLY_JIT_PYTHON to that executable,
CUBUTTERFLY_COMPILE_SOCKET to PATH, and the usual broker/per-attempt events.
The service serializes requests; it never compiles or loads a GPU module.
"""
from contextlib import redirect_stderr, redirect_stdout
import argparse
import io
import os
from pathlib import Path
import signal
import socket
import struct

import semantic_compile_broker as broker
from semantic_worker_resources import exceeded, limits, snapshot

MAX_FRAME = 1024 * 1024
ENV_NAMES = ("CUBUTTERFLY_PREPARE_BROKER", "CUBUTTERFLY_PREPARE_EVENTS",
             "CUBUTTERFLY_WORKER_MAX_VMAS", "CUBUTTERFLY_WORKER_MAX_RSS_BYTES", "PATH")


def _receive(connection, size):
    result = bytearray()
    while len(result) < size:
        chunk = connection.recv(size - len(result))
        if not chunk:
            raise ValueError("truncated compile service frame")
        result.extend(chunk)
    return bytes(result)


def _frame(connection):
    size, = struct.unpack("!I", _receive(connection, 4))
    if size > MAX_FRAME:
        raise ValueError("compile service frame exceeds limit")
    return _receive(connection, size)


def _strings(values):
    return b"".join(struct.pack("!I", len(value.encode())) + value.encode() for value in values)


def _decode(payload):
    values = []
    offset = 0
    while offset < len(payload):
        if offset + 4 > len(payload):
            raise ValueError("truncated compile service string length")
        size, = struct.unpack_from("!I", payload, offset)
        offset += 4
        if offset + size > len(payload):
            raise ValueError("truncated compile service string")
        value = payload[offset:offset + size].decode("utf-8")
        if "\0" in value:
            raise ValueError("NUL in compile service string")
        values.append(value)
        offset += size
    if len(values) < 8 or values[0] != "cubutterfly-compile-service-v1":
        raise ValueError("unsupported compile service request")
    return values


def _parent(pid):
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    raise ValueError("cannot verify compile client parent")


def dispatch(directory, values, peer_pid, peer_uid):
    """Run the frozen resource guard and runtime CLI for an authenticated peer."""
    events = None
    previous_env = {name: os.environ.get(name) for name in ENV_NAMES}
    previous_cwd = os.getcwd()
    out, err = io.StringIO(), io.StringIO()
    try:
        if peer_uid != os.getuid() or int(values[1]) != _parent(peer_pid):
            raise ValueError("compile client credentials or parent mismatch")
        os.chdir(values[2])
        for name, value in zip(ENV_NAMES, values[3:8]):
            if value:
                os.environ[name] = value
            else:
                os.environ.pop(name, None)
        events = os.environ.get("CUBUTTERFLY_PREPARE_EVENTS")
        if not events:
            raise ValueError("CUBUTTERFLY_PREPARE_EVENTS is required")
        if Path(values[3]).resolve() != directory:
            raise ValueError("runtime request differs from service broker")
        args = values[8:]
        forbidden = {"--init", "--worker", "--watch", "--status", "--retry-failed",
                     "--seed-cache-since", "--jobs"}
        if any(arg.split("=", 1)[0] in forbidden for arg in args):
            raise ValueError("compile service accepts runtime requests only")
        for index, arg in enumerate(args):
            if arg == "--broker" or arg.startswith("--broker="):
                value = args[index + 1] if arg == "--broker" else arg.split("=", 1)[1]
                if Path(value).resolve() != directory:
                    raise ValueError("runtime request differs from service broker")
        budget = limits(max_vmas=int(values[5] or "45000"),
                        max_rss_bytes=int(values[6] or str(8 * 1024**3)))
        sample = snapshot(int(values[1]))
        reason = exceeded(sample, budget)
        if reason:
            # Match the frozen proxy's event fields exactly.
            import json
            with open(events, "a") as target:
                target.write(json.dumps({"kind": "worker_resource_limit", "reason": reason,
                                         "resources": sample}) + "\n")
            return 75, "", "worker resource limit: " + reason + "\n"
        with redirect_stdout(out), redirect_stderr(err):
            try:
                code = broker.main(args)
            except SystemExit as error:
                code = int(error.code or 0)
                if code:
                    broker._event(events, {"kind": "broker_error", "error": "invalid runtime arguments"})
        return code, out.getvalue(), err.getvalue()
    except Exception as error:
        if events:
            broker._event(events, {"kind": "broker_error", "error": str(error)})
        return 1, "", "semantic compile service: " + str(error) + "\n"
    finally:
        os.chdir(previous_cwd)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def serve(directory, endpoint):
    directory, endpoint = Path(directory).resolve(), Path(endpoint).absolute()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    broker._validated_worker(directory)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    identity = None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            # bind refuses both active and stale endpoints; never overwrite one.
            listener.bind(str(endpoint))
            identity = endpoint.stat()
            endpoint.chmod(0o600)
            listener.listen(16)
            listener.settimeout(0.25)
            while not stopping:
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(10)
                    try:
                        pid, uid, _ = struct.unpack("3i", connection.getsockopt(
                            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
                        values = _decode(_frame(connection))
                        code, out, err = dispatch(directory, values, pid, uid)
                    except Exception as error:
                        code, out, err = 1, "", "semantic compile service: " + str(error) + "\n"
                    payload = struct.pack("!I", code) + _strings([out, err])
                    try:
                        connection.sendall(struct.pack("!I", len(payload)) + payload)
                    except OSError:
                        pass
    finally:
        if identity is not None:
            try:
                current = endpoint.stat()
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    endpoint.unlink()
            except FileNotFoundError:
                pass
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broker", required=True, type=Path)
    parser.add_argument("--socket", required=True, type=Path)
    args = parser.parse_args(argv)
    serve(args.broker, args.socket)


if __name__ == "__main__":
    main()
