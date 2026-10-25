#!/usr/bin/env python3
"""Run a finite list of prepared experiment commands, optionally after a writer.

This does not resume failed experiments or wait for foreign GPU jobs. A failed
prerequisite or exclusivity check stops the queue. Individual runners retain
their measurement compatibility checks and checkpoints.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from sweep_cubutterfly_designs import require_exclusive_gpu


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--command-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--after-session")
    parser.add_argument("--after-journal", type=Path)
    parser.add_argument("--after-status", default="complete-measured-matrix")
    parser.add_argument("--wait-seconds", type=float, default=43200)
    args = parser.parse_args()
    if bool(args.after_session) != bool(args.after_journal):
        parser.error("--after-session and --after-journal must be supplied together")
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    by_id = {command["id"]: command for command in plan["commands"]}
    commands = [by_id[identifier] for identifier in args.command_id]
    if len(set(args.command_id)) != len(args.command_id):
        parser.error("duplicate command IDs")
    for command in commands:
        if command.get("requires_gpu") and command.get("env", {}).get("CUDA_VISIBLE_DEVICES", "").lower() != args.gpu_uuid.lower():
            parser.error(f"GPU target mismatch: {command['id']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.output.exists():
            raise ValueError("queue journal exists; review it and select a new continuation journal")
        document = dict(schema="cubutterfly-prepared-queue-v1", status="starting", started_at=now(),
                        plan=str(args.plan.resolve()), plan_sha256=hashlib.sha256(raw).hexdigest(),
                        gpu_uuid=args.gpu_uuid, command_ids=args.command_id, records=[],
                        prerequisite=dict(session=args.after_session,
                                          journal=str(args.after_journal) if args.after_journal else None,
                                          required_status=args.after_status))
        write(args.output, document)
        try:
            if args.after_session:
                document["status"] = "waiting-for-own-writer"
                write(args.output, document)
                deadline = time.monotonic() + args.wait_seconds
                while subprocess.run(["tmux", "has-session", "-t", "=" + args.after_session],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("prerequisite wait expired; no commands started")
                    time.sleep(min(15, max(0, deadline - time.monotonic())))
                prerequisite = json.loads(args.after_journal.read_text())
                if prerequisite.get("status") != args.after_status:
                    raise RuntimeError(f"prerequisite ended with status {prerequisite.get('status')}")
            document["status"] = "running"
            write(args.output, document)
            for index, command in enumerate(commands):
                environment = dict(os.environ, **command.get("env", {}))
                if command.get("requires_gpu"):
                    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
                    require_exclusive_gpu()
                record = dict(id=command["id"], status="running", started_at=now(),
                              argv=command["argv"], cwd=command.get("cwd", str(ROOT)),
                              env=command.get("env", {}))
                log = args.output.parent / f"{args.output.stem}-{index:02d}.log"
                record["log"] = str(log.resolve())
                document["records"].append(record)
                write(args.output, document)
                started = time.monotonic()
                with log.open("x") as stream:
                    result = subprocess.run(command["argv"], cwd=record["cwd"], env=environment,
                                            stdout=stream, stderr=subprocess.STDOUT)
                record.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - started,
                              ended_at=now(), status="completed" if result.returncode == 0 else "failed")
                write(args.output, document)
                if result.returncode:
                    raise RuntimeError(f"command failed: {command['id']}, see {log}")
                if command.get("requires_gpu"):
                    require_exclusive_gpu()
            document.update(status="completed-commands", ended_at=now())
            write(args.output, document)
        except BaseException as error:
            document.update(status="stopped", reason=str(error), ended_at=now())
            write(args.output, document)
            raise


if __name__ == "__main__":
    main()
