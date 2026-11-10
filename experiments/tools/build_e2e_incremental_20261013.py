#!/usr/bin/env python3
"""Isolated three-unit research rebuild; dry-run by default, never invokes CMake."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
UNITS = ("src/c_api.cpp", "src/xd_real_fft.cu", "src/xd_axis_fft.cu")
BASELINE_SHA = "ddb7d70cc8e0fac900c0b912f03d7aae18855a807d06f0dff4a0ebad1bcb665b"


def within(path, parent):
    return path == parent or parent in path.parents


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stamp(paths):
    return {str(path): sha(path) for path in sorted(set(paths))}


def expand_make(command, variables):
    def replace(match):
        name = match.group(1)
        if name not in variables:
            raise ValueError("unknown Make variable: " + name)
        return variables[name]
    expanded = re.sub(r"\$\(([^)]+)\)", replace, command)
    if "$" in expanded:
        raise ValueError("unexpanded shell/Make expression")
    return shlex.split(expanded)


def response_inputs(argv, cwd):
    paths = []
    for index, token in enumerate(argv):
        if token in ("--options-file", "-optf"):
            paths.append((cwd / argv[index + 1]).resolve())
        elif token.startswith("@"):
            paths.append((cwd / token[1:]).resolve())
        elif token.startswith("--options-file="):
            paths.append((cwd / token.split("=", 1)[1]).resolve())
    return paths


def expand_link_responses(argv, cwd):
    result = []
    for token in argv:
        if token.startswith("@"):
            result.extend(expand_link_responses(shlex.split(
                (cwd / token[1:]).read_text()), cwd))
        else:
            result.append(token)
    return result


def prepare(baseline, destination):
    target = baseline / "CMakeFiles/cuntt.dir"
    flags = target / "flags.make"
    build = target / "build.make"
    link_file = target / "link.txt"
    variables = dict(re.findall(r"^(\w+)\s*=\s*(.*)$", flags.read_text(), re.M))
    recipes = build.read_text().splitlines()
    commands = []
    changed = {}
    metadata = [flags, build, link_file]
    headers = []
    for unit in UNITS:
        source = (ROOT / unit).resolve()
        old_relative = "CMakeFiles/cuntt.dir/" + unit + ".o"
        matches = [line.strip() for line in recipes if line.startswith("\t")
                   and " -c " in line and str(source) in line
                   and old_relative in line and not line.lstrip().startswith("@")]
        if len(matches) != 1:
            raise ValueError("expected one compile recipe for " + unit)
        argv = expand_make(matches[0], variables)
        metadata.extend(response_inputs(argv, baseline))
        old_object = (baseline / old_relative).resolve()
        new_object = (destination / old_relative).resolve()
        for option in ("-o", "-MF", "-MT"):
            index = argv.index(option) + 1
            original = argv[index]
            expected = old_relative + (".d" if option == "-MF" else "")
            if original != expected:
                raise ValueError("unexpected output operand: " + original)
            argv[index] = str(new_object) + (".d" if option == "-MF" else "")
        if unit == "src/xd_axis_fft.cu" and "-DCUBUTTERFLY_XD_CUFFTDX_SM=800" not in argv:
            raise ValueError("baseline axis recipe lacks the per-file SM800 define")
        changed[old_object] = new_object
        commands.append({"source": str(source), "object": str(new_object),
                         "cwd": str(baseline), "argv": argv})
        dependency = Path(str(old_object) + ".d")
        if not dependency.exists():
            raise ValueError("missing baseline dependency file: " + str(dependency))
        metadata.append(dependency)
        # Make dependency escaping is handled by shlex after joining continuations.
        deps = shlex.split(dependency.read_text().replace("\\\n", " "))[1:]
        for token in deps:
            path = (baseline / token).resolve()
            if path.is_file() and path.suffix in (".h", ".hpp", ".cuh") and (
                    within(path, ROOT / "src") or within(path, ROOT / "include")):
                headers.append(path)
    raw_link = shlex.split(link_file.read_text())
    metadata.extend(response_inputs(raw_link, baseline))
    link = expand_link_responses(raw_link, baseline)
    old_objects = []
    for index, token in enumerate(link):
        if token.endswith(".o"):
            original = (baseline / token).resolve()
            old_objects.append(original)
            link[index] = str(changed.get(original, original))
    if set(changed) - set(old_objects):
        raise ValueError("changed objects absent from link command")
    output_index = link.index("-o") + 1
    if (baseline / link[output_index]).resolve() != baseline / "libcuntt.so":
        raise ValueError("unexpected baseline library output")
    link[output_index] = str(destination / "libcuntt.so")
    fingerprint = baseline / "snapshots/build_fingerprint.hpp"
    return {
        "schema_version": 1, "state": "prepared_dry_run", "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_directory": str(baseline), "destination": str(destination),
        "baseline_library": {"path": str(baseline / "libcuntt.so"),
                             "sha256": sha(baseline / "libcuntt.so")},
        "reused_objects": stamp(set(old_objects) - set(changed)),
        "baseline_replaced_objects": stamp(changed),
        "source_and_header_sha256": stamp([ROOT / unit for unit in UNITS] + headers),
        "build_metadata_sha256": stamp(metadata),
        "embedded_build_fingerprint": {"inherited": True, "path": str(fingerprint),
                                       "sha256": sha(fingerprint), "contents": fingerprint.read_text()},
        "identity_note": "Incremental research build inherits the baseline embedded fingerprint; "
                         "use this receipt/library SHA as build identity. The c_api real cache version "
                         "is separately maintained by the source owner (planned v19).",
        "compile_commands": commands,
        "link_command": {"cwd": str(baseline), "argv": link},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "build-e2e-20261013")
    parser.add_argument("--destination", type=Path, default=ROOT / "build-e2e-scale-20261013")
    parser.add_argument("--execute", action="store_true", help="compile and link only after sources freeze")
    parser.add_argument("--axis-receipt", type=Path,
        help="reuse the axis worker's source/header/object SHA and cwd/argv success receipt")
    args = parser.parse_args()
    baseline, destination = args.baseline.resolve(), args.destination.resolve()
    if within(destination, baseline) or within(baseline, destination):
        parser.error("destination must be separate from baseline")
    if not within(destination, ROOT) or destination == ROOT:
        parser.error("destination must be a new directory inside the repository")
    previous_receipt = destination / "incremental_build_receipt.json"
    if (destination / "libcuntt.so").exists():
        parser.error("never overwrite a previous library")
    if previous_receipt.exists() and (not args.execute or
            json.loads(previous_receipt.read_text())["state"] != "prepared_dry_run"):
        parser.error("only --execute may replace a preparation-only receipt")
    if destination.exists() and any(destination.iterdir()) and not args.axis_receipt:
        parser.error("nonempty destination requires --axis-receipt")
    receipt = prepare(baseline, destination)
    if receipt["baseline_library"]["sha256"] != BASELINE_SHA:
        raise ValueError("baseline library differs from the frozen expected SHA")
    if args.axis_receipt:
        axis_receipt_path = args.axis_receipt.resolve()
        axis = json.loads(axis_receipt_path.read_text())
        if axis["status"] != "success":
            raise ValueError("axis compile receipt is not successful")
        argv = list(axis["argv"])
        axis_object = Path(argv[argv.index("-o") + 1]).resolve()
        if not within(axis_object, destination) or not axis_object.is_file():
            raise ValueError("prebuilt axis object must exist inside the isolated destination")
        if sha(axis_object) != axis["object_sha256"]:
            raise ValueError("prebuilt axis object SHA mismatch")
        axis_inputs = {str(ROOT / "src/xd_axis_fft.cu"): axis["source_sha256"],
                       str(ROOT / "src/xd_axis_fft.hpp"): axis["header_sha256"]}
        for path in (ROOT / "src/xd_axis_fft.cu", ROOT / "src/xd_axis_fft.hpp"):
            if axis_inputs.get(str(path)) != sha(path):
                raise ValueError("axis receipt must match current source/header: " + str(path))
        if stamp(map(Path, axis_inputs)) != axis_inputs:
            raise ValueError("axis receipt input SHA mismatch")
        original_command = receipt["compile_commands"].pop()
        expected = list(original_command["argv"])
        for option in ("-o", "-MF", "-MT"):
            index = expected.index(option) + 1
            expected[index] = str(axis_object) + (".d" if option == "-MF" else "")
        if argv != expected or Path(axis["cwd"]).resolve() != baseline:
            raise ValueError("axis receipt compiler command differs from baseline recipe")
        link = receipt["link_command"]["argv"]
        link[link.index(original_command["object"])] = str(axis_object)
        receipt["prebuilt_axis"] = {"receipt": str(axis_receipt_path),
            "receipt_sha256": sha(axis_receipt_path), "record": axis}
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "incremental_build_receipt.json"
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    commands = receipt["compile_commands"] + [receipt["link_command"]]
    for command in commands:
        print("cwd=" + command["cwd"] + "\n" + shlex.join(command["argv"]), flush=True)
    if not args.execute:
        print("DRY RUN: no compiler/linker invoked; receipt=" + str(receipt_path))
        return
    receipt["state"] = "building"
    save()
    try:
        for index, command in enumerate(commands):
            for option in ("-o", "-MF", "-MT"):
                if option in command["argv"]:
                    output = Path(command["argv"][command["argv"].index(option) + 1])
                    if not within(output, destination):
                        raise ValueError("output escaped isolated directory")
                    output.parent.mkdir(parents=True, exist_ok=True)
            with (destination / ("command_%02d.log" % index)).open("w") as log:
                subprocess.run(command["argv"], cwd=command["cwd"], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            receipt["completed_commands"] = index + 1
            save()
        for field in ("reused_objects", "baseline_replaced_objects", "source_and_header_sha256",
                      "build_metadata_sha256"):
            if stamp(map(Path, receipt[field])) != receipt[field]:
                raise RuntimeError("inputs changed during build: " + field)
        if args.axis_receipt and (sha(axis_object) != axis["object_sha256"] or
                                  sha(axis_receipt_path) != receipt["prebuilt_axis"]["receipt_sha256"]):
            raise RuntimeError("prebuilt axis object/receipt changed during build")
        if sha(baseline / "libcuntt.so") != BASELINE_SHA:
            raise RuntimeError("baseline library changed during build")
        receipt["new_object_sha256"] = stamp(Path(command["object"]) for command in receipt["compile_commands"])
        receipt["library_sha256"] = sha(destination / "libcuntt.so")
        receipt["state"] = "built_inputs_verified"
    except Exception as error:
        receipt["state"] = "failed"
        receipt["error"] = str(error)
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
