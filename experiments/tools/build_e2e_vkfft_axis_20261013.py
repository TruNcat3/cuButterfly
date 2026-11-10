#!/usr/bin/env python3
"""Isolated VkFFT provider rebuild; dry-run by default, without CMake or axis compilation."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

from build_e2e_incremental_20261013 import ROOT, sha, stamp, within, response_inputs

BASELINE = ROOT / "build-e2e-scale-20261013"
DESTINATION = ROOT / "build-e2e-vkfft-20261013"
BASELINE_SHA = "89a038905a45e2d2ab072c142f9bde17ed97b42ed86106bdac656e5ec0f0f33b"
PROVIDER = ROOT / "src/xd_vkfft_axis.cpp"
VKFFT = ROOT / "external/VkFFT/vkFFT"
CUDA = Path("/usr/local/cuda-12.4")


def dependency_paths(path, cwd):
    tokens = shlex.split(path.read_text().replace("\\\n", " "))[1:]
    return [(cwd / token).resolve() for token in tokens if
            (cwd / token).is_file()]


def prepare():
    old_receipt_path = BASELINE / "incremental_build_receipt.json"
    old = json.loads(old_receipt_path.read_text())
    if old["state"] != "built_inputs_verified" or old["library_sha256"] != BASELINE_SHA:
        raise ValueError("scale receipt is not the frozen verified build")
    if sha(BASELINE / "libcuntt.so") != BASELINE_SHA:
        raise ValueError("scale library SHA mismatch")
    expected = dict(old["reused_objects"])
    expected.update(old["new_object_sha256"])
    axis = old["prebuilt_axis"]
    if axis["record"]["status"] != "success":
        raise ValueError("scale axis receipt was not successful")
    axis_argv = axis["record"]["argv"]
    axis_object = Path(axis_argv[axis_argv.index("-o") + 1]).resolve()
    if not within(axis_object, BASELINE):
        raise ValueError("axis object must be the scale build's precompiled axis")
    expected[str(axis_object)] = axis["record"]["object_sha256"]
    if stamp(map(Path, expected)) != expected:
        raise ValueError("recorded scale link object SHA mismatch")
    if sha(Path(axis["receipt"])) != axis["receipt_sha256"]:
        raise ValueError("scale axis receipt SHA mismatch")
    link_record = old["link_command"]
    link = list(link_record["argv"])
    link_objects = [str((Path(link_record["cwd"]) / token).resolve())
                    for token in link if token.endswith(".o")]
    if len(link_objects) != len(set(link_objects)) or set(link_objects) != set(expected):
        raise ValueError("scale receipt object set differs from actual link command")
    if link[link.index("-o") + 1] != str(BASELINE / "libcuntt.so"):
        raise ValueError("unexpected scale library output")
    link[link.index("-o") + 1] = str(DESTINATION / "libcuntt.so")
    commands = []
    replaced = {}
    inputs = [PROVIDER]
    metadata = [old_receipt_path, Path(axis["receipt"])]
    metadata.extend(map(Path, old["build_metadata_sha256"]))
    for previous in old["compile_commands"]:
        source = Path(previous["source"])
        if source.name not in ("c_api.cpp", "xd_real_fft.cu"):
            raise ValueError("unexpected scale compile unit")
        argv = list(previous["argv"])
        previous_object = Path(previous["object"])
        new_object = DESTINATION / previous_object.relative_to(BASELINE)
        for option in ("-o", "-MF", "-MT"):
            index = argv.index(option) + 1
            if argv[index] != str(previous_object) + (".d" if option == "-MF" else ""):
                raise ValueError("unexpected scale compile output")
            argv[index] = str(new_object) + (".d" if option == "-MF" else "")
        commands.append({"source": str(source), "object": str(new_object),
                         "cwd": previous["cwd"], "argv": argv})
        replaced[str(previous_object)] = expected[str(previous_object)]
        link[link.index(str(previous_object))] = str(new_object)
        inputs.append(source)
        dependency = Path(str(previous_object) + ".d")
        metadata.append(dependency)
        inputs.extend(dependency_paths(dependency, Path(previous["cwd"])))
        metadata.extend(response_inputs(argv, Path(previous["cwd"])))
    if {Path(command["source"]).name for command in commands} != {"c_api.cpp", "xd_real_fft.cu"}:
        raise ValueError("both scale compile recipes are required")
    cpp = next(command for command in commands if command["source"].endswith("c_api.cpp"))
    provider_argv = list(cpp["argv"])
    provider_object = DESTINATION / "CMakeFiles/cuntt.dir/src/xd_vkfft_axis.cpp.o"
    provider_argv[provider_argv.index("-c") + 1] = str(PROVIDER)
    for option in ("-o", "-MF", "-MT"):
        provider_argv[provider_argv.index(option) + 1] = str(provider_object) + (
            ".d" if option == "-MF" else "")
    provider_argv[1:1] = ["-DVKFFT_BACKEND=1", "-I" + str(VKFFT),
                          "-I" + str(CUDA / "include")]
    commands.append({"source": str(PROVIDER), "object": str(provider_object),
                     "cwd": cpp["cwd"], "argv": provider_argv})
    # Include every vendored VkFFT header, even inactive backend branches.
    vkfft_headers = [path for path in VKFFT.rglob("*") if path.is_file() and
                     path.suffix in (".h", ".hpp", ".cuh", ".inc")]
    if not vkfft_headers or not (VKFFT / "vkFFT.h").is_file():
        raise ValueError("complete external VkFFT header tree is required")
    inputs.extend(path for directory in (ROOT / "src", ROOT / "include", CUDA / "include")
                  for path in directory.rglob("*") if path.is_file() and
                  path.suffix in (".h", ".hpp", ".cuh", ".inc"))
    libraries = [Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"),
                 CUDA / "targets/x86_64-linux/lib/libnvrtc.so",
                 CUDA / "targets/x86_64-linux/lib/libcudart.so"]
    link.insert(link.index("-L/usr/local/cuda/targets/x86_64-linux/lib/stubs"), str(provider_object))
    link.extend([str(path) for path in libraries])
    link.extend(["-Wl,-rpath," + str(CUDA / "targets/x86_64-linux/lib"), "-Wl,-z,defs"])
    fingerprint = dict(old["embedded_build_fingerprint"])
    fingerprint["inherited_from_scale_receipt"] = str(old_receipt_path)
    inputs.append(Path(fingerprint["path"]))
    return {
        "schema_version": 1, "state": "prepared_dry_run",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_directory": str(BASELINE), "destination": str(DESTINATION),
        "baseline_library": {"path": str(BASELINE / "libcuntt.so"), "sha256": BASELINE_SHA},
        "baseline_link_objects": expected,
        "reused_objects": {path: digest for path, digest in expected.items() if path not in replaced},
        "baseline_replaced_objects": replaced, "prebuilt_axis": axis,
        "source_and_header_sha256": stamp(inputs),
        "external_vkfft_header_sha256": stamp(vkfft_headers),
        "build_metadata_sha256": stamp(metadata), "link_library_sha256": stamp(libraries),
        "embedded_build_fingerprint": fingerprint,
        "identity_note": "Embedded fingerprint is inherited from the original precompiled build, "
                         "via the verified scale build. Use this receipt and output library SHA as "
                         "build identity; c_api cache v20 is maintained by the source owner.",
        "compile_commands": commands,
        "link_command": {"cwd": link_record["cwd"], "argv": link},
        "dlopen_command": {"cwd": str(DESTINATION), "argv": [sys.executable, "-c",
            "import ctypes, os; ctypes.CDLL(" + repr(str(DESTINATION / "libcuntt.so")) +
            ", mode=os.RTLD_NOW | os.RTLD_LOCAL)"]},
    }


def verify_inputs(receipt):
    for field in ("baseline_link_objects", "source_and_header_sha256",
                  "external_vkfft_header_sha256", "build_metadata_sha256", "link_library_sha256"):
        if stamp(map(Path, receipt[field])) != receipt[field]:
            raise RuntimeError("inputs changed: " + field)
    if sha(BASELINE / "libcuntt.so") != BASELINE_SHA:
        raise RuntimeError("scale library changed")


def main():
    global DESTINATION
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="compile only after sources freeze")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    DESTINATION = args.destination.absolute()
    if (not within(DESTINATION, ROOT) or DESTINATION == ROOT or
            within(DESTINATION, BASELINE) or within(BASELINE, DESTINATION)):
        parser.error("destination must be separate from the baseline and inside the repository")
    receipt_path = DESTINATION / "vkfft_axis_build_receipt.json"
    if DESTINATION.is_symlink() or DESTINATION.resolve() != DESTINATION:
        parser.error("destination must be the exact new isolated directory")
    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        if not receipt_path.is_file() or any(path != receipt_path for path in DESTINATION.iterdir()):
            parser.error("never overwrite previous build outputs or unrelated files")
        if json.loads(receipt_path.read_text())["state"] != "prepared_dry_run":
            parser.error("only a preparation receipt may be replaced")
    receipt = prepare()
    DESTINATION.mkdir(exist_ok=True)
    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    save()
    commands = receipt["compile_commands"] + [receipt["link_command"], receipt["dlopen_command"]]
    for command in commands:
        print("cwd=" + command["cwd"] + "\n" + " ".join(map(shlex.quote, command["argv"])), flush=True)
    if not args.execute:
        print("DRY RUN: no compiler, linker, or dlopen invoked; receipt=" + str(receipt_path))
        return
    receipt["state"] = "building"
    save()
    try:
        verify_inputs(receipt)
        for index, command in enumerate(commands):
            for option in ("-o", "-MF", "-MT"):
                if option in command["argv"]:
                    output = Path(command["argv"][command["argv"].index(option) + 1])
                    if not within(output.resolve(), DESTINATION):
                        raise ValueError("command output escaped isolated directory")
                    output.parent.mkdir(parents=True, exist_ok=True)
            with (DESTINATION / ("command_%02d.log" % index)).open("w") as log:
                subprocess.run(command["argv"], cwd=command["cwd"], stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            receipt["completed_commands"] = index + 1
            save()
        verify_inputs(receipt)
        receipt["new_object_sha256"] = stamp(Path(command["object"])
                                             for command in receipt["compile_commands"])
        dependencies = []
        for command in receipt["compile_commands"]:
            dependencies.extend(dependency_paths(Path(command["object"] + ".d"), Path(command["cwd"])))
        receipt["compiled_dependency_sha256"] = stamp(dependencies)
        receipt["library_sha256"] = sha(DESTINATION / "libcuntt.so")
        receipt["dlopen_verified"] = True
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
