#!/usr/bin/env python3
"""CPU-only isolated one-TU experiment build with frozen-object replay checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import re

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build-e13-cufftdx"
OUTPUT = ROOT / "results/paper_targeted_20261012/c2r_copy_elision_build"
TARGET = "CMakeFiles/cuntt.dir/src/xd_real_fft.cu.o"
FROZEN = ROOT / "results/search_budget_20261011/frozen_runtime/libcuntt.so"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha_ignoring_nvcc_temp(path):
    data = Path(path).read_bytes()
    # nvcc embeds a per-invocation tmpxft_<pid> name in ELF .strtab. It is
    # metadata only; normalize just the fixed-width hexadecimal token.
    data = re.sub(rb"(tmpxft_)[0-9a-f]{8}", rb"\g<1>00000000", data)
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def run(command, label):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(command, cwd=BUILD, env=environment, capture_output=True, text=True)
    (OUTPUT / (label + ".stdout.log")).write_text(result.stdout)
    (OUTPUT / (label + ".stderr.log")).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(label + " failed; inspect isolated logs")


def compiler_command(output, source=ROOT / "src/xd_real_fft.cu"):
    variables = {}
    for line in (BUILD / "CMakeFiles/cuntt.dir/flags.make").read_text().splitlines():
        if " = " in line:
            key, value = line.split(" = ", 1)
            variables[key] = value
    command = ["/usr/local/cuda/bin/nvcc", "-forward-unknown-to-host-compiler"]
    for key in ("CUDA_DEFINES", "CUDA_INCLUDES", "CUDA_FLAGS"):
        command += shlex.split(variables[key])
    return command + ["-MD", "-MT", str(output), "-MF", str(output) + ".d", "-x", "cu", "-c",
                      str(source), "-o", str(output)]


def link_command(manifest, output, target=None):
    command = list(manifest["link_argv"])
    for index, token in enumerate(command):
        if token == "-o":
            command[index + 1] = str(output)
        elif token.endswith(".o"):
            command[index] = str(target if token == TARGET and target is not None else OUTPUT / "objects" / token)
    return command


def capture(output=OUTPUT):
    global OUTPUT
    OUTPUT = Path(output).resolve()
    if (OUTPUT / "capture.json").exists():
        raise ValueError("capture already exists; do not overwrite experiment inputs")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_object = BUILD / TARGET
    dependencies = shlex.split(Path(str(original_object) + ".d").read_text().replace("\\\n", " "))[2:]
    source_path = str(ROOT / "src/xd_real_fft.cu")
    if any(path != source_path and Path(path).stat().st_mtime_ns > original_object.stat().st_mtime_ns for path in dependencies):
        raise ValueError("TU dependencies changed after original object; cannot safely reuse")
    if sha(BUILD / "libcuntt.so") != sha(FROZEN):
        raise ValueError("current build library differs from frozen baseline")
    argv = shlex.split((BUILD / "CMakeFiles/cuntt.dir/link.txt").read_text())
    inputs = []
    for token in argv:
        if token.endswith(".o"):
            path = BUILD / token
            destination = OUTPUT / "objects" / token
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            inputs.append(dict(record(path), snapshot=str(destination), snapshot_sha256=sha(destination)))
    source = ROOT / "src/xd_real_fft.cu"
    shutil.copy2(source, OUTPUT / "original_xd_real_fft.cu")
    manifest = dict(schema="cubutterfly-c2r-copy-elision-build-v1", status="capturing", experiment_only=True,
                    source_snapshot_relation="shared_source lacks xd_real_fft.cu; frozen binary reproduced from existing CMake objects instead",
                    source_before=record(source), frozen_library=record(FROZEN), link_argv=argv, input_objects=inputs,
                    dependencies=[record(path) for path in dependencies],
                    compiler=record("/usr/local/cuda/bin/nvcc"), host_linker=record(argv[0]),
                    flags=record(BUILD / "CMakeFiles/cuntt.dir/flags.make"),
                    includes=record(BUILD / "CMakeFiles/cuntt.dir/includes_CUDA.rsp"),
                    link_recipe=record(BUILD / "CMakeFiles/cuntt.dir/link.txt"),
                    linked_libraries=[record(token) for token in argv if token.startswith("/") and token.endswith(".so")],
                    scale_arithmetic_changed=False, compact_workspace_retained=True, gpu_used=False)
    try:
        baseline = OUTPUT / "libcuntt.c2r-copy-baseline.so"
        manifest["baseline_link_command"] = link_command(manifest, baseline)
        run(manifest["baseline_link_command"], "baseline-relink")
        manifest["baseline_relinked"] = record(baseline)
        if sha(baseline) != sha(FROZEN):
            raise ValueError("original-object relink does not reproduce frozen library bit-for-bit")
        control = OUTPUT / "xd_real_fft.control.o"
        original_text = (ROOT / "src/xd_real_fft.cu").read_text()
        patched = '''        if (!axis_plans_.empty()) {
            // Axis plans can read the caller's compact spectrum while writing
            // their first result into owned scratch; avoid the redundant D2D
            // copy that is required by the legacy no-axis path.
            execute_c2c_axes(const_cast<Complex*>(input_complex), scratch,
                             CUFFT_INVERSE, true);
        } else {
            check_cuda(cudaMemcpyAsync(scratch, input_complex, compact_bytes_, cudaMemcpyDeviceToDevice, stream),
                       "copy xD inverse compact input");
            execute_c2c_axes(scratch, scratch, CUFFT_INVERSE);
        }'''
        original = '''        check_cuda(cudaMemcpyAsync(scratch, input_complex, compact_bytes_, cudaMemcpyDeviceToDevice, stream),
                   "copy xD inverse compact input");
        execute_c2c_axes(scratch, scratch, CUFFT_INVERSE);'''
        if patched not in original_text:
            raise ValueError("expected copy-elision patch is not present")
        control_source = OUTPUT / "control_source" / "xd_real_fft.cu"
        control_source.parent.mkdir(parents=True, exist_ok=True)
        control_source.write_text(original_text.replace(patched, original, 1))
        manifest["control_source"] = record(control_source)
        manifest["control_compile_command"] = compiler_command(control, control_source)
        run(manifest["control_compile_command"], "control-compile")
        manifest["control_object"] = record(control)
        manifest["control_object_matches_after_nvcc_normalization"] = object_sha_ignoring_nvcc_temp(control) == object_sha_ignoring_nvcc_temp(original_object)
        manifest["control_object_note"] = "Diagnostic recompile uses an isolated source path; CUDA symbol file-path hash prevents byte-equivalence. Frozen-object replay is established by bit-identical baseline relink; candidate is linked with only the changed TU."
        manifest["control_object_normalized_sha256"] = object_sha_ignoring_nvcc_temp(control)
        manifest["original_object_normalized_sha256"] = object_sha_ignoring_nvcc_temp(original_object)
        manifest["status"] = "compatible-inputs-captured"
    except Exception as error:
        manifest.update(status="blocked_reuse", error=str(error), fallback="Configure a new independent CMake build and build cuntt normally; preserve this capture and original build.")
    (OUTPUT / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def build(output=OUTPUT):
    global OUTPUT
    OUTPUT = Path(output).resolve()
    manifest = json.loads((OUTPUT / "capture.json").read_text())
    if manifest["status"] != "compatible-inputs-captured":
        raise ValueError("capture did not establish safe one-TU reuse")
    for identity in manifest["dependencies"]:
        if identity["path"] != str(ROOT / "src/xd_real_fft.cu") and sha(identity["path"]) != identity["sha256"]:
            raise ValueError("unchanged TU dependency changed: " + identity["path"])
    for identity in manifest["input_objects"]:
        if sha(identity["snapshot"]) != identity["snapshot_sha256"]:
            raise ValueError("frozen object snapshot changed")
    output = OUTPUT / "libcuntt.c2r-copy-elision.experimental.so"
    target = OUTPUT / "xd_real_fft.copy-elision.o"
    manifest["changed_source"] = record(ROOT / "src/xd_real_fft.cu")
    manifest["compile_command"] = compiler_command(target)
    run(manifest["compile_command"], "candidate-compile")
    manifest["candidate_object"] = record(target)
    manifest["candidate_link_command"] = link_command(manifest, output, target)
    run(manifest["candidate_link_command"], "candidate-link")
    manifest.update(status="built-not-gpu-validated", candidate_library=record(output),
                    inherited_runtime_fingerprint=True,
                    identity_boundary="Internal fingerprint in untouched c_api object is historical; compare explicit experimental file SHA and manifest, not that fingerprint alone.")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = capture(args.output_dir) if args.capture else build(args.output_dir)
    print(json.dumps({key: result.get(key) for key in ("status", "error", "baseline_relinked", "candidate_library")}, indent=2))
    return 1 if result["status"] == "blocked_reuse" else 0


if __name__ == "__main__":
    raise SystemExit(main())
