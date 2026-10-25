#!/usr/bin/env python3
"""Validate the E13 end-to-end preparation contract without side effects.

This checker validates JSON and optionally writes a readiness report. It does not import GPU libraries, access the
network, download assets, build source trees, or execute an application.
Commands in the readiness report are a human checklist; network commands are
always marked as requiring explicit opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
OPERATORS = {"fft", "fwht", "ntt"}
LICENSES = {"MIT", "Apache-2.0"}


def get_path(obj: dict[str, Any], *keys: str) -> Any:
    value: Any = obj
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def validate(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema") != "cubutterfly-end-to-end-v1":
        errors.append("schema must be cubutterfly-end-to-end-v1")
    if manifest.get("experiment_id") != "E13":
        errors.append("experiment_id must be E13")
    if manifest.get("status") != "prepared_not_run":
        errors.append("manifest status must remain prepared_not_run")

    scope = manifest.get("scope", {})
    for key in ("gpu_execution_performed", "application_results_claimed",
                "external_data_downloaded", "model_or_weight_downloaded",
                "network_application_runs_performed"):
        if scope.get(key) is not False:
            errors.append(f"scope.{key} must be false")
    if scope.get("pinned_source_checkout_performed") is not True:
        errors.append("scope.pinned_source_checkout_performed must be true")

    repositories = manifest.get("source_validation", {}).get("repositories", [])
    repo_by_id: dict[str, dict[str, Any]] = {}
    if len(repositories) != 3:
        errors.append("exactly three pinned repositories are required")
    for repo in repositories:
        repo_id = repo.get("id")
        if repo_id in repo_by_id:
            errors.append(f"duplicate repository id: {repo_id}")
        repo_by_id[repo_id] = repo
        if not isinstance(repo.get("url"), str) or not repo["url"].startswith("https://"):
            errors.append(f"{repo_id}: repository URL must be HTTPS")
        if not COMMIT_RE.fullmatch(str(repo.get("commit", ""))):
            errors.append(f"{repo_id}: commit must be a 40-character lowercase SHA")
        if repo.get("license") not in LICENSES:
            errors.append(f"{repo_id}: unsupported or missing source license")

    applications = manifest.get("applications", [])
    seen_operators: set[str] = set()
    if len(applications) != 3:
        errors.append("exactly three applications are required")
    for app in applications:
        operator = app.get("operator")
        app_id = app.get("id", "<missing>")
        if operator in seen_operators:
            errors.append(f"duplicate operator application: {operator}")
        seen_operators.add(operator)
        if operator not in OPERATORS:
            errors.append(f"{app_id}: operator must be one of {sorted(OPERATORS)}")
        if app.get("repository") not in repo_by_id:
            errors.append(f"{app_id}: repository is not pinned in source_validation")
        elif app.get("license") != repo_by_id[app["repository"]].get("license"):
            errors.append(f"{app_id}: application/source license mismatch")
        payloads = app.get("payloads", [])
        if len(payloads) != 2:
            errors.append(f"{app_id}: exactly two finite payload definitions are required")
        for payload in payloads:
            status = str(payload.get("status", ""))
            if not ("pending" in status or "blocked" in status):
                errors.append(f"{app_id}/{payload.get('id')}: status must be pending or blocked")
        for blocker in app.get("blockers", []):
            if not str(blocker).strip():
                errors.append(f"{app_id}: blockers must not contain empty entries")
        for command in app.get("commands", []):
            if command.get("network") and command.get("requires_explicit_opt_in") is not True:
                errors.append(f"{app_id}: network command lacks explicit opt-in")
            if str(command.get("status", "")).lower() in {"completed", "measured", "run"}:
                errors.append(f"{app_id}: command cannot claim execution in preparation")

        if operator == "fft":
            baseline = app.get("baseline", {})
            transform = baseline.get("transform", {})
            boundary = app.get("replacement_boundary", {})
            data = app.get("external_data", {})
            if baseline.get("complex_data") is not False:
                errors.append("FNO baseline must keep complex_data=false")
            if "rfftn" not in transform.get("forward", "") or "irfftn" not in transform.get("inverse", ""):
                errors.append("FNO contract must name rfftn and irfftn")
            if boundary.get("real_fft_required") is not True:
                errors.append("FNO replacement must require the real-FFT adapter")
            resolutions = set(data.get("available_resolutions", []))
            if not {64, 128}.issubset(resolutions):
                errors.append("FNO data contract must include verified 64 and 128 resolutions")
            for asset in data.get("verified_files", []):
                if not isinstance(asset.get("bytes"), int) or not re.fullmatch(r"[0-9a-f]{32}", str(asset.get("md5", ""))):
                    errors.append("FNO verified archives need byte count and MD5")

        if operator == "fwht":
            baseline = app.get("baseline", {})
            shape = app.get("replacement_boundary", {}).get("llama2_7b_shape_contract", {})
            if "no_init_weights" not in baseline.get("upstream_warning", ""):
                errors.append("QuaRot random no_init_weights warning is required")
            if shape.get("intermediate_size") != 11008 or shape.get("small_hadamard_factor") != 172 or shape.get("power_of_two_subpath") != 64:
                errors.append("QuaRot must preserve the 11008 = 172 x 64 decomposition")
            preserved = " ".join(app.get("replacement_boundary", {}).get("preserve", []))
            if "Linear4bit" not in preserved or "GEMM" not in preserved:
                errors.append("QuaRot boundary must preserve the quantized GEMM path")

        if operator == "ntt":
            baseline = app.get("baseline", {})
            contract = baseline.get("contract", {})
            boundary = app.get("replacement_boundary", {})
            coverage = app.get("dispatch_coverage", {})
            if boundary.get("word64_modulus_max_exclusive") != 2**63:
                errors.append("NTT word64 modulus bound must be exactly 2^63 exclusive")
            if "X^N+1" not in contract.get("ring", ""):
                errors.append("NTT ring must preserve X^N+1")
            if contract.get("layout") != "gpuntt::PerPolynomial" or contract.get("zero_padding") is not False:
                errors.append("NTT layout and zero-padding contract is incomplete")
            if not contract.get("stream"):
                errors.append("NTT stream contract is required")
            if len(coverage.get("dispatch_families", [])) < 3:
                errors.append("NTT complete-call dispatch coverage is incomplete")
            policy = app.get("payload_policy", {})
            if policy.get("main_payload_ids") != []:
                errors.append("NTT main payload must remain unselected until the audit")
            if not policy.get("candidate_payload_ids"):
                errors.append("NTT resource candidate must be recorded")

    if seen_operators != OPERATORS:
        errors.append(f"operator coverage must be exactly {sorted(OPERATORS)}")
    return errors


def build_report(manifest: dict[str, Any], manifest_path: Path, errors: list[str]) -> dict[str, Any]:
    commands = []
    blockers = []
    for app in manifest.get("applications", []):
        blockers.extend({"application": app.get("id"), "blocker": blocker}
                         for blocker in app.get("blockers", []))
        for command in app.get("commands", []):
            commands.append({
                "application": app.get("id"),
                "phase": command.get("phase"),
                "command": command.get("command"),
                "network": bool(command.get("network", False)),
                "gpu": bool(command.get("gpu", False)),
                "working_directory": ("parent of the isolated source checkouts"
                                      if command.get("phase") == "source_checkout"
                                      else f"{app.get('repository')} checkout root"),
                "prerequisite_status": command.get("status", "source_checkout_recipe"),
                "application_benchmark_ready": False,
                "requires_explicit_opt_in": bool(command.get("requires_explicit_opt_in", False)),
                "execution": "not_run",
            })
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        "schema": "cubutterfly-end-to-end-readiness-v1",
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "status": "invalid_manifest" if errors else "blocked_pending_adapters_assets_and_gpu_validation",
        "application_ready": False,
        "cpu_validation_performed": True,
        "network_actions_performed": False,
        "gpu_queries_performed": False,
        "application_runs_performed": False,
        "measurement_protocol": manifest.get("measurement_protocol", {}),
        "checks": {
            "manifest_contract": not errors,
            "source_and_license_fields": not errors,
            "payload_and_semantic_guards": not errors,
            "commands_are_unrun_checklist": not errors,
        },
        "errors": errors,
        "blockers": blockers,
        "commands": commands,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    default_manifest = Path(__file__).resolve().parents[1] / "experiments" / "end_to_end.json"
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "invalid_manifest", "error": str(exc)}, indent=2))
        return 1
    if not isinstance(manifest, dict):
        print(json.dumps({"status": "invalid_manifest", "error": "manifest root must be an object"}, indent=2))
        return 1

    errors = validate(manifest)
    report = build_report(manifest, args.manifest, errors)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
