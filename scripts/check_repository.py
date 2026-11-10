#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import py_compile
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def local_artifact_paths():
    manifest = json.loads((ROOT / "config/repository_local_artifacts.json").read_text())
    if manifest.get("schema") != "cubutterfly-repository-local-artifacts-v1":
        raise ValueError("unsupported repository local-artifact manifest schema")
    paths = manifest.get("artifacts")
    if not isinstance(paths, list):
        raise ValueError("local artifacts must be an explicit list of exact results/ paths")
    for value in paths:
        if (not isinstance(value, str) or not value.startswith("results/")
                or any(char in value for char in "*?[]\\")
                or any(part in ("", ".", "..") for part in value.split("/"))
                or pathlib.PurePosixPath(value).as_posix() != value):
            raise ValueError("local artifacts require normalized exact results/ paths: " + str(value))
    if len(set(paths)) != len(paths):
        raise ValueError("duplicate local artifact paths")
    return set(paths)


def check_markdown_links(strict_local_artifacts=False):
    registered = local_artifact_paths()
    root = ROOT.resolve()
    missing = []
    unbundled = set()
    documents = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "ROADMAP.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]
    pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for document in documents:
        try:
            document.resolve().relative_to(root)
        except ValueError:
            missing.append(f"{document.relative_to(ROOT)}: outside repository")
            continue
        for target in pattern.findall(document.read_text()):
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = target.split("#", 1)[0]
            if not local:
                continue
            # Check lexical containment before following symlinks, then check
            # resolved containment before probing the target's existence.
            candidate = pathlib.Path(os.path.abspath(document.parent / local))
            try:
                candidate.relative_to(root)
                resolved = candidate.resolve()
                relative = resolved.relative_to(root).as_posix()
            except ValueError:
                missing.append(f"{document.relative_to(ROOT)}: outside repository: {target}")
                continue
            if resolved.exists():
                continue
            if relative in registered:
                unbundled.add(relative)
                if not strict_local_artifacts:
                    continue
            missing.append(f"{document.relative_to(ROOT)}: missing {target}")
    if missing:
        raise RuntimeError("\n".join(missing))
    if strict_local_artifacts:
        # Strict mode checks the entire declared artifact inventory, even if a
        # documentation link has since been removed. Directories do not cover children.
        absent = [value for value in registered if not (ROOT / value).exists()]
        if absent:
            raise RuntimeError("registered raw artifacts absent:\n" + "\n".join(sorted(absent)))
    return len(documents), sorted(unbundled)


def check_json():
    paths = sorted((*ROOT.glob("config/**/*.json"), *ROOT.glob("configs/**/*.json")))
    for path in paths:
        json.loads(path.read_text())
    return len(paths)


def check_python():
    paths = sorted((ROOT / "scripts").glob("*.py"))
    for path in paths:
        py_compile.compile(path, doraise=True)
    return len(paths)


def check_citation():
    text = (ROOT / "CITATION.cff").read_text()
    required = ("cff-version:", "title:", "authors:", "repository-code:", "version:", "license:")
    missing = [key for key in required if not any(line.startswith(key) for line in text.splitlines())]
    if missing:
        raise RuntimeError(f"CITATION.cff is missing required keys: {', '.join(missing)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check packaged source links and report explicitly unbundled raw artifacts")
    parser.add_argument("--strict-local-artifacts", action="store_true",
                        help="Require all registered historical raw artifacts to exist locally too")
    args = parser.parse_args(argv)
    markdown, unbundled = check_markdown_links(args.strict_local_artifacts)
    json_files = check_json()
    python_files = check_python()
    check_citation()
    print(f"repository checks passed: {markdown} Markdown, {json_files} JSON, {python_files} Python files")
    print(f"unbundled historical raw artifacts: {len(unbundled)} exact linked paths "
          "(explicit inventory; contents and experimental validity not checked)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
