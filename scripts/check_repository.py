#!/usr/bin/env python3
import json
import pathlib
import py_compile
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]


def check_markdown_links():
    missing = []
    documents = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "ROADMAP.md",
        *sorted((ROOT / "docs").glob("*.md")),
    ]
    pattern = re.compile(r"!?\[[^]]*\]\(([^)]+)\)")
    for document in documents:
        for target in pattern.findall(document.read_text()):
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = target.split("#", 1)[0]
            if local and not (document.parent / local).resolve().exists():
                missing.append(f"{document.relative_to(ROOT)}: missing {target}")
    if missing:
        raise RuntimeError("\n".join(missing))
    return len(documents)


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


def main():
    markdown = check_markdown_links()
    json_files = check_json()
    python_files = check_python()
    check_citation()
    print(f"repository checks passed: {markdown} Markdown, {json_files} JSON, {python_files} Python files")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
