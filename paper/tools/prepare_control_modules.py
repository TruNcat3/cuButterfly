#!/usr/bin/env python3
"""Create CPU-only module requests for explicitly bound paper studies."""
import argparse
import json
from pathlib import Path
from prepare_experiments import canonical, module_manifest, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, action="append", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    studies = {str(p.resolve()): json.loads(p.read_text()) for p in args.study}
    if any(s.get("schema") != "cubutterfly-controlled-experiment-v1" for s in studies.values()):
        raise ValueError("expected explicit controlled studies")
    target = json.loads(args.target.read_text())
    document = module_manifest(studies, target)
    output = args.output_dir / "modules.json"
    if output.exists() and canonical(json.loads(output.read_text())) != canonical(document):
        raise ValueError("module requests changed; choose a new output directory")
    write(output, document)
    print(json.dumps(dict(modules=document["module_count"], gpu_queried=False, output=str(output))))


if __name__ == "__main__":
    main()
