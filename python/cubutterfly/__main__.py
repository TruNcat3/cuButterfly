"""CPU descriptor validation and capability inspection."""
import argparse
import json
from pathlib import Path

from . import Descriptor, Library


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate and inspect a cuButterfly descriptor")
    parser.add_argument("descriptor", type=Path, help="descriptor JSON file")
    parser.add_argument("--library", help="shared library path (or CUBUTTERFLY_LIBRARY)")
    args = parser.parse_args(argv)
    library = Library(args.library)
    with Descriptor.from_json(library, args.descriptor.read_text(encoding="utf-8")) as descriptor:
        descriptor.validate()
        print(json.dumps({"descriptor": json.loads(descriptor.json()),
                          "capabilities": json.loads(descriptor.capabilities_json())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
