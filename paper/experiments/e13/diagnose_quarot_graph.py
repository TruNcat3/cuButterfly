"""Run the application runner with a bounded reduced model for Graph diagnosis."""
import argparse
import json
import sys
from pathlib import Path

import run_quarot_random_end_to_end as runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    args, remaining = parser.parse_known_args()
    if not 1 <= args.layers <= 4:
        parser.error("Graph diagnosis requires 1 to 4 layers")
    original_config = runner._config

    def reduced_config(module):
        config = original_config(module)
        config.num_hidden_layers = args.layers
        return config

    runner._config = reduced_config
    sys.argv = [sys.argv[0], *remaining]
    status = runner.main()
    output = Path(remaining[remaining.index("--output") + 1])
    report = json.loads(output.read_text())
    report.update(full_model_run=False, diagnostic_layers=args.layers,
                  scope="reduced model Graph diagnosis; not full 7B application evidence")
    output.write_text(json.dumps(report, indent=2) + "\n")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
