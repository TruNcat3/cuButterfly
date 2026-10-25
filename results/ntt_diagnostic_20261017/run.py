#!/usr/bin/env python3
"""Compatibility entry point for the reusable NTT supplement runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_ntt_comparison_supplement import main, mappings

if __name__ == '__main__':
    raise SystemExit(main())
