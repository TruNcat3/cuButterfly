#!/usr/bin/env python3
"""Compatibility entry point for the reusable grouped FFT supplement runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_fft_grouped_supplement import main

if __name__ == '__main__':
    raise SystemExit(main())
