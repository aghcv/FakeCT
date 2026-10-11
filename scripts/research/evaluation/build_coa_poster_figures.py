#!/usr/bin/env python3
"""Export source-checked CoA development poster figures without model inference."""
import argparse
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from fakect_coa_poster_figures import build_figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path,
                        default=ROOT / "outputs/experiments/coa/focused-pilot-r1")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/poster/coa/pilot-r1")
    args = parser.parse_args()
    try:
        result = build_figures(args.experiment, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"error: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
