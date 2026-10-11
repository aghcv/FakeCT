#!/usr/bin/env python3
"""Build an interactive baseline anatomy review for a prepared context cohort."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from fakect_context_report import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--context-mm', type=float, default=30., help='Additional vertical context around the anatomical crop')
    parser.add_argument('--attenuation-max', type=float, default=.3, help='Upper grayscale display limit in cm^-1')
    args = parser.parse_args()
    try:
        result = render_report(args.manifest, args.output, context_mm=args.context_mm,
                               attenuation_max=args.attenuation_max)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({k: result[k] for k in ('viewer', 'case_count', 'scenarios', 'omitted_surfaces',
                                          'anatomical_extent_reviewed')}, indent=2))


if __name__ == '__main__':
    main()
