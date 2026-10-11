#!/usr/bin/env python3
"""Select thoracic aorta and proximal branches from labeled XCAT, without an ROI."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from fakect_phantom_anatomy import load_config, validate, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        result = validate(config) if args.validate_only else run(config)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'error: {exc}\n')
    print(json.dumps({k: result[k] for k in ('status', 'case_id', 'scope', 'geometry',
                                           'z_slab_mm', 'branch_reach_mm', 'limitations')}, indent=2))
    if not args.validate_only:
        print(json.dumps({'output_directory': str(config['output_directory']), 'counts': result['counts']}, indent=2))


if __name__ == '__main__':
    main()
