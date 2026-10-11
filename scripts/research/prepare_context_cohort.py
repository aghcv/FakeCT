#!/usr/bin/env python3
"""Prepare the broader-aorta cohort without duplicating whole-body images."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from fakect_context_data import load_config, prepare, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--stage', choices=['prepare'], default='prepare')
    parser.add_argument('--resume', action='store_true', help='Reuse matching completed case artifacts after an interrupted run')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        config['resume'] = args.resume
        result = validate(config) if args.validate_only else prepare(config)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({k: result[k] for k in ('status', 'case_count', 'sample_count', 'anatomical_extent_reviewed')}, indent=2))
    if 'manifest_path' in result:
        print(result['manifest_path'])


if __name__ == '__main__':
    main()
