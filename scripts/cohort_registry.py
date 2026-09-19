#!/usr/bin/env python3
"""Register, list, or verify immutable prepared FakeCT cohort revisions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fakect_cohort_registry import register_dataset, load_registry_entry, list_registry_entries


def _summary(entry):
    return {key: value for key, value in entry.items() if key != 'manifest'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    register = commands.add_parser('register', help='Copy and pin a completed dataset as an immutable revision')
    register.add_argument('--registry', required=True)
    register.add_argument('--dataset', required=True, help='Path to dataset-manifest.json')
    register.add_argument('--name', required=True, help='Immutable revision name, e.g. coa-260602-v1')
    register.add_argument('--anatomy-family', required=True)
    register.add_argument('--family-verified', action='store_true', help='Assert the source-family identity has been independently reviewed')
    listing = commands.add_parser('list', help='List metadata without decoding NPZ payloads')
    listing.add_argument('--registry', required=True)
    verify = commands.add_parser('verify', help='Verify all saved bytes and array contracts')
    verify.add_argument('--registry', required=True)
    verify.add_argument('--name', required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'register':
            result = _summary(register_dataset(args.registry, args.dataset, args.name,
                                               args.anatomy_family, args.family_verified))
        elif args.command == 'verify':
            result = {'verified': True, **_summary(load_registry_entry(args.registry, args.name))}
        else:
            result = {'entries': [_summary(entry) for entry in list_registry_entries(args.registry)]}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
