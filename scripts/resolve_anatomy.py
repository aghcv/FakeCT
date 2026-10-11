#!/usr/bin/env python3
"""Resolve an optional anatomical label filter before ROI preview or editing."""
import argparse
import configparser
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_config import load_preview_config
from fakect_anatomy import create_anatomy_selection


def load_config(path):
    """Choose the strict preview or study parser from the declared INI schema.

    Parameters
    ----------
    path : str or pathlib.Path
        UTF-8 INI input whose study.schema_version selects the parser.

    Returns
    -------
    dict
        Normalized configuration with resolved paths; no voxel volumes are read.

    Notes
    -----
    Malformed INI syntax is reported as ValueError. This reads configuration
    metadata and does not write an anatomical selection.
    """
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    try:
        with Path(path).open(encoding='utf-8-sig') as stream:
            parser.read_file(stream)
    except configparser.Error as error:
        raise ValueError(f'Invalid anatomy input: {error}') from error
    schema = parser.get('study', 'schema_version', fallback='')
    if schema in ('fakect.study/1', 'fakect.recipe-study/1', 'fakect.recipe-cohort/1'):
        from fakect_study_config import load_study_config
        return load_study_config(path)
    return load_preview_config(path)


def main():
    """Resolve and save an optional anatomical selector from stable input bytes.

    Parameters
    ----------
    None
        Arguments are read from the process command line.

    Returns
    -------
    None
        Writes the requested files or prints CLI output; no return value.

    Notes
    -----
    Writes the configured selection artifact and prints IDs, counts and warnings
    as JSON. Does not edit geometry or prepare a dataset. CLI failures exit with 2.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        raw = args.config.read_bytes()
        config = load_config(args.config)
        if raw != args.config.read_bytes():
            raise ValueError('Input changed while loading')
        result = create_anatomy_selection(config, raw)
        if raw != args.config.read_bytes():
            raise ValueError('Input changed during resolution; saved input.ini describes the completed artifact')
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({'anatomy_file': str(config['selection']['anatomy_file']),
                      'selected_source_ids': result['selected_source_ids'],
                      'counts': result['counts'], 'warnings': result['warnings'],
                      'geometry_edited': False, 'dataset_prepared': False}, indent=2))


if __name__ == '__main__':
    main()
