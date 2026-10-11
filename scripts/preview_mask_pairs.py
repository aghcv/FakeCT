#!/usr/bin/env python3
"""Review frozen edit deltas on a new automatic anatomical target."""
import argparse
import configparser
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fakect_mask_pair_preview import write_preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args()
    try:
        ini=configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',))
        raw=args.config.read_bytes()
        ini.read_string(raw.decode('utf-8-sig'))
        if set(ini.sections())!={'mask_pairs'} or ini.defaults():
            raise ValueError('Expected only [mask_pairs] without defaults')
        required={'case_id','anatomy_npz','dataset_manifest','output_directory'}
        if set(ini['mask_pairs'])!=required:
            raise ValueError('mask_pairs requires exactly: '+', '.join(sorted(required)))
        def path(key):
            value=Path(ini['mask_pairs'][key].strip()).expanduser()
            return (value if value.is_absolute() else ROOT/value).resolve()
        result=write_preview(path('anatomy_npz'),case_id=ini['mask_pairs']['case_id'].strip(),
                             dataset_manifest=path('dataset_manifest'),output_directory=path('output_directory'))
        if args.config.read_bytes()!=raw:
            raise ValueError('INI changed during mask-pair export')
        (path('output_directory')/'input.ini').write_bytes(raw)
        counts={status:sum(r['status']==status for r in result['samples']) for status in
                ('ready_for_visual_review','requires_review')}
        print(json.dumps({'output_directory':str(path('output_directory')),'counts':counts,
                          'training_ready':False},indent=2))
        return 0 if not counts['requires_review'] else 2
    except (OSError,ValueError,KeyError,configparser.Error) as error:
        parser.exit(2,f'error: {error}\n')


if __name__=='__main__':raise SystemExit(main())
