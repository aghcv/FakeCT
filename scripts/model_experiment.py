#!/usr/bin/env python3
"""Plan, fit and evaluate models with family-grouped INI protocols."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_model_experiment import (fit_model_experiment, freeze_model_experiment,
                                     load_model_experiment_config, plan_model_experiment)


def run(config_path, *, stage='plan', validate_only=False, preview_splits=False):
    """Execute a model stage against separately prepared immutable cohort revisions.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Model-experiment INI file; its bytes are checked before and after execution.
    stage : str, optional
        plan, fit, test or summarize; grouped evaluation enables test/summarize.
    validate_only : bool, optional
        Validate the model plan without freezing output or fitting a model.
    preview_splits : bool, optional
        Preview provisional family assignments without loading dataset payloads; mutually exclusive with validate_only.

    Returns
    -------
    dict
        Stage-specific planning, training, testing or summary metadata.

    Notes
    -----
    Stages may create fresh model artifacts; fitting lazily activates the
    selected learning backend. This entry point never generates phantoms or
    modifies prepared datasets. Invalid stages or changed INI bytes raise ValueError.
    """
    path = Path(config_path).expanduser().resolve()
    original = path.read_bytes()
    config = load_model_experiment_config(path)
    if path.read_bytes() != original:
        raise ValueError('Model input changed while being parsed')
    if preview_splits and validate_only:
        raise ValueError('Choose --preview-splits or --validate-only')
    if 'evaluation' in config:
        from fakect_model_evaluation import (preview_evaluation, plan_evaluation,
            freeze_evaluation, fit_evaluation, test_evaluation, summarize_evaluation)
        if preview_splits:
            result = preview_evaluation(config)
        elif validate_only:
            result = plan_evaluation(config)
            result['base_plan'].pop('samples', None)
        else:
            stages = {'plan': freeze_evaluation, 'fit': fit_evaluation,
                      'test': test_evaluation, 'summarize': summarize_evaluation}
            if stage not in stages:
                raise ValueError('Evaluation stage must be plan, fit, test or summarize')
            result = stages[stage](config, original)
    elif preview_splits:
        raise ValueError('--preview-splits requires an [evaluation] section')
    elif validate_only:
        result = plan_model_experiment(config)
        result.pop('samples', None)
    elif stage == 'plan':
        result = freeze_model_experiment(config, original)
    elif stage == 'fit':
        result = fit_model_experiment(config, original)
    else:
        raise ValueError('Single model stage must be plan or fit; add [evaluation] for test/summarize')
    if path.read_bytes() != original:
        raise ValueError('Model input changed during execution; frozen outputs must be reviewed')
    return result


def main(argv=None):
    """Parse and execute one model experiment command.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments excluding the executable name; None uses sys.argv[1:].

    Returns
    -------
    int
        Zero for successful/ready plans; two for handled failures or a nonready result.

    Notes
    -----
    Prints finite JSON stage results to stdout and handled errors to stderr.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--stage', choices=('plan', 'fit', 'test', 'summarize'), default='plan')
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--validate-only', action='store_true')
    checks.add_argument('--preview-splits', action='store_true',
                        help='Preview provisional families from the INI planning_project without loading data')
    args = parser.parse_args(argv)
    try:
        result = run(args.config, stage=args.stage, validate_only=args.validate_only,
                     preview_splits=args.preview_splits)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result.get('planning_only') or result.get('ready', True) else 2


if __name__ == '__main__':
    raise SystemExit(main())
