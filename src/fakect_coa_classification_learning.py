"""Classification-only instrumentation around the frozen matched-budget fitter.

The model, fresh initialization, paired Adam update, crops and archive policy
remain in the pinned modules. Only the existing severity loss multiplier changes
to zero. The unused output remains in the graph for initialization/reload parity;
it must never be represented as a trained severity prediction.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fakect_coa_budget_learning as budget


SCHEMA = 'fakect.coa-classification-only-learning/1'
learning = budget.learning


def classification_metrics(records):
    """Frozen classification calculations, without reading severity outputs.

    Match the pinned metric definitions and clipping exactly. In particular,
    this function accepts exported predictions whose ``score_prediction`` is
    ``None`` and does not calculate a severity error before discarding it.
    """
    if not records:
        return None
    y = np.array([r['label'] for r in records], float)
    p = np.array([r['probability'] for r in records], float)
    decision = p >= .5
    positive, negative = y == 1, y == 0
    eps = np.finfo(np.float32).eps
    bce = float(np.mean(-y * np.log(np.clip(p, eps, 1 - eps))
                        - (1 - y) * np.log(np.clip(1 - p, eps, 1 - eps))))
    return {'samples': len(records), 'bce': bce,
            'accuracy': float(np.mean(decision == y)),
            'sensitivity': float(np.mean(decision[positive])) if positive.any() else None,
            'specificity': float(np.mean(~decision[negative])) if negative.any() else None,
            'positive_probability_mean': float(p[positive].mean()) if positive.any() else None,
            'negative_probability_mean': float(p[negative].mean()) if negative.any() else None}


def sanitize_predictions(records):
    """Copy inference rows and explicitly withhold the untrained severity output.

    Ground-truth ``score_label`` is retained as annotation metadata. Raw numeric
    auxiliary outputs are allowed only in initialization/parity evidence, never
    in held-out inference exports or severity performance metrics.
    """
    return [{**record, 'score_prediction': None, 'severity_trained': False}
            for record in records]


def _require_zero_weight(c):
    value = c.get('training', {}).get('regression_weight')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
        raise ValueError('Classification-only control requires regression_weight = 0')


def fit_classifier(tf, runtime, rows, proposals, output, c, *, epochs=200,
                   checkpoint_interval=50, fixed_budget=True,
                   expected_initial_weights_sha256=None):
    """Run the identical fresh budget fitter, suppressing severity performance.

    Patching is scoped to this synchronous call and restored on any exception.
    The raw initialization predictions and both-output checkpoint parity are
    preserved. Training histories, receipt metrics and console progress contain
    classification performance only; the severity head is explicitly untrained.
    """
    _require_zero_weight(c)
    with patch.object(learning.legacy, 'metrics', new=classification_metrics):
        receipt = budget.fit_classifier(
            tf, runtime, rows, proposals, output, c, epochs=epochs,
            checkpoint_interval=checkpoint_interval, fixed_budget=fixed_budget,
            expected_initial_weights_sha256=expected_initial_weights_sha256)
    output = Path(output)
    evidence = {'classification_only_schema': SCHEMA,
                'regression_weight': 0., 'severity_trained': False,
                'severity_performance_evaluated': False,
                'auxiliary_output_contract':
                    'Untrained graph output retained only for initialization and reload parity.'}
    receipt = {**receipt, **evidence}
    learning.geometry.write_json(output / 'receipt.json', receipt)
    history = learning.geometry.checked_json(output / 'history.json')
    learning.geometry.write_json(output / 'history.json', {**history, **evidence})
    initialization = learning.geometry.checked_json(output / 'initialization.json')
    learning.geometry.write_json(output / 'initialization.json', {
        **initialization, **evidence,
        'raw_auxiliary_predictions_for_parity_only': True})
    return receipt
