"""Fixed classifier budgets with descriptive milestones and no model selection.

The pinned rotation fitter performs every build, update, input read and metric
calculation. This wrapper only intercepts that freshly constructed model's
existing saves, then copies the saved bytes at requested epoch milestones.
Milestones are never scored on held-out data during training or selected by
their eventual performance. No classifier or optimizer state is inherited.
"""
from contextlib import contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch

import fakect_coa_rotation_learning as learning


SCHEMA = 'fakect.coa-fixed-budget-learning/1'


def _milestones(epochs, checkpoint_interval):
    if type(epochs) is not int or epochs <= 0:
        raise ValueError('epochs must be a positive integer')
    if type(checkpoint_interval) is not int or checkpoint_interval <= 0:
        raise ValueError('checkpoint_interval must be a positive integer')
    if checkpoint_interval > epochs or epochs % checkpoint_interval:
        raise ValueError('checkpoint_interval must divide epochs exactly')
    return tuple(range(checkpoint_interval, epochs + 1, checkpoint_interval))


@contextmanager
def _archive_saves(network, output, milestones, expected_initial_weights_sha256):
    """Copy already serialized, optimizer-free models without extra TF work."""
    checkpoint = output / 'classifier.keras'
    state = {'save_calls': 0, 'checkpoints': [], 'models_built': 0}
    build = network.build_classifier

    def build_with_archive(*args, **kwargs):
        state['models_built'] += 1
        if state['models_built'] != 1:
            raise ValueError('Fixed-budget training must build exactly one fresh classifier')
        model = build(*args, **kwargs)
        if (expected_initial_weights_sha256 is not None and
                learning.weight_fingerprint(model) != expected_initial_weights_sha256):
            raise ValueError('Fresh classifier initialization differs from the matched baseline')
        save = model.save

        def save_with_archive(filepath, *save_args, **save_kwargs):
            if Path(filepath) != checkpoint:
                raise ValueError('Unexpected classifier checkpoint path')
            result = save(filepath, *save_args, **save_kwargs)
            state['save_calls'] += 1
            epoch = state['save_calls']
            if epoch in milestones:
                archive = output / 'checkpoints' / f'epoch-{epoch:04d}.keras'
                archive.parent.mkdir(exist_ok=True)
                if archive.exists():
                    raise FileExistsError(str(archive))
                shutil.copyfile(checkpoint, archive)
                state['checkpoints'].append({
                    'epoch': epoch,
                    'checkpoint': str(archive),
                    'checkpoint_sha256': learning.geometry.sha256(archive),
                })
            return result

        # This modifies only this new instance's save method. The graph, losses,
        # optimizer, initialized weights and training functions are untouched.
        model.save = save_with_archive
        return model

    with patch.object(network, 'build_classifier', new=build_with_archive):
        yield state


def fit_classifier(tf, runtime, rows, proposals, output, c, *, epochs=200,
                   checkpoint_interval=50, fixed_budget=True,
                   expected_initial_weights_sha256=None):
    """Run the pinned fresh fitter for exactly ``epochs`` with train-only rows.

    The final epoch is the sole classifier selected for inference. Copies at
    ``checkpoint_interval`` multiples preserve the existing Keras model artifact
    (graph and weights, without training-optimizer state); they are descriptive
    artifacts, not resumable optimizer checkpoints or candidates for selection.
    Returned receipt extends the pinned fitter's initialization, history and
    optimizer-count evidence with archive paths and hashes.
    """
    milestones = _milestones(epochs, checkpoint_interval)
    if (expected_initial_weights_sha256 is not None and
            (not isinstance(expected_initial_weights_sha256, str) or
             len(expected_initial_weights_sha256) != 64 or
             any(char not in '0123456789abcdef' for char in expected_initial_weights_sha256))):
        raise ValueError('expected_initial_weights_sha256 must be a lowercase SHA256 digest')
    if fixed_budget is not True:
        raise ValueError('Only fixed-budget training without checkpoint selection is supported')
    if not rows or any(row.get('split') != 'train' for row in rows):
        raise ValueError('Fixed-budget classifier fitting requires train-only rows; no held-out inputs')
    pairs = learning.matched_pairs(rows)
    output = Path(output)
    import fakect_coa_paired_steps as network

    with _archive_saves(network, output, milestones, expected_initial_weights_sha256) as state:
        receipt = learning.fit_classifier(tf, runtime, rows, proposals, output, c,
                                         epochs=epochs, fixed_budget=True)

    if (state['models_built'] != 1 or state['save_calls'] != epochs or
            tuple(record['epoch'] for record in state['checkpoints']) != milestones):
        raise ValueError('Pinned fitter did not save every fixed-budget epoch as expected')
    if (receipt['selected_epoch'] != epochs or receipt['epochs_completed'] != epochs or
            receipt['validation_sample_ids'] or receipt['validation'] is not None or
            receipt['optimizer_updates'] != epochs * len(pairs)):
        raise ValueError('Fixed-budget receipt violated the train-only final-epoch contract')
    history = learning.geometry.checked_json(output / 'history.json')
    epoch_records = history['epochs']
    if len(epoch_records) != epochs:
        raise ValueError('Fixed-budget history is incomplete')
    for epoch, record in enumerate(epoch_records, 1):
        if (record['epoch'] != epoch or record['selection_bce'] is not None or
                record['validation'] is not None or
                record['optimizer_updates'] != epoch * len(pairs)):
            raise ValueError('Fixed-budget history violated the epoch/update contract')
    for record in state['checkpoints']:
        record['optimizer_updates'] = epoch_records[record['epoch'] - 1]['optimizer_updates']
        if learning.geometry.sha256(record['checkpoint']) != record['checkpoint_sha256']:
            raise ValueError('Archived classifier checkpoint changed during training')
    if state['checkpoints'][-1]['checkpoint_sha256'] != receipt['checkpoint_sha256']:
        raise ValueError('Final archived model differs from the selected final-epoch model')
    receipt = {**receipt,
               'fixed_budget_schema': SCHEMA,
               'checkpoint_interval': checkpoint_interval,
               'model_save_calls': state['save_calls'],
               'budget_checkpoints': state['checkpoints'],
               'checkpoint_selection': 'final fixed-budget epoch',
               'milestones_selection_allowed': False,
               'parent_classifier_checkpoint_reused': False,
               'expected_initial_weights_sha256': expected_initial_weights_sha256}
    learning.geometry.write_json(output / 'receipt.json', receipt)
    return receipt
