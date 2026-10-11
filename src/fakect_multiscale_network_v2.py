"""Stable locator revision for the multiresolution CoA engineering pilot.

The first pilot's predicted boxes did not intersect their targets. Its IoU
penalty was therefore constant, while small post-sigmoid Huber gradients
struggled to recover a saturated superior/inferior coordinate. This revision
supervises the *raw* bounded-box parameters with stable fractional-target
cross-entropy and an explicit normalized-coordinate error.

The runner must derive ``prior_box`` exclusively from TRAINING baselines and
record those sample IDs. It must compare the learned locator against that
constant prior. A good prior or training fit does not establish image-based
localization on unseen anatomy. Predictor inputs remain attenuation/validity;
no per-example target boxes, labels or edit coordinates are input channels.

The frozen spatial encoder and aligned feature sampling reuse the unchanged
r1 implementation. The classifier graph preserves its architecture and handles
restored Keras models that return a one-element feature list. Pin this module AND fakect_multiscale_network
when locking experiments. TensorFlow imports and GPU configuration remain lazy
and controlled by the runner.
"""
import numpy as np

import fakect_multiscale_network as base


MODEL_SCHEMA = 'fakect.multiscale-coa-network/2'
MINIMUM_EXTENT = .001
COORDINATE_WEIGHT = 10.

# Preserve the classifier architecture to isolate the locator correction.
_tensorflow = base._tensorflow
compile_classifier = base.compile_classifier
predict_boxes = base.predict_boxes
load_model = base.load_model
feature_grid = base.feature_grid


def prior_parameters(prior_box=None):
    """Validate a caller-supplied train-only prior and return inverse logits.

    The generic fallback is data-independent. Deriving an empirical prior or
    choosing the training partition is deliberately not done inside a model.
    """
    if prior_box is None:
        prior_box = [.5, .5, .5, .15, .15, .15]
    box = np.asarray(prior_box, dtype=np.float64)
    if (box.shape != (6,) or not np.isfinite(box).all()
            or np.any(box[3:] < MINIMUM_EXTENT) or np.any(box[3:] > 1)
            or np.any(box[:3] - box[3:] / 2 < -1e-7)
            or np.any(box[:3] + box[3:] / 2 > 1 + 1e-7)):
        raise ValueError('prior_box needs valid normalized center_kji + extent_kji')
    extent = box[3:]
    center_probability = np.divide(box[:3] - extent / 2, 1 - extent,
        out=np.full(3, .5), where=(1 - extent) > 1e-7)
    extent_probability = (extent - MINIMUM_EXTENT) / (1 - MINIMUM_EXTENT)
    probability = np.clip(np.concatenate([center_probability, extent_probability]), 1e-6, 1 - 1e-6)
    return (np.log(probability) - np.log1p(-probability)).astype(np.float32)


def _box_probabilities(box):
    """Inverse fractional targets for the r1 BoundedBox parameterization."""
    tf = _tensorflow()
    box = tf.cast(box, tf.float32)
    extent = box[..., 3:]
    center_probability = tf.where(1 - extent > 1e-7,
        tf.math.divide_no_nan(box[..., :3] - extent / 2, 1 - extent), .5)
    extent_probability = (extent - MINIMUM_EXTENT) / (1 - MINIMUM_EXTENT)
    return tf.clip_by_value(tf.concat([center_probability, extent_probability], axis=-1), 0., 1.)


def locator_loss(y_true, y_pred, *, logits=None):
    """Per-example box-parameter KL plus 10x mean squared coordinate error.

    TRAINING MUST supply the raw ``logits`` from ``locator_outputs``. Stable
    cross-entropy then retains corrective gradients even when the sigmoid has
    rounded to zero or one. The optional inverse-box fallback is convenient
    for ordinary numerical evaluation, but cannot recover saturated logits.
    There is no IoU term and no gradient discontinuity when boxes are disjoint.
    """
    tf = _tensorflow()
    truth, prediction = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    target_probability = _box_probabilities(truth)
    if logits is None:
        probability = tf.clip_by_value(_box_probabilities(prediction), 1e-7, 1 - 1e-7)
        logits = tf.math.log(probability) - tf.math.log1p(-probability)
    logits = tf.cast(logits, tf.float32)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=target_probability, logits=logits)
    target_entropy = -(tf.math.xlogy(target_probability, target_probability)
                       + tf.math.xlogy(1 - target_probability, 1 - target_probability))
    # Subtract the fixed target entropy so a correct box has near-zero loss.
    # This subtraction changes neither optimization gradients nor its optimum.
    divergence = tf.maximum(cross_entropy - target_entropy, 0.)
    coordinate_error = tf.reduce_mean(tf.square(truth - prediction), axis=-1)
    return tf.reduce_mean(divergence, axis=-1) + COORDINATE_WEIGHT * coordinate_error


def build_locator(overview_shape, base_filters=4, seed=42, *, prior_box=None):
    """Build an image-only locator initialized near a supplied training prior."""
    shape = base._settings(overview_shape, base_filters, seed)
    prior_logits = prior_parameters(prior_box)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    BoundedBox, _, _ = base._components()
    layers = tf.keras.layers
    image = layers.Input(shape + (2,), name='overview')
    encoder = base._encoder(shape, base_filters, 'overview_encoder')
    features = encoder(image)
    x = layers.Flatten(name='locator_spatial_features')(features)
    x = layers.LayerNormalization(epsilon=1e-5, name='locator_spatial_norm')(x)
    # A bounded hidden head and small final weights avoid sending an untrained
    # large flattened feature vector directly into saturated box sigmoids.
    x = layers.Dense(64, activation='tanh', name='locator_hidden')(x)
    logits = layers.Dense(6, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=.001),
        bias_initializer=tf.keras.initializers.Constant(prior_logits), name='locator_raw_box')(x)
    box = BoundedBox(minimum_extent=MINIMUM_EXTENT, name='anatomy_box')(logits)
    return tf.keras.Model({'overview': image}, box, name='fakect_multiscale_locator_v2')


def locator_outputs(locator):
    """One shared forward pass for a physical box and its unbounded logits.

    Returns a Keras view with the same input and weights. Outputs are exactly
    ``{'box': [B,6], 'box_logits': [B,6]}``. Construct this view once before the
    runner's fixed-signature tf.function; do not recreate it per training step.
    """
    tf = _tensorflow()
    return tf.keras.Model(locator.input,
        {'box': _single_feature_output(locator.output, 'locator'),
         'box_logits': _single_feature_output(locator.get_layer('locator_raw_box').output, 'locator_raw_box')},
        name='fakect_multiscale_locator_v2_training_view')


def _single_feature_output(value, name):
    """Normalize Keras 3.4 restored nested models without copying their weights."""
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f'{name} must return exactly one feature tensor')
        value = value[0]
    return value


def build_classifier(overview_shape, detail_shape, num_proposals=2, base_filters=4,
                     seed=42, *, locator):
    """Freeze/reuse locator features and fuse aligned coarse/fine image features.

    Inputs are overview, proposal_boxes and detail_0 ... detail_N. The proposal
    boxes describe the actual detail crops, including padded parts, in the
    overview FOV. They must be produced from images by the frozen locator and a
    deterministic crop policy, never from edit/condition ground truth.
    """
    shape = base._settings(overview_shape, base_filters, seed)
    detail_shape = base._shape(detail_shape, 'detail_shape')
    if isinstance(num_proposals, bool) or not isinstance(num_proposals, int) or not 1 <= num_proposals <= 4:
        raise ValueError('num_proposals must be an integer in [1,4]')
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    _, AlignedFeatureSamples, ProposalIndex = base._components()
    layers = tf.keras.layers
    encoder = locator.get_layer('overview_encoder')
    if tuple(encoder.input_shape[1:]) != shape + (2,):
        raise ValueError('Locator overview shape differs from classifier overview shape')
    encoder.trainable = False
    image = layers.Input(shape + (2,), name='overview')
    boxes = layers.Input((num_proposals, 6), name='proposal_boxes')
    features = _single_feature_output(encoder(image, training=False), 'overview_encoder')
    samples = AlignedFeatureSamples(shape, name='aligned_coarse_samples')([features, boxes])
    fine_encoder = base._encoder(detail_shape, base_filters, 'detail_encoder')
    fine_mean = layers.GlobalAveragePooling3D(name='detail_mean')
    fine_max = layers.GlobalMaxPooling3D(name='detail_max')
    coarse_projection = layers.Dense(32, activation='relu', name='coarse_projection')
    proposal_projection = layers.Dense(32, activation='relu', name='proposal_projection')
    inputs = {'overview': image, 'proposal_boxes': boxes}
    fused = []
    for index in range(num_proposals):
        detail = layers.Input(detail_shape + (2,), name=f'detail_{index}')
        inputs[f'detail_{index}'] = detail
        coarse = ProposalIndex(index, name=f'coarse_proposal_{index}')(samples)
        coarse = coarse_projection(coarse)
        local_features = _single_feature_output(fine_encoder(detail), 'detail_encoder')
        # Mean summarizes the crop; max preserves sparse feature responses that
        # averaging alone can dilute. Clinical focal-lesion sensitivity still
        # requires the held-out study; pooling is an engineering design choice.
        local = layers.Concatenate(name=f'detail_summary_{index}')([
            fine_mean(local_features), fine_max(local_features)])
        coordinates = ProposalIndex(index, name=f'proposal_coordinates_{index}')(boxes)
        combined = layers.Concatenate(name=f'proposal_fusion_{index}')([coarse, local, coordinates])
        fused.append(proposal_projection(combined))
    x = fused[0] if len(fused) == 1 else layers.Concatenate(name='ordered_proposals')(fused)
    x = layers.Dense(32, activation='relu', name='classification_hidden')(x)
    logit = layers.Dense(1, name='coa_logit')(x)
    score = layers.Dense(1, activation='sigmoid', name='local_volume_loss')(x)
    return tf.keras.Model(inputs, {'coa_logit': logit, 'local_volume_loss': score},
                          name='fakect_multiscale_classifier')

