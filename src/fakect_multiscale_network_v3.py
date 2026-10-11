"""Spatially supervised, image-only anatomy locator for the CoA pilot.

A shared fully convolutional overview encoder retains a stride-four spatial
feature map. A center heatmap, explicitly supervised during training, replaces
the flattened coordinate head. Soft-argmax uses the actual SAME-convolution
voxel-center grid. Extents use heatmap-weighted image features. The center never
uses a training-coordinate prior; a train-only prior may initialize extents.

Architecture alone does not prove image dependence or translation robustness.
Those claims require the runner's held-out and image-ablation gates. Supervision
boxes are loss targets only; attenuation and validity are the predictor inputs.
Pin this module and its unchanged r1/v2 dependencies with each experiment.
"""
from functools import lru_cache

import numpy as np

import fakect_multiscale_network as base
import fakect_multiscale_network_v2 as previous


MODEL_SCHEMA = 'fakect.multiscale-coa-network/3'
ENCODER_LEVELS = 2
MINIMUM_EXTENT = .001
COORDINATE_WEIGHT = 10.
_tensorflow = base._tensorflow
compile_classifier = base.compile_classifier
predict_boxes = base.predict_boxes
feature_grid = base.feature_grid
_single_feature_output = previous._single_feature_output


def heatmap_grid(overview_shape):
    """Stride-four feature centers in normalized overview voxel-edge coordinates."""
    shape = np.asarray(base._shape(tuple(overview_shape), 'overview_shape'), dtype=np.int64)
    if np.any(shape % 4):
        raise ValueError('Spatial locator overview dimensions must be divisible by 4')
    feature_shape, offset, stride = feature_grid(shape, ENCODER_LEVELS)
    axes = [(np.arange(n, dtype=np.float32) * stride + start + .5) / length
            for n, start, length in zip(feature_shape, offset, shape)]
    return np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).astype(np.float32)


@lru_cache(maxsize=1)
def _components():
    tf = _tensorflow()
    register = tf.keras.utils.register_keras_serializable

    @register(package='FakeCTv3')
    class GridValidity(tf.keras.layers.Layer):
        def __init__(self, overview_shape, **kwargs):
            super().__init__(**kwargs)
            self.overview_shape = tuple(int(v) for v in overview_shape)
            heatmap_grid(self.overview_shape)

        def call(self, image):
            # For dimensions divisible by four, two 3x3 SAME stride-two
            # convolutions have native voxel centers 3,7,11,... .
            return tf.cast(image[:, 3::4, 3::4, 3::4, 1:2] > 0, self.compute_dtype)

        def get_config(self):
            return {**super().get_config(), 'overview_shape': self.overview_shape}

    @register(package='FakeCTv3')
    class MaskHeatmap(tf.keras.layers.Layer):
        def call(self, inputs):
            logits, valid = inputs
            return tf.where(valid > 0, logits, tf.cast(-10000., logits.dtype))

    @register(package='FakeCTv3')
    class SpatialExpectation(tf.keras.layers.Layer):
        def __init__(self, overview_shape, **kwargs):
            super().__init__(**kwargs)
            self.overview_shape = tuple(int(v) for v in overview_shape)
            self.coordinates = heatmap_grid(self.overview_shape).reshape(-1, 3).tolist()

        def call(self, logits):
            probability = tf.nn.softmax(tf.reshape(logits, [tf.shape(logits)[0], -1]), axis=-1)
            return tf.linalg.matmul(probability, tf.constant(self.coordinates, dtype=logits.dtype))

        def get_config(self):
            return {**super().get_config(), 'overview_shape': self.overview_shape}

    @register(package='FakeCTv3')
    class AttentionPool(tf.keras.layers.Layer):
        def call(self, inputs):
            features, logits = inputs
            probability = tf.nn.softmax(tf.reshape(logits, [tf.shape(logits)[0], -1]), axis=-1)
            flattened = tf.reshape(features, [tf.shape(features)[0], -1, tf.shape(features)[-1]])
            return tf.reduce_sum(flattened * probability[..., None], axis=1)

    @register(package='FakeCTv3')
    class SpatialBox(tf.keras.layers.Layer):
        def call(self, inputs):
            center, extent_logits = inputs
            extent = MINIMUM_EXTENT + (1 - MINIMUM_EXTENT) * tf.sigmoid(extent_logits)
            # Keep center unchanged; only trim a box extent that would cross
            # the overview boundary. Centers already lie inside the grid.
            maximum = 2 * tf.minimum(center, 1 - center)
            extent = tf.minimum(extent, maximum)
            return tf.concat([center, extent], axis=-1)

    return {'GridValidity': GridValidity, 'MaskHeatmap': MaskHeatmap,
            'SpatialExpectation': SpatialExpectation, 'AttentionPool': AttentionPool,
            'SpatialBox': SpatialBox}


def _spatial_encoder(shape, base_filters):
    tf = _tensorflow()
    layers = tf.keras.layers
    image = layers.Input(shape + (2,), name='overview_encoder_image_valid')
    x = image
    for level in range(ENCODER_LEVELS):
        x = layers.Conv3D(base_filters * 2 ** level, 3, strides=2, padding='same',
            use_bias=False, kernel_initializer='he_normal', name=f'overview_encoder_down_{level}')(x)
        # Preserve local attenuation amplitude in the shallow path. With very
        # few channels, per-voxel LayerNorm can reduce a contrast change almost
        # entirely to channel ordering and remove the spatial image signal.
        x = layers.Activation('relu', name=f'overview_encoder_relu_{level}')(x)
    shallow = x
    filters = base_filters * 4
    x = layers.Conv3D(filters, 1, padding='same', name='overview_context_projection')(x)
    # The grid stays at stride four. Dilation supplies broad context without a
    # fourth spatial downsampling or a large flattened coordinate regressor.
    for index, dilation in enumerate((1, 2, 4, 8)):
        residual = layers.Conv3D(filters, 3, dilation_rate=dilation, padding='same',
            use_bias=False, kernel_initializer='he_normal', name=f'overview_context_{index}')(x)
        residual = layers.LayerNormalization(epsilon=1e-5, name=f'overview_context_norm_{index}')(residual)
        residual = layers.Activation('relu', name=f'overview_context_relu_{index}')(residual)
        x = layers.Add(name=f'overview_context_add_{index}')([x, residual])
        x = layers.Activation('relu', name=f'overview_context_out_{index}')(x)
    x = layers.Concatenate(name='overview_shallow_context')([shallow, x])
    return tf.keras.Model(image, x, name='overview_encoder')


def build_locator(overview_shape, base_filters=4, seed=42, *, prior_box=None):
    """Build an image-to-box predictor with spatial center supervision.

    ``prior_box`` must be derived solely from training originals by the runner.
    Its center is deliberately unused; only its three extents initialize the
    extent head. The generic extent fallback is independent of all data.
    """
    shape = base._settings(overview_shape, base_filters, seed)
    heatmap_grid(shape)
    prior_extent_logits = previous.prior_parameters(prior_box)[3:]
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    components, layers = _components(), tf.keras.layers
    image = layers.Input(shape + (2,), name='overview')
    encoder = _spatial_encoder(shape, base_filters)
    features = _single_feature_output(encoder(image), 'overview_encoder')
    raw = layers.Conv3D(1, 1, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', name='locator_heatmap_raw')(features)
    valid = components['GridValidity'](shape, name='locator_heatmap_valid')(image)
    logits = components['MaskHeatmap'](name='locator_heatmap_logits')([raw, valid])
    center = components['SpatialExpectation'](shape, name='locator_center')(logits)
    pooled = components['AttentionPool'](name='locator_spatial_pool')([features, logits])
    pooled = layers.LayerNormalization(epsilon=1e-5, name='locator_extent_norm')(pooled)
    extent_logits = layers.Dense(3, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=.001),
        bias_initializer=tf.keras.initializers.Constant(prior_extent_logits), name='locator_extent_logits')(pooled)
    box = components['SpatialBox'](name='anatomy_box')([center, extent_logits])
    return tf.keras.Model({'overview': image}, box, name='fakect_multiscale_locator_v3')


def locator_outputs(locator):
    """Shared training view; construct once before making a fixed-signature step."""
    tf = _tensorflow()
    outputs = {'box': _single_feature_output(locator.output, 'locator')}
    for key in ('heatmap_logits', 'heatmap_valid', 'extent_logits'):
        outputs[key] = _single_feature_output(locator.get_layer('locator_' + key).output, key)
    return tf.keras.Model(locator.input, outputs, name='fakect_multiscale_locator_v3_training_view')


def target_heatmap(y_true, overview_shape, validity=None):
    """Normalized Gaussian supervision; no ground-truth image is a model input."""
    tf = _tensorflow()
    truth = tf.cast(y_true, tf.float32)
    coordinates = tf.constant(heatmap_grid(tuple(overview_shape)).reshape(-1, 3), tf.float32)
    minimum_sigma = tf.constant(3. / np.asarray(overview_shape, dtype=np.float32))
    sigma = tf.maximum(truth[..., 3:] * .2, minimum_sigma)
    distance = (coordinates[None] - truth[:, None, :3]) / sigma[:, None]
    logits = -.5 * tf.reduce_sum(distance ** 2, axis=-1)
    if validity is not None:
        mask = tf.reshape(validity, [tf.shape(validity)[0], -1]) > 0
        tf.debugging.assert_positive(tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1),
                                     message='Heatmap needs at least one valid spatial cell')
        logits = tf.where(mask, logits, -10000.)
    return tf.nn.softmax(logits, axis=-1)


def locator_loss(y_true, outputs):
    """Per-example spatial heatmap KL + extent KL + normalized coordinate MSE.

    The spatial cross-entropy is evaluated from raw logits; a confidently wrong
    distant location therefore retains a corrective gradient. The Gaussian's
    width is at least .75 feature cells and can grow with the target box size.
    """
    tf = _tensorflow()
    truth, prediction = tf.cast(y_true, tf.float32), tf.cast(outputs['box'], tf.float32)
    heatmap_logits = tf.cast(outputs['heatmap_logits'], tf.float32)
    spatial_shape = heatmap_logits.shape[1:4]
    if any(n is None for n in spatial_shape):
        raise ValueError('The locator heatmap requires fixed spatial dimensions')
    overview_shape = tuple(int(n) * 4 for n in spatial_shape)
    target = target_heatmap(truth, overview_shape, outputs['heatmap_valid'])
    flattened = tf.reshape(heatmap_logits, [tf.shape(heatmap_logits)[0], -1])
    heatmap_ce = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=flattened)
    heatmap_entropy = -tf.reduce_sum(tf.math.xlogy(target, target), axis=-1)
    heatmap_kl = tf.maximum(heatmap_ce - heatmap_entropy, 0.)
    extent_target = tf.clip_by_value((truth[..., 3:] - MINIMUM_EXTENT) / (1 - MINIMUM_EXTENT), 0., 1.)
    extent_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=extent_target,
        logits=tf.cast(outputs['extent_logits'], tf.float32))
    extent_entropy = -(tf.math.xlogy(extent_target, extent_target)
                      + tf.math.xlogy(1 - extent_target, 1 - extent_target))
    extent_kl = tf.reduce_mean(tf.maximum(extent_ce - extent_entropy, 0.), axis=-1)
    coordinates = tf.reduce_mean(tf.square(truth - prediction), axis=-1)
    return heatmap_kl + extent_kl + COORDINATE_WEIGHT * coordinates


def load_model(path, *, compile=False):
    """Register both model generations before safe Keras checkpoint loading."""
    tf = _tensorflow()
    base._components()
    _components()
    return tf.keras.models.load_model(path, compile=compile, safe_mode=True)


def build_classifier(overview_shape, detail_shape, num_proposals=2, base_filters=4,
                     seed=42, *, locator):
    """Freeze/reuse locator features and fuse aligned coarse/fine image features.

    Inputs are overview, proposal_boxes and detail_0 ... detail_N. The proposal
    boxes describe the actual detail crops, including padded parts, in the
    overview FOV. They must be produced from images by the frozen locator and a
    deterministic crop policy, never from edit/condition ground truth.
    """
    shape = base._settings(overview_shape, base_filters, seed)
    heatmap_grid(shape)
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
    if (len(encoder.outputs) != 1
            or tuple(encoder.outputs[0].shape[1:4]) != tuple(n // 4 for n in shape)):
        raise ValueError('Classifier requires the v3 stride-four overview feature grid')
    encoder.trainable = False
    image = layers.Input(shape + (2,), name='overview')
    boxes = layers.Input((num_proposals, 6), name='proposal_boxes')
    features = _single_feature_output(encoder(image, training=False), 'overview_encoder')
    samples = AlignedFeatureSamples(shape, levels=ENCODER_LEVELS, name='aligned_coarse_samples')([features, boxes])
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
                          name='fakect_multiscale_classifier_v3')

