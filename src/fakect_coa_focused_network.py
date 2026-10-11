"""Fine-image-only CoA classifier preserving a stride-four spatial feature map.

Each input is ``detail_N`` with normalized attenuation and acquisition validity
channels. The frozen locator chooses those image crops outside this module.
No overview, crop coordinates, anatomy masks or edit labels enter this network.
TensorFlow imports/registration remain lazy so planning is CPU-only.

At the production 216x192x192 shape, the final spatial map is 54x48x48 (2.4 mm
pitch for 0.6 mm samples). A trainable local evidence/attention path and ordered
6x6x6 spatial bins retain focal information before the scalar heads. Masking is
applied before convolution and to both aggregation paths. Zero-validity padding
cannot contribute attenuation values or become a pooling maximum.
"""
from functools import lru_cache
import math

import numpy as np

from fakect_tensorflow_steps import StableSteps, _apply, _optimizer, _positive, _regularization


MODEL_SCHEMA = 'fakect.coa-focused-network/1'
ENCODER_STRIDE = 4
SPATIAL_BINS = 6


def _tensorflow():
    import tensorflow as tf
    return tf


def _settings(detail_shape, num_proposals, base_filters, seed):
    if (not isinstance(detail_shape, (tuple, list)) or len(detail_shape) != 3
            or any(isinstance(n, bool) or not isinstance(n, (int, np.integer))
                   or n < 8 or n % 4 for n in detail_shape)):
        raise ValueError('detail_shape needs three integer dimensions >=8 and divisible by4')
    shape = tuple(int(n) for n in detail_shape)
    if math.prod(shape) > 8_388_608:
        raise ValueError('detail_shape exceeds the input voxel budget')
    if type(num_proposals) is not int or not 1 <= num_proposals <= 4:
        raise ValueError('num_proposals must be an integer in [1,4]')
    if type(base_filters) is not int or not 2 <= base_filters <= 8:
        raise ValueError('base_filters must be an integer in [2,8]')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return shape


def feature_geometry(detail_shape):
    shape = _settings(detail_shape, 2, 4, 42)
    fine = tuple(n // ENCODER_STRIDE for n in shape)
    windows = tuple(int(math.ceil(n / SPATIAL_BINS)) for n in fine)
    return {'encoder_stride': ENCODER_STRIDE, 'feature_shape_kji': list(fine),
            'spatial_bin_shape_kji': [int(math.ceil(n / p)) for n, p in zip(fine, windows)],
            'spatial_bin_window_kji': list(windows)}


@lru_cache(maxsize=1)
def _components():
    tf = _tensorflow()
    register = tf.keras.utils.register_keras_serializable

    @register(package='FakeCTFocused')
    class SanitizeImage(tf.keras.layers.Layer):
        def call(self, image):
            validity = tf.clip_by_value(image[..., 1:2], 0., 1.)
            attenuation = tf.where(validity > 0, image[..., 0:1], tf.zeros_like(validity))
            return tf.concat([attenuation, validity], axis=-1)

    @register(package='FakeCTFocused')
    class GridValidity(tf.keras.layers.Layer):
        def __init__(self, stride=4, **kwargs):
            super().__init__(**kwargs)
            self.stride = int(stride)

        def call(self, image):
            return tf.nn.avg_pool3d(image[..., 1:2], self.stride, self.stride, 'SAME')

        def get_config(self):
            return {**super().get_config(), 'stride': self.stride}

    @register(package='FakeCTFocused')
    class MaskFeatures(tf.keras.layers.Layer):
        def call(self, inputs):
            features, valid = inputs
            return tf.where(valid > 0, features, tf.zeros_like(features))

    @register(package='FakeCTFocused')
    class AmplitudePath(tf.keras.layers.Layer):
        def call(self, image):
            valid = image[..., 1:2]
            numerator = tf.nn.avg_pool3d(image[..., 0:1] * valid, 4, 4, 'SAME')
            denominator = tf.nn.avg_pool3d(valid, 4, 4, 'SAME')
            return tf.math.divide_no_nan(numerator, denominator)

    @register(package='FakeCTFocused')
    class ChannelRMS(tf.keras.layers.Layer):
        def call(self, features):
            return tf.sqrt(tf.reduce_mean(tf.square(features), axis=-1, keepdims=True))

    @register(package='FakeCTFocused')
    class ValidAttention(tf.keras.layers.Layer):
        def call(self, inputs):
            logits, valid = inputs
            mask = tf.reshape(valid, [tf.shape(valid)[0], -1])
            values = tf.reshape(logits, [tf.shape(logits)[0], -1])
            values = tf.where(mask > 0, values, tf.cast(-10000., values.dtype))
            weights = tf.nn.softmax(values, axis=-1) * mask
            weights = tf.math.divide_no_nan(weights, tf.reduce_sum(weights, axis=-1, keepdims=True))
            return tf.reshape(weights, tf.shape(valid))

    @register(package='FakeCTFocused')
    class SpatialEvidence(tf.keras.layers.Layer):
        """Attention-weighted feature/evidence vectors and spatial moments."""
        def __init__(self, feature_shape, **kwargs):
            super().__init__(**kwargs)
            self.feature_shape = tuple(int(n) for n in feature_shape)
            axes = [(np.arange(n, dtype=np.float32) + .5) / n for n in self.feature_shape]
            self.coordinates = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)

        def call(self, inputs):
            features, evidence, attention, valid = inputs
            weights = tf.reshape(attention, [tf.shape(attention)[0], -1])
            flattened = tf.reshape(features, [tf.shape(features)[0], -1, tf.shape(features)[-1]])
            logits = tf.reshape(evidence, [tf.shape(evidence)[0], -1])
            mask = tf.reshape(valid, [tf.shape(valid)[0], -1])
            pooled = tf.reduce_sum(flattened * weights[..., None], axis=1)
            local = tf.reduce_sum(logits * weights, axis=1, keepdims=True)
            coords = tf.constant(self.coordinates, dtype=features.dtype)
            center = tf.linalg.matmul(weights, coords)
            second_moment = tf.linalg.matmul(weights, tf.square(coords))
            variance = tf.maximum(second_moment - tf.square(center), 0.)
            # Clipped logarithms keep gradients finite at zero-validity cells.
            entropy = -tf.reduce_sum(weights * tf.math.log(tf.maximum(weights, 1e-30)),
                                     axis=-1, keepdims=True)
            mass = tf.reduce_sum(mask, axis=-1, keepdims=True)
            masked_logits = tf.where(mask > 0, logits, tf.cast(-10000., logits.dtype))
            maximum = tf.reduce_max(masked_logits, axis=-1, keepdims=True)
            maximum = tf.where(mass > 0, maximum, tf.zeros_like(maximum))
            return tf.concat([pooled, local, maximum, center, variance, entropy], axis=-1)

        def get_config(self):
            return {**super().get_config(), 'feature_shape': self.feature_shape}

    @register(package='FakeCTFocused')
    class SpatialBins(tf.keras.layers.Layer):
        """Ordered local means/maxima; invalid cells never become extrema."""
        def __init__(self, feature_shape, bins=6, **kwargs):
            super().__init__(**kwargs)
            self.feature_shape = tuple(int(n) for n in feature_shape)
            self.bins = int(bins)
            self.window = tuple(int(math.ceil(n / self.bins)) for n in self.feature_shape)

        def call(self, inputs):
            features, valid = inputs
            kernel = [1, *self.window, 1]
            mass = tf.nn.avg_pool3d(valid, kernel, kernel, 'SAME')
            mean = tf.math.divide_no_nan(tf.nn.avg_pool3d(features * valid, kernel, kernel, 'SAME'), mass)
            # Finite min is safe for float32 attenuation-derived features, and
            # empty bins are explicitly zero rather than retaining the sentinel.
            masked = tf.where(valid > 0, features, tf.cast(-1e30, features.dtype))
            maximum = tf.nn.max_pool3d(masked, kernel, kernel, 'SAME')
            maximum = tf.where(mass > 0, maximum, tf.zeros_like(maximum))
            return tf.concat([mean, maximum], axis=-1)

        def get_config(self):
            return {**super().get_config(), 'feature_shape': self.feature_shape, 'bins': self.bins}

    return {c.__name__: c for c in (SanitizeImage, GridValidity, MaskFeatures, AmplitudePath,
            ChannelRMS, ValidAttention, SpatialEvidence, SpatialBins)}


def _encoder(shape, base_filters):
    tf = _tensorflow()
    layers, components = tf.keras.layers, _components()
    image = layers.Input(shape + (2,), name='focused_attenuation_validity')
    clean = components['SanitizeImage'](name='clean_image')(image)
    valid2 = components['GridValidity'](2, name='valid_down2')(clean)
    valid4 = components['GridValidity'](4, name='valid_down4')(clean)
    down2 = layers.Conv3D(base_filters, 3, strides=2, padding='same', kernel_initializer='he_normal',
                          name='fine_down2_conv')(clean)
    down2 = layers.LeakyReLU(negative_slope=.1, name='fine_down2_activation')(down2)
    down2 = components['MaskFeatures'](name='fine_down2')([down2, valid2])
    down4 = layers.Conv3D(base_filters * 2, 3, strides=2, padding='same', kernel_initializer='he_normal',
                          name='fine_down4_conv')(down2)
    down4 = layers.LeakyReLU(negative_slope=.1, name='fine_down4_activation')(down4)
    down4 = components['MaskFeatures'](name='fine_down4')([down4, valid4])
    context = layers.Conv3D(base_filters * 4, 1, padding='same', kernel_initializer='he_normal',
                            name='context_projection')(down4)
    for index, dilation in enumerate((1, 2, 4)):
        residual = layers.Conv3D(base_filters * 4, 3, dilation_rate=dilation, padding='same',
                                kernel_initializer='he_normal', name=f'context_conv_{index}')(context)
        residual = layers.LeakyReLU(negative_slope=.1, name=f'context_activation_{index}')(residual)
        residual = layers.Rescaling(.25, name=f'context_residual_scale_{index}')(residual)
        context = layers.Add(name=f'context_add_{index}')([context, residual])
        context = components['MaskFeatures'](name=f'context_valid_{index}')([context, valid4])
    amplitude = components['AmplitudePath'](name='amplitude_path')(clean)
    features = layers.Concatenate(name='fine_amplitude_features')([down4, context, amplitude])
    evidence = layers.Conv3D(1, 1, name='local_evidence_logit')(features)
    evidence = components['MaskFeatures'](name='local_evidence_valid')([evidence, valid4])
    attention_logits = layers.Conv3D(1, 1, name='local_attention_logit')(features)
    attention = components['ValidAttention'](name='local_attention')([attention_logits, valid4])
    fine_shape = tuple(n // 4 for n in shape)
    attended = components['SpatialEvidence'](fine_shape, name='attended_evidence')(
        [features, evidence, attention, valid4])
    local_channels = layers.Conv3D(base_filters, 1, name='spatial_bin_projection')(features)
    spatial_channels = layers.Concatenate(name='spatial_bin_channels')([local_channels, evidence, amplitude])
    grid = components['SpatialBins'](fine_shape, bins=SPATIAL_BINS, name='spatial_bins')([spatial_channels, valid4])
    flat_grid = layers.Flatten(name='ordered_spatial_bins')(grid)
    pooled = layers.Concatenate(name='retained_spatial_evidence')([attended, flat_grid])
    outputs = {'amplitude': amplitude, 'down2': components['ChannelRMS'](name='down2_rms')(down2),
               'down4': components['ChannelRMS'](name='down4_rms')(down4),
               'context': components['ChannelRMS'](name='context_rms')(context),
               'local_logit': evidence, 'attention': attention,
               'spatial_grid': components['ChannelRMS'](name='spatial_grid_rms')(grid), 'pooled': pooled}
    return tf.keras.Model(image, outputs, name='focused_detail_encoder')


def build_classifier(detail_shape, num_proposals=2, base_filters=4, seed=42):
    """Build a bounded classifier using only high-resolution attenuation crops."""
    shape = _settings(detail_shape, num_proposals, base_filters, seed)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    encoder = _encoder(shape, base_filters)
    inputs, pooled = {}, []
    for index in range(num_proposals):
        key = f'detail_{index}'
        inputs[key] = tf.keras.layers.Input(shape + (2,), name=key)
        pooled.append(encoder(inputs[key])['pooled'])
    combined = pooled[0] if len(pooled) == 1 else tf.keras.layers.Concatenate(name='ordered_detail_blocks')(pooled)
    hidden = tf.keras.layers.Dense(32, name='classification_hidden')(combined)
    hidden = tf.keras.layers.LeakyReLU(negative_slope=.1, name='classification_activation')(hidden)
    outputs = {'coa_logit': tf.keras.layers.Dense(1, name='coa_logit')(hidden),
               'local_volume_loss': tf.keras.layers.Dense(1, activation='sigmoid', name='local_volume_loss')(hidden)}
    return tf.keras.Model(inputs, outputs, name='fakect_coa_focused_classifier')


def feature_probe(model):
    """Shared-weight inference view with reduced per-stage maps for each block.

    Keys ``detail_N_{amplitude,down2,down4,context,local_logit,attention,
    spatial_grid,pooled}`` expose sensitivity without retaining full feature
    channels. Construction does not change the classifier's weights.
    """
    tf = _tensorflow()
    encoder = model.get_layer('focused_detail_encoder')
    outputs = {}
    inputs = model.input
    if not isinstance(inputs, dict) or any(not name.startswith('detail_') for name in inputs):
        raise ValueError('feature_probe needs a focused detail-only classifier')
    for name, image in inputs.items():
        outputs.update({name + '_' + stage: value for stage, value in encoder(image).items()})
    return tf.keras.Model(inputs, outputs, name='fakect_coa_focused_feature_probe')


def load_model(path, *, compile=False):
    _components()
    return _tensorflow().keras.models.load_model(path, compile=compile, safe_mode=True)


def make_steps(model, detail_shape, num_proposals=2, learning_rate=.001,
               clipnorm=1., regression_weight=1.):
    """Fixed microbatch1 BCE + Huber(.1) steps for detail-only inputs."""
    shape = _settings(detail_shape, num_proposals, 4, 42)
    learning_rate = _positive(learning_rate, 'learning_rate')
    clipnorm = _positive(clipnorm, 'clipnorm')
    regression_weight = _positive(regression_weight, 'regression_weight', allow_zero=True)
    tf = _tensorflow()
    optimizer = _optimizer(tf, model, learning_rate)
    specs = {f'detail_{index}': tf.TensorSpec((1,) + shape + (2,), tf.float32)
             for index in range(num_proposals)}
    if set(model.input) != set(specs):
        raise ValueError('Model inputs must exactly match detail-only signatures')
    targets_spec = {name: tf.TensorSpec((1, 1), tf.float32) for name in ('coa_logit', 'local_volume_loss')}

    @tf.function(input_signature=[specs, targets_spec], autograph=False, jit_compile=False)
    def train(inputs, targets):
        inputs = tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite fine image'), inputs)
        targets = tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite focused target'), targets)
        with tf.GradientTape() as tape:
            output = model(inputs, training=True)
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets['coa_logit'], logits=output['coa_logit']))
            difference = output['local_volume_loss'] - targets['local_volume_loss']
            error = tf.abs(difference)
            huber = tf.reduce_mean(tf.where(error <= .1, .5 * difference ** 2, .1 * (error - .05)))
            loss = tf.debugging.check_numerics(bce + regression_weight * huber + _regularization(tf, model),
                                              'Nonfinite focused classifier loss')
        norm = _apply(tf, model, optimizer, tape, loss, clipnorm)
        return {'loss': loss, 'binary_crossentropy': bce, 'huber': huber, 'gradient_global_norm': norm}

    @tf.function(input_signature=[specs], autograph=False, jit_compile=False)
    def predict(inputs):
        inputs = tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite fine image'), inputs)
        return tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite focused output'),
                                     model(inputs, training=False))

    return StableSteps(train, predict, optimizer)
