"""Compact image-only TensorFlow components for the multiresolution CoA pilot.

Importing this module does not import TensorFlow. Images have two channels:
normalized attenuation and acquisition validity. No anatomical or edit labels
are predictor inputs. Boxes are center_kji + extent_kji in the overview's
voxel-edge field of view: voxel k has coordinate (k + .5) / shape_k.

The anatomy locator is trained first. The classifier then reuses that exact
frozen overview encoder and samples its spatial features at the supplied,
image-predicted detail crops. Callers own physical image sampling and provenance;
this module never chooses crops from supervision masks.
"""
from functools import lru_cache
import math

import numpy as np


MODEL_SCHEMA = 'fakect.multiscale-coa-network/1'


def _tensorflow():
    # The runner owns device selection and memory limits before initialization.
    # Setting memory growth here would conflict with its logical-device cap.
    import tensorflow as tf
    return tf


def _shape(value, name):
    if (not isinstance(value, (tuple, list)) or len(value) != 3 or
            any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 8
                for n in value)):
        raise ValueError(f'{name} needs three integer dimensions >= 8 in k,j,i order')
    if math.prod(value) > 8_388_608:
        raise ValueError(f'{name} exceeds the 8,388,608-voxel input budget')
    return tuple(int(n) for n in value)


def _settings(overview_shape, base_filters, seed):
    shape = _shape(overview_shape, 'overview_shape')
    if isinstance(base_filters, bool) or not isinstance(base_filters, int) or not 2 <= base_filters <= 16:
        raise ValueError('base_filters must be an integer in [2,16]')
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    return shape


def feature_grid(overview_shape, levels=4):
    """Native voxel-center offsets/strides of 3x3 SAME strided convolutions."""
    shape = np.asarray(overview_shape, dtype=np.int64)
    offset, stride = np.zeros(3, dtype=float), 1
    for _ in range(levels):
        output = (shape + 1) // 2
        padding_before = np.maximum((output - 1) * 2 + 3 - shape, 0) // 2
        offset += (1 - padding_before) * stride
        shape, stride = output, stride * 2
    return tuple(shape.tolist()), tuple(offset.tolist()), stride


@lru_cache(maxsize=1)
def _components():
    """Register serializable layers lazily, including during checkpoint loading."""
    tf = _tensorflow()
    register = tf.keras.utils.register_keras_serializable

    @register(package='FakeCT')
    class BoundedBox(tf.keras.layers.Layer):
        def __init__(self, minimum_extent=.001, **kwargs):
            super().__init__(**kwargs)
            self.minimum_extent = float(minimum_extent)

        def call(self, raw):
            extent = self.minimum_extent + (1 - self.minimum_extent) * tf.sigmoid(raw[..., 3:])
            center = extent / 2 + (1 - extent) * tf.sigmoid(raw[..., :3])
            return tf.concat([center, extent], axis=-1)

        def get_config(self):
            return {**super().get_config(), 'minimum_extent': self.minimum_extent}

    @register(package='FakeCT')
    class AlignedFeatureSamples(tf.keras.layers.Layer):
        """Trilinear, spatially ordered feature samples per physical crop.

        A fixed 3^3 grid avoids allocating resized dense feature volumes. Points
        outside the encoded feature domain use zero-padded interpolation
        corners rather than repeated edges.
        Gradients propagate to valid feature values and proposal coordinates.
        """
        def __init__(self, overview_shape, levels=4, samples_per_axis=3, **kwargs):
            super().__init__(**kwargs)
            self.overview_shape = tuple(int(v) for v in overview_shape)
            self.levels = int(levels)
            self.samples_per_axis = int(samples_per_axis)
            _, self.offset, self.stride = feature_grid(self.overview_shape, self.levels)
            points = np.linspace(-.5, .5, self.samples_per_axis, dtype=np.float32)
            self.grid_values = np.stack(np.meshgrid(points, points, points, indexing='ij'), -1).reshape(-1, 3).tolist()

        def call(self, inputs):
            features, boxes = inputs
            features = tf.cast(features, self.compute_dtype)
            boxes = tf.cast(boxes, self.compute_dtype)
            grid = tf.constant(self.grid_values, dtype=features.dtype)
            coordinates = boxes[..., None, :3] + boxes[..., None, 3:] * grid
            positions = ((coordinates * tf.constant(self.overview_shape, features.dtype) - .5
                          - tf.constant(self.offset, features.dtype)) / self.stride)
            lower = tf.floor(positions)
            fraction = positions - lower
            lower = tf.cast(lower, tf.int32)
            shape = tf.shape(features)
            output_shape = tf.concat([tf.shape(boxes)[:2], [len(self.grid_values)], [shape[-1]]], 0)
            result = tf.zeros(output_shape, features.dtype)
            batch = tf.broadcast_to(tf.range(shape[0])[:, None, None], tf.shape(lower)[:-1])
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        corner = tf.constant([dz, dy, dx], tf.int32)
                        index = lower + corner
                        valid = tf.reduce_all((index >= 0) & (index < shape[1:4]), axis=-1)
                        index = tf.maximum(0, tf.minimum(index, shape[1:4] - 1))
                        gather = tf.concat([batch[..., None], index], axis=-1)
                        values = tf.gather_nd(features, gather)
                        weight = tf.reduce_prod(tf.where(tf.cast(corner, tf.bool), fraction, 1 - fraction), axis=-1)
                        result += values * (weight * tf.cast(valid, features.dtype))[..., None]
            return tf.reshape(result, [shape[0], tf.shape(boxes)[1], len(self.grid_values) * shape[-1]])

        def compute_output_shape(self, input_shape):
            feature_shape, box_shape = input_shape
            return (feature_shape[0], box_shape[1], self.samples_per_axis ** 3 * feature_shape[-1])

        def get_config(self):
            return {**super().get_config(), 'overview_shape': self.overview_shape,
                    'levels': self.levels, 'samples_per_axis': self.samples_per_axis}

    @register(package='FakeCT')
    class ProposalIndex(tf.keras.layers.Layer):
        def __init__(self, index, **kwargs):
            super().__init__(**kwargs)
            self.index = int(index)

        def call(self, inputs):
            return inputs[:, self.index]

        def get_config(self):
            return {**super().get_config(), 'index': self.index}

    register(package='FakeCT')(locator_loss)
    return BoundedBox, AlignedFeatureSamples, ProposalIndex


def locator_loss(y_true, y_pred):
    """Per-example center/extent Huber plus box-IoU loss (normalized units)."""
    tf = _tensorflow()
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    error = tf.abs(y_true - y_pred)
    huber = tf.reduce_mean(tf.where(error <= .1, .5 * error ** 2, .1 * (error - .05)), axis=-1)
    low_true, high_true = y_true[..., :3] - y_true[..., 3:] / 2, y_true[..., :3] + y_true[..., 3:] / 2
    low_pred, high_pred = y_pred[..., :3] - y_pred[..., 3:] / 2, y_pred[..., :3] + y_pred[..., 3:] / 2
    intersection = tf.reduce_prod(tf.maximum(0., tf.minimum(high_true, high_pred) - tf.maximum(low_true, low_pred)), axis=-1)
    union = tf.reduce_prod(y_true[..., 3:], -1) + tf.reduce_prod(y_pred[..., 3:], -1) - intersection
    iou = tf.math.divide_no_nan(intersection, union)
    return huber + .25 * (1 - iou)


def _encoder(shape, base_filters, name, levels=4):
    tf = _tensorflow()
    layers = tf.keras.layers
    image = layers.Input(shape + (2,), name=f'{name}_image_valid')
    x = image
    # Stride before the first activation avoids full-resolution feature stacks.
    # Layer normalization is independent of the microbatch population.
    for level in range(levels):
        filters = base_filters * min(2 ** level, 8)
        x = layers.Conv3D(filters, 3, strides=2, padding='same', use_bias=False,
                          kernel_initializer='he_normal', name=f'{name}_down_{level}')(x)
        x = layers.LayerNormalization(axis=-1, epsilon=1e-5, name=f'{name}_norm_{level}')(x)
        x = layers.Activation('relu', name=f'{name}_relu_{level}')(x)
    return tf.keras.Model(image, x, name=name)


def build_locator(overview_shape, base_filters=4, seed=42):
    """Image-to-box anatomy locator; train before building the classifier."""
    shape = _settings(overview_shape, base_filters, seed)
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    BoundedBox, _, _ = _components()
    layers = tf.keras.layers
    image = layers.Input(shape + (2,), name='overview')
    encoder = _encoder(shape, base_filters, 'overview_encoder')
    features = encoder(image)
    # Flatten retains anatomical position; global average alone would lose it.
    x = layers.Flatten(name='locator_spatial_features')(features)
    x = layers.Dense(64, activation='relu', name='locator_hidden')(x)
    raw = layers.Dense(6, name='locator_raw_box')(x)
    box = BoundedBox(name='anatomy_box')(raw)
    return tf.keras.Model({'overview': image}, box, name='fakect_multiscale_locator')


def predict_boxes(locator, overview):
    """Return checked normalized boxes from image/validity input, without masks."""
    overview = np.asarray(overview, dtype=np.float32)
    if overview.ndim != 5 or overview.shape[-1] != 2 or not np.isfinite(overview).all():
        raise ValueError('overview must be finite [B,K,J,I,2] image and validity')
    result = np.asarray(locator({'overview': overview}, training=False), dtype=np.float32)
    if (result.shape != (len(overview), 6) or not np.isfinite(result).all()
            or np.any(result[:, 3:] <= 0)
            or np.any(result[:, :3] - result[:, 3:] / 2 < -1e-6)
            or np.any(result[:, :3] + result[:, 3:] / 2 > 1 + 1e-6)):
        raise ValueError('Locator returned an invalid normalized anatomy box')
    return result


def build_classifier(overview_shape, detail_shape, num_proposals=2, base_filters=4,
                     seed=42, *, locator):
    """Freeze/reuse locator features and fuse aligned coarse/fine image features.

    Inputs are overview, proposal_boxes and detail_0 ... detail_N. The proposal
    boxes describe the actual detail crops, including padded parts, in the
    overview FOV. They must be produced from images by the frozen locator and a
    deterministic crop policy, never from edit/condition ground truth.
    """
    shape = _settings(overview_shape, base_filters, seed)
    detail_shape = _shape(detail_shape, 'detail_shape')
    if isinstance(num_proposals, bool) or not isinstance(num_proposals, int) or not 1 <= num_proposals <= 4:
        raise ValueError('num_proposals must be an integer in [1,4]')
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(seed)
    _, AlignedFeatureSamples, ProposalIndex = _components()
    layers = tf.keras.layers
    encoder = locator.get_layer('overview_encoder')
    if tuple(encoder.input_shape[1:]) != shape + (2,):
        raise ValueError('Locator overview shape differs from classifier overview shape')
    encoder.trainable = False
    image = layers.Input(shape + (2,), name='overview')
    boxes = layers.Input((num_proposals, 6), name='proposal_boxes')
    features = encoder(image, training=False)
    samples = AlignedFeatureSamples(shape, name='aligned_coarse_samples')([features, boxes])
    fine_encoder = _encoder(detail_shape, base_filters, 'detail_encoder')
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
        local_features = fine_encoder(detail)
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


def compile_locator(model, learning_rate=.001, clipnorm=1.):
    tf = _tensorflow()
    _components()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=clipnorm),
                  loss=locator_loss, jit_compile=False)
    return model


def compile_classifier(model, learning_rate=.001, clipnorm=1., regression_weight=1.):
    tf = _tensorflow()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=clipnorm),
                  loss={'coa_logit': tf.keras.losses.BinaryCrossentropy(from_logits=True),
                        'local_volume_loss': tf.keras.losses.Huber(delta=.1)},
                  loss_weights={'coa_logit': 1., 'local_volume_loss': regression_weight},
                  jit_compile=False)
    return model


def load_model(path, *, compile=False):
    """Load registered components in safe mode, without unsafe Lambda layers."""
    tf = _tensorflow()
    _components()
    return tf.keras.models.load_model(path, compile=compile, safe_mode=True)
