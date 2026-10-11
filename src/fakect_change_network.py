"""Focused CT to mutually exclusive net added/removed voxel probabilities.

The input contains normalized current attenuation and acquisition validity only.
The paired baseline is supervision, never an inference input. Class zero means
unchanged, class one means added relative to baseline, and class two removed.
This compact three-level 3D U-Net restores the input resolution after a stride-eight
encoder. It does not require a target mask or crop coordinates at inference.

TensorFlow imports and custom-layer registration are lazy. Inputs and outputs
are float32; training steps use a fixed batch of one to avoid retracing. Padding
is excluded from supervision, and an entirely invalid patch does not advance
the optimizer or alter its momentum. Loss weights are fixed, bounded settings,
not inverse frequencies computed from an almost-empty change mask.
"""
from functools import lru_cache
import math

from fakect_tensorflow_steps import StableSteps, _apply, _optimizer, _positive


MODEL_SCHEMA = 'fakect.focused-net-change/1'
CLASS_NAMES = ('unchanged', 'added', 'removed')


def _shape(patch_shape):
    if (not isinstance(patch_shape, (tuple, list)) or len(patch_shape) != 3
            or any(type(n) is not int or n < 8 or n % 8 for n in patch_shape)):
        raise ValueError('patch_shape needs three integer dimensions >=8 and divisible by8')
    if math.prod(patch_shape) > 1_048_576:
        raise ValueError('patch_shape exceeds the 1,048,576 voxel training patch budget')
    return tuple(patch_shape)


def _loss_settings(change_weight, dice_weight):
    change_weight = _positive(change_weight, 'change_weight')
    dice_weight = _positive(dice_weight, 'dice_weight', allow_zero=True)
    if not 1. <= change_weight <= 64.:
        raise ValueError('change_weight must be between1 and64')
    if dice_weight > 10.:
        raise ValueError('dice_weight must not exceed10')
    return change_weight, dice_weight


@lru_cache(maxsize=1)
def _components():
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package='FakeCTChange')
    class SanitizeImage(tf.keras.layers.Layer):
        def call(self, image):
            validity = tf.clip_by_value(image[..., 1:2], 0., 1.)
            attenuation = tf.where(validity > 0., image[..., 0:1], tf.zeros_like(validity))
            return tf.concat([attenuation, validity], axis=-1)

    return {'SanitizeImage': SanitizeImage}


def build_model(patch_shape=(64, 64, 64), base_filters=4, seed=42):
    """Build an uncompiled U-Net: ``[N,K,J,I,2] -> [N,K,J,I,3]`` logits.

    Dimensions are K,J,I. The two channels are attenuation and validity; every
    output voxel has one softmax distribution, so added and removed cannot both
    be its predicted class. Use :func:`load_model` for safe archive restoration.
    """
    shape = _shape(patch_shape)
    if type(base_filters) is not int or not 2 <= base_filters <= 16:
        raise ValueError('base_filters must be an integer between2 and16')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    layers = tf.keras.layers

    def block(tensor, width, name):
        for i in range(2):
            tensor = layers.Conv3D(width, 3, padding='same', kernel_initializer='he_normal',
                                   dtype='float32', name=f'{name}_conv{i}')(tensor)
            tensor = layers.LeakyReLU(negative_slope=.1, dtype='float32',
                                      name=f'{name}_activation{i}')(tensor)
        return tensor

    image = layers.Input(shape + (2,), dtype='float32', name='attenuation_validity')
    clean = _components()['SanitizeImage'](dtype='float32', name='clean_image')(image)
    skip0 = block(clean, base_filters, 'fine')
    down0 = layers.MaxPool3D(2, dtype='float32', name='down2')(skip0)
    skip1 = block(down0, base_filters * 2, 'middle')
    down1 = layers.MaxPool3D(2, dtype='float32', name='down4')(skip1)
    skip2 = block(down1, base_filters * 4, 'coarse')
    down2 = layers.MaxPool3D(2, dtype='float32', name='down8')(skip2)
    encoded = block(down2, base_filters * 8, 'bottleneck')
    up2 = layers.UpSampling3D(2, dtype='float32', name='up4')(encoded)
    joined2 = layers.Concatenate(dtype='float32', name='skip_coarse')([up2, skip2])
    decoded2 = block(joined2, base_filters * 4, 'decode_coarse')
    up1 = layers.UpSampling3D(2, dtype='float32', name='up2')(decoded2)
    joined1 = layers.Concatenate(dtype='float32', name='skip_middle')([up1, skip1])
    decoded1 = block(joined1, base_filters * 2, 'decode_middle')
    up0 = layers.UpSampling3D(2, dtype='float32', name='up_full')(decoded1)
    joined0 = layers.Concatenate(dtype='float32', name='skip_fine')([up0, skip0])
    decoded0 = block(joined0, base_filters, 'decode_fine')
    logits = layers.Conv3D(3, 1, dtype='float32', name='change_logits')(decoded0)
    return tf.keras.Model(image, logits, name='focused_net_change_unet')


def load_model(path):
    """Load an uncompiled .keras archive with registered local layers only."""
    import tensorflow as tf
    _components()
    return tf.keras.models.load_model(path, compile=False, safe_mode=True)


def loss_components(logits, labels, validity, change_weight=8., dice_weight=1.):
    """Return weighted CE plus mean foreground Dice loss.

    ``logits`` is [N,K,J,I,3]; integer ``labels`` and float ``validity`` are
    [N,K,J,I]. Validity can be fractional in [0,1]. Invalid labels are ignored,
    including a padding sentinel of -1. CE divides by total class-weighted valid
    mass, rather than the count of foreground voxels. For a present foreground
    class, Dice uses the usual soft overlap. For an absent class it uses mean
    false-positive probability on valid voxels, retaining a useful gradient on
    unchanged controls. All components are zero for an entirely invalid patch.
    """
    change_weight, dice_weight = _loss_settings(change_weight, dice_weight)
    import tensorflow as tf
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)
    raw_labels = tf.convert_to_tensor(labels)
    if not raw_labels.dtype.is_integer:
        raise ValueError('labels must have an integer dtype')
    labels = tf.cast(raw_labels, tf.int32)
    validity = tf.convert_to_tensor(validity, dtype=tf.float32)
    assertions = [tf.debugging.assert_rank(logits, 5),
                  tf.debugging.assert_rank(labels, 4),
                  tf.debugging.assert_rank(validity, 4),
                  tf.debugging.assert_equal(tf.shape(logits)[-1], 3),
                  tf.debugging.assert_equal(tf.shape(logits)[:-1], tf.shape(labels)),
                  tf.debugging.assert_equal(tf.shape(labels), tf.shape(validity)),
                  tf.debugging.assert_greater_equal(validity, 0.),
                  tf.debugging.assert_less_equal(validity, 1.)]
    with tf.control_dependencies(assertions):
        validity = tf.debugging.check_numerics(validity, 'Nonfinite change validity')
        labels = tf.where(validity > 0., labels, tf.zeros_like(labels))
        logits = tf.where(validity[..., None] > 0., logits, tf.zeros_like(logits))
        logits = tf.debugging.check_numerics(logits, 'Nonfinite valid change logits')
    assertions = [tf.debugging.assert_greater_equal(labels, 0),
                  tf.debugging.assert_less(labels, 3)]
    with tf.control_dependencies(assertions):
        labels = tf.identity(labels)
    weights = tf.where(labels > 0, tf.cast(change_weight, tf.float32), 1.) * validity
    ce_voxel = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    ce = tf.math.divide_no_nan(tf.reduce_sum(ce_voxel * weights), tf.reduce_sum(weights))
    probability = tf.nn.softmax(logits, axis=-1)
    valid_mass = tf.reduce_sum(validity)
    dice_losses = []
    for cls in (1, 2):
        target = tf.cast(labels == cls, tf.float32) * validity
        predicted = probability[..., cls] * validity
        target_mass = tf.reduce_sum(target)
        predicted_mass = tf.reduce_sum(predicted)
        intersection = tf.reduce_sum(probability[..., cls] * target)
        # Branching avoids a zero-target Dice singularity and penalizes empty-
        # control false positives according to their probability volume.
        dice_losses.append(tf.cond(target_mass > 0.,
            lambda: 1. - (2. * intersection + 1e-6) / (target_mass + predicted_mass + 1e-6),
            lambda: tf.math.divide_no_nan(predicted_mass, valid_mass)))
    dice = tf.add_n(dice_losses) / 2.
    loss = tf.debugging.check_numerics(ce + dice_weight * dice, 'Nonfinite change loss')
    return {'loss': loss, 'cross_entropy': ce, 'foreground_dice_loss': dice,
            'added_soft_dice': tf.where(valid_mass > 0., 1. - dice_losses[0], 0.),
            'removed_soft_dice': tf.where(valid_mass > 0., 1. - dice_losses[1], 0.),
            'valid_voxels': valid_mass,
            'changed_voxels': tf.reduce_sum(tf.cast(labels > 0, tf.float32) * validity)}


def make_steps(model, learning_rate=.001, change_weight=8., dice_weight=1., clipnorm=1.):
    """Create fixed-signature ``train(x,y,valid)`` and ``predict(x)`` steps.

    x: [1,K,J,I,2] float32; y: [1,K,J,I] int32; valid: [1,K,J,I]
    float32. Predict returns float32 logits, not thresholded masks. The validity
    tensor must equal x's second channel. Invalid attenuation is sanitized
    before numerical checks. Returned train values are the loss components and
    ``gradient_global_norm``. No optimizer update occurs for all-invalid x.
    """
    shape = _shape(list(model.input_shape[1:4]))
    if tuple(model.input_shape[-1:]) != (2,) or tuple(model.output_shape[1:]) != shape + (3,):
        raise ValueError('model must map attenuation/validity patches to three class logits')
    learning_rate = _positive(learning_rate, 'learning_rate')
    clipnorm = _positive(clipnorm, 'clipnorm')
    change_weight, dice_weight = _loss_settings(change_weight, dice_weight)
    import tensorflow as tf
    optimizer = _optimizer(tf, model, learning_rate)
    image_spec = tf.TensorSpec((1,) + shape + (2,), tf.float32, name='focused_ct')
    label_spec = tf.TensorSpec((1,) + shape, tf.int32, name='net_change_labels')
    validity_spec = tf.TensorSpec((1,) + shape, tf.float32, name='acquisition_validity')

    def clean_input(image):
        valid = tf.debugging.check_numerics(image[..., 1:2], 'Nonfinite input validity')
        with tf.control_dependencies([tf.debugging.assert_greater_equal(valid, 0.),
                                     tf.debugging.assert_less_equal(valid, 1.)]):
            image = tf.concat([tf.where(valid > 0., image[..., 0:1], tf.zeros_like(valid)), valid], -1)
        return tf.debugging.check_numerics(image, 'Nonfinite valid input attenuation')

    @tf.function(input_signature=[image_spec, label_spec, validity_spec], autograph=False,
                 jit_compile=False)
    def train(image, labels, valid):
        image = clean_input(image)
        with tf.control_dependencies([tf.debugging.assert_near(image[..., 1], valid,
                                      message='Input and supervision validity differ')]):
            valid = tf.identity(valid)
        with tf.GradientTape() as tape:
            logits = model(image, training=True)
            components = loss_components(logits, labels, valid, change_weight, dice_weight)
            loss = components['loss']
        norm = tf.cond(tf.reduce_sum(valid) > 0.,
                       lambda: _apply(tf, model, optimizer, tape, loss, clipnorm),
                       lambda: tf.constant(0., tf.float32))
        return {**components, 'gradient_global_norm': norm}

    @tf.function(input_signature=[image_spec], autograph=False, jit_compile=False)
    def predict(image):
        logits = model(clean_input(image), training=False)
        return tf.debugging.check_numerics(logits, 'Nonfinite change prediction')

    return StableSteps(train, predict, optimizer)
