"""Matched-sibling, fixed-batch-two training for the unchanged focused graph.

Both siblings are evaluated at the same weights. BCE and Huber are averaged
before computing and clipping ONE combined gradient. This preserves cancellation
of common-background positive/negative gradients, unlike clipping and updating
each example separately. The caller verifies matching case and edit metadata.
This module verifies balanced binary targets and fixes the tensor batch at two.
"""
import fakect_coa_focused_network as focused
from fakect_tensorflow_steps import StableSteps, _apply, _optimizer, _positive, _regularization


MODEL_SCHEMA = 'fakect.coa-paired-steps/1'
build_classifier = focused.build_classifier
load_model = focused.load_model
feature_probe = focused.feature_probe
feature_geometry = focused.feature_geometry


def make_steps(model, detail_shape, num_proposals=2, learning_rate=.001,
               clipnorm=1., regression_weight=1., batch_size=2):
    """Return fixed batch2 image-only train/predict steps with one pair update.

    ``train`` targets contain one negative and one positive sibling in either
    order. Labels are supervision only. ``predict`` accepts any two image rows.
    """
    if type(batch_size) is not int or batch_size != 2:
        raise ValueError('Matched-sibling training requires fixed batch_size=2')
    shape = focused._settings(detail_shape, num_proposals, 4, 42)
    learning_rate = _positive(learning_rate, 'learning_rate')
    clipnorm = _positive(clipnorm, 'clipnorm')
    regression_weight = _positive(regression_weight, 'regression_weight', allow_zero=True)
    tf = focused._tensorflow()
    optimizer = _optimizer(tf, model, learning_rate)
    specs = {f'detail_{index}': tf.TensorSpec((batch_size,) + shape + (2,), tf.float32)
             for index in range(num_proposals)}
    if set(model.input) != set(specs):
        raise ValueError('Model inputs must exactly match detail-only signatures')
    targets_spec = {name: tf.TensorSpec((batch_size, 1), tf.float32)
                    for name in ('coa_logit', 'local_volume_loss')}

    @tf.function(input_signature=[specs, targets_spec], autograph=False, jit_compile=False)
    def train(inputs, targets):
        inputs = tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite paired image'), inputs)
        targets = tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite paired target'), targets)
        labels = targets['coa_logit']
        checks = [tf.debugging.assert_equal(tf.reduce_sum(labels), 1.,
                                           message='A matched pair needs one positive and one negative'),
                  tf.debugging.assert_equal(tf.reduce_all(tf.logical_or(labels == 0., labels == 1.)), True,
                                           message='Matched pair classification targets must be binary')]
        with tf.control_dependencies(checks):
            labels = tf.identity(labels)
        with tf.GradientTape() as tape:
            output = model(inputs, training=True)
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output['coa_logit']))
            difference = output['local_volume_loss'] - targets['local_volume_loss']
            error = tf.abs(difference)
            huber = tf.reduce_mean(tf.where(error <= .1, .5 * difference ** 2, .1 * (error - .05)))
            loss = tf.debugging.check_numerics(bce + regression_weight * huber + _regularization(tf, model),
                                              'Nonfinite paired classifier loss')
        # The loss already averages BOTH siblings at the same parameter values.
        # Clip only the resulting combined gradient, then advance Adam once.
        norm = _apply(tf, model, optimizer, tape, loss, clipnorm)
        return {'loss': loss, 'binary_crossentropy': bce, 'huber': huber, 'gradient_global_norm': norm}

    @tf.function(input_signature=[specs], autograph=False, jit_compile=False)
    def predict(inputs):
        inputs = tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite paired image'), inputs)
        return tf.nest.map_structure(lambda value: tf.debugging.check_numerics(value, 'Nonfinite paired output'),
                                     model(inputs, training=False))

    return StableSteps(train, predict, optimizer)
