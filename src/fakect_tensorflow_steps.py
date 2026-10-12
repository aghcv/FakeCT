"""Fixed-signature TensorFlow steps without per-batch Python iterators.

Factories create one train and one inference function. Adam slot variables are
built before tracing; input signatures fix the microbatch and every tensor
shape. No Keras ``train_on_batch`` wrapper or learned-feature cache is used.
TensorFlow remains an optional, lazy import for CPU data preparation.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StableSteps:
    """Hold compiled training/inference steps and their shared optimizer.

    Attributes
    ----------
    train : tensorflow.types.experimental.GenericFunction
        Fixed-signature training callable produced by a step factory.
    predict : tensorflow.types.experimental.GenericFunction
        Fixed-signature inference callable using the same model.
    optimizer : tensorflow.keras.optimizers.Optimizer
        Optimizer whose slot variables were built before tracing.

    Notes
    -----
    Field references are immutable; training still updates model and optimizer
    variables. Trace-count methods check graph reuse without importing TensorFlow.
    """
    train: object
    predict: object
    optimizer: object

    def tracing_counts(self):
        """Inspect the number of traces used by the fixed-signature TensorFlow steps.

        Parameters
        ----------
        None
            This function takes no explicit arguments.

        Returns
        -------
        dict of str to int
            Current train and predict tracing counts.
        """
        return {'train': int(self.train.experimental_get_tracing_count()),
                'predict': int(self.predict.experimental_get_tracing_count())}

    def assert_stable_traces(self, maximum=1):
        """Reject unexpected retracing of fixed-signature model steps.

        Parameters
        ----------
        maximum : int
            Positive maximum permitted trace count per step.

        Returns
        -------
        dict of str to int
            Verified train/predict tracing counts.

        Raises
        ------
        ValueError
            The maximum is not a positive integer.
        RuntimeError
            One or more functions exceed the trace limit.
        """
        if type(maximum) is not int or maximum < 1:
            raise ValueError('maximum must be a positive integer')
        counts = self.tracing_counts()
        if any(count > maximum for count in counts.values()):
            raise RuntimeError(f'TensorFlow retraced fixed-signature steps: {counts}')
        return counts


def _shape(values, name):
    """Validate bounded three-dimensional fixed TensorFlow input geometry.

    Parameters
    ----------
    values : list or tuple of int
        Three positive Python integer sizes; product at most 8,388,608.
    name : str
        Signature shape label for errors.

    Returns
    -------
    tuple of int
        Positive dimensions for the factory input signature.

    Raises
    ------
    ValueError
        Shape values or voxel budget are invalid.
    """
    if (not isinstance(values, (tuple, list)) or len(values) != 3 or
            any(type(v) is not int or v <= 0 for v in values)):
        raise ValueError(f'{name} must contain three positive integers')
    if math.prod(values) > 8_388_608:
        raise ValueError(f'{name} exceeds the input voxel budget')
    return tuple(values)


def _positive(value, name, *, allow_zero=False):
    """Validate finite optimizer/loss scalars.

    Parameters
    ----------
    value : int or float
        Numeric setting, with Boolean values rejected.
    name : str
        Setting label for errors.
    allow_zero : bool
        Permit zero in addition to positive values.

    Returns
    -------
    float
        Converted finite positive or explicitly allowed zero value.

    Raises
    ------
    ValueError
        Type, finiteness or sign is invalid.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be numeric')
    value = float(value)
    if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f'{name} must be {"nonnegative" if allow_zero else "positive"}')
    return value


def _optimizer(tf, model, learning_rate):
    """Create Adam and build its slot variables before tracing model steps.

    Parameters
    ----------
    tf : module
        Imported TensorFlow module; supplied to keep optional imports explicit.
    model : tensorflow.keras.Model
        Built model containing trainable variables.
    learning_rate : float
        Positive optimizer update rate.

    Returns
    -------
    tensorflow.keras.optimizers.Adam
        Optimizer initialized for every model trainable variable.

    Raises
    ------
    ValueError
        The model is unbuilt or lacks trainable variables.
    """
    if not model.built or not model.trainable_variables:
        raise ValueError('Build a model with trainable variables before creating steps')
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    optimizer.build(model.trainable_variables)
    return optimizer


def _apply(tf, model, optimizer, tape, loss, clipnorm):
    """Validate, clip and apply gradients with ordered finite-loss checks.

    Parameters
    ----------
    tf : module
        Imported TensorFlow module; supplied to keep optional imports explicit.
    model : tensorflow.keras.Model
        Model whose trainable variables are updated.
    optimizer : tensorflow.keras.optimizers.Optimizer
        Prebuilt optimizer whose state is updated.
    tape : tensorflow.GradientTape
        Tape recording the supplied loss.
    loss : tensorflow.Tensor
        Checked scalar loss tensor used to gate all updates.
    clipnorm : float
        Positive global gradient clipping threshold.

    Returns
    -------
    tensorflow.Tensor
        Scalar float gradient global norm before clipping.

    Raises
    ------
    ValueError
        A trainable variable is disconnected from the loss.

    Notes
    -----
    Mutates model and optimizer variables only after finite-gradient/loss dependencies pass; TensorFlow numeric checks raise on nonfinite gradients.
    """
    variables = model.trainable_variables
    gradients = tape.gradient(loss, variables)
    if any(g is None for g in gradients):
        names = [v.name for v, g in zip(variables, gradients) if g is None]
        raise ValueError(f'Disconnected trainable variables: {names}')
    checked = []
    for gradient in gradients:
        if isinstance(gradient, tf.IndexedSlices):
            checked.append(tf.IndexedSlices(tf.debugging.check_numerics(gradient.values, 'Nonfinite gradient'),
                                            gradient.indices, gradient.dense_shape))
        else:
            checked.append(tf.debugging.check_numerics(gradient, 'Nonfinite gradient'))
    norm = tf.debugging.check_numerics(tf.linalg.global_norm(checked), 'Nonfinite gradient global norm')
    clipped, _ = tf.clip_by_global_norm(checked, clipnorm, use_norm=norm)
    # Adam's iteration counter is otherwise independent of the gradient data
    # edges and may advance before a failing check completes. Gate every update
    # explicitly on both the checked loss and checked gradient norm.
    with tf.control_dependencies([loss, norm]):
        optimizer.apply_gradients(zip(clipped, variables))
    return norm


def _regularization(tf, model):
    """Collect optional model regularization terms for a traced loss.

    Parameters
    ----------
    tf : module
        Imported TensorFlow module; supplied to keep optional imports explicit.
    model : tensorflow.keras.Model
        Model exposing its current regularization loss terms.

    Returns
    -------
    tensorflow.Tensor
        Sum of model losses or a zero float32 scalar.
    """
    return tf.add_n(model.losses) if model.losses else tf.constant(0., tf.float32)


def make_locator_steps(model, overview_shape, learning_rate=.001, clipnorm=1., loss_function=None):
    """Create fixed-signature locator training and inference functions.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Built locator accepting an overview input mapping; all variables must connect to
        its loss.
    overview_shape : list or tuple of int
        Three fixed overview dimensions; image tensor shape is (1, K, J, I, 2).
    learning_rate : float
        Positive Adam update rate.
    clipnorm : float
        Positive global gradient clipping threshold.
    loss_function : callable or None
        Loss(target_boxes, model_output); None uses locator_loss. Box supervision is
        float32 shape (1, 6).

    Returns
    -------
    StableSteps
        Train(image, boxes), predict(image) and prebuilt Adam optimizer; training
        returns loss and gradient norm.

    Raises
    ------
    ValueError
        Shapes, optimizer settings or model readiness are invalid.

    Notes
    -----
    Predict returns the model output structure, including auxiliary logits where supplied. TensorFlow is imported lazily; no per-batch Python iterator is traced.

    Return stable ``train(image, boxes)`` and ``predict(image)`` functions.

    Image signature: ``[1,K,J,I,2]`` float32; supervision: ``[1,6]`` float32.
    The injected loss receives ``(target_boxes, model_output)`` and may accept
    tensor or dictionary model outputs. Predict returns the same output structure
    as the model, including optional auxiliary logits. The default loss supports
    the original tensor-output syncoCT locator.
    """
    shape = _shape(overview_shape, 'overview_shape')
    learning_rate = _positive(learning_rate, 'learning_rate')
    clipnorm = _positive(clipnorm, 'clipnorm')
    import tensorflow as tf
    if loss_function is None:
        from fakect_multiscale_network import locator_loss
        loss_function = locator_loss
    optimizer = _optimizer(tf, model, learning_rate)
    image_spec = tf.TensorSpec((1,) + shape + (2,), tf.float32, name='overview')
    target_spec = tf.TensorSpec((1, 6), tf.float32, name='target_boxes')

    @tf.function(input_signature=[image_spec, target_spec], autograph=False, jit_compile=False)
    def train(image, boxes):
        image = tf.debugging.check_numerics(image, 'Nonfinite overview input')
        boxes = tf.debugging.check_numerics(boxes, 'Nonfinite locator supervision')
        with tf.GradientTape() as tape:
            output = model({'overview': image}, training=True)
            loss = tf.reduce_mean(tf.cast(loss_function(boxes, output), tf.float32))
            loss = tf.debugging.check_numerics(loss + _regularization(tf, model), 'Nonfinite locator loss')
        norm = _apply(tf, model, optimizer, tape, loss, clipnorm)
        return {'loss': loss, 'gradient_global_norm': norm}

    @tf.function(input_signature=[image_spec], autograph=False, jit_compile=False)
    def predict(image):
        image = tf.debugging.check_numerics(image, 'Nonfinite overview input')
        output = model({'overview': image}, training=False)
        return tf.nest.map_structure(lambda x: tf.debugging.check_numerics(x, 'Nonfinite locator prediction'), output)

    return StableSteps(train, predict, optimizer)


def make_classifier_steps(model, overview_shape, detail_shape, num_proposals=2,
                          learning_rate=.001, clipnorm=1., regression_weight=1.):
    """Create fixed-dictionary classifier training and inference functions.

    Parameters
    ----------
    model : tensorflow.keras.Model
        Built model returning coa_logit and local_volume_loss tensors.
    overview_shape : list or tuple of int
        Fixed overview K,J,I dimensions; overview tensors have shape (1, K, J, I, 2).
    detail_shape : list or tuple of int
        Fixed detail K,J,I dimensions; each detail tensor has shape (1, K, J, I, 2).
    num_proposals : int
        One to four detail proposals; proposal_boxes has shape (1, num_proposals, 6).
    learning_rate : float
        Positive Adam update rate.
    clipnorm : float
        Positive global gradient clipping threshold.
    regression_weight : float
        Nonnegative multiplier of the Huber regression term.

    Returns
    -------
    StableSteps
        Train(inputs, targets), predict(inputs) and Adam; loss is sigmoid BCE plus
        weighted Huber(delta=0.1) and regularization.

    Raises
    ------
    ValueError
        Shapes, proposal count, optimizer/loss settings or model readiness are invalid.

    Notes
    -----
    Input mappings contain overview, proposal_boxes and detail_0 through detail_N-1. Target keys coa_logit and local_volume_loss each hold float32 tensors of shape (1, 1).

    Return fixed-dictionary classifier train/inference functions.

    Input keys match syncoCT's classifier: ``overview``, ``proposal_boxes`` and
    ``detail_0`` ... ``detail_N``. Target keys are ``coa_logit`` (binary float
    supervision) and ``local_volume_loss``. Loss is mean sigmoid cross entropy
    plus weighted Huber(delta=.1), matching the original Keras configuration.
    """
    overview_shape = _shape(overview_shape, 'overview_shape')
    detail_shape = _shape(detail_shape, 'detail_shape')
    if type(num_proposals) is not int or not 1 <= num_proposals <= 4:
        raise ValueError('num_proposals must be between 1 and 4')
    learning_rate = _positive(learning_rate, 'learning_rate')
    clipnorm = _positive(clipnorm, 'clipnorm')
    regression_weight = _positive(regression_weight, 'regression_weight', allow_zero=True)
    import tensorflow as tf
    optimizer = _optimizer(tf, model, learning_rate)
    input_spec = {'overview': tf.TensorSpec((1,) + overview_shape + (2,), tf.float32),
                  'proposal_boxes': tf.TensorSpec((1, num_proposals, 6), tf.float32)}
    for i in range(num_proposals):
        input_spec['detail_' + str(i)] = tf.TensorSpec((1,) + detail_shape + (2,), tf.float32)
    target_spec = {key: tf.TensorSpec((1, 1), tf.float32) for key in ('coa_logit', 'local_volume_loss')}

    @tf.function(input_signature=[input_spec, target_spec], autograph=False, jit_compile=False)
    def train(inputs, targets):
        inputs = tf.nest.map_structure(lambda x: tf.debugging.check_numerics(x, 'Nonfinite classifier input'), inputs)
        targets = tf.nest.map_structure(lambda x: tf.debugging.check_numerics(x, 'Nonfinite classifier supervision'), targets)
        with tf.GradientTape() as tape:
            output = model(inputs, training=True)
            bce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets['coa_logit'], logits=output['coa_logit']))
            difference = output['local_volume_loss'] - targets['local_volume_loss']
            error = tf.abs(difference)
            huber = tf.reduce_mean(tf.where(error <= .1, .5 * difference ** 2, .1 * (error - .05)))
            loss = tf.debugging.check_numerics(bce + regression_weight * huber + _regularization(tf, model),
                                             'Nonfinite classifier loss')
        norm = _apply(tf, model, optimizer, tape, loss, clipnorm)
        return {'loss': loss, 'gradient_global_norm': norm, 'binary_crossentropy': bce, 'huber': huber}

    @tf.function(input_signature=[input_spec], autograph=False, jit_compile=False)
    def predict(inputs):
        inputs = tf.nest.map_structure(lambda x: tf.debugging.check_numerics(x, 'Nonfinite classifier input'), inputs)
        output = model(inputs, training=False)
        return tf.nest.map_structure(lambda x: tf.debugging.check_numerics(x, 'Nonfinite classifier prediction'), output)

    return StableSteps(train, predict, optimizer)
