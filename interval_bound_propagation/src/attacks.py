# coding=utf-8
# Copyright 2019 The Interval Bound Propagation Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to define attacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import six
import sonnet as snt
import tensorflow.compat.v1 as tf

nest = tf.nest


@six.add_metaclass(abc.ABCMeta)
class UnrolledOptimizer(object):
  """In graph optimizer to be used in tf.while_loop."""

  def __init__(self, colocate_gradients_with_ops=False):
    self._colocate_gradients_with_ops = colocate_gradients_with_ops

  @abc.abstractmethod
  def minimize(self, loss, x, optim_state):
    """Compute a new value of `x` to minimize `loss`.

    Args:
      loss: A scalar Tensor, the value to be minimized. `loss` should be a
        continuous function of `x` which supports gradients, `loss = f(x)`.
      x: A list of Tensors, the values to be updated. This is analogous to the
        `var_list` argument in standard TF Optimizer.
      optim_state: A (possibly nested) dict, containing any state info needed
        for the optimizer.

    Returns:
      new_x: A list of Tensors, the same length as `x`, which are updated
      new_optim_state: A new state, with the same structure as `optim_state`,
        which have been updated.
    """

  @abc.abstractmethod
  def init_state(self, x):
    """Returns the initial state of the optimizer.

    Args:
      x: A list of Tensors, which will be optimized.

    Returns:
      Any structured output.
    """


class UnrolledGradientDescent(UnrolledOptimizer):
  """Vanilla gradient descent optimizer."""

  _State = collections.namedtuple('State', ['iteration'])  # pylint: disable=invalid-name

  def __init__(self, lr=.1, lr_fn=None, fgsm=False,
               colocate_gradients_with_ops=False):
    super(UnrolledGradientDescent, self).__init__(
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    self._lr_fn = (lambda i: lr) if lr_fn is None else lr_fn
    self._fgsm = fgsm

  def init_state(self, unused_x):
    return self._State(tf.constant(0, dtype=tf.int64))

  def minimize(self, loss, x, optim_state):
    """Refer to parent class documentation."""
    lr = self._lr_fn(optim_state.iteration)
    grads = self.gradients(loss, x)
    if self._fgsm:
      grads = [tf.sign(g) for g in grads]
    new_x = [None] * len(x)
    for i in range(len(x)):
      new_x[i] = x[i] - lr * grads[i]
    new_optim_state = self._State(optim_state.iteration + 1)
    return new_x, new_optim_state

  def gradients(self, loss, x):
    return tf.gradients(
        loss, x, colocate_gradients_with_ops=self._colocate_gradients_with_ops)


# Syntactic sugar.
class UnrolledFGSMDescent(UnrolledGradientDescent):
  """Identical to UnrolledGradientDescent but forces FGM steps."""

  def __init__(self, lr=.1, lr_fn=None,
               colocate_gradients_with_ops=False):
    super(UnrolledFGSMDescent, self).__init__(
        lr, lr_fn, True, colocate_gradients_with_ops)


class UnrolledAdam(UnrolledOptimizer):
  """The Adam optimizer defined in https://arxiv.org/abs/1412.6980."""

  _State = collections.namedtuple('State', ['t', 'm', 'u'])  # pylint: disable=invalid-name

  def __init__(self, lr=0.1, lr_fn=None, beta1=0.9, beta2=0.999, epsilon=1e-9,
               colocate_gradients_with_ops=False):
    super(UnrolledAdam, self).__init__(
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    self._lr_fn = (lambda i: lr) if lr_fn is None else lr_fn
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

  def init_state(self, x):
    return self._State(
        t=tf.constant(0, dtype=tf.int64),
        m=[tf.zeros_like(v) for v in x],
        u=[tf.zeros_like(v) for v in x])

  def _apply_gradients(self, grads, x, optim_state):
    """Applies gradients."""
    lr = self._lr_fn(optim_state.t)
    new_optim_state = self._State(
        t=optim_state.t + 1,
        m=[None] * len(x),
        u=[None] * len(x))
    t = tf.cast(new_optim_state.t, tf.float32)
    new_x = [None] * len(x)
    for i in range(len(x)):
      g = grads[i]
      m_old = optim_state.m[i]
      u_old = optim_state.u[i]
      new_optim_state.m[i] = self._beta1 * m_old + (1. - self._beta1) * g
      new_optim_state.u[i] = self._beta2 * u_old + (1. - self._beta2) * g * g
      m_hat = new_optim_state.m[i] / (1. - tf.pow(self._beta1, t))
      u_hat = new_optim_state.u[i] / (1. - tf.pow(self._beta2, t))
      new_x[i] = x[i] - lr * m_hat / (tf.sqrt(u_hat) + self._epsilon)
    return new_x, new_optim_state

  def minimize(self, loss, x, optim_state):
    grads = self.gradients(loss, x)
    return self._apply_gradients(grads, x, optim_state)

  def gradients(self, loss, x):
    return tf.gradients(
        loss, x, colocate_gradients_with_ops=self._colocate_gradients_with_ops)


def _spsa_gradients(loss_fn, x, delta=0.01, num_samples=16, num_iterations=4):
  """Compute gradient estimates using SPSA.

  Args:
    loss_fn: Callable that takes a single argument of shape [batch_size, ...]
      and returns the loss contribution of each element of the batch as a
      tensor of shape [batch_size].
    x: List of tensors with a single element. We only support computation of
      the gradient of the loss with respect to x[0]. We take a list as input to
      keep the same API call as tf.gradients.
    delta: The gradients are computed by computing the loss within x - delta and
      x + delta.
    num_samples: The total number of random samples used to compute the gradient
      is `num_samples` times `num_iterations`. `num_samples` contributes to the
      gradient by tiling `x` `num_samples` times.
    num_iterations: The total number of random samples used to compute the
      gradient is `num_samples` times `num_iterations`. `num_iterations`
      contributes to the gradient by iterating using a `tf.while_loop`.

  Returns:
    List of tensors with a single element corresponding to the gradient of
    loss_fn(x[0]) with respect to x[0].
  """

  if len(x) != 1:
    raise NotImplementedError('SPSA gradients with respect to multiple '
                              'variables is not supported.')
  # loss_fn takes a single argument.
  tensor = x[0]

  def _get_delta(x):
    return delta * tf.sign(
        tf.random_uniform(tf.shape(x), minval=-1., maxval=1., dtype=x.dtype))

  # Process batch_size samples at a time.
  def cond(i, *_):
    return tf.less(i, num_iterations)

  def loop_body(i, total_grad):
    """Compute gradient estimate."""
    batch_size = tf.shape(tensor)[0]
    # The tiled tensor has shape [num_samples, batch_size, ...]
    tiled_tensor = tf.expand_dims(tensor, axis=0)
    tiled_tensor = tf.tile(tiled_tensor,
                           [num_samples] + [1] * len(tensor.shape))
    # The tiled tensor has now shape [2, num_samples, batch_size, ...].
    delta = _get_delta(tiled_tensor)
    tiled_tensor = tf.stack(
        [tiled_tensor + delta, tiled_tensor - delta], axis=0)
    # Compute loss with shape [2, num_samples, batch_size].
    losses = loss_fn(
        tf.reshape(tiled_tensor,
                   [2 * num_samples, batch_size] + tensor.shape.as_list()[1:]))
    losses = tf.reshape(losses, [2, num_samples, batch_size])

    # Compute approximate gradient using broadcasting.
    shape = losses.shape.as_list() + [1] * (len(tensor.shape) - 1)
    shape = [(s or -1) for s in shape]  # Remove None.
    losses = tf.reshape(losses, shape)
    g = tf.reduce_mean((losses[0] - losses[1]) / (2. * delta), axis=0)
    return [i + 1, g / num_iterations + total_grad]

  _, g = tf.while_loop(
      cond,
      loop_body,
      loop_vars=[tf.constant(0.), tf.zeros_like(tensor)],
      parallel_iterations=1,
      back_prop=False)
  return [g]


@six.add_metaclass(abc.ABCMeta)
class UnrolledSPSA(object):
  """Abstract class that represents an optimizer based on SPSA."""


class UnrolledSPSAGradientDescent(UnrolledGradientDescent, UnrolledSPSA):
  """Optimizer for gradient-free attacks in https://arxiv.org/abs/1802.05666.

  Gradients estimates are computed using Simultaneous Perturbation Stochastic
  Approximation (SPSA).
  """

  def __init__(self, lr=0.1, lr_fn=None, fgsm=False,
               colocate_gradients_with_ops=False, delta=0.01, num_samples=32,
               num_iterations=4, loss_fn=None):
    super(UnrolledSPSAGradientDescent, self).__init__(
        lr, lr_fn, fgsm, colocate_gradients_with_ops)
    assert num_samples % 2 == 0, 'Number of samples must be even'
    self._delta = delta
    self._num_samples = num_samples // 2  # Since we mirror +/- delta later.
    self._num_iterations = num_iterations
    assert loss_fn is not None, 'loss_fn must be specified.'
    self._loss_fn = loss_fn

  def gradients(self, loss, x):
    return _spsa_gradients(self._loss_fn, x, self._delta, self._num_samples,
                           self._num_iterations)


# Syntactic sugar.
class UnrolledSPSAFGSMDescent(UnrolledSPSAGradientDescent):
  """Identical to UnrolledSPSAGradientDescent but forces FGSM steps."""

  def __init__(self, lr=.1, lr_fn=None,
               colocate_gradients_with_ops=False, delta=0.01, num_samples=32,
               num_iterations=4, loss_fn=None):
    super(UnrolledSPSAFGSMDescent, self).__init__(
        lr, lr_fn, True, colocate_gradients_with_ops, delta, num_samples,
        num_iterations, loss_fn)


class UnrolledSPSAAdam(UnrolledAdam, UnrolledSPSA):
  """Optimizer for gradient-free attacks in https://arxiv.org/abs/1802.05666.

  Gradients estimates are computed using Simultaneous Perturbation Stochastic
  Approximation (SPSA), combined with the ADAM update rule.
  """

  def __init__(self, lr=0.1, lr_fn=None, beta1=0.9, beta2=0.999, epsilon=1e-9,
               colocate_gradients_with_ops=False, delta=0.01, num_samples=32,
               num_iterations=4, loss_fn=None):
    super(UnrolledSPSAAdam, self).__init__(lr, lr_fn, beta1, beta2, epsilon,
                                           colocate_gradients_with_ops)
    assert num_samples % 2 == 0, 'Number of samples must be even'
    self._delta = delta
    self._num_samples = num_samples // 2  # Since we mirror +/- delta later.
    self._num_iterations = num_iterations
    assert loss_fn is not None, 'loss_fn must be specified.'
    self._loss_fn = loss_fn

  def gradients(self, loss, x):
    return _spsa_gradients(self._loss_fn, x, self._delta, self._num_samples,
                           self._num_iterations)


def _is_spsa_optimizer(cls):
  return issubclass(cls, UnrolledSPSA)


def wrap_optimizer(cls, **default_kwargs):
  """Wraps an optimizer such that __init__ uses the specified kwargs."""

  class WrapperUnrolledOptimizer(cls):

    def __init__(self, *args, **kwargs):
      new_kwargs = default_kwargs.copy()
      new_kwargs.update(kwargs)
      super(WrapperUnrolledOptimizer, self).__init__(*args, **new_kwargs)
  return WrapperUnrolledOptimizer


def _project_perturbation(perturbation, epsilon, input_image, image_bounds):
  """Project `perturbation` onto L-infinity ball of radius `epsilon`."""
  clipped_perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
  new_image = tf.clip_by_value(input_image + clipped_perturbation,
                               image_bounds[0], image_bounds[1])
  return new_image - input_image


def pgd_attack(loss_fn, input_image, epsilon, num_steps,
               optimizer=UnrolledGradientDescent(),
               project_perturbation=_project_perturbation,
               image_bounds=None, random_init=1.):
  """Projected gradient descent for generating adversarial images.

  Args:
    loss_fn: A callable which takes `input_image` and `label` as arguments, and
      returns the loss, a scalar Tensor, we will be minimized
    input_image: Tensor, a batch of images
    epsilon: float, the L-infinity norm of the maximum allowable perturbation
    num_steps: int, the number of steps of gradient descent
    optimizer: An `UnrolledOptimizer` object
    project_perturbation: A function, which will be used to enforce some
      constraint. It should have the same signature as `_project_perturbation`.
      Note that if you use a custom projection function, you should double-check
      your implementation, since an incorrect implementation will not error,
      and will appear to work fine.
    image_bounds: A pair of floats: minimum and maximum pixel value. If None
      (default), the bounds are assumed to be 0 and 1.
    random_init: Probability of starting from random location rather than
      nominal input image.

  Returns:
    adversarial version of `input_image`, with L-infinity difference less than
      epsilon, which tries to minimize loss_fn.
  """
  image_bounds = image_bounds or (0., 1.)
  random_shape = [tf.shape(input_image)[0]] + [1] * (len(input_image.shape) - 1)
  use_random_init = tf.cast(
      tf.random_uniform(random_shape) < float(random_init), tf.float32)
  init_perturbation = use_random_init * tf.random_uniform(
      tf.shape(input_image), minval=-epsilon, maxval=epsilon)
  init_perturbation = project_perturbation(init_perturbation,
                                           epsilon, input_image, image_bounds)
  init_optim_state = optimizer.init_state([init_perturbation])

  def loop_body(i, perturbation, flat_optim_state):
    """Update perturbation to input image."""
    optim_state = nest.pack_sequence_as(structure=init_optim_state,
                                        flat_sequence=flat_optim_state)
    loss = loss_fn(input_image + perturbation)
    new_perturbation_list, new_optim_state = optimizer.minimize(
        loss, [perturbation], optim_state)
    projected_perturbation = project_perturbation(
        new_perturbation_list[0], epsilon, input_image, image_bounds)
    return i + 1, projected_perturbation, nest.flatten(new_optim_state)

  def cond(i, *_):
    return tf.less(i, num_steps)

  flat_init_optim_state = nest.flatten(init_optim_state)
  _, final_perturbation, _ = tf.while_loop(
      cond,
      loop_body,
      loop_vars=[tf.constant(0.), init_perturbation, flat_init_optim_state],
      parallel_iterations=1,
      back_prop=False)

  adversarial_image = input_image + final_perturbation
  return tf.stop_gradient(adversarial_image)


@six.add_metaclass(abc.ABCMeta)
class Attack(snt.AbstractModule):
  """Defines an attack as a Sonnet module."""

  def __init__(self, predictor, specification, name, predictor_kwargs=None):
    super(Attack, self).__init__(name=name)
    self._predictor = predictor
    self._specification = specification
    if predictor_kwargs is None:
      self._kwargs = {'intermediate': {}, 'final': {}}
    else:
      self._kwargs = predictor_kwargs
    self._forced_mode = None
    self._target_class = None

  def _eval_fn(self, x, mode='intermediate'):
    """Runs the logits corresponding to `x`.

    Args:
      x: input to the predictor network.
      mode: Either "intermediate" or "final". Selects the desired predictor
        arguments.

    Returns:
      Tensor of logits.
    """
    if self._forced_mode is not None:
      mode = self._forced_mode
    return self._predictor(x, **self._kwargs[mode])

  @abc.abstractmethod
  def _build(self, inputs, labels):
    """Returns the adversarial attack around inputs."""

  @abc.abstractproperty
  def logits(self):
    """Returns the logits corresponding to the best attack."""

  @abc.abstractproperty
  def attack(self):
    """Returns the best attack."""

  @abc.abstractproperty
  def success(self):
    """Returns whether the attack was successful."""

  def force_mode(self, mode):
    """Only used by RestartedAttack to force the evaluation mode."""
    self._forced_mode = mode

  @property
  def target_class(self):
    """Returns the target class if this attack is a targeted attacks."""
    return self._target_class

  @target_class.setter
  def target_class(self, t):
    self._target_class = t


@six.add_metaclass(abc.ABCMeta)
class PGDAttack(Attack):
  """Defines a PGD attack."""

  def __init__(self, predictor, specification, epsilon, lr=.1, lr_fn=None,
               num_steps=20, num_restarts=1, input_bounds=(0., 1.),
               random_init=1., optimizer_builder=UnrolledGradientDescent,
               project_perturbation=_project_perturbation,
               predictor_kwargs=None):
    super(PGDAttack, self).__init__(predictor, specification, name='pgd',
                                    predictor_kwargs=predictor_kwargs)
    self._num_steps = num_steps
    self._num_restarts = num_restarts
    self._epsilon = epsilon
    self._lr = lr
    self._lr_fn = lr_fn
    self._input_bounds = input_bounds
    self._random_init = random_init
    self._optimizer_builder = optimizer_builder
    self._project_perturbation = project_perturbation

  # Helper functions.
  def prepare_inputs(self, inputs):
    """Tiles inputs according to number of restarts."""
    batch_size = tf.shape(inputs)[0]
    input_shape = list(inputs.shape.as_list()[1:])
    duplicated_inputs = tf.expand_dims(inputs, axis=0)
    # Shape is [num_restarts, batch_size, ...]
    duplicated_inputs = tf.tile(
        duplicated_inputs,
        [self._num_restarts, 1] + [1] * len(input_shape))
    # Shape is [num_restarts * batch_size, ...]
    duplicated_inputs = tf.reshape(
        duplicated_inputs, [self._num_restarts * batch_size] + input_shape)
    return batch_size, input_shape, duplicated_inputs

  def prepare_labels(self, labels):
    """Tiles labels according to number of restarts."""
    return tf.tile(labels, [self._num_restarts])

  def find_worst_attack(self, objective_fn, adversarial_input, batch_size,
                        input_shape):
    """Returns the attack that maximizes objective_fn."""
    adversarial_objective = objective_fn(adversarial_input)
    adversarial_objective = tf.reshape(adversarial_objective, [-1, batch_size])
    adversarial_input = tf.reshape(adversarial_input,
                                   [-1, batch_size] + input_shape)
    i = tf.argmax(adversarial_objective, axis=0)
    j = tf.cast(tf.range(tf.shape(adversarial_objective)[1]), i.dtype)
    ij = tf.stack([i, j], axis=1)
    return tf.gather_nd(adversarial_input, ij)


def _maximize_margin(bounds):
  # Bounds has shape [num_restarts, batch_size, num_specs].
  return tf.reduce_max(bounds, axis=-1)


def _any_greater(bounds):
  # Bounds has shape [batch_size, num_specs].
  bounds = tf.reduce_max(bounds, axis=-1)
  return bounds > 0.


def _maximize_topk_hinge_margin(bounds, k=5, margin=.1):
  # Bounds has shape [num_restarts, batch_size, num_specs].
  b = tf.nn.top_k(bounds, k=k, sorted=False).values
  return tf.reduce_sum(tf.minimum(b, margin), axis=-1)


def _topk_greater(bounds, k=5):
  # Bounds has shape [batch_size, num_specs].
  b = tf.nn.top_k(bounds, k=k, sorted=False).values
  return tf.reduce_min(b, axis=-1) > 0.


class UntargetedPGDAttack(PGDAttack):
  """Defines an untargeted PGD attack."""

  def __init__(self, predictor, specification, epsilon, lr=.1, lr_fn=None,
               num_steps=20, num_restarts=1, input_bounds=(0., 1.),
               random_init=1., optimizer_builder=UnrolledGradientDescent,
               project_perturbation=_project_perturbation,
               objective_fn=_maximize_margin, success_fn=_any_greater,
               predictor_kwargs=None):
    super(UntargetedPGDAttack, self).__init__(
        predictor, specification, epsilon, lr, lr_fn, num_steps, num_restarts,
        input_bounds, random_init, optimizer_builder, project_perturbation,
        predictor_kwargs)
    self._objective_fn = objective_fn
    self._success_fn = success_fn

  def _build(self, inputs, labels):
    batch_size, input_shape, duplicated_inputs = self.prepare_inputs(inputs)
    duplicated_labels = self.prepare_labels(labels)

    # Define objectives.
    def objective_fn(x):
      model_logits = self._eval_fn(x)  # [restarts * batch_size, output].
      model_logits = tf.reshape(
          model_logits, [self._num_restarts, batch_size, -1])
      bounds = self._specification.evaluate(model_logits)
      # Output has dimension [num_restarts, batch_size].
      return self._objective_fn(bounds)

    # Only used for SPSA.
    # The input to this loss is the perturbation (not the image).
    # The first dimension corresponds to the number of SPSA samples.
    # Shape of perturbations is [num_samples, restarts * batch_size, ...]
    def spsa_loss_fn(perturbation):
      """Computes the loss per SPSA sample."""
      x = tf.reshape(
          perturbation + tf.expand_dims(duplicated_inputs, axis=0),
          [-1] + duplicated_inputs.shape.as_list()[1:])
      model_logits = self._eval_fn(x)
      num_outputs = tf.shape(model_logits)[1]
      model_logits = tf.reshape(
          model_logits, [-1, batch_size, num_outputs])
      bounds = self._specification.evaluate(model_logits)
      losses = -self._objective_fn(bounds)
      return tf.reshape(losses, [-1])

    def reduced_loss_fn(x):
      # Pick worse attack, output has shape [num_restarts, batch_size].
      return -tf.reduce_sum(objective_fn(x))

    # Use targeted attacks as specified by the specification.
    if _is_spsa_optimizer(self._optimizer_builder):
      optimizer = self._optimizer_builder(lr=self._lr, lr_fn=self._lr_fn,
                                          loss_fn=spsa_loss_fn)
    else:
      optimizer = self._optimizer_builder(lr=self._lr, lr_fn=self._lr_fn)

    adversarial_input = pgd_attack(
        reduced_loss_fn, duplicated_inputs, epsilon=self._epsilon,
        num_steps=self._num_steps, image_bounds=self._input_bounds,
        random_init=self._random_init, optimizer=optimizer,
        project_perturbation=self._project_perturbation)
    adversarial_input = self.adapt(duplicated_inputs, adversarial_input,
                                   duplicated_labels)

    self._attack = self.find_worst_attack(objective_fn, adversarial_input,
                                          batch_size, input_shape)
    self._logits = self._eval_fn(self._attack, mode='final')
    self._success = self._success_fn(self._specification.evaluate(self._logits))
    return self._attack

  @property
  def logits(self):
    self._ensure_is_connected()
    return self._logits

  @property
  def attack(self):
    self._ensure_is_connected()
    return self._attack

  @property
  def success(self):
    self._ensure_is_connected()
    return self._success

  def adapt(self, original_inputs, adversarial_inputs, labels):
    """Function called after PGD to adapt adversarial examples."""
    return adversarial_inputs


class UntargetedTop5PGDAttack(UntargetedPGDAttack):
  """Defines an untargeted PGD attack on top-5."""

  def __init__(self, predictor, specification, epsilon, lr=.1, lr_fn=None,
               num_steps=20, num_restarts=1, input_bounds=(0., 1.),
               random_init=1., optimizer_builder=UnrolledGradientDescent,
               project_perturbation=_project_perturbation,
               objective_fn=_maximize_topk_hinge_margin, predictor_kwargs=None):
    super(UntargetedTop5PGDAttack, self).__init__(
        predictor, specification, epsilon, lr=lr, lr_fn=lr_fn,
        num_steps=num_steps, num_restarts=num_restarts,
        input_bounds=input_bounds, random_init=random_init,
        optimizer_builder=optimizer_builder,
        project_perturbation=project_perturbation, objective_fn=objective_fn,
        success_fn=_topk_greater, predictor_kwargs=predictor_kwargs)


class UntargetedAdaptivePGDAttack(UntargetedPGDAttack):
  """Uses an adaptive scheme to pick attacks that are just strong enough."""

  def adapt(self, original_inputs, adversarial_inputs, labels):
    """Runs binary search to find the first misclassified input."""
    batch_size = tf.shape(original_inputs)[0]
    binary_search_iterations = 10

    def cond(i, *_):
      return tf.less(i, binary_search_iterations)

    def get(m):
      m = tf.reshape(m, [batch_size] + [1] * (len(original_inputs.shape) - 1))
      return (adversarial_inputs - original_inputs) * m + original_inputs

    def is_attack_successful(m):
      logits = self._eval_fn(get(m))
      return self._success_fn(self._specification.evaluate(logits))

    def loop_body(i, lower, upper):
      m = (lower + upper) * .5
      success = is_attack_successful(m)
      new_lower = tf.where(success, lower, m)
      new_upper = tf.where(success, m, upper)
      return i + 1, new_lower, new_upper

    lower = tf.zeros(shape=[batch_size])
    upper = tf.ones(shape=[batch_size])
    _, lower, upper = tf.while_loop(
        cond,
        loop_body,
        loop_vars=[tf.constant(0.), lower, upper],
        parallel_iterations=1,
        back_prop=False)
    # If lower is incorrectly classified, pick lower; otherwise pick upper.
    success = is_attack_successful(lower)
    return get(tf.where(success, lower, upper))


class MultiTargetedPGDAttack(PGDAttack):
  """Runs targeted attacks for each specification."""

  def __init__(self, predictor, specification, epsilon, lr=.1, lr_fn=None,
               num_steps=20, num_restarts=1, input_bounds=(0., 1.),
               random_init=1., optimizer_builder=UnrolledGradientDescent,
               project_perturbation=_project_perturbation,
               max_specifications=0, random_specifications=False,
               predictor_kwargs=None):
    super(MultiTargetedPGDAttack, self).__init__(
        predictor, specification, epsilon, lr=lr, lr_fn=lr_fn,
        num_steps=num_steps, num_restarts=num_restarts,
        input_bounds=input_bounds, random_init=random_init,
        optimizer_builder=optimizer_builder,
        project_perturbation=project_perturbation,
        predictor_kwargs=predictor_kwargs)
    self._max_specifications = max_specifications
    self._random_specifications = random_specifications

  def _build(self, inputs, labels):
    batch_size = tf.shape(inputs)[0]
    num_specs = self._specification.num_specifications
    if self._max_specifications > 0 and self._max_specifications < num_specs:
      model_logits = self._eval_fn(inputs)
      bounds = self._specification.evaluate(model_logits)
      _, idx = tf.math.top_k(bounds, k=self._max_specifications, sorted=False)
      if self._random_specifications:
        idx = tf.random.uniform(shape=tf.shape(idx),
                                maxval=self._specification.num_specifications,
                                dtype=idx.dtype)
      idx = tf.tile(tf.expand_dims(idx, 0), [self._num_restarts, 1, 1])
      select_fn = lambda x: tf.gather(x, idx, batch_dims=len(idx.shape) - 1)
    else:
      select_fn = lambda x: x

    input_shape = list(inputs.shape.as_list()[1:])
    duplicated_inputs = tf.expand_dims(inputs, axis=0)
    # Shape is [num_restarts * num_specifications, batch_size, ...]
    duplicated_inputs = tf.tile(
        duplicated_inputs,
        [self._num_restarts * num_specs, 1] + [1] * len(input_shape))
    # Shape is [num_restarts * num_specifications * batch_size, ...]
    duplicated_inputs = tf.reshape(duplicated_inputs, [-1] + input_shape)

    def objective_fn(x):
      # Output has shape [restarts * num_specs * batch_size, output].
      model_logits = self._eval_fn(x)
      model_logits = tf.reshape(
          model_logits, [self._num_restarts, num_specs, batch_size, -1])
      # Output has shape [num_restarts, batch_size, num_specs].
      return self._specification.evaluate(model_logits)

    def reduced_loss_fn(x):
      # Negate as we minimize.
      return -tf.reduce_sum(select_fn(objective_fn(x)))

    # Use targeted attacks as specified by the specification.
    if _is_spsa_optimizer(self._optimizer_builder):
      raise ValueError('"UnrolledSPSA*" unsupported in '
                       'MultiTargetedPGDAttack')
    optimizer = self._optimizer_builder(lr=self._lr, lr_fn=self._lr_fn)
    adversarial_input = pgd_attack(
        reduced_loss_fn, duplicated_inputs,
        epsilon=self._epsilon, num_steps=self._num_steps,
        image_bounds=self._input_bounds, random_init=self._random_init,
        optimizer=optimizer, project_perturbation=self._project_perturbation)
    # Get best attack.
    adversarial_objective = objective_fn(adversarial_input)
    adversarial_objective = tf.transpose(adversarial_objective, [0, 2, 1])
    adversarial_objective = tf.reshape(adversarial_objective, [-1, batch_size])
    adversarial_input = tf.reshape(adversarial_input,
                                   [-1, batch_size] + input_shape)
    i = tf.argmax(adversarial_objective, axis=0)
    j = tf.cast(tf.range(tf.shape(adversarial_objective)[1]), i.dtype)
    ij = tf.stack([i, j], axis=1)
    self._attack = tf.gather_nd(adversarial_input, ij)
    self._logits = self._eval_fn(self._attack, mode='final')
    # Count the number of sample that violate any specification.
    bounds = tf.reduce_max(self._specification.evaluate(self._logits), axis=1)
    self._success = (bounds > 0.)
    return self._attack

  @property
  def logits(self):
    self._ensure_is_connected()
    return self._logits

  @property
  def attack(self):
    self._ensure_is_connected()
    return self._attack

  @property
  def success(self):
    self._ensure_is_connected()
    return self._success


class MemoryEfficientMultiTargetedPGDAttack(PGDAttack):
  """Defines a targeted PGD attack for each specification using while_loop."""

  def __init__(self, predictor, specification, epsilon, lr=.1, lr_fn=None,
               num_steps=20, num_restarts=1, input_bounds=(0., 1.),
               random_init=1., optimizer_builder=UnrolledGradientDescent,
               project_perturbation=_project_perturbation,
               max_specifications=0, random_specifications=False,
               predictor_kwargs=None):
    super(MemoryEfficientMultiTargetedPGDAttack, self).__init__(
        predictor, specification, epsilon, lr=lr, lr_fn=lr_fn,
        num_steps=num_steps, num_restarts=num_restarts,
        input_bounds=input_bounds, random_init=random_init,
        optimizer_builder=optimizer_builder,
        project_perturbation=project_perturbation,
        predictor_kwargs=predictor_kwargs)
    self._max_specifications = max_specifications
    self._random_specifications = random_specifications

  def _build(self, inputs, labels):
    batch_size, input_shape, duplicated_inputs = self.prepare_inputs(inputs)
    if (self._max_specifications > 0 and
        self._max_specifications < self._specification.num_specifications):
      num_specs = self._max_specifications
      model_logits = self._eval_fn(inputs)
      bounds = self._specification.evaluate(model_logits)
      _, idx = tf.math.top_k(bounds, k=num_specs, sorted=False)
      if self._random_specifications:
        idx = tf.random.uniform(shape=tf.shape(idx),
                                maxval=self._specification.num_specifications,
                                dtype=idx.dtype)
      idx = tf.tile(tf.expand_dims(idx, 0), [self._num_restarts, 1, 1])
      def select_fn(x, i):
        return tf.squeeze(
            tf.gather(x, tf.expand_dims(idx[:, :, i], -1),
                      batch_dims=len(idx.shape) - 1),
            axis=-1)
    else:
      num_specs = self._specification.num_specifications
      select_fn = lambda x, i: x[:, :, i]

    def objective_fn(x):
      model_logits = self._eval_fn(x)  # [restarts * batch_size, output].
      model_logits = tf.reshape(
          model_logits, [self._num_restarts, batch_size, -1])
      # Output has dimension [num_restarts, batch_size, num_specifications].
      return self._specification.evaluate(model_logits)

    def flat_objective_fn(x):
      return _maximize_margin(objective_fn(x))

    def build_loss_fn(idx):
      def _reduced_loss_fn(x):
        # Pick worse attack, output has shape [num_restarts, batch_size].
        return -tf.reduce_sum(select_fn(objective_fn(x), idx))
      return _reduced_loss_fn

    if _is_spsa_optimizer(self._optimizer_builder):
      raise ValueError('"UnrolledSPSA*" unsupported in '
                       'MultiTargetedPGDAttack')
    optimizer = self._optimizer_builder(lr=self._lr, lr_fn=self._lr_fn)

    # Run a separate PGD attack for each specification.
    def cond(spec_idx, unused_attack, success):
      # If we are already successful, we break.
      return tf.logical_and(spec_idx < num_specs,
                            tf.logical_not(tf.reduce_all(success)))

    def body(spec_idx, attack, success):
      """Runs a separate PGD attack for each specification."""
      adversarial_input = pgd_attack(
          build_loss_fn(spec_idx), duplicated_inputs,
          epsilon=self._epsilon, num_steps=self._num_steps,
          image_bounds=self._input_bounds, random_init=self._random_init,
          optimizer=optimizer, project_perturbation=self._project_perturbation)
      new_attack = self.find_worst_attack(flat_objective_fn, adversarial_input,
                                          batch_size, input_shape)
      new_logits = self._eval_fn(new_attack)
      # Count the number of sample that violate any specification.
      new_success = _any_greater(self._specification.evaluate(new_logits))
      # The first iteration always sets the attack and logits.
      use_new_values = tf.logical_or(tf.equal(spec_idx, 0), new_success)
      print_op = tf.print('Processed specification #', spec_idx)
      with tf.control_dependencies([print_op]):
        new_spec_idx = spec_idx + 1
      return (new_spec_idx,
              tf.where(use_new_values, new_attack, attack),
              tf.logical_or(success, new_success))

    _, self._attack, self._success = tf.while_loop(
        cond, body, back_prop=False, parallel_iterations=1,
        loop_vars=[
            tf.constant(0, dtype=tf.int32),
            inputs,
            tf.zeros([tf.shape(inputs)[0]], dtype=tf.bool),
        ])
    self._logits = self._eval_fn(self._attack, mode='final')
    return self._attack

  @property
  def logits(self):
    self._ensure_is_connected()
    return self._logits

  @property
  def attack(self):
    self._ensure_is_connected()
    return self._attack

  @property
  def success(self):
    self._ensure_is_connected()
    return self._success


class RestartedAttack(Attack):
  """Wraps an attack to run it multiple times using a tf.while_loop."""

  def __init__(self, inner_attack, num_restarts=1):
    super(RestartedAttack, self).__init__(
        inner_attack._predictor,  # pylint: disable=protected-access
        inner_attack._specification,  # pylint: disable=protected-access
        name='restarted_' + inner_attack.module_name,
        predictor_kwargs=inner_attack._kwargs)  # pylint: disable=protected-access
    self._inner_attack = inner_attack
    self._num_restarts = num_restarts
    # Prevent the inner attack from updating batch normalization statistics.
    self._inner_attack.force_mode('intermediate')

  def _build(self, inputs, labels):

    def cond(i, unused_attack, success):
      # If we are already successful, we break.
      return tf.logical_and(i < self._num_restarts,
                            tf.logical_not(tf.reduce_all(success)))

    def body(i, attack, success):
      new_attack = self._inner_attack(inputs, labels)
      new_success = self._inner_attack.success
      # The first iteration always sets the attack.
      use_new_values = tf.logical_or(tf.equal(i, 0), new_success)
      return (i + 1,
              tf.where(use_new_values, new_attack, attack),
              tf.logical_or(success, new_success))

    _, self._attack, self._success = tf.while_loop(
        cond, body, back_prop=False, parallel_iterations=1,
        loop_vars=[
            tf.constant(0, dtype=tf.int32),
            inputs,
            tf.zeros([tf.shape(inputs)[0]], dtype=tf.bool),
        ])
    self._logits = self._eval_fn(self._attack, mode='final')
    return self._attack

  @property
  def logits(self):
    self._ensure_is_connected()
    return self._logits

  @property
  def attack(self):
    self._ensure_is_connected()
    return self._attack

  @property
  def success(self):
    self._ensure_is_connected()
    return self._success
