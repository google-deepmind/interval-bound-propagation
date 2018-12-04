# coding=utf-8
# Copyright 2018 The Interval Bound Propagation Authors.
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

"""Utilies to define PGD attacks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

import sonnet as snt
import tensorflow as tf

nest = tf.contrib.framework.nest


class UnrolledOptimizer(object):
  """In graph optimizer to be used in tf.while_loop."""

  __metaclass__ = abc.ABCMeta

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

  def __init__(self, lr=.1, lr_fn=None,
               colocate_gradients_with_ops=False):
    super(UnrolledGradientDescent, self).__init__(
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    self._lr = lr
    self._lr_fn = (lambda l, i: l) if lr_fn is None else lr_fn

  def init_state(self, unused_x):
    return self._State(tf.constant(0, dtype=tf.float32))

  def minimize(self, loss, x, optim_state):
    """Refer to parent class documentation."""
    lr = self._lr_fn(self._lr, optim_state.iteration)
    grads = tf.gradients(
        loss, x, colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    new_x = [None] * len(x)
    for i in xrange(len(x)):
      new_x[i] = x[i] - lr * grads[i]
    new_optim_state = self._State(optim_state.iteration + 1)
    return new_x, new_optim_state


class UnrolledAdam(UnrolledOptimizer):
  """The Adam optimizer defined in https://arxiv.org/abs/1412.6980."""

  _State = collections.namedtuple('State', ['t', 'm', 'u'])  # pylint: disable=invalid-name

  def __init__(self, lr=0.1, lr_fn=None,
               beta1=0.9, beta2=0.999, epsilon=1e-9,
               colocate_gradients_with_ops=False):
    super(UnrolledAdam, self).__init__(
        colocate_gradients_with_ops=colocate_gradients_with_ops)
    self._lr = lr
    self._lr_fn = (lambda l, i: l) if lr_fn is None else lr_fn
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

  def init_state(self, x):
    return self._State(
        t=tf.constant(0, dtype=tf.float32),
        m=[tf.zeros_like(v) for v in x],
        u=[tf.zeros_like(v) for v in x])

  def _apply_gradients(self, grads, x, optim_state):
    """Refer to parent class documentation."""
    lr = self._lr_fn(self._lr, optim_state.t)
    new_optim_state = self._State(
        t=optim_state.t + 1.,
        m=[None] * len(x),
        u=[None] * len(x))
    t = new_optim_state.t
    new_x = [None] * len(x)
    for i in xrange(len(x)):
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
    grads = tf.gradients(
        loss, x, colocate_gradients_with_ops=self._colocate_gradients_with_ops)
    return self._apply_gradients(grads, x, optim_state)


def _project_perturbation(perturbation, epsilon, input_image, image_bounds):
  """Project `perturbation` onto L-infinity ball of radius `epsilon`."""
  clipped_perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
  new_image = tf.clip_by_value(input_image + clipped_perturbation,
                               image_bounds[0], image_bounds[1])
  return new_image - input_image


def pgd_attack(loss_fn, input_image, epsilon, num_steps,
               optimizer=UnrolledGradientDescent(),
               project_perturbation=_project_perturbation,
               image_bounds=None,
               random_init=True):
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
    random_init: See module docstring.

  Returns:
    adversarial version of `input_image`, with L-infinity difference less than
      epsilon, which tries to minimize loss_fn.
  """
  image_bounds = image_bounds or (0., 1.)
  if random_init:
    init_perturbation = tf.random_uniform(tf.shape(input_image),
                                          minval=-epsilon, maxval=epsilon)
  else:
    init_perturbation = tf.zeros_like(input_image)
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


class Attack(snt.AbstractModule):
  """Defines an attack as a Sonnet module."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, name):
    super(Attack, self).__init__(name=name)

  @abc.abstractproperty
  def logits(self):
    """Returns the logits corresponding to the best attack."""

  @abc.abstractproperty
  def attack(self):
    """Returns the best attack."""

  @abc.abstractproperty
  def accuracy(self):
    """Returns the accuracy under attack."""


class PGDAttack(Attack):
  """Defines a PGD attack."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, predictor, specification, epsilon, lr=.1, lr_fn=None,
               num_steps=20, num_restarts=1, input_bounds=(0., 1.),
               optimizer_builder=UnrolledGradientDescent):
    super(PGDAttack, self).__init__(name='pgd')
    self._predictor = predictor
    self._specification = specification
    self._num_steps = num_steps
    self._num_restarts = num_restarts
    self._epsilon = epsilon
    self._lr = lr
    self._lr_fn = lr_fn
    self._input_bounds = input_bounds
    self._optimizer_builder = optimizer_builder


class UntargetedPGDAttack(PGDAttack):
  """Defines an untargeted PGD attack."""

  def _build(self, labels):
    batch_size = tf.shape(self._specification.c)[0]
    input_shape = list(self._predictor.inputs.shape.as_list()[1:])
    duplicated_inputs = tf.expand_dims(self._predictor.inputs, axis=0)
    # Shape is [num_restarts, batch_size, ...]
    duplicated_inputs = tf.tile(
        duplicated_inputs,
        [self._num_restarts, 1] + [1] * len(input_shape))
    # Shape is [num_restarts * batch_size, ...]
    duplicated_inputs = tf.reshape(duplicated_inputs, [-1] + input_shape)

    def eval_fn(x):
      return self._predictor(x, is_training=False, passthrough=True)

    def objective_fn(x):
      model_logits = eval_fn(x)  # [restarts * batch_size, output].
      model_logits = tf.reshape(
          model_logits, [self._num_restarts, batch_size, -1])
      # c has shape [batch_size, num_specs, output].
      obj = tf.einsum('rbo,bso->rsb', model_logits, self._specification.c)
      # Output has dimension [num_restarts, batch_size].
      return tf.reduce_max(obj, axis=1)

    def reduced_loss_fn(x):
      # Pick worse attack, output has shape [num_restarts, batch_size].
      return -tf.reduce_sum(objective_fn(x))

    # Use targeted attacks as specified by the specification.
    optimizer = self._optimizer_builder(lr=self._lr, lr_fn=self._lr_fn)
    adversarial_input = pgd_attack(
        reduced_loss_fn, duplicated_inputs,
        epsilon=self._epsilon, num_steps=self._num_steps,
        image_bounds=self._input_bounds, random_init=True, optimizer=optimizer)
    # Get best attack.
    adversarial_objective = objective_fn(adversarial_input)
    adversarial_objective = tf.reshape(adversarial_objective, [-1, batch_size])
    adversarial_input = tf.reshape(adversarial_input,
                                   [-1, batch_size] + input_shape)
    i = tf.argmax(adversarial_objective, axis=0)
    j = tf.cast(tf.range(tf.shape(adversarial_objective)[1]), i.dtype)
    ij = tf.stack([i, j], axis=1)
    self._attack = tf.gather_nd(adversarial_input, ij)
    self._logits = eval_fn(self._attack)
    correct_examples = tf.equal(labels, tf.argmax(self._logits, 1))
    self._accuracy = tf.reduce_mean(tf.cast(correct_examples, tf.float32))
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
  def accuracy(self):
    self._ensure_is_connected()
    return self._accuracy


class TargetedPGDAttack(PGDAttack):
  """Runs targeted attacks for each specification."""

  def _build(self, labels):
    batch_size = tf.shape(self._specification.c)[0]
    num_specs = tf.shape(self._specification.c)[1]
    input_shape = list(self._predictor.inputs.shape.as_list()[1:])
    duplicated_inputs = tf.expand_dims(self._predictor.inputs, axis=0)
    # Shape is [num_restarts * num_specifications, batch_size, ...]
    duplicated_inputs = tf.tile(
        duplicated_inputs,
        [self._num_restarts * num_specs, 1] + [1] * len(input_shape))
    # Shape is [num_restarts * num_specifications * batch_size, ...]
    duplicated_inputs = tf.reshape(duplicated_inputs, [-1] + input_shape)

    def eval_fn(x):
      return self._predictor(x, is_training=False, passthrough=True)

    def objective_fn(x):
      model_logits = eval_fn(x)  # [restarts * num_specs * batch_size, output].
      model_logits = tf.reshape(
          model_logits, [self._num_restarts, num_specs, batch_size, -1])
      # c has shape [batch_size, num_specs, output].
      obj = tf.einsum('rsbo,bso->rsb', model_logits, self._specification.c)
      # Output has dimension [num_restarts, num_objectives, batch_size]
      return obj

    def reduced_loss_fn(x):
      # Negate as we minimize.
      return -tf.reduce_sum(objective_fn(x))

    # Use targeted attacks as specified by the specification.
    optimizer = self._optimizer_builder(lr=self._lr, lr_fn=self._lr_fn)
    adversarial_input = pgd_attack(
        reduced_loss_fn, duplicated_inputs,
        epsilon=self._epsilon, num_steps=self._num_steps,
        image_bounds=self._input_bounds, random_init=True, optimizer=optimizer)
    # Get best attack.
    adversarial_objective = objective_fn(adversarial_input)
    adversarial_objective = tf.reshape(adversarial_objective, [-1, batch_size])
    adversarial_input = tf.reshape(adversarial_input,
                                   [-1, batch_size] + input_shape)
    i = tf.argmax(adversarial_objective, axis=0)
    j = tf.cast(tf.range(tf.shape(adversarial_objective)[1]), i.dtype)
    ij = tf.stack([i, j], axis=1)
    self._attack = tf.gather_nd(adversarial_input, ij)
    self._logits = eval_fn(self._attack)
    correct_examples = tf.equal(labels, tf.argmax(self._logits, 1))
    self._accuracy = tf.reduce_mean(tf.cast(correct_examples, tf.float32))
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
  def accuracy(self):
    self._ensure_is_connected()
    return self._accuracy
