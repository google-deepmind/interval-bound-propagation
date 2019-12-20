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

"""CROWN-IBP implementation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
from interval_bound_propagation.src import bounds
from interval_bound_propagation.src import fastlin
from interval_bound_propagation.src import loss
from interval_bound_propagation.src import model
from interval_bound_propagation.src import specification as specification_lib
from interval_bound_propagation.src import utils
from interval_bound_propagation.src import verifiable_wrapper
import tensorflow.compat.v1 as tf


class BackwardBounds(bounds.AbstractBounds):
  """Implementation of backward bound propagation used by CROWN."""

  def __init__(self, lower, upper):
    super(BackwardBounds, self).__init__()
    # Setting "lower" or "upper" to None will avoid creating the computation
    # graph for CROWN lower or upper bounds. For verifiable training, only the
    # upper bound is necessary.
    self._lower = lower
    self._upper = upper

  @property
  def lower(self):
    return self._lower

  @property
  def upper(self):
    return self._upper

  @property
  def shape(self):
    return self.lower.shape.as_list()

  def concretize(self):
    """Returns lower and upper interval bounds."""
    lb = ub = None
    if self.lower is not None:
      lb = (
          tf.einsum('nsi,ni->ns',
                    self._reshape_to_rank(tf.maximum(self.lower.w, 0), 3),
                    self._reshape_to_rank(self.lower.lower, 2)) +
          tf.einsum('nsi,ni->ns',
                    self._reshape_to_rank(tf.minimum(self.lower.w, 0), 3),
                    self._reshape_to_rank(self.lower.upper, 2)))
      lb += self.lower.b
    if self.upper is not None:
      ub = (
          tf.einsum('nsi,ni->ns',
                    self._reshape_to_rank(tf.maximum(self.upper.w, 0), 3),
                    self._reshape_to_rank(self.upper.upper, 2)) +
          tf.einsum('nsi,ni->ns',
                    self._reshape_to_rank(tf.minimum(self.upper.w, 0), 3),
                    self._reshape_to_rank(self.upper.lower, 2)))
      ub += self.upper.b
    return bounds.IntervalBounds(lb, ub)

  @classmethod
  def convert(cls, other_bounds):
    if isinstance(other_bounds, cls):
      return other_bounds
    raise RuntimeError('BackwardBounds does not support conversion from any '
                       'other bound type.')

  def apply_linear(self, wrapper, w, b):
    """Propagate CROWN bounds backward through a linear layer."""
    def _linear_propagate(bound):
      """Propagate one side of the bound."""
      new_bound_w = tf.einsum('nsk,lk->nsl', bound.w, w)
      if b is not None:
        bias = tf.tensordot(bound.w, b, axes=1)
      return fastlin.LinearExpression(w=new_bound_w, b=bias + bound.b,
                                      lower=wrapper.input_bounds.lower,
                                      upper=wrapper.input_bounds.upper)
    ub_expr = _linear_propagate(self.upper) if self.upper else None
    lb_expr = _linear_propagate(self.lower) if self.lower else None
    return BackwardBounds(lb_expr, ub_expr)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    """Propagate CROWN bounds backward through a convolution layer."""
    def _conv2d_propagate(bound):
      """Propagate one side of the bound."""
      s = tf.shape(bound.w)
      # Variable bound.w has shape (batch_size, num_specs, H, W, C),
      # resize it to (batch_size * num_specs, H, W, C) for batch processing.
      effective_batch_size = tf.reshape(s[0] * s[1], [1])
      batched_shape = tf.concat([effective_batch_size, s[2:]], 0)
      # The output of a deconvolution is the input shape of the corresponding
      # convolution.
      output_shape = wrapper.input_bounds.lower.shape
      batched_output_shape = tf.concat([effective_batch_size, output_shape[1:]],
                                       0)
      # Batched transpose convolution for efficiency.
      bound_batch = tf.nn.conv2d_transpose(tf.reshape(bound.w, batched_shape),
                                           filter=w,
                                           output_shape=batched_output_shape,
                                           strides=[1] + list(strides) + [1],
                                           padding=padding)
      # Reshape results to (batch_size, num_specs, new_H, new_W, new_C).
      new_shape = tf.concat(
          [tf.reshape(s[0], [1]), tf.reshape(s[1], [1]), output_shape[1:]], 0)
      new_bound_w = tf.reshape(bound_batch, new_shape)
      # If this convolution has bias, multiplies it with current w.
      bias = 0
      if b is not None:
        # Variable bound.w has dimension (batch_size, num_specs, H, W, C),
        # accumulate H and W, and do a dot product for each channel C.
        bias = tf.tensordot(tf.reduce_sum(bound.w, [2, 3]), b, axes=1)
      return fastlin.LinearExpression(w=new_bound_w, b=bias + bound.b,
                                      lower=wrapper.input_bounds.lower,
                                      upper=wrapper.input_bounds.upper)
    ub_expr = _conv2d_propagate(self.upper) if self.upper else None
    lb_expr = _conv2d_propagate(self.lower) if self.lower else None
    return BackwardBounds(lb_expr, ub_expr)

  def _get_monotonic_fn_bound(self, wrapper, fn):
    """Compute CROWN upper and lower linear bounds for a given function fn."""
    # Get lower and upper bounds from forward IBP pass.
    lb, ub = wrapper.input_bounds.lower, wrapper.input_bounds.upper
    if fn.__name__ == 'relu':
      # CROWN upper and lower linear bounds for ReLU.
      f_lb = tf.minimum(lb, 0)
      f_ub = tf.maximum(ub, 0)
      # When both ub and lb are very close to 0 we might have NaN issue,
      # so we have to avoid this happening.
      f_ub = tf.maximum(f_ub, f_lb + 1e-8)
      # CROWN upper/lower scaling matrices and biases.
      ub_scaling_matrix = f_ub / (f_ub - f_lb)
      ub_bias = -f_lb * ub_scaling_matrix
      # Expand dimension for using broadcast later.
      ub_scaling_matrix = tf.expand_dims(ub_scaling_matrix, 1)
      lb_scaling_matrix = tf.cast(tf.greater(ub_scaling_matrix, .5),
                                  dtype=tf.float32)
      lb_bias = 0.
    # For 'apply' fn we need to differentiate them through the wrapper.
    elif isinstance(wrapper, verifiable_wrapper.ImageNormWrapper):
      inner_module = wrapper.inner_module
      ub_scaling_matrix = lb_scaling_matrix = inner_module.scale
      ub_bias = - inner_module.offset * inner_module.scale
      lb_bias = ub_bias
    else:
      raise NotImplementedError('monotonic fn {} is not supported '
                                'by BackwardBounds'.format(fn.__name__))
    return ub_scaling_matrix, lb_scaling_matrix, ub_bias, lb_bias

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args):
    """Propagate CROWN bounds backward through a increasing monotonic fn."""
    # Function _get_monotonic_fn_bound returns matrix and bias term for linear
    # relaxation.
    (ub_scaling_matrix, lb_scaling_matrix,
     ub_bias, lb_bias) = self._get_monotonic_fn_bound(wrapper, fn)
    def _propagate_monotonic_fn(bound, ub_mult, lb_mult):
      # Matrix multiplication by a diagonal matrix.
      new_bound_w = ub_mult * ub_scaling_matrix + lb_mult * lb_scaling_matrix
      # Matrix vector product for the bias term. ub_bias or lb_bias might be 0
      # or a constant, or need broadcast. They will be handled optimally.
      b = self._matvec(ub_mult, ub_bias) + self._matvec(lb_mult, lb_bias)
      return fastlin.LinearExpression(w=new_bound_w, b=bound.b + b,
                                      lower=wrapper.input_bounds.lower,
                                      upper=wrapper.input_bounds.upper)
    # Multiplies w to upper or lower scaling terms according to its sign.
    ub_expr = _propagate_monotonic_fn(
        self.upper, tf.maximum(self.upper.w, 0),
        tf.minimum(self.upper.w, 0)) if self.upper else None
    lb_expr = _propagate_monotonic_fn(
        self.lower, tf.minimum(self.lower.w, 0),
        tf.maximum(self.lower.w, 0)) if self.lower else None
    return BackwardBounds(lb_expr, ub_expr)

  def apply_batch_reshape(self, wrapper, shape):
    """Propagate CROWN bounds backward through a reshape layer."""
    input_shape = wrapper.input_bounds.lower.shape[1:]
    def _propagate_batch_flatten(bound):
      new_bound_w = tf.reshape(
          bound.w, tf.concat([tf.shape(bound.w)[:2], input_shape], 0))
      return fastlin.LinearExpression(w=new_bound_w, b=bound.b,
                                      lower=wrapper.input_bounds.lower,
                                      upper=wrapper.input_bounds.upper)
    ub_expr = _propagate_batch_flatten(self.upper) if self.upper else None
    lb_expr = _propagate_batch_flatten(self.lower) if self.lower else None
    return BackwardBounds(lb_expr, ub_expr)

  @staticmethod
  def _reshape_to_rank(a, rank):
    """Reshapes to the given rank while keeping the first (rank-1) dims."""
    shape = tf.concat([tf.shape(a)[0:(rank - 1)], [-1]], axis=-1)
    return tf.reshape(a, shape)

  @staticmethod
  def _matvec(a, b):
    """Specialized matvec detecting the case where b is 0 or constant."""
    if isinstance(b, int) or isinstance(b, float):
      if b == 0:
        # For efficiency we directly return constant 0, no graph generated.
        return 0
      else:
        # Broadcasting a constant.
        return a * b
    elif len(b.shape) == 1:
      # Need to broadcast against all examples in the batch. This can be done
      # using an einsum "tf.einsum('ns...c,c->ns', a, b)" but it currently
      # triggers a compiler bug on TPUs, thus we use the following instead.
      return tf.einsum('nsc,c->ns', tf.reduce_sum(a, [2, 3]), b)
    else:
      # Normal 1D or 3D mat-vec product.
      return tf.einsum('nsi,ni->ns',
                       BackwardBounds._reshape_to_rank(a, 3),
                       BackwardBounds._reshape_to_rank(b, 2))


ScalarMetrics = collections.namedtuple('ScalarMetrics', [
    'nominal_accuracy',
    # Verified accuracy using pure IBP bounds.
    'verified_accuracy',
    # Verified accuracy using CROWN and IBP mixture.
    'crown_ibp_verified_accuracy',
    'attack_accuracy',
    'attack_success'])

ScalarLosses = collections.namedtuple('ScalarLosses', [
    'nominal_cross_entropy',
    'attack_cross_entropy',
    'verified_loss'])


class Losses(loss.Losses):
  """Helper to compute CROWN-IBP losses."""

  def __init__(self, predictor, specification=None, pgd_attack=None,
               interval_bounds_loss_type='xent',
               interval_bounds_hinge_margin=10.,
               label_smoothing=0.,
               use_crown_ibp=False,
               crown_bound_schedule=None):
    super(Losses, self).__init__(predictor, specification, pgd_attack,
                                 interval_bounds_loss_type,
                                 interval_bounds_hinge_margin,
                                 label_smoothing)
    self._use_crown_ibp = use_crown_ibp
    self._crown_bound_schedule = crown_bound_schedule

  def _get_specification_bounds(self):
    """Get upper bounds on specification. Used for building verified loss."""
    ibp_bounds = self._specification(self._predictor.modules)
    # Compute verified accuracy using IBP bounds.
    v = tf.reduce_max(ibp_bounds, axis=1)
    self._interval_bounds_accuracy = tf.reduce_mean(
        tf.cast(v <= 0., tf.float32))
    # CROWN-IBP bounds.
    if self._use_crown_ibp:
      logging.info('CROWN-IBP active')
      def _build_crown_ibp_bounds():
        """Create the computationally expensive CROWN bounds for tf.cond."""
        predictor = self._predictor
        # CROWN is computed backwards so we need to start with a
        # initial bound related to the specification.
        init_crown_bounds = create_initial_backward_bounds(self._specification,
                                                           predictor.modules)
        # Now propagate the specification matrix layer by layer;
        # we only need the CROWN upper bound, do not need lower bound.
        crown_bound = predictor.propagate_bound_backward(init_crown_bounds,
                                                         compute_upper=True,
                                                         compute_lower=False)
        # A linear mixture of the two bounds with a schedule.
        return self._crown_bound_schedule * crown_bound.upper + \
               (1. - self._crown_bound_schedule) * ibp_bounds
      # If the coefficient for CROWN bound is close to 0, compute IBP only.
      mixture_bounds = tf.cond(self._crown_bound_schedule < 1e-6,
                               lambda: ibp_bounds, _build_crown_ibp_bounds)
      v = tf.reduce_max(mixture_bounds, axis=1)
      self._crown_ibp_accuracy = tf.reduce_mean(tf.cast(v <= 0., tf.float32))
    else:
      mixture_bounds = ibp_bounds
      self._crown_ibp_accuracy = tf.constant(0.)
    return mixture_bounds

  @property
  def scalar_metrics(self):
    self._ensure_is_connected()
    return ScalarMetrics(self._nominal_accuracy,
                         self._interval_bounds_accuracy,
                         self._crown_ibp_accuracy,
                         self._attack_accuracy,
                         self._attack_success)

  @property
  def scalar_losses(self):
    self._ensure_is_connected()
    return ScalarLosses(self._cross_entropy,
                        self._attack_cross_entropy,
                        self._verified_loss)


class VerifiableModelWrapper(model.VerifiableModelWrapper):
  """Model wrapper with CROWN-IBP backward bound propagation."""

  def _propagate(self, current_module, current_bounds):
    """Propagate CROWN bounds in a backwards manner."""
    # Construct bounds for this layer.
    if isinstance(current_module, verifiable_wrapper.ModelInputWrapper):
      if current_module.index != 0:
        raise NotImplementedError('CROWN backpropagation does not support '
                                  'multiple inputs.')
      return current_bounds
    # Propagate the bounds through the current layer.
    new_bounds = current_module.propagate_bounds(current_bounds)
    prev_modules = self._module_depends_on[current_module]
    # We assume that each module only depends on one module.
    if len(prev_modules) != 1:
      raise NotImplementedError('CROWN for non-sequential networks is not '
                                'implemented.')
    return self._propagate(prev_modules[0], new_bounds)

  def propagate_bound_backward(self, initial_bound,
                               compute_upper=True, compute_lower=False):
    """Propagates CROWN bounds backward through the network.

    This function assumes that we have obtained bounds for all intermediate
    layers using IBP. Currently only sequential networks are implemented.

    Args:

      initial_bound: A BackwardBounds object containing the initial matrices
        and biases to start bound propagation.
      compute_upper: Set to True to construct the computation graph for the
        CROWN upper bound. For verified training, only the upper bound is
        needed. Default is True.
      compute_lower: Set to True to construct the computation graph for the
        CROWN lower bound. Default is False.

    Returns:
      IntervalBound instance corresponding to bounds on the specification.
    """
    if (not compute_upper) and (not compute_lower):
      raise ValueError('At least one of "compute_upper" or "compute_lower" '
                       'needs to be True')
    self._ensure_is_connected()
    # We start bound propagation from the logit layer.
    logit_layer = self._produced_by[self._logits.name]
    # If only one of ub or lb is needed, we set the unnecessary one to None.
    ub = initial_bound.upper if compute_upper else None
    lb = initial_bound.lower if compute_lower else None
    bound = BackwardBounds(lb, ub)
    crown_bound = self._propagate(logit_layer, bound)
    return crown_bound.concretize()


def create_initial_backward_bounds(spec, modules):
  """Create the initial BackwardBounds according to specification."""
  last_bounds = bounds.IntervalBounds.convert(modules[-1].input_bounds)
  if isinstance(spec, specification_lib.ClassificationSpecification):
    c_correct = tf.expand_dims(
        tf.one_hot(spec.correct_idx[:, 1], spec.num_specifications + 1), 1)
    c_wrong = tf.one_hot(spec.wrong_idx[:, :, 1], spec.num_specifications + 1)
    c = c_wrong - c_correct
    b = tf.zeros(spec.num_specifications)
    lb = ub = fastlin.LinearExpression(w=c, b=b, lower=last_bounds.lower,
                                       upper=last_bounds.upper)
  elif isinstance(spec, specification_lib.LinearSpecification):
    b = spec.d if spec.d is not None else tf.zeros(spec.num_specifications)
    lb = ub = fastlin.LinearExpression(w=spec.c, b=b, lower=last_bounds.lower,
                                       upper=last_bounds.upper)
  else:
    raise ValueError('Unknown specification class type "{}"'.format(str(spec)))
  return BackwardBounds(lb, ub)


def create_classification_losses(
    global_step, inputs, label, predictor_network, epsilon, loss_weights,
    warmup_steps=0, rampup_steps=-1, input_bounds=(0., 1.), options=None):
  """Create the training loss for CROWN-IBP."""
  def _is_loss_active(init, final, warmup=None):
    return init > 0. or final > 0. or (warmup is not None and warmup > 0.)
  if 'crown_bound' in loss_weights:
    schedule = utils.build_loss_schedule(global_step, warmup_steps,
                                         rampup_steps,
                                         **loss_weights.get('crown_bound'))
    use_crown_ibp = _is_loss_active(**loss_weights.get('crown_bound'))
  else:
    schedule = None
    use_crown_ibp = False
  # Use the loss builder for CROWN-IBP with additional kwargs.
  def _loss_builder(*args, **kwargs):
    kwargs.update(dict(use_crown_ibp=use_crown_ibp,
                       crown_bound_schedule=schedule))
    return Losses(*args, **kwargs)
  return utils.create_classification_losses(
      global_step, inputs, label, predictor_network, epsilon,
      loss_weights, warmup_steps, rampup_steps, input_bounds,
      loss_builder=_loss_builder, options=options)
