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

"""Interval bounds expressed relative to a nominal value."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from interval_bound_propagation.src import bounds as basic_bounds
import sonnet as snt
import tensorflow.compat.v1 as tf


class RelativeIntervalBounds(basic_bounds.AbstractBounds):
  """Upper and lower bounds, as a delta relative to nominal values."""

  def __init__(self, lower_offset, upper_offset, nominal):
    super(RelativeIntervalBounds, self).__init__()
    self._lower_offset = lower_offset
    self._upper_offset = upper_offset
    self._nominal = nominal

  @property
  def lower_offset(self):
    """Returns lower bounds, expressed relative to nominal values."""
    return self._lower_offset

  @property
  def upper_offset(self):
    """Returns upper bounds, expressed relative to nominal values."""
    return self._upper_offset

  @property
  def nominal(self):
    return self._nominal

  @property
  def lower(self):
    """Returns absolute lower bounds."""
    return self.nominal + self.lower_offset

  @property
  def upper(self):
    """Returns absolute upper bounds."""
    return self.nominal + self.upper_offset

  @property
  def shape(self):
    return self.lower_offset.shape.as_list()

  @classmethod
  def convert(cls, bounds):
    if isinstance(bounds, tf.Tensor):
      return cls(tf.zeros_like(bounds), tf.zeros_like(bounds), bounds)
    bounds = bounds.concretize()
    if not isinstance(bounds, cls):
      raise ValueError('Cannot convert "{}" to "{}"'.format(bounds,
                                                            cls.__name__))
    return bounds

  def apply_batch_reshape(self, wrapper, shape):
    """Propagates the bounds through a reshape.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      shape: output shape, excluding the batch dimension.

    Returns:
      Output bounds.
    """
    reshape = snt.BatchReshape(shape)
    return RelativeIntervalBounds(
        reshape(self.lower_offset),
        reshape(self.upper_offset),
        reshape(self.nominal))

  def apply_linear(self, wrapper, w, b):
    """Propagates the bounds through a linear layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      w: 2D tensor of shape (input_size, output_size) containing
        weights for the linear layer.
      b: 1D tensor of shape (output_size) containing biases for the linear
        layer, or `None` if no bias.

    Returns:
      Output bounds.
    """
    w_pos = tf.maximum(w, 0)
    w_neg = tf.minimum(w, 0)
    lb = (tf.matmul(self.lower_offset, w_pos) +
          tf.matmul(self.upper_offset, w_neg))
    ub = (tf.matmul(self.upper_offset, w_pos) +
          tf.matmul(self.lower_offset, w_neg))

    nominal_out = tf.matmul(self.nominal, w)
    if b is not None:
      nominal_out += b

    return RelativeIntervalBounds(lb, ub, nominal_out)

  def apply_conv1d(self, wrapper, w, b, padding, stride):
    """Propagates the bounds through a 1D convolution layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      w: 3D tensor of shape (kernel_length, input_channels, output_channels)
        containing weights for the convolution.
      b: 1D tensor of shape (output_channels) containing biases for the
        convolution, or `None` if no bias.
      padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
      stride: Integer stride.

    Returns:
      Output bounds.
    """
    w_pos = tf.maximum(w, 0)
    w_neg = tf.minimum(w, 0)
    lb = (tf.nn.conv1d(self.lower_offset, w_pos,
                       padding=padding, stride=stride) +
          tf.nn.conv1d(self.upper_offset, w_neg,
                       padding=padding, stride=stride))
    ub = (tf.nn.conv1d(self.upper_offset, w_pos,
                       padding=padding, stride=stride) +
          tf.nn.conv1d(self.lower_offset, w_neg,
                       padding=padding, stride=stride))

    nominal_out = tf.nn.conv1d(self.nominal, w,
                               padding=padding, stride=stride)
    if b is not None:
      nominal_out += b

    return RelativeIntervalBounds(lb, ub, nominal_out)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    """Propagates the bounds through a 2D convolution layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      w: 4D tensor of shape (kernel_height, kernel_width, input_channels,
        output_channels) containing weights for the convolution.
      b: 1D tensor of shape (output_channels) containing biases for the
        convolution, or `None` if no bias.
      padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
      strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

    Returns:
      Output bounds.
    """
    w_pos = tf.maximum(w, 0)
    w_neg = tf.minimum(w, 0)
    lb = (tf.nn.convolution(self.lower_offset, w_pos,
                            padding=padding, strides=strides) +
          tf.nn.convolution(self.upper_offset, w_neg,
                            padding=padding, strides=strides))
    ub = (tf.nn.convolution(self.upper_offset, w_pos,
                            padding=padding, strides=strides) +
          tf.nn.convolution(self.lower_offset, w_neg,
                            padding=padding, strides=strides))

    nominal_out = tf.nn.convolution(self.nominal, w,
                                    padding=padding, strides=strides)
    if b is not None:
      nominal_out += b

    return RelativeIntervalBounds(lb, ub, nominal_out)

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
    """Propagates the bounds through a non-linear activation layer or `add` op.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      fn: String specifying non-linear activation function.
        May be one of: sig, relu, tanh, elu, leaky_relu.
        Anything else denotes identity.
      *args: Other inputs' bounds, for a multi-input node (e.g. Add).
      **parameters: Optional parameters if activation is parameterised, e.g.
        `{'alpha': 0.2}` for leaky ReLu.

    Returns:
      Output bounds.
    """
    if fn.__name__ in ('add', 'reduce_mean', 'reduce_sum', 'avg_pool'):
      return RelativeIntervalBounds(
          fn(self.lower_offset, *[bounds.lower_offset for bounds in args]),
          fn(self.upper_offset, *[bounds.upper_offset for bounds in args]),
          fn(self.nominal, *[bounds.nominal for bounds in args]))
    else:
      assert not args, 'unary function expected'
      nominal_out = fn(self.nominal)
      if fn.__name__ == 'reduce_max':
        lb, ub = _maxpool_bounds(fn, None, None,
                                 self.lower_offset, self.upper_offset,
                                 nominal_in=self.nominal,
                                 nominal_out=nominal_out)
      elif fn.__name__ == 'max_pool':
        lb, ub = _maxpool_bounds(fn,
                                 parameters['ksize'][1:-1],
                                 parameters['strides'][1:-1],
                                 self.lower_offset, self.upper_offset,
                                 nominal_in=self.nominal,
                                 nominal_out=nominal_out)
      else:
        lb, ub = _activation_bounds(fn, self.lower_offset, self.upper_offset,
                                    nominal_in=self.nominal,
                                    parameters=parameters)
      return RelativeIntervalBounds(lb, ub, nominal_out)

  def apply_batch_norm(self, wrapper, mean, variance, scale, bias, epsilon):
    """Propagates the bounds through a batch norm layer.

    Args:
      wrapper: Contains prior bounds from a previous iteration.
      mean: Learnt batch mean.
      variance: Learnt batch variance.
      scale: Trained component-wise scale variable.
      bias: Trained component-wise bias variable.
      epsilon: Epsilon for avoiding instability when `variance` is very small.

    Returns:
      Output bounds.
    """
    lb = tf.nn.batch_normalization(self.lower_offset,
                                   tf.zeros_like(mean), variance,
                                   None, scale, epsilon)
    ub = tf.nn.batch_normalization(self.upper_offset,
                                   tf.zeros_like(mean), variance,
                                   None, scale, epsilon)
    # It's just possible that the batchnorm's scale is negative.
    lb, ub = tf.minimum(lb, ub), tf.maximum(lb, ub)

    nominal_out = tf.nn.batch_normalization(self.nominal,
                                            mean, variance,
                                            bias, scale, epsilon)
    return RelativeIntervalBounds(lb, ub, nominal_out)

  def _set_up_cache(self):
    self._lower_offset, update_lower = self._cache_with_update_op(
        self._lower_offset)
    self._upper_offset, update_upper = self._cache_with_update_op(
        self._upper_offset)
    return tf.group([update_lower, update_upper])


def _maxpool_bounds(module, kernel_shape, strides, lb_in, ub_in,
                    nominal_in, nominal_out):
  """Calculates naive bounds on output of an N-D max pool layer.

  Args:
    module: Callable for max-pool operation.
    kernel_shape: Integer list of `[kernel_height, kernel_width]`,
      or `None` to aggregate over the layer`s entire spatial extent.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.
    lb_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing lower bounds on the inputs to the
      max pool layer.
    ub_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing upper bounds on the inputs to the
      max pool layer.
    nominal_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing nominal input values.
      Inputs bounds are interpreted relative to this.
    nominal_out: (N+2)D tensor of shape (batch_size, output_height,output_width,
      layer_channels) containing nominal input values.
      The returned output bounds are expressed relative to this.

  Returns:
    lb_out: (N+2)D tensor of shape (batch_size, output_height, output_width,
      layer_channels) with lower bounds on the outputs of the max pool layer.
    ub_out: (N+2)D tensor of shape (batch_size, output_height, output_width,
      layer_channels) with upper bounds on the outputs of the max pool layer.
  """
  if kernel_shape is None:
    nominal_out = tf.reduce_max(nominal_in,
                                axis=list(range(1, nominal_in.shape.ndims-1)),
                                keepdims=True)
    return (module((nominal_in - nominal_out) + lb_in),
            module((nominal_in - nominal_out) + ub_in))
  else:
    # Must perform the max on absolute bounds, as the kernels may overlap.
    # TODO(stanforth) investigate a more numerically stable implementation
    del strides
    return (module(nominal_in + lb_in) - nominal_out,
            module(nominal_in + ub_in) - nominal_out)


def _activation_bounds(nl_fun, lb_in, ub_in, nominal_in, parameters=None):
  """Calculates naive bounds on output of an activation layer.

  Inputs bounds are interpreted relative to `nominal_in`, and the returned
  output bounds are expressed relative to `nominal_out=nl(nominal_in)`.

  Args:
    nl_fun: Callable implementing the activation function itself.
    lb_in: (N+2)D tensor of shape (batch_size, layer_height, layer_width,
      layer_channels) containing lower bounds on the pre-activations.
    ub_in: (N+2)D tensor of shape (batch_size, layer_height, layer_width,
      layer_channels) containing upper bounds on the pre-activations.
    nominal_in: (N+2)D tensor of shape (batch_size, input_height, input_width,
      layer_channels) containing nominal input values.
    parameters: Optional parameter dict if activation is parameterised, e.g.
      `{'alpha': 0.2}` for leaky ReLu.

  Returns:
    lb_out: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      with lower bounds on the activations.
    ub_out: 2D tensor of shape (batch_size, layer_size) or
      4D tensor of shape (batch_size, layer_height, layer_width, layer_channels)
      with upper bounds on the activations.
  """
  if nl_fun.__name__ == 'relu':
    return (
        tf.maximum(tf.minimum(nominal_in, 0.) + lb_in,
                   tf.minimum(-nominal_in, 0.)),  # pylint:disable=invalid-unary-operand-type
        tf.maximum(tf.minimum(nominal_in, 0.) + ub_in,
                   tf.minimum(-nominal_in, 0.)))  # pylint:disable=invalid-unary-operand-type
  elif nl_fun.__name__ == 'leaky_relu':
    alpha = parameters['alpha']
    return (
        tf.maximum(
            lb_in + tf.minimum(nominal_in, 0.) * (1. - alpha),
            alpha * lb_in + tf.minimum(-nominal_in, 0.) * (1. - alpha)),  # pylint:disable=invalid-unary-operand-type
        tf.maximum(
            ub_in + tf.minimum(nominal_in, 0.) * (1. - alpha),
            alpha * ub_in + tf.minimum(-nominal_in, 0.) * (1. - alpha)))  # pylint:disable=invalid-unary-operand-type
  else:
    nominal_out = nl_fun(nominal_in)
    return (nl_fun(nominal_in + lb_in) - nominal_out,
            nl_fun(nominal_in + ub_in) - nominal_out)
