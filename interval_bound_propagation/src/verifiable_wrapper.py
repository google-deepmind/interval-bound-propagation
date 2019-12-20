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

"""Wrapper around modules that provides additional facilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import types

from absl import logging
from interval_bound_propagation.src import layers
import six
import sonnet as snt
import tensorflow.compat.v1 as tf


@six.add_metaclass(abc.ABCMeta)
class VerifiableWrapper(object):
  """Abstract wrapper class."""

  def __init__(self, module):
    self._module = module
    self._input_bounds = None
    self._output_bounds = None

  @property
  def input_bounds(self):
    assert self._input_bounds is not None
    return self._input_bounds

  @property
  def output_bounds(self):
    return self._output_bounds

  @property
  def module(self):
    return self._module

  def __str__(self):
    if isinstance(self._module, tf.Tensor):
      return str(self._module)
    if isinstance(self._module, types.LambdaType):
      return self._module.__name__
    if isinstance(self._module, snt.AbstractModule):
      return self._module.module_name
    if hasattr(self._module, '__class__'):
      return self._module.__class__.__name__
    return str(self._module)

  def propagate_bounds(self, *input_bounds):
    """Propagates bounds and saves input and output bounds."""
    output_bounds = self._propagate_through(self.module, *input_bounds)

    if len(input_bounds) == 1:
      self._input_bounds = input_bounds[0]
    else:
      self._input_bounds = tuple(input_bounds)
    self._output_bounds = output_bounds

    return output_bounds

  @abc.abstractmethod
  def _propagate_through(self, module, *input_bounds):
    """Propagates bounds through a verifiable wrapper.

    Args:
      module: This wrapped module, through which bounds are to be propagated.
      *input_bounds: Bounds on the node's input(s).

    Returns:
      New bounds on the node's output.
    """


class ModelInputWrapper(object):
  """Virtual node representing the network's inputs."""

  def __init__(self, index):
    super(ModelInputWrapper, self).__init__()
    self._index = index
    self._output_bounds = None

  @property
  def index(self):
    return self._index

  @property
  def output_bounds(self):
    return self._output_bounds

  @output_bounds.setter
  def output_bounds(self, bounds):
    self._output_bounds = bounds

  def __str__(self):
    return 'Model input {}'.format(self.index)


class ConstWrapper(VerifiableWrapper):
  """Wraps a constant tensor."""

  def _propagate_through(self, module):
    # Make sure that the constant value can be converted to a tensor.
    return tf.convert_to_tensor(module)


class LinearFCWrapper(VerifiableWrapper):
  """Wraps fully-connected layers."""

  def __init__(self, module):
    if not isinstance(module, snt.Linear):
      raise ValueError('Cannot wrap {} with a LinearFCWrapper.'.format(module))
    super(LinearFCWrapper, self).__init__(module)

  def _propagate_through(self, module, input_bounds):
    w = module.w
    b = module.b if module.has_bias else None
    return input_bounds.apply_linear(self, w, b)


class LinearConvWrapper(VerifiableWrapper):
  """Wraps convolutional layers."""


class LinearConv1dWrapper(LinearConvWrapper):
  """Wraps 1-D convolutional layers."""

  def __init__(self, module):
    if not isinstance(module, snt.Conv1D):
      raise ValueError('Cannot wrap {} with a LinearConv1dWrapper.'.format(
          module))
    super(LinearConv1dWrapper, self).__init__(module)

  def _propagate_through(self, module, input_bounds):
    w = module.w
    b = module.b if module.has_bias else None
    padding = module.padding
    stride = module.stride[1]
    return input_bounds.apply_conv1d(self, w, b, padding, stride)


class LinearConv2dWrapper(LinearConvWrapper):
  """Wraps 2-D convolutional layers."""

  def __init__(self, module):
    if not isinstance(module, snt.Conv2D):
      raise ValueError('Cannot wrap {} with a LinearConv2dWrapper.'.format(
          module))
    super(LinearConv2dWrapper, self).__init__(module)

  def _propagate_through(self, module, input_bounds):
    w = module.w
    b = module.b if module.has_bias else None
    padding = module.padding
    strides = module.stride[1:-1]
    return input_bounds.apply_conv2d(self, w, b, padding, strides)


class IncreasingMonotonicWrapper(VerifiableWrapper):
  """Wraps monotonically increasing functions of the inputs."""

  def __init__(self, module, **parameters):
    super(IncreasingMonotonicWrapper, self).__init__(module)
    self._parameters = parameters

  @property
  def parameters(self):
    return self._parameters

  def _propagate_through(self, module, main_bounds, *other_input_bounds):
    return main_bounds.apply_increasing_monotonic_fn(self, module,
                                                     *other_input_bounds,
                                                     **self.parameters)


class SoftmaxWrapper(VerifiableWrapper):
  """Wraps softmax layers."""

  def __init__(self):
    super(SoftmaxWrapper, self).__init__(None)

  def _propagate_through(self, module, input_bounds):
    return input_bounds.apply_softmax(self)


class PiecewiseMonotonicWrapper(VerifiableWrapper):
  """Wraps a piecewise (not necessarily increasing) monotonic function."""

  def __init__(self, module, boundaries=()):
    super(PiecewiseMonotonicWrapper, self).__init__(module)
    self._boundaries = boundaries

  @property
  def boundaries(self):
    return self._boundaries

  def _propagate_through(self, module, main_bounds, *other_input_bounds):
    return main_bounds.apply_piecewise_monotonic_fn(self, module,
                                                    self.boundaries,
                                                    *other_input_bounds)


class ImageNormWrapper(IncreasingMonotonicWrapper):
  """Convenience wrapper for getting track of the ImageNorm layer."""

  def __init__(self, module):
    if not isinstance(module, layers.ImageNorm):
      raise ValueError('Cannot wrap {} with a ImageNormWrapper.'.format(module))
    super(ImageNormWrapper, self).__init__(module.apply)
    self._inner_module = module

  @property
  def inner_module(self):
    return self._inner_module


class BatchNormWrapper(VerifiableWrapper):
  """Wraps batch normalization."""

  def __init__(self, module):
    if not isinstance(module, snt.BatchNorm):
      raise ValueError('Cannot wrap {} with a BatchNormWrapper.'.format(
          module))
    super(BatchNormWrapper, self).__init__(module)

  def _propagate_through(self, module, input_bounds):
    if isinstance(module, layers.BatchNorm):
      # This IBP-specific batch-norm implementation exposes stats recorded
      # the most recent time the BatchNorm module was connected.
      # These will be either the batch stats (e.g. if training) or the moving
      # averages, depending on how the module was called.
      mean = module.mean
      variance = module.variance
      epsilon = module.epsilon
      scale = module.scale
      bias = module.bias

    else:
      # This plain Sonnet batch-norm implementation only exposes the
      # moving averages.
      logging.warn('Sonnet BatchNorm module encountered: %s. '
                   'IBP will always use its moving averages, not the local '
                   'batch stats, even in training mode.', str(module))
      mean = module.moving_mean
      variance = module.moving_variance
      epsilon = module._eps  # pylint: disable=protected-access
      try:
        bias = module.beta
      except snt.Error:
        bias = None
      try:
        scale = module.gamma
      except snt.Error:
        scale = None

    return input_bounds.apply_batch_norm(self, mean, variance,
                                         scale, bias, epsilon)


class BatchReshapeWrapper(VerifiableWrapper):
  """Wraps batch reshape."""

  def __init__(self, module, shape):
    if not isinstance(module, snt.BatchReshape):
      raise ValueError('Cannot wrap {} with a BatchReshapeWrapper.'.format(
          module))
    super(BatchReshapeWrapper, self).__init__(module)
    self._shape = shape

  @property
  def shape(self):
    return self._shape

  def _propagate_through(self, module, input_bounds):
    return input_bounds.apply_batch_reshape(self, self.shape)


class BatchFlattenWrapper(BatchReshapeWrapper):
  """Wraps batch flatten."""

  def __init__(self, module):
    if not isinstance(module, snt.BatchFlatten):
      raise ValueError('Cannot wrap {} with a BatchFlattenWrapper.'.format(
          module))
    super(BatchFlattenWrapper, self).__init__(module, [-1])
