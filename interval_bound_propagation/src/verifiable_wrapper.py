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

"""Wrapper around modules that provides additional facilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import types

from interval_bound_propagation.src import layers
import sonnet as snt


class VerifiableWrapper(object):
  """Abstract wrapper class."""

  __metaclass__ = abc.ABCMeta

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
    assert self._output_bounds is not None
    return self._output_bounds

  @property
  def module(self):
    return self._module

  def __str__(self):
    if isinstance(self._module, types.LambdaType):
      return self._module.__name__
    if isinstance(self._module, snt.AbstractModule):
      return self._module.module_name
    if hasattr(self._module, '__class__'):
      return self._module.__class__.__name__
    return str(self._module)

  def propagate_bounds(self, *input_bounds):
    """Propagates bounds and saves input and output bounds."""
    if not input_bounds:
      raise RuntimeError('propagate_bounds expects at least one argument.')
    main_bounds = input_bounds[0]
    if len(input_bounds) == 1:
      self._input_bounds = main_bounds
    else:
      self._input_bounds = tuple(input_bounds)
    self._output_bounds = main_bounds.propagate_through(
        self, *input_bounds[1:])
    return self._output_bounds


class LinearFCWrapper(VerifiableWrapper):
  """Wraps fully-connected layers."""

  def __init__(self, module):
    if not isinstance(module, snt.Linear):
      raise ValueError('Cannot wrap {} with a LinearFCWrapper.'.format(module))
    super(LinearFCWrapper, self).__init__(module)


class LinearConv2dWrapper(VerifiableWrapper):
  """Wraps convolutional layers."""

  def __init__(self, module):
    if not isinstance(module, snt.Conv2D):
      raise ValueError('Cannot wrap {} with a LinearConv2dWrapper.'.format(
          module))
    super(LinearConv2dWrapper, self).__init__(module)


class MonotonicWrapper(VerifiableWrapper):
  """Wraps monotonically increasing functions of the inputs."""


class ImageNormWrapper(MonotonicWrapper):
  """Convinence wrapper for getting track of the ImageNorm layer."""

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
    if not isinstance(module, layers.BatchNorm):
      raise ValueError('Cannot wrap {} with a BatchNormWrapper.'.format(
          module))
    super(BatchNormWrapper, self).__init__(module)


class BatchFlattenWrapper(VerifiableWrapper):
  """Wraps batch flatten."""

  def __init__(self, module):
    if not isinstance(module, snt.BatchFlatten):
      raise ValueError('Cannot wrap {} with a BatchFlattenWrapper.'.format(
          module))
    super(BatchFlattenWrapper, self).__init__(module)
