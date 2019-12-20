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

"""Naive bound calculation for common neural network layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from interval_bound_propagation.src import bounds as basic_bounds
from interval_bound_propagation.src import relative_bounds
import sonnet as snt
import tensorflow.compat.v1 as tf


class SimplexBounds(basic_bounds.AbstractBounds):
  """Specifies a bounding simplex within an embedding space."""

  def __init__(self, vertices, nominal, r):
    """Initialises the simplex bounds.

    Args:
      vertices: Tensor of shape (num_vertices, *input_shape)
        or of shape (batch_size, num_vertices, *input_shape)
        containing the vertices in embedding space.
      nominal: Tensor of shape (batch_size, *input_shape) specifying
        the unperturbed inputs in embedding space, where `*input_shape`
        denotes either (embedding_size,) for flat input (e.g. bag-of-words)
        or (input_length, embedding_channels) for sequence input.
      r: Scalar specifying the dilation factor of the simplex. The dilated
        simplex will have vertices `nominal + r * (vertices-nominal)`.
    """
    super(SimplexBounds, self).__init__()
    self._vertices = vertices
    self._nominal = nominal
    self._r = r

  @property
  def vertices(self):
    return self._vertices

  @property
  def nominal(self):
    return self._nominal

  @property
  def r(self):
    return self._r

  @property
  def shape(self):
    return self.nominal.shape.as_list()

  @classmethod
  def convert(cls, bounds):
    if not isinstance(bounds, cls):
      raise ValueError('Cannot convert "{}" to "{}"'.format(bounds,
                                                            cls.__name__))
    return bounds

  def apply_batch_reshape(self, wrapper, shape):
    reshape = snt.BatchReshape(shape)
    if self.vertices.shape.ndims == self.nominal.shape.ndims:
      reshape_vertices = reshape
    else:
      reshape_vertices = snt.BatchReshape(shape, preserve_dims=2)
    return SimplexBounds(reshape_vertices(self.vertices),
                         reshape(self.nominal),
                         self.r)

  def apply_linear(self, wrapper, w, b):
    mapped_centres = tf.matmul(self.nominal, w)
    mapped_vertices = tf.tensordot(self.vertices, w, axes=1)

    lb, ub = _simplex_bounds(mapped_vertices, mapped_centres, self.r, -2)

    nominal_out = tf.matmul(self.nominal, w)
    if b is not None:
      nominal_out += b

    return relative_bounds.RelativeIntervalBounds(lb, ub, nominal_out)

  def apply_conv1d(self, wrapper, w, b, padding, stride):
    mapped_centres = tf.nn.conv1d(self.nominal, w,
                                  padding=padding, stride=stride)
    if self.vertices.shape.ndims == 3:
      # `self.vertices` has no batch dimension; its shape is
      # (num_vertices, input_length, embedding_channels).
      mapped_vertices = tf.nn.conv1d(self.vertices, w,
                                     padding=padding, stride=stride)
    elif self.vertices.shape.ndims == 4:
      # `self.vertices` has shape
      # (batch_size, num_vertices, input_length, embedding_channels).
      # Vertices are different for each example in the batch,
      # e.g. for word perturbations.
      mapped_vertices = snt.BatchApply(
          lambda x: tf.nn.conv1d(x, w, padding=padding, stride=stride))(
              self.vertices)
    else:
      raise ValueError('"vertices" must have either 3 or 4 dimensions.')

    lb, ub = _simplex_bounds(mapped_vertices, mapped_centres, self.r, -3)

    nominal_out = tf.nn.conv1d(self.nominal, w,
                               padding=padding, stride=stride)
    if b is not None:
      nominal_out += b

    return relative_bounds.RelativeIntervalBounds(lb, ub, nominal_out)

  def apply_conv2d(self, wrapper, w, b, padding, strides):
    mapped_centres = tf.nn.convolution(self.nominal, w,
                                       padding=padding, strides=strides)
    if self.vertices.shape.ndims == 4:
      # `self.vertices` has no batch dimension; its shape is
      # (num_vertices, input_height, input_width, input_channels).
      mapped_vertices = tf.nn.convolution(self.vertices, w,
                                          padding=padding, strides=strides)
    elif self.vertices.shape.ndims == 5:
      # `self.vertices` has shape
      # (batch_size, num_vertices, input_height, input_width, input_channels).
      # Vertices are different for each example in the batch.
      mapped_vertices = snt.BatchApply(
          lambda x: tf.nn.convolution(x, w, padding=padding, strides=strides))(
              self.vertices)
    else:
      raise ValueError('"vertices" must have either 4 or 5 dimensions.')

    lb, ub = _simplex_bounds(mapped_vertices, mapped_centres, self.r, -4)

    nominal_out = tf.nn.convolution(self.nominal, w,
                                    padding=padding, strides=strides)
    if b is not None:
      nominal_out += b

    return relative_bounds.RelativeIntervalBounds(lb, ub, nominal_out)

  def apply_increasing_monotonic_fn(self, wrapper, fn, *args, **parameters):
    if fn.__name__ in ('add', 'reduce_mean', 'reduce_sum', 'avg_pool'):
      if self.vertices.shape.ndims == self.nominal.shape.ndims:
        vertices_fn = fn
      else:
        vertices_fn = snt.BatchApply(fn, n_dims=2)
      return SimplexBounds(
          vertices_fn(self.vertices, *[bounds.vertices for bounds in args]),
          fn(self.nominal, *[bounds.nominal for bounds in args]),
          self.r)

    elif fn.__name__ == 'quotient':
      return SimplexBounds(
          self.vertices / tf.expand_dims(parameters['denom'], axis=1),
          fn(self.nominal),
          self.r)

    else:
      return super(SimplexBounds, self).apply_increasing_monotonic_fn(
          wrapper, fn, *args, **parameters)


def _simplex_bounds(mapped_vertices, mapped_centres, r, axis):
  """Calculates naive bounds on the given layer-mapped vertices.

  Args:
    mapped_vertices: Tensor of shape (num_vertices, *output_shape)
      or of shape (batch_size, num_vertices, *output_shape)
      containing the vertices in the layer's output space.
    mapped_centres: Tensor of shape (batch_size, *output_shape)
      containing the layer's nominal outputs.
    r: Scalar in [0, 1) specifying the radius (in vocab space) of the simplex.
    axis: Index of the `num_vertices` dimension of `mapped_vertices`.

  Returns:
    lb_out: Tensor of shape (batch_size, *output_shape) with lower bounds
      on the outputs of the affine layer.
    ub_out: Tensor of shape (batch_size, *output_shape) with upper bounds
      on the outputs of the affine layer.
  """
  # Use the negative of r, instead of the complement of r, as
  # we're shifting the input domain to be centred at the origin.
  lb_out = -r * mapped_centres + r * tf.reduce_min(mapped_vertices, axis=axis)
  ub_out = -r * mapped_centres + r * tf.reduce_max(mapped_vertices, axis=axis)
  return lb_out, ub_out

