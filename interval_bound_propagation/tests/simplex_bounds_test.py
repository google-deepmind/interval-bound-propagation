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

"""Tests for naive_bounds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import interval_bound_propagation as ibp
from interval_bound_propagation import layer_utils
import numpy as np
import tensorflow.compat.v1 as tf


class SimplexBoundsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_linear_simplex_bounds_shape(self, dtype):
    vocab_size = 103
    batch_size = 11
    input_size = 7
    output_size = 5

    w = tf.placeholder(dtype=dtype, shape=(input_size, output_size))
    b = tf.placeholder(dtype=dtype, shape=(output_size,))
    embedding = tf.placeholder(dtype=dtype, shape=(vocab_size, input_size))
    centres = tf.placeholder(dtype=dtype, shape=(batch_size, input_size))
    r = .2

    bounds_in = ibp.SimplexBounds(embedding, centres, r)
    bounds_out = bounds_in.apply_linear(None, w, b)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    self.assertEqual(dtype, lb_out.dtype)
    self.assertEqual(dtype, ub_out.dtype)
    self.assertEqual((batch_size, output_size), lb_out.shape)
    self.assertEqual((batch_size, output_size), ub_out.shape)

  @parameterized.named_parameters(('float32', tf.float32, 1.e-6),
                                  ('float64', tf.float64, 1.e-8))
  def test_linear_bounds_on_embedding_layer(self, dtype, tol):
    w = tf.constant([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=dtype)
    b = tf.constant([0.01, -0.02, 0.03], dtype=dtype)
    embedding = tf.constant([[0.0, 0.0], [10.0, 10.0], [0.0, -20.0]],
                            dtype=dtype)
    centres = tf.constant([[7.0, 6.0]], dtype=dtype)
    r = .1
    # Simplex vertices: [6.3, 5.4], [7.3, 6.4], and [6.3, 3.4].
    # They map to: [27.91, -14.42, 51.33], [32.91, -17.42, 60.33],
    # and [19.91, -4.42, 39.33].

    bounds_in = ibp.SimplexBounds(embedding, centres, r)
    bounds_out = bounds_in.apply_linear(None, w, b)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    lb_out_exp = np.array([[19.91, -17.42, 39.33]])
    ub_out_exp = np.array([[32.91, -4.42, 60.33]])

    with self.test_session() as session:
      lb_out_act, ub_out_act = session.run((lb_out, ub_out))
      self.assertAllClose(lb_out_exp, lb_out_act, atol=tol, rtol=tol)
      self.assertAllClose(ub_out_exp, ub_out_act, atol=tol, rtol=tol)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_conv1d_simplex_bounds_shape(self, dtype):
    num_vertices = 41
    batch_size = 11
    input_length = 13
    kernel_length = 5
    input_channels = 3
    output_channels = 2
    padding = 'VALID'
    strides = (2,)

    # Expected output dimensions, based on convolution settings.
    output_length = 5

    w = tf.placeholder(dtype=dtype, shape=(
        kernel_length, input_channels, output_channels))
    b = tf.placeholder(dtype=dtype, shape=(output_channels,))
    vertices = tf.placeholder(dtype=dtype, shape=(
        batch_size, num_vertices, input_length, input_channels))
    centres = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    r = .2

    bounds_in = ibp.SimplexBounds(vertices, centres, r)
    bounds_out = bounds_in.apply_conv1d(None, w, b, padding, strides)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    self.assertEqual(dtype, lb_out.dtype)
    self.assertEqual(dtype, ub_out.dtype)
    self.assertEqual((batch_size, output_length, output_channels),
                     lb_out.shape)
    self.assertEqual((batch_size, output_length, output_channels),
                     ub_out.shape)

  @parameterized.named_parameters(('float32', tf.float32, 2.e-6),
                                  ('float64', tf.float64, 1.e-8))
  def test_conv1d_simplex_bounds(self, dtype, tol):
    num_vertices = 37
    batch_size = 53
    input_length = 17
    kernel_length = 7
    input_channels = 3
    output_channels = 2
    padding = 'VALID'
    strides = (2,)

    w = tf.random_normal(dtype=dtype, shape=(
        kernel_length, input_channels, output_channels))
    b = tf.random_normal(dtype=dtype, shape=(output_channels,))
    vertices = tf.random_normal(dtype=dtype, shape=(
        batch_size, num_vertices, input_length, input_channels))
    centres = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    r = .2

    bounds_in = ibp.SimplexBounds(vertices, centres, r)
    bounds_out = bounds_in.apply_conv1d(None, w, b, padding, strides[0])
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    # Compare against equivalent linear layer.
    bounds_out_lin = _materialised_conv_simplex_bounds(
        w, b, padding, strides, bounds_in)
    lb_out_lin, ub_out_lin = bounds_out_lin.lower, bounds_out_lin.upper

    with self.test_session() as session:
      (lb_out_val, ub_out_val,
       lb_out_lin_val, ub_out_lin_val) = session.run((lb_out, ub_out,
                                                      lb_out_lin, ub_out_lin))
      self.assertAllClose(lb_out_val, lb_out_lin_val, atol=tol, rtol=tol)
      self.assertAllClose(ub_out_val, ub_out_lin_val, atol=tol, rtol=tol)


def _materialised_conv_simplex_bounds(w, b, padding, strides, bounds_in):
  """Calculates naive bounds on output of an N-D convolution layer.

  The calculation is performed by first materialising the convolution as a
  (sparse) fully-connected linear layer. Doing so will affect performance, but
  may be useful for investigating numerical stability issues.

  The layer inputs and the vertices are assumed to be (N-D) sequences in an
  embedding space. The input domain is taken to be the simplex of perturbations
  of the centres (true inputs) towards the given vertices.

  Specifically, the input domain is the convex hull of this set of vertices::
    { (1-r)*centres + r*vertices[j] : j<num_vertices }

  Args:
    w: (N+2)D tensor of shape (kernel_length, input_channels, output_channels)
      containing weights for the convolution.
    b: 1D tensor of shape (output_channels) containing biases for the
      convolution, or `None` if no bias.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.
    bounds_in: bounds of shape (batch_size, input_length, input_channels)
      containing bounds on the inputs to the convolution layer.

  Returns:
    bounds of shape (batch_size, output_length, output_channels)
      with bounds on the outputs of the convolution layer.

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  # Flatten the inputs, as the materialised convolution will have no
  # spatial structure.
  bounds_in_flat = bounds_in.apply_batch_reshape(None, [-1])

  # Materialise the convolution as a (sparse) fully connected linear layer.
  input_shape = bounds_in.shape[1:]
  w_lin, b_lin = layer_utils.materialise_conv(w, b, input_shape,
                                              padding=padding, strides=strides)
  bounds_out_flat = bounds_in_flat.apply_linear(None, w_lin, b_lin)

  # Unflatten the output bounds.
  output_shape = layer_utils.conv_output_shape(input_shape, w, padding, strides)
  return bounds_out_flat.apply_batch_reshape(None, output_shape)


if __name__ == '__main__':
  tf.test.main()
