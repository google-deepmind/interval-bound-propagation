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

"""Tests for relative_bounds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import interval_bound_propagation as ibp
from interval_bound_propagation import layer_utils
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf


class RelativeIntervalBoundsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_linear_bounds_shape(self, dtype):
    batch_size = 11
    input_size = 7
    output_size = 5

    w = tf.placeholder(dtype=dtype, shape=(input_size, output_size))
    b = tf.placeholder(dtype=dtype, shape=(output_size,))
    lb_rel_in = tf.placeholder(dtype=dtype, shape=(batch_size, input_size))
    ub_rel_in = tf.placeholder(dtype=dtype, shape=(batch_size, input_size))
    nominal = tf.placeholder(dtype=dtype, shape=(batch_size, input_size))

    bounds_in = ibp.RelativeIntervalBounds(lb_rel_in, ub_rel_in, nominal)
    bounds_out = bounds_in.apply_linear(None, w, b)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    self.assertEqual(dtype, lb_out.dtype)
    self.assertEqual(dtype, ub_out.dtype)
    self.assertEqual((batch_size, output_size), lb_out.shape)
    self.assertEqual((batch_size, output_size), ub_out.shape)

  @parameterized.named_parameters(('float32', tf.float32, 1.e-6),
                                  ('float64', tf.float64, 1.e-8))
  def test_linear_bounds(self, dtype, tol):
    w = tf.constant([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]], dtype=dtype)
    b = tf.constant([0.1, 0.2, 0.3], dtype=dtype)
    lb_in = tf.constant([[-1.0, -1.0]], dtype=dtype)
    ub_in = tf.constant([[2.0, 2.0]], dtype=dtype)
    nominal = tf.constant([[3.1, 4.2]], dtype=dtype)

    bounds_in = ibp.RelativeIntervalBounds(lb_in - nominal,
                                           ub_in - nominal, nominal)
    bounds_out = bounds_in.apply_linear(None, w, b)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    lb_out_exp = np.array([[-4.9, -11.8, -8.7]])
    ub_out_exp = np.array([[10.1, 9.2, 18.3]])

    with self.test_session() as session:
      lb_out_act, ub_out_act = session.run((lb_out, ub_out))
      self.assertAllClose(lb_out_exp, lb_out_act, atol=tol, rtol=tol)
      self.assertAllClose(ub_out_exp, ub_out_act, atol=tol, rtol=tol)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_conv2d_bounds_shape(self, dtype):
    batch_size = 23
    input_height = 17
    input_width = 7
    kernel_height = 3
    kernel_width = 4
    input_channels = 3
    output_channels = 5
    padding = 'VALID'
    strides = (2, 1)

    # Expected output dimensions, based on convolution settings.
    output_height = 8
    output_width = 4

    w = tf.placeholder(dtype=dtype, shape=(
        kernel_height, kernel_width, input_channels, output_channels))
    b = tf.placeholder(dtype=dtype, shape=(output_channels,))
    lb_rel_in = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))
    ub_rel_in = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))
    nominal = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))

    bounds_in = ibp.RelativeIntervalBounds(lb_rel_in, ub_rel_in, nominal)
    bounds_out = bounds_in.apply_conv2d(None, w, b, padding, strides)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    self.assertEqual(dtype, lb_out.dtype)
    self.assertEqual(dtype, ub_out.dtype)
    self.assertEqual((batch_size, output_height, output_width, output_channels),
                     lb_out.shape)
    self.assertEqual((batch_size, output_height, output_width, output_channels),
                     ub_out.shape)

  @parameterized.named_parameters(('float32', tf.float32, 1.e-5),
                                  ('float64', tf.float64, 1.e-8))
  def test_conv2d_bounds(self, dtype, tol):
    batch_size = 53
    input_height = 17
    input_width = 7
    kernel_height = 3
    kernel_width = 4
    input_channels = 3
    output_channels = 2
    padding = 'VALID'
    strides = (2, 1)

    w = tf.random_normal(dtype=dtype, shape=(
        kernel_height, kernel_width, input_channels, output_channels))
    b = tf.random_normal(dtype=dtype, shape=(output_channels,))
    lb_in = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))
    ub_in = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))
    lb_in, ub_in = tf.minimum(lb_in, ub_in), tf.maximum(lb_in, ub_in)
    nominal = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_height, input_width, input_channels))

    bounds_in = ibp.RelativeIntervalBounds(lb_in - nominal,
                                           ub_in - nominal, nominal)
    bounds_out = bounds_in.apply_conv2d(None, w, b, padding, strides)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    # Compare against equivalent linear layer.
    bounds_out_lin = _materialised_conv_bounds(
        w, b, padding, strides, bounds_in)
    lb_out_lin, ub_out_lin = bounds_out_lin.lower, bounds_out_lin.upper

    with self.test_session() as session:
      (lb_out_val, ub_out_val,
       lb_out_lin_val, ub_out_lin_val) = session.run((lb_out, ub_out,
                                                      lb_out_lin, ub_out_lin))
      self.assertAllClose(lb_out_val, lb_out_lin_val, atol=tol, rtol=tol)
      self.assertAllClose(ub_out_val, ub_out_lin_val, atol=tol, rtol=tol)

  @parameterized.named_parameters(('float32', tf.float32),
                                  ('float64', tf.float64))
  def test_conv1d_bounds_shape(self, dtype):
    batch_size = 23
    input_length = 13
    kernel_length = 3
    input_channels = 3
    output_channels = 5
    padding = 'VALID'
    strides = (2,)

    # Expected output dimensions, based on convolution settings.
    output_length = 6

    w = tf.placeholder(dtype=dtype, shape=(
        kernel_length, input_channels, output_channels))
    b = tf.placeholder(dtype=dtype, shape=(output_channels,))
    lb_rel_in = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    ub_rel_in = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    nominal = tf.placeholder(dtype=dtype, shape=(
        batch_size, input_length, input_channels))

    bounds_in = ibp.RelativeIntervalBounds(lb_rel_in, ub_rel_in, nominal)
    bounds_out = bounds_in.apply_conv1d(None, w, b, padding, strides[0])
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    self.assertEqual(dtype, lb_out.dtype)
    self.assertEqual(dtype, ub_out.dtype)
    self.assertEqual((batch_size, output_length, output_channels),
                     lb_out.shape)
    self.assertEqual((batch_size, output_length, output_channels),
                     ub_out.shape)

  @parameterized.named_parameters(('float32', tf.float32, 1.e-5),
                                  ('float64', tf.float64, 1.e-8))
  def test_conv1d_bounds(self, dtype, tol):
    batch_size = 53
    input_length = 13
    kernel_length = 5
    input_channels = 3
    output_channels = 2
    padding = 'VALID'
    strides = (2,)

    w = tf.random_normal(dtype=dtype, shape=(
        kernel_length, input_channels, output_channels))
    b = tf.random_normal(dtype=dtype, shape=(output_channels,))
    lb_in = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    ub_in = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_length, input_channels))
    lb_in, ub_in = tf.minimum(lb_in, ub_in), tf.maximum(lb_in, ub_in)
    nominal = tf.random_normal(dtype=dtype, shape=(
        batch_size, input_length, input_channels))

    bounds_in = ibp.RelativeIntervalBounds(lb_in - nominal,
                                           ub_in - nominal, nominal)
    bounds_out = bounds_in.apply_conv1d(None, w, b, padding, strides[0])
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    # Compare against equivalent linear layer.
    bounds_out_lin = _materialised_conv_bounds(
        w, b, padding, strides, bounds_in)
    lb_out_lin, ub_out_lin = bounds_out_lin.lower, bounds_out_lin.upper

    with self.test_session() as session:
      (lb_out_val, ub_out_val,
       lb_out_lin_val, ub_out_lin_val) = session.run((lb_out, ub_out,
                                                      lb_out_lin, ub_out_lin))
      self.assertAllClose(lb_out_val, lb_out_lin_val, atol=tol, rtol=tol)
      self.assertAllClose(ub_out_val, ub_out_lin_val, atol=tol, rtol=tol)

  @parameterized.named_parameters(
      ('float32_snt', snt.BatchNorm, tf.float32, 1.e-5, False),
      ('float64_snt', snt.BatchNorm, tf.float64, 1.e-8, False),
      ('float32', ibp.BatchNorm, tf.float32, 1.e-5, False),
      ('float64', ibp.BatchNorm, tf.float64, 1.e-8, False),
      ('float32_train', ibp.BatchNorm, tf.float32, 1.e-5, True),
      ('float64_train', ibp.BatchNorm, tf.float64, 1.e-8, True))
  def test_batchnorm_bounds(self, batchnorm_class, dtype, tol, is_training):
    batch_size = 11
    input_size = 7
    output_size = 5

    lb_in = tf.random_normal(dtype=dtype, shape=(batch_size, input_size))
    ub_in = tf.random_normal(dtype=dtype, shape=(batch_size, input_size))
    lb_in, ub_in = tf.minimum(lb_in, ub_in), tf.maximum(lb_in, ub_in)
    nominal = tf.random_normal(dtype=dtype, shape=(batch_size, input_size))

    # Linear layer.
    w = tf.random_normal(dtype=dtype, shape=(input_size, output_size))
    b = tf.random_normal(dtype=dtype, shape=(output_size,))

    # Batch norm layer.
    epsilon = 1.e-2
    bn_initializers = {
        'beta': tf.random_normal_initializer(),
        'gamma': tf.random_uniform_initializer(.1, 3.),
        'moving_mean': tf.random_normal_initializer(),
        'moving_variance': tf.random_uniform_initializer(.1, 3.)
    }
    batchnorm_module = batchnorm_class(offset=True, scale=True, eps=epsilon,
                                       initializers=bn_initializers)
    # Connect the batchnorm module to the graph.
    batchnorm_module(tf.random_normal(dtype=dtype,
                                      shape=(batch_size, output_size)),
                     is_training=is_training)

    bounds_in = ibp.RelativeIntervalBounds(lb_in - nominal,
                                           ub_in - nominal, nominal)
    bounds_out = bounds_in.apply_linear(None, w, b)
    bounds_out = bounds_out.apply_batch_norm(
        batchnorm_module,
        batchnorm_module.mean if is_training else batchnorm_module.moving_mean,
        batchnorm_module.variance if is_training
        else batchnorm_module.moving_variance,
        batchnorm_module.gamma,
        batchnorm_module.beta,
        epsilon)
    lb_out, ub_out = bounds_out.lower, bounds_out.upper

    # Separately, calculate dual objective by adjusting the linear layer.
    wn, bn = layer_utils.combine_with_batchnorm(w, b, batchnorm_module)
    bounds_out_lin = bounds_in.apply_linear(None, wn, bn)
    lb_out_lin, ub_out_lin = bounds_out_lin.lower, bounds_out_lin.upper

    init_op = tf.global_variables_initializer()

    with self.test_session() as session:
      session.run(init_op)
      (lb_out_val, ub_out_val,
       lb_out_lin_val, ub_out_lin_val) = session.run((lb_out, ub_out,
                                                      lb_out_lin, ub_out_lin))
      self.assertAllClose(lb_out_val, lb_out_lin_val, atol=tol, rtol=tol)
      self.assertAllClose(ub_out_val, ub_out_lin_val, atol=tol, rtol=tol)


def _materialised_conv_bounds(w, b, padding, strides, bounds_in):
  """Calculates bounds on output of an N-D convolution layer.

  The calculation is performed by first materialising the convolution as a
  (sparse) fully-connected linear layer. Doing so will affect performance, but
  may be useful for investigating numerical stability issues.

  Args:
    w: (N+2)D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing weights for the convolution.
    b: 1D tensor of shape (output_channels) containing biases for the
      convolution, or `None` if no bias.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.
    bounds_in: bounds of shape (batch_size, input_height, input_width,
      input_channels) containing bounds on the inputs to the
      convolution layer.

  Returns:
    bounds of shape (batch_size, output_height, output_width,
      output_channels) with bounds on the outputs of the
      convolution layer.

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
