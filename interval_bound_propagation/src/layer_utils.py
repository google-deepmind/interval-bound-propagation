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

"""Graph construction for dual verification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from interval_bound_propagation.src import layers
import sonnet as snt
import tensorflow.compat.v1 as tf


def conv_output_shape(input_shape, w, padding, strides):
  """Calculates the output shape of the given N-D convolution.

  Args:
    input_shape: Integer list of length N+1 specifying the non-batch dimensions
      of the inputs: [input_height, input_width, input_channels].
    w: (N+2)D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing weights for the convolution.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

  Returns:
    Integer list of length N+1 specifying the non-batch dimensions
      of the outputs: [output_height, output_width, output_channels].

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  # Connect a convolution (never to be run) to infer the output's
  # spatial structure.
  dummy_inputs = tf.zeros(dtype=w.dtype, shape=([1] + input_shape))
  if len(w.shape) == 4:
    dummy_outputs = tf.nn.convolution(dummy_inputs,
                                      w, padding=padding, strides=strides)
  elif len(w.shape) == 3:
    dummy_outputs = tf.nn.conv1d(dummy_inputs,
                                 w, padding=padding, stride=strides[0])
  else:
    raise ValueError()
  return dummy_outputs.shape.as_list()[1:]


def materialise_conv(w, b, input_shape, padding, strides):
  """Converts an N-D convolution to an equivalent linear layer.

  Args:
    w: (N+2)D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing the convolution weights.
    b: 1D tensor of shape (output_channels) containing the convolution biases,
      or `None` if no biases.
    input_shape: Integer list of length N+1 specifying the non-batch dimensions
      of the inputs: [input_height, input_width, input_channels].
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of length N: `[vertical_stride, horizontal_stride]`.

  Returns:
    w: 2D tensor of shape (input_height * input_width * input_channels,
      output_height * output_width * output_channels) containing weights.
    b: 1D tensor of shape (output_height * output_width * output_channels)
      containing biases, or `None` if no biases.

  Raises:
    ValueError: if an unsupported convolution dimensionality is encountered.
  """
  if len(input_shape) == 3:
    return _materialise_conv2d(w, b, input_shape[0], input_shape[1],
                               padding, strides)
  elif len(input_shape) == 2:
    return _materialise_conv1d(w, b, input_shape[0], padding, strides[0])
  else:
    raise ValueError()


def _materialise_conv2d(w, b, input_height, input_width, padding, strides):
  """Converts a convolution to an equivalent linear layer.

  Args:
    w: 4D tensor of shape (kernel_height, kernel_width, input_channels,
      output_channels) containing the convolution weights.
    b: 1D tensor of shape (output_channels) containing the convolution biases,
      or `None` if no biases.
    input_height: height of the input tensor.
    input_width: width of the input tensor.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    strides: Integer list of `[vertical_stride, horizontal_stride]`.

  Returns:
    w: 2D tensor of shape (input_height * input_width * input_channels,
      output_height * output_width * output_channels) containing weights.
    b: 1D tensor of shape (output_height * output_width * output_channels)
      containing biases, or `None` if no biases.
  """
  kernel_height = w.shape[0].value
  kernel_width = w.shape[1].value
  input_channels = w.shape[2].value
  output_channels = w.shape[3].value

  # Temporarily move the input_channels dimension to output_channels.
  w = tf.reshape(w, shape=(kernel_height, kernel_width, 1,
                           input_channels * output_channels))
  # Apply the convolution to elementary (i.e. one-hot) inputs.
  diagonal_input = tf.reshape(
      tf.eye(input_height * input_width, dtype=w.dtype),
      shape=[input_height * input_width, input_height, input_width, 1])
  conv = tf.nn.convolution(
      diagonal_input, w,
      padding=padding, strides=strides)
  output_height = conv.shape[1].value
  output_width = conv.shape[2].value
  # conv is of shape (input_height * input_width, output_height, output_width,
  #                   input_channels * output_channels).
  # Reshape it to (input_height * input_width * input_channels,
  #                output_height * output_width * output_channels).
  w = tf.reshape(conv, shape=(
      [input_height * input_width,
       output_height, output_width,
       input_channels, output_channels]))
  w = tf.transpose(w, perm=[0, 3, 1, 2, 4])
  w = tf.reshape(w, shape=(
      [input_height * input_width * input_channels,
       output_height * output_width * output_channels]))

  # Broadcast b over spatial dimensions.
  b = tf.tile(b, [output_height * output_width]) if b is not None else None

  return w, b


def _materialise_conv1d(w, b, input_length, padding, stride):
  """Converts a convolution to an equivalent linear layer.

  Args:
    w: 3D tensor of shape (kernel_length, input_channels,
      output_channels) containing the convolution weights.
    b: 1D tensor of shape (output_channels) containing the convolution biases,
      or `None` if no biases.
    input_length: length of the input tensor.
    padding: `"VALID"` or `"SAME"`, the convolution's padding algorithm.
    stride: Integer stride.

  Returns:
    w: 2D tensor of shape (input_length * input_channels,
      output_length * output_channels) containing weights.
    b: 1D tensor of shape (output_length * output_channels)
      containing biases, or `None` if no biases.
  """
  kernel_length = w.shape[0].value
  input_channels = w.shape[1].value
  output_channels = w.shape[2].value

  # Temporarily move the input_channels dimension to output_channels.
  w = tf.reshape(w, shape=(kernel_length, 1,
                           input_channels * output_channels))
  # Apply the convolution to elementary (i.e. one-hot) inputs.
  diagonal_input = tf.reshape(
      tf.eye(input_length, dtype=w.dtype),
      shape=[input_length, input_length, 1])
  conv = tf.nn.conv1d(
      diagonal_input, w,
      padding=padding, stride=stride)
  output_length = conv.shape[1].value
  # conv is of shape (input_length, output_length,
  #                   input_channels * output_channels).
  # Reshape it to (input_length * input_channels,
  #                output_length * output_channels).
  w = tf.reshape(conv, shape=(
      [input_length,
       output_length,
       input_channels, output_channels]))
  w = tf.transpose(w, perm=[0, 2, 1, 3])
  w = tf.reshape(w, shape=(
      [input_length * input_channels,
       output_length * output_channels]))

  # Broadcast b over spatial dimensions.
  b = tf.tile(b, [output_length]) if b is not None else None

  return w, b


def decode_batchnorm(batchnorm_module):
  """Calculates the neuron-wise multipliers and biases of the batch norm layer.

  Note that, in the case of a convolution, the returned bias will have
  spatial dimensions.

  Args:
    batchnorm_module: `snt.BatchNorm` module.

  Returns:
    w: 1D tensor of shape (output_size) or 3D tensor of shape
      (output_height, output_width, output_channels) containing
      neuron-wise multipliers for the batch norm layer.
    b: 1D tensor of shape (output_size) or 3D tensor of shape
      (output_height, output_width, output_channels) containing
      neuron-wise biases for the batch norm layer.
  """
  if isinstance(batchnorm_module, layers.BatchNorm):
    mean = batchnorm_module.mean
    variance = batchnorm_module.variance
    variance_epsilon = batchnorm_module.epsilon
    scale = batchnorm_module.scale
    offset = batchnorm_module.bias

  else:
    assert isinstance(batchnorm_module, snt.BatchNorm)
    mean = batchnorm_module.moving_mean
    variance = batchnorm_module.moving_variance
    variance_epsilon = batchnorm_module._eps  # pylint: disable=protected-access
    try:
      scale = batchnorm_module.gamma
    except snt.Error:
      scale = None
    try:
      offset = batchnorm_module.beta
    except snt.Error:
      offset = None

  w = tf.rsqrt(variance + variance_epsilon)
  if scale is not None:
    w *= scale

  b = -w * mean
  if offset is not None:
    b += offset

  # Batchnorm vars have a redundant leading dim.
  w = tf.squeeze(w, axis=0)
  b = tf.squeeze(b, axis=0)
  return w, b


def combine_with_batchnorm(w, b, batchnorm_module):
  """Combines a linear layer and a batch norm into a single linear layer.

  Calculates the weights and biases of the linear layer formed by
  applying the specified linear layer followed by the batch norm.

  Note that, in the case of a convolution, the returned bias will have
  spatial dimensions.

  Args:
    w: 2D tensor of shape (input_size, output_size) or 4D tensor of shape
      (kernel_height, kernel_width, input_channels, output_channels) containing
      weights for the linear layer.
    b: 1D tensor of shape (output_size) or (output_channels) containing biases
      for the linear layer, or `None` if no bias.
    batchnorm_module: `snt.BatchNorm` module.

  Returns:
    w: 2D tensor of shape (input_size, output_size) or 4D tensor of shape
      (kernel_height, kernel_width, input_channels, output_channels) containing
      weights for the combined layer.
    b: 1D tensor of shape (output_size) or 3D tensor of shape
      (output_height, output_width, output_channels) containing
      biases for the combined layer.
  """
  if b is None:
    b = tf.zeros(dtype=w.dtype, shape=())

  w_bn, b_bn = decode_batchnorm(batchnorm_module)
  return w * w_bn, b * w_bn + b_bn
