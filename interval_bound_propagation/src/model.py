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

"""Sonnet modules that represent the predictor network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
from interval_bound_propagation.src import layers
from interval_bound_propagation.src import verifiable_wrapper
import numpy as np
import sonnet as snt
import tensorflow as tf


# Set of supported activations. Must be monotonic and attributes of `tf.nn`.
_ALLOWED_ACTIVATIONS = set([
    'elu',
    'leaky_relu',
    'relu',
    'relu6',
    'selu',
    'sigmoid',
    'softsign',
    'tanh',
])

# Mapping between graph node ops and their TensorFlow function.
_MONOTONIC_NODE_OPS = {
    'Elu': tf.nn.elu,
    'LeakyRelu': tf.nn.leaky_relu,
    'Relu': tf.nn.relu,
    'Relu6': tf.nn.relu6,
    'Selu': tf.nn.selu,
    'Sigmoid': tf.nn.sigmoid,
    'Softsign': tf.nn.softsign,
    'Tanh': tf.nn.tanh,
}


class VerifiableModelWrapper(snt.AbstractModule):
  """Wraps a predictor network."""

  def __init__(self, net_builder, name='verifiable_predictor'):
    """Constructor for the verifiable model.

    Args:
      net_builder: A callable that returns output logits from an input.
        net_builder must accept two arguments: the input (as the first
        argument) and is_training (as the second).
      name: Sonnet module name.
    """
    super(VerifiableModelWrapper, self).__init__(name=name)
    self._net_builder = net_builder

  @property
  def wrapped_network(self):
    return self._net_builder

  @property
  def output_size(self):
    self._ensure_is_connected()
    return self._num_classes

  @property
  def logits(self):
    self._ensure_is_connected()
    return self._logits

  @property
  def inputs(self):
    self._ensure_is_connected()
    return self._inputs

  @property
  def modules(self):
    self._ensure_is_connected()
    return self._modules

  def _build(self, z0, is_training=False, passthrough=False):
    """Outputs logits from input z0.

    Args:
      z0: input.
      is_training: Boolean to indicate to `BatchNorm` if we are
        currently training. By default `False`.
      passthrough: If True, this function does not update any internal state.

    Returns:
      logits resulting from using z0 as input.
    """
    if passthrough:
      # Must have been connected once before.
      self._ensure_is_connected()
    else:
      self._inputs = z0
    if not passthrough:
      # Build underlying verifiable modules.
      self._modules = []
      self._produced_by = {z0.name: None}  # Connection graph.
      self._module_depends_on = collections.defaultdict(list)
      self._output_by_module = {}  # For logging only.
      with snt.observe_connections(self._observer):
        logits = self._net_builder(z0, is_training=is_training)
      # Log analysis.
      for m in self._modules:
        logging.info('Found: %s', m)
        logging.info('  Output shape: %s', self._output_by_module[m].shape)
        for depends in self._module_depends_on[m]:
          logging.info('  Depends on: %s', z0 if depends is None else depends)
      logging.info('Final logits produced by: %s',
                   self._produced_by[logits.name])
      self._logits = logits
      self._num_classes = logits.shape[-1].value
    else:
      logits = self._net_builder(z0, is_training=is_training)
    return logits

  def _observer(self, subgraph):
    m = subgraph.module
    # Only support a few operations for now.
    if not (isinstance(m, snt.BatchFlatten) or
            isinstance(m, snt.Linear) or
            isinstance(m, snt.Conv2D) or
            isinstance(m, layers.ImageNorm) or
            isinstance(m, layers.BatchNorm)):
      # We do not fail as we want to allow higher-level Sonnet components.
      # In practice, the rest of the logic will fail if we are unable to
      # connect all low-level modules.
      logging.warn('Unprocessed module "%s"', str(m))
      return

    # If the input is unknown, we must come from support sequences of
    # TF operations.
    if isinstance(m, layers.BatchNorm):
      input_node = subgraph.inputs['input_batch']
    else:
      input_node = subgraph.inputs['inputs']
    if input_node.name not in self._produced_by:
      self._backtrack(input_node, max_depth=10)

    if isinstance(m, snt.BatchFlatten):
      self._modules.append(verifiable_wrapper.BatchFlattenWrapper(m))
    elif isinstance(m, snt.Linear):
      self._modules.append(verifiable_wrapper.LinearFCWrapper(m))
    elif isinstance(m, snt.Conv2D):
      self._modules.append(verifiable_wrapper.LinearConv2dWrapper(m))
    elif isinstance(m, layers.ImageNorm):
      self._modules.append(verifiable_wrapper.ImageNormWrapper(m))
    else:
      assert isinstance(m, layers.BatchNorm)
      self._modules.append(verifiable_wrapper.BatchNormWrapper(m))
    self._produced_by[subgraph.outputs.name] = self._modules[-1]
    self._module_depends_on[self._modules[-1]].append(
        self._produced_by[input_node.name])
    self._output_by_module[self._modules[-1]] = subgraph.outputs

  def _backtrack(self, node, max_depth=100):
    if node.name in self._produced_by:
      return
    if max_depth <= 0:
      raise ValueError('Unable to backtrack through the graph. Consider using '
                       'more basic Sonnet modules.')

    if node.op.type in _MONOTONIC_NODE_OPS:
      input_node = node.op.inputs[0]
      self._backtrack(input_node, max_depth - 1)
      # Leaky ReLUs are a special case since they have a second argument.
      if node.op.type == 'LeakyRelu':
        alpha = node.op.get_attr('alpha')
        # Use function definition instead of lambda for clarity.
        def leaky_relu(x):
          return tf.nn.leaky_relu(x, alpha)
        fn = leaky_relu
      else:
        fn = _MONOTONIC_NODE_OPS[node.op.type]
      self._modules.append(verifiable_wrapper.MonotonicWrapper(fn))
      self._produced_by[node.name] = self._modules[-1]
      self._module_depends_on[self._modules[-1]].append(
          self._produced_by[input_node.name])
      self._output_by_module[self._modules[-1]] = node
    elif node.op.type == 'Add':
      input_node0 = node.op.inputs[0]
      input_node1 = node.op.inputs[1]
      self._backtrack(input_node0, max_depth - 1)
      self._backtrack(input_node1, max_depth - 1)
      self._modules.append(verifiable_wrapper.MonotonicWrapper(tf.add))
      self._produced_by[node.name] = self._modules[-1]
      self._module_depends_on[self._modules[-1]].append(
          self._produced_by[input_node0.name])
      self._module_depends_on[self._modules[-1]].append(
          self._produced_by[input_node1.name])
      self._output_by_module[self._modules[-1]] = node
    elif node.op.type in ('Mean', 'Max'):
      # reduce_mean/max have two inputs. The first one should be produced by a
      # upstream node, while the two one should represent the axis.
      input_node = node.op.inputs[0]
      self._backtrack(input_node, max_depth - 1)
      axis = node.op.inputs[1]
      keep_dims = node.op.get_attr('keep_dims')
      # Use function definition instead of lambda for clarity.
      def reduce_mean(x):
        return tf.reduce_mean(x, axis=axis, keep_dims=keep_dims)
      def reduce_max(x):
        return tf.reduce_max(x, axis=axis, keep_dims=keep_dims)
      self._modules.append(verifiable_wrapper.MonotonicWrapper(
          reduce_mean if node.op.type == 'Mean' else reduce_max))
      self._produced_by[node.name] = self._modules[-1]
      self._module_depends_on[self._modules[-1]].append(
          self._produced_by[input_node.name])
      self._output_by_module[self._modules[-1]] = node
    elif node.op.type == 'ConcatV2':
      num_inputs = node.op.get_attr('N')
      assert num_inputs == len(node.op.inputs) - 1
      inputs = node.op.inputs[:num_inputs]
      axis = node.op.inputs[num_inputs]
      for i in inputs:
        self._backtrack(i, max_depth - 1)
      def concat(*args):
        return tf.concat(args, axis=axis)
      self._modules.append(verifiable_wrapper.MonotonicWrapper(concat))
      self._produced_by[node.name] = self._modules[-1]
      for i in inputs:
        self._module_depends_on[self._modules[-1]].append(
            self._produced_by[i.name])
      self._output_by_module[self._modules[-1]] = node
    elif node.op.type == 'ExpandDims':
      input_node = node.op.inputs[0]
      self._backtrack(input_node, max_depth - 1)
      axis = node.op.inputs[1]
      def expand_dims(x):
        return tf.expand_dims(x, axis=axis)
      self._modules.append(verifiable_wrapper.MonotonicWrapper(expand_dims))
      self._produced_by[node.name] = self._modules[-1]
      self._module_depends_on[self._modules[-1]].append(
          self._produced_by[input_node.name])
      self._output_by_module[self._modules[-1]] = node
    elif node.op.type == 'Squeeze':
      input_node = node.op.inputs[0]
      self._backtrack(input_node, max_depth - 1)
      squeeze_dims = node.op.get_attr('squeeze_dims')
      def squeeze(x):
        return tf.squeeze(x, axis=squeeze_dims)
      self._modules.append(verifiable_wrapper.MonotonicWrapper(squeeze))
      self._produced_by[node.name] = self._modules[-1]
      self._module_depends_on[self._modules[-1]].append(
          self._produced_by[input_node.name])
      self._output_by_module[self._modules[-1]] = node
    else:
      raise NotImplementedError(
          'Unsupported operation: "{}".'.format(node.op.type))

  def propagate_bounds(self, input_bounds):
    self._ensure_is_connected()
    def _get_bounds(input_module):
      return (
          input_bounds if input_module is None else input_module.output_bounds)
    # By construction, this list is topologically sorted.
    for m in self._modules:
      # Construct combined input bounds.
      upstream_bounds = [_get_bounds(b) for b in self._module_depends_on[m]]
      m.propagate_bounds(*upstream_bounds)
    # We assume that the last module is the final output layer.
    return self._produced_by[self._logits.name].output_bounds


class DNN(snt.AbstractModule):
  """Simple feed-forward neural network."""

  def __init__(self, num_classes, layer_types, l2_regularization_scale=0.,
               name='predictor'):
    """Constructor for the DNN.

    Args:
      num_classes: Output size.
      layer_types: Iterable of tuples. Each tuple must be one of the following:
        * ('conv2d', (kernel_height, width), channels, padding, stride)
        * ('linear', output_size)
        * ('batch_normalization',)
        * ('activation', activation)
        Convolutional layers must precede all linear layers.
      l2_regularization_scale: Scale of the L2 regularization on the weights
        of each layer.
      name: Sonnet module name.
    """
    super(DNN, self).__init__(name=name)
    self._layer_types = list(layer_types)
    self._layer_types.append(('linear', num_classes))
    if l2_regularization_scale > 0.:
      regularizer = tf.contrib.layers.l2_regularizer(
          scale=l2_regularization_scale)
      self._regularizers = {'w': regularizer}
    else:
      self._regularizers = None

  def _build(self, z0, is_training=False):
    """Outputs logits."""
    zk = z0
    conv2d_id = 0
    linear_id = 0
    name = None
    for spec in self._layer_types:
      if spec[0] == 'conv2d':
        if linear_id > 0:
          raise ValueError('Convolutional layers must precede fully connected '
                           'layers.')
        name = 'conv2d_{}'.format(conv2d_id)
        conv2d_id += 1
        (_, (kernel_height, kernel_width), channels, padding, stride) = spec
        m = snt.Conv2D(output_channels=channels,
                       kernel_shape=(kernel_height, kernel_width),
                       padding=padding, stride=stride, use_bias=True,
                       regularizers=self._regularizers,
                       initializers=_create_conv2d_initializer(
                           zk.get_shape().as_list()[1:], channels,
                           (kernel_height, kernel_width)),
                       name=name)
        zk = m(zk)
      elif spec[0] == 'linear':
        must_flatten = (linear_id == 0 and len(zk.shape) > 2)
        if must_flatten:
          zk = snt.BatchFlatten()(zk)
        name = 'linear_{}'.format(linear_id)
        linear_id += 1
        output_size = spec[1]
        m = snt.Linear(output_size,
                       regularizers=self._regularizers,
                       initializers=_create_linear_initializer(
                           np.prod(zk.get_shape().as_list()[1:]), output_size),
                       name=name)
        zk = m(zk)
      elif spec[0] == 'batch_normalization':
        if name is None:
          raise ValueError('Batch normalization only supported after linear '
                           'layers.')
        name += '_batch_norm'
        m = layers.BatchNorm(name=name)
        zk = m(zk, is_training=is_training)
      elif spec[0] == 'activation':
        if spec[1] not in _ALLOWED_ACTIVATIONS:
          raise NotImplementedError(
              'Only the following activations are supported {}'.format(
                  list(_ALLOWED_ACTIVATIONS)))
        name = None
        m = getattr(tf.nn, spec[1])
        zk = m(zk)
    return zk


def _create_conv2d_initializer(
    input_shape, output_channels, kernel_shape, dtype=tf.float32):  # pylint: disable=unused-argument
  """Returns a default initializer for the weights of a convolutional module."""
  return {
      'w': tf.orthogonal_initializer(),
      'b': tf.zeros_initializer(dtype=dtype),
  }


def _create_linear_initializer(input_size, output_size, dtype=tf.float32):  # pylint: disable=unused-argument
  """Returns a default initializer for the weights of a linear module."""
  return {
      'w': tf.orthogonal_initializer(),
      'b': tf.zeros_initializer(dtype=dtype),
  }
