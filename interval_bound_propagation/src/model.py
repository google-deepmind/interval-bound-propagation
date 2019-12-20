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
import tensorflow.compat.v1 as tf


# Set of supported activations. Must be monotonic and attributes of `tf.nn`.
_ALLOWED_ACTIVATIONS = set([
    'elu',
    'leaky_relu',
    'relu',
    'relu6',
    'selu',
    'sigmoid',
    'softplus',
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
    'Softplus': tf.nn.softplus,
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
  def input_wrappers(self):
    self._ensure_is_connected()
    return self._model_inputs

  @property
  def modules(self):
    self._ensure_is_connected()
    return self._modules

  def dependencies(self, module):
    self._ensure_is_connected()
    return self._module_depends_on[module]

  @property
  def output_module(self):
    self._ensure_is_connected()
    return self._produced_by[self._logits.name]

  def fanout_of(self, node):
    """Looks up fan-out for a given node.

    Args:
      node: `ibp.VerifiableWrapper` occurring in the network either as an
        operation, or as the initial input.

    Returns:
      Number of times `node` occurs as the input of another operation within
        the network, or 1 if `node` is the overall output.
    """
    return self._fanouts[node]

  def _build(self, *z0, **kwargs):
    """Outputs logits from input z0.

    Args:
      *z0: inputs as `Tensor`.
      **kwargs: Other arguments passed directly to the _build() function of the
        wrapper model. Assumes the possible presence of `override` (defaults to
        False). However, if False, this function does not update any internal
        state and reuses any components computed by a previous call to _build().
        If there were no previous calls to _build(), behaves as if it was set to
        True.

    Returns:
      logits resulting from using z0 as inputs.
    """
    override = not self.is_connected
    if 'override' in kwargs:
      override = kwargs['override'] or override
      del kwargs['override']
    if override:
      self._inputs = z0[0] if len(z0) == 1 else z0
      # Build underlying verifiable modules.
      self._model_inputs = []
      self._modules = []
      self._produced_by = {}  # Connection graph.
      self._fanouts = collections.Counter()
      for i, z in enumerate(z0):
        self._model_inputs.append(verifiable_wrapper.ModelInputWrapper(i))
        self._produced_by[z.name] = self._model_inputs[-1]
      self._module_depends_on = collections.defaultdict(list)
      self._output_by_module = {}
      with snt.observe_connections(self._observer):
        logits = self._net_builder(*z0, **kwargs)
      # Logits might be produced by a non-Sonnet module.
      self._backtrack(logits, max_depth=100)
      # Log analysis.
      for m in self._modules:
        logging.info('Found: %s', m)
        output_shape = self._output_by_module[m].shape.as_list()[1:]
        logging.info('  Output shape: %s => %d units', output_shape,
                     np.prod(output_shape))
        for depends in self._module_depends_on[m]:
          logging.info('  Depends on: %s', depends)
      logging.info('Final logits produced by: %s',
                   self._produced_by[logits.name])
      self._logits = logits
      self._num_classes = logits.shape[-1].value
    else:
      # Must have been connected once before.
      self._ensure_is_connected()
      logits = self._net_builder(*z0, **kwargs)
    return logits

  def _observer(self, subgraph):
    input_nodes = self._inputs_for_observed_module(subgraph)
    if input_nodes is None:
      # We do not fail as we want to allow higher-level Sonnet components.
      # In practice, the rest of the logic will fail if we are unable to
      # connect all low-level modules.
      logging.warn('Unprocessed module "%s"', str(subgraph.module))
      return
    if subgraph.outputs in input_nodes:
      # The Sonnet module is just returning its input as its output.
      # This may happen with a reshape in which the shape does not change.
      return

    self._add_module(self._wrapper_for_observed_module(subgraph),
                     subgraph.outputs, *input_nodes)

  def _inputs_for_observed_module(self, subgraph):
    """Extracts input tensors from a connected Sonnet module.

    This default implementation supports common layer types, but should be
    overridden if custom layer types are to be supported.

    Args:
      subgraph: `snt.ConnectedSubGraph` specifying the Sonnet module being
        connected, and its inputs and outputs.

    Returns:
      List of input tensors, or None if not a supported Sonnet module.
    """
    m = subgraph.module
    # Only support a few operations for now.
    if not (isinstance(m, snt.BatchReshape) or
            isinstance(m, snt.Linear) or
            isinstance(m, snt.Conv1D) or
            isinstance(m, snt.Conv2D) or
            isinstance(m, snt.BatchNorm) or
            isinstance(m, layers.ImageNorm)):
      return None

    if isinstance(m, snt.BatchNorm):
      return subgraph.inputs['input_batch'],
    else:
      return subgraph.inputs['inputs'],

  def _wrapper_for_observed_module(self, subgraph):
    """Creates a wrapper for a connected Sonnet module.

    This default implementation supports common layer types, but should be
    overridden if custom layer types are to be supported.

    Args:
      subgraph: `snt.ConnectedSubGraph` specifying the Sonnet module being
        connected, and its inputs and outputs.

    Returns:
      `ibp.VerifiableWrapper` for the Sonnet module.
    """
    m = subgraph.module
    if isinstance(m, snt.BatchReshape):
      shape = subgraph.outputs.get_shape()[1:].as_list()
      return verifiable_wrapper.BatchReshapeWrapper(m, shape)
    elif isinstance(m, snt.Linear):
      return verifiable_wrapper.LinearFCWrapper(m)
    elif isinstance(m, snt.Conv1D):
      return verifiable_wrapper.LinearConv1dWrapper(m)
    elif isinstance(m, snt.Conv2D):
      return verifiable_wrapper.LinearConv2dWrapper(m)
    elif isinstance(m, layers.ImageNorm):
      return verifiable_wrapper.ImageNormWrapper(m)
    else:
      assert isinstance(m, snt.BatchNorm)
      return verifiable_wrapper.BatchNormWrapper(m)

  def _backtrack(self, node, max_depth=100):
    if node.name not in self._produced_by:
      if max_depth <= 0:
        raise ValueError('Unable to backtrack through the graph. '
                         'Consider using more basic Sonnet modules.')
      self._wrap_node(node, max_depth=(max_depth - 1))
    self._fanouts[self._produced_by[node.name]] += 1

  def _wrap_node(self, node, **kwargs):
    """Adds an IBP wrapper for the node, and backtracks through its inputs.

    This default implementation supports common layer types, but should be
    overridden if custom layer types are to be supported.

    Implementations should create a `ibp.VerifiableWrapper` and then invoke
    `self._add_module(wrapper, node, *input_node, **kwargs)`.

    Args:
      node: TensorFlow graph node to wrap for IBP.
      **kwargs: Context to pass to `self._add_module`.
    """
    # Group all unary monotonic ops at the end.
    if node.op.type in ('Add', 'AddV2', 'Mul', 'Sub', 'Maximum', 'Minimum'):
      input_node0 = node.op.inputs[0]
      input_node1 = node.op.inputs[1]
      if node.op.type in ('Add', 'AddV2'):
        w = verifiable_wrapper.IncreasingMonotonicWrapper(tf.add)
      elif node.op.type == 'Mul':
        w = verifiable_wrapper.PiecewiseMonotonicWrapper(tf.multiply)
      elif node.op.type == 'Sub':
        w = verifiable_wrapper.PiecewiseMonotonicWrapper(tf.subtract)
      elif node.op.type == 'Maximum':
        w = verifiable_wrapper.IncreasingMonotonicWrapper(tf.maximum)
      elif node.op.type == 'Minimum':
        w = verifiable_wrapper.IncreasingMonotonicWrapper(tf.minimum)

      self._add_module(w, node, input_node0, input_node1, **kwargs)
      return
    elif node.op.type == 'ConcatV2':
      num_inputs = node.op.get_attr('N')
      assert num_inputs == len(node.op.inputs) - 1
      inputs = node.op.inputs[:num_inputs]
      axis = node.op.inputs[num_inputs]
      def concat(*args):
        return tf.concat(args, axis=axis)
      self._add_module(
          verifiable_wrapper.IncreasingMonotonicWrapper(concat, axis=axis),
          node, *inputs, **kwargs)
      return
    elif node.op.type == 'Softmax':
      input_node = node.op.inputs[0]
      self._add_module(verifiable_wrapper.SoftmaxWrapper(), node, input_node,
                       **kwargs)
      return
    elif node.op.type == 'Const':
      self._add_module(verifiable_wrapper.ConstWrapper(node), node, **kwargs)
      return

    # The rest are all unary monotonic ops.
    parameters = dict()
    if node.op.type in _MONOTONIC_NODE_OPS:
      input_node = node.op.inputs[0]
      # Leaky ReLUs are a special case since they have a second argument.
      if node.op.type == 'LeakyRelu':
        parameters = dict(alpha=node.op.get_attr('alpha'))
        # Use function definition instead of lambda for clarity.
        def leaky_relu(x):
          return tf.nn.leaky_relu(x, **parameters)
        fn = leaky_relu
      else:
        fn = _MONOTONIC_NODE_OPS[node.op.type]

    elif node.op.type in ('Mean', 'Max', 'Sum', 'Min'):
      # reduce_mean/max have two inputs. The first one should be produced by a
      # upstream node, while the two one should represent the axis.
      input_node = node.op.inputs[0]
      parameters = dict(axis=node.op.inputs[1],
                        keep_dims=node.op.get_attr('keep_dims'))
      # Use function definition instead of lambda for clarity.
      def reduce_max(x):
        return tf.reduce_max(x, **parameters)
      def reduce_mean(x):
        return tf.reduce_mean(x, **parameters)
      def reduce_min(x):
        return tf.reduce_min(x, **parameters)
      def reduce_sum(x):
        return tf.reduce_sum(x, **parameters)
      fn = dict(
          Max=reduce_max, Mean=reduce_mean, Sum=reduce_sum,
          Min=reduce_min)[node.op.type]

    elif node.op.type == 'ExpandDims':
      input_node = node.op.inputs[0]
      parameters = dict(axis=node.op.inputs[1])
      def expand_dims(x):
        return tf.expand_dims(x, **parameters)
      fn = expand_dims

    elif node.op.type == 'Transpose':
      input_node = node.op.inputs[0]
      parameters = dict(perm=node.op.inputs[1])
      def transpose(x):
        return tf.transpose(x, **parameters)
      fn = transpose

    elif node.op.type == 'Squeeze':
      input_node = node.op.inputs[0]
      parameters = dict(axis=node.op.get_attr('squeeze_dims'))
      def squeeze(x):
        return tf.squeeze(x, **parameters)
      fn = squeeze

    elif node.op.type == 'Pad':
      input_node = node.op.inputs[0]
      parameters = dict(paddings=node.op.inputs[1])
      def pad(x):
        return tf.pad(x, **parameters)
      fn = pad

    elif node.op.type in ('MaxPool', 'AvgPool'):
      input_node = node.op.inputs[0]
      parameters = dict(
          ksize=node.op.get_attr('ksize'),
          strides=node.op.get_attr('strides'),
          padding=node.op.get_attr('padding'),
          data_format=node.op.get_attr('data_format'),
      )
      if node.op.type == 'MaxPool':
        def max_pool(x):
          return tf.nn.max_pool(x, **parameters)
        fn = max_pool
      elif node.op.type == 'AvgPool':
        def avg_pool(x):
          return tf.nn.avg_pool(x, **parameters)
        fn = avg_pool

    elif node.op.type == 'Reshape':
      input_node = node.op.inputs[0]
      parameters = dict(shape=node.op.inputs[1])
      def reshape(x):
        return tf.reshape(x, **parameters)
      fn = reshape

    elif node.op.type == 'Identity':
      input_node = node.op.inputs[0]
      def identity(x):
        return tf.identity(x)
      fn = identity

    elif node.op.type == 'MatrixDiag':
      input_node = node.op.inputs[0]
      def matrix_diag(x):
        return tf.matrix_diag(x)
      fn = matrix_diag

    elif node.op.type == 'Slice':
      input_node = node.op.inputs[0]
      parameters = dict(
          begin=node.op.inputs[1],
          size=node.op.inputs[2],
      )
      def regular_slice(x):
        return tf.slice(x, **parameters)
      fn = regular_slice

    elif node.op.type == 'StridedSlice':
      input_node = node.op.inputs[0]
      parameters = dict(
          begin=node.op.inputs[1],
          end=node.op.inputs[2],
          strides=node.op.inputs[3],
          begin_mask=node.op.get_attr('begin_mask'),
          end_mask=node.op.get_attr('end_mask'),
          ellipsis_mask=node.op.get_attr('ellipsis_mask'),
          new_axis_mask=node.op.get_attr('new_axis_mask'),
          shrink_axis_mask=node.op.get_attr('shrink_axis_mask'),
      )
      def strided_slice(x):
        return tf.strided_slice(x, **parameters)
      fn = strided_slice

    elif node.op.type == 'Fill':
      input_node = node.op.inputs[1]  # Shape is the first argument.
      dims = node.op.inputs[0]
      parameters = dict(dims=dims)
      def fill(x):
        return tf.fill(dims, x)
      fn = fill

    elif node.op.type == 'RealDiv':
      # The denominator is assumed to be constant but is permitted to be
      # example-dependent, for example a sequence's length prior to padding.
      input_node = node.op.inputs[0]
      denom = node.op.inputs[1]
      parameters = dict(denom=denom)
      def quotient(x):
        return x / denom
      fn = quotient

    else:
      raise NotImplementedError(
          'Unsupported operation: "{}" with\n{}.'.format(node.op.type, node.op))

    self._add_module(
        verifiable_wrapper.IncreasingMonotonicWrapper(fn, **parameters),
        node, input_node, **kwargs)

  def _add_module(self, wrapper, node, *input_nodes, **kwargs):
    """Adds the given node wrapper, first backtracking through its inputs.

    Args:
      wrapper: `ibp.VerifiableWrapper` for the node.
      node: TensorFlow graph node.
      *input_nodes: Input nodes for `node`.
      **kwargs: Contains the `max_depth` argument for recursive _backtrack call.
    """
    for input_node in input_nodes:
      self._backtrack(input_node, **kwargs)
    self._modules.append(wrapper)
    self._produced_by[node.name] = self._modules[-1]
    self._module_depends_on[self._modules[-1]].extend(
        [self._produced_by[input_node.name] for input_node in input_nodes])
    self._output_by_module[self._modules[-1]] = node

  def propagate_bounds(self, *input_bounds):
    """Propagates input bounds through the network.

    Args:
      *input_bounds: `AbstractBounds` instance corresponding to z0.

    Returns:
      The final output bounds corresponding to the output logits.
    """
    self._ensure_is_connected()
    def _get_bounds(input_module):
      """Retrieves the bounds corresponding to a module."""
      # All bounds need to be canonicalized to the same type. In particular, we
      # need to handle the case of constant bounds specially. We convert them
      # to the same type as input_bounds.
      if isinstance(input_module, verifiable_wrapper.ConstWrapper):
        return input_bounds[0].convert(input_module.output_bounds)
      return input_module.output_bounds

    # Initialise inputs' bounds.
    for model_input in self._model_inputs:
      model_input.output_bounds = input_bounds[model_input.index]
    # By construction, this list is topologically sorted.
    for m in self._modules:
      # Construct combined input bounds.
      upstream_bounds = [_get_bounds(b) for b in self._module_depends_on[m]]
      m.propagate_bounds(*upstream_bounds)
    # We assume that the last module is the final output layer.
    return self._produced_by[self._logits.name].output_bounds


class StandardModelWrapper(snt.AbstractModule):
  """Wraps a predictor network that keeps track of inputs and logits."""

  def __init__(self, net_builder, name='verifiable_predictor'):
    """Constructor for a non-verifiable model.

    This wrapper can be used to seamlessly use loss.py and utils.py without
    IBP verification.

    Args:
      net_builder: A callable that returns output logits from an input.
        net_builder must accept two arguments: the input (as the first
        argument) and is_training (as the second).
      name: Sonnet module name.
    """
    super(StandardModelWrapper, self).__init__(name=name)
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
    raise RuntimeError('Model is not wrapped by a VerifiableModelWrapper. '
                       'Bounds cannot be propagated.')

  def propagate_bounds(self, *input_bounds):
    raise RuntimeError('Model is not wrapped by a VerifiableModelWrapper. '
                       'Bounds cannot be propagated.')

  def _build(self, *z0, **kwargs):
    """Outputs logits from input z0.

    Args:
      *z0: inputs as `Tensor`.
      **kwargs: Other arguments passed directly to the _build() function of the
        wrapper model. Assumes the possible presence of `override` (defaults to
        False). However, if False, this function does not update any internal
        state and reuses any components computed by a previous call to _build().
        If there were no previous calls to _build(), behaves as if it was set to
        True.

    Returns:
      logits resulting from using z0 as inputs.
    """
    override = not self.is_connected
    if 'override' in kwargs:
      override = kwargs['override'] or override
      del kwargs['override']
    if override:
      self._inputs = z0[0] if len(z0) == 1 else z0
      logits = self._net_builder(*z0, **kwargs)
      self._logits = logits
      self._num_classes = logits.shape[-1].value
    else:
      # Must have been connected once before.
      self._ensure_is_connected()
      logits = self._net_builder(*z0, **kwargs)
    return logits


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
      regularizer = tf.keras.regularizers.l2(l=0.5*l2_regularization_scale)
      self._regularizers = {'w': regularizer}
    else:
      self._regularizers = None
    # The following allows to reuse previous batch norm statistics.
    self._batch_norms = {}

  def _build(self, z0, is_training=True, test_local_stats=False, reuse=False):
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
        if reuse:
          if m.scope_name not in self._batch_norms:
            raise ValueError('Cannot set reuse to True without connecting the '
                             'module once before.')
          m = self._batch_norms[m.scope_name]
        else:
          self._batch_norms[m.scope_name] = m
        zk = m(zk, is_training=is_training, test_local_stats=test_local_stats,
               reuse=reuse)
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
