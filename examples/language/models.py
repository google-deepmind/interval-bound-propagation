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

"""Models for sentence representation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow.compat.v1 as tf


def _max_pool_1d(x, pool_size=2, name='max_pool_1d'):
  with tf.name_scope(name, 'MaxPool1D', [x, pool_size]):
    return tf.squeeze(
        tf.nn.max_pool(tf.expand_dims(x, 1),
                       [1, 1, pool_size, 1],
                       [1, 1, pool_size, 1],
                       'VALID'),
        axis=1)


class SentenceRepresenterConv(snt.AbstractModule):
  """Use stacks of 1D Convolutions to build a sentence representation."""

  def __init__(self,
               config,
               keep_prob=1.,
               pooling='max',
               name='sentence_rep_conv'):
    super(SentenceRepresenterConv, self).__init__(name=name)
    self._config = config
    self._pooling = pooling
    self._keep_prob = keep_prob

  def _build(self, padded_word_embeddings, length):
    x = padded_word_embeddings
    for layer in self._config['conv_architecture']:
      if isinstance(layer, tuple) or isinstance(layer, list):
        filters, kernel_size, pooling_size = layer
        conv = snt.Conv1D(
            output_channels=filters,
            kernel_shape=kernel_size)
        x = conv(x)
        if pooling_size and pooling_size > 1:
          x = _max_pool_1d(x, pooling_size)
      elif layer == 'relu':
        x = tf.nn.relu(x)
        if self._keep_prob < 1:
          x = tf.nn.dropout(x, keep_prob=self._keep_prob)
      else:
        raise RuntimeError('Bad layer type {} in conv'.format(layer))
    # Final layer pools over the remaining sequence length to get a
    # fixed sized vector.
    if self._pooling == 'max':
      x = tf.reduce_max(x, axis=1)
    elif self._pooling == 'average':
      x = tf.reduce_sum(x, axis=1)
      lengths = tf.expand_dims(tf.cast(length, tf.float32), axis=1)
      x = x / lengths

    if self._config['conv_fc1']:
      fc1_layer = snt.Linear(output_size=self._config['conv_fc1'])
      x = tf.nn.relu(fc1_layer(x))
      if self._keep_prob < 1:
        x = tf.nn.dropout(x, keep_prob=self._keep_prob)
    if self._config['conv_fc2']:
      fc2_layer = snt.Linear(output_size=self._config['conv_fc2'])
      x = tf.nn.relu(fc2_layer(x))
      if self._keep_prob < 1:
        x = tf.nn.dropout(x, keep_prob=self._keep_prob)

    return x



