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

"""Utilities for sentence representation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from absl import logging
import sonnet as snt
import tensorflow as tf
from tensorflow.contrib import lookup as contrib_lookup


def get_padded_embeddings(embeddings,
                          vocabulary_table,
                          tokens, batch_size,
                          token_indexes=None):
  """Reshapes and pads 'raw' word embeddings.

  Say we have batch of B tokenized sentences, of variable length, with a total
  of W tokens. For example, B = 2 and W = 3 + 4 = 7:
  [['The',   'cat', 'eats'],
   [  'A', 'black',  'cat', 'jumps']]

  Since rows have variable length, this cannot be represented as a tf.Tensor.
  It is represented as a tf.SparseTensor, with 7 values & indexes:
  indices: [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [1,3]]
  values: ['The', 'cat', 'eats', 'A', 'black', 'cat', 'jumps']

  We have also built a vocabulary table:
  vocabulary table: ['cat', 'The', 'A', 'black', 'eats', 'jumps']

  We also have the embeddings, a WxD matrix of floats
  representing each word in the vocabulary table as a normal tf.Tensor.

  For example, with D=3, embeddings could be:
  [[0.4, 0.5, -0.6],  # This is the embedding for word 0 = 'cat'
   [0.1, -0.3, 0.6],  # This is the embedding for word 1 = 'The''
   [0.7, 0.8, -0.9],  # This is the embedding for word 2 = 'A'
   [-0.1, 0.9, 0.7],  # This is the embedding for word 3 = 'black'
   [-0.2, 0.4, 0.7],  # This is the embedding for word 4 = 'eats
   [0.3, -0.5, 0.2]]  # This is the embedding for word 5 = 'jumps'

  This function builds a normal tf.Tensor containing the embeddings for the
  tokens provided, in the correct order, with appropriate 0 padding.

  In our example, the returned tensor would be:
  [[[0.1, -0.3, 0.6], [0.4, 0.5, -0.6], [-0.2, 0.4, 0.7], [0.0, 0.0, 0.0]],
   [[0.7, 0.8, -0.9], [-0.1, 0.9, 0.7], [0.4, 0.5, -0.6], [0.3, -0.5, 0.2]]]

  Note that since the first sentence has only 3 words, the 4th embedding gets
  replaced by a D-dimensional vector of 0.

  Args:
    embeddings: [W, D] Tensor of floats, containing the embeddings, initialized
        with the same vocabulary file as vocabulary_table.
    vocabulary_table: a tf.contrib.lookup.LookupInterface,
        containing the vocabulary, initialized with the same vocabulary file as
        embeddings.
    tokens: [B, ?] SparseTensor of strings, the tokens.
    batch_size: Python integer.
    token_indexes: A Boolean, indicating whether the input tokens are
        token ids or string.

  Returns:
    [B, L, D] Tensor of floats: the embeddings in the correct order,
    appropriately padded with 0.0, where L = max(num_tokens) and B = batch_size
  """
  embedding_dim = embeddings.get_shape()[1].value  # D in docstring above.
  num_tokens_in_batch = tf.shape(tokens.indices)[0]  # W in the docstring above.
  max_length = tokens.dense_shape[1]  # This is L in the docstring above.

  # Get indices of tokens in vocabulary_table.
  if token_indexes is not None:
    indexes = token_indexes
  else:
    indexes = vocabulary_table.lookup(tokens.values)

  # Get word embeddings.
  tokens_embeddings = tf.gather(embeddings, indexes)

  # Shape of the return tensor.
  new_shape = tf.cast(
      tf.stack([batch_size, max_length, embedding_dim], axis=0), tf.int32)

  # Build the vector of indices for the return Tensor.
  # In the example above, indices_final would be:
  # [[[0,0,0], [0,0,1], [0,0,2]],
  #  [[0,1,0], [0,1,1], [0,1,2]],
  #  [[0,2,0], [0,2,1], [0,2,2]],
  #  [[1,0,0], [1,0,1], [1,0,2]],
  #  [[1,1,0], [1,1,1], [1,1,2]],
  #  [[1,2,0], [1,2,1], [1,2,2]],
  #  [[1,3,0], [1,3,1], [1,3,2]]]
  tiled = tf.tile(tokens.indices, [1, embedding_dim])
  indices_tiled = tf.cast(
      tf.reshape(tiled, [num_tokens_in_batch * embedding_dim, 2]), tf.int32)
  indices_linear = tf.expand_dims(
      tf.tile(tf.range(0, embedding_dim), [num_tokens_in_batch]), axis=1)
  indices_final = tf.concat([indices_tiled, indices_linear], axis=1)

  # Build the dense Tensor.
  embeddings_padded = tf.sparse_to_dense(
      sparse_indices=indices_final,
      output_shape=new_shape,
      sparse_values=tf.reshape(tokens_embeddings,
                               [num_tokens_in_batch * embedding_dim]))
  embeddings_padded.set_shape((batch_size, None, embedding_dim))

  return embeddings_padded


def get_padded_indexes(vocabulary_table,
                       tokens, batch_size,
                       token_indexes=None):
  """Get the indices of tokens from vocabulary table.

  Args:
    vocabulary_table: a tf.contrib.lookup.LookupInterface,
        containing the vocabulary, initialized with the same vocabulary file as
        embeddings.
    tokens: [B, ?] SparseTensor of strings, the tokens.
    batch_size: Python integer.
    token_indexes: A Boolean, indicating whether the input tokens are
        token ids or string.

  Returns:
    [B, L] Tensor of integers: indices of tokens in the correct order,
    appropriately padded with 0, where L = max(num_tokens) and B = batch_size
  """
  num_tokens_in_batch = tf.shape(tokens.indices)[0]
  max_length = tokens.dense_shape[1]

  # Get indices of tokens in vocabulary_table.
  if token_indexes is not None:
    indexes = token_indexes
  else:
    indexes = vocabulary_table.lookup(tokens.values)

  # Build the dense Tensor.
  indexes_padded = tf.sparse_to_dense(
      sparse_indices=tokens.indices,
      output_shape=[batch_size, max_length],
      sparse_values=tf.reshape(indexes,
                               [num_tokens_in_batch]))
  indexes_padded.set_shape((batch_size, None))

  return indexes_padded


class EmbedAndPad(snt.AbstractModule):
  """Embed and pad tokenized words.

  This class primary functionality is similar to get_padded_embeddings.
  It stores references to the embeddings and vocabulary table for convenience,
  so that the user does not have to keep and pass them around.
  """

  def __init__(self,
               batch_size,
               vocabularies,
               embedding_dim,
               num_oov_buckets=1000,
               fine_tune_embeddings=False,
               padded_token=None,
               name='embed_and_pad'):
    super(EmbedAndPad, self).__init__(name=name)
    self._batch_size = batch_size
    vocab_file, vocab_size = get_merged_vocabulary_file(vocabularies,
                                                        padded_token)
    self._vocab_size = vocab_size
    self._num_oov_buckets = num_oov_buckets

    # Load vocabulary table for index lookup.
    self._vocabulary_table = contrib_lookup.index_table_from_file(
        vocabulary_file=vocab_file,
        num_oov_buckets=num_oov_buckets,
        vocab_size=self._vocab_size)

    def create_initializer(initializer_range=0.02):
      """Creates a `truncated_normal_initializer` with the given range."""
      # The default value is chosen from language/bert/modeling.py.
      return tf.truncated_normal_initializer(stddev=initializer_range)

    self._embeddings = tf.get_variable('embeddings_matrix',
                                       [self._vocab_size + num_oov_buckets,
                                        embedding_dim],
                                       trainable=fine_tune_embeddings,
                                       initializer=create_initializer())

  def _build(self, tokens):
    padded_embeddings = get_padded_embeddings(
        self._embeddings, self._vocabulary_table, tokens, self._batch_size)
    return padded_embeddings

  @property
  def vocab_table(self):
    return self._vocabulary_table

  @property
  def vocab_size(self):
    return self._vocab_size + self._num_oov_buckets


def get_accuracy(logits, labels):
  """Top 1 accuracy from logits and labels."""
  return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32))


def get_num_correct_predictions(logits, labels):
  """Get the number of correct predictions over a batch."""
  predictions = tf.cast(tf.argmax(logits, axis=1), tf.int64)
  evals = tf.equal(predictions, labels)
  num_correct = tf.reduce_sum(tf.cast(evals, tf.float64))
  return num_correct


def get_merged_vocabulary_file(vocabularies, padded_token=None):
  """Merges several vocabulary files into one temporary file.

  The TF object that loads the embedding expects a vocabulary file, to know
  which embeddings it should load.
  See tf.contrib.embedding.load_embedding_initializer.

  When we want to train/test on several datasets simultaneously we need to merge
  their vocabulary files into a single file.

  Args:
    vocabularies: Iterable of vocabularies. Each vocabulary should be
        a list of tokens.
    padded_token: If not None, add the padded_token to the first index.
  Returns:
    outfilename: Name of the merged file. Contains the union of all tokens in
        filenames, without duplicates, one token per line.
    vocabulary_size: Count of tokens in the merged file.
  """
  uniques = [set(vocabulary) for vocabulary in vocabularies]
  unique_merged = frozenset().union(*uniques)
  unique_merged_sorted = sorted(unique_merged)
  if padded_token is not None:
    # Add padded token as 0 index.
    unique_merged_sorted = [padded_token] + unique_merged_sorted
  vocabulary_size = len(unique_merged_sorted)
  outfile = tempfile.NamedTemporaryFile(delete=False)
  outfile.write(b'\n'.join(unique_merged_sorted))
  outfilename = outfile.name
  logging.info('Merged vocabulary file with %d tokens: %s', vocabulary_size,
               outfilename)
  outfile.close()
  return outfilename, vocabulary_size
