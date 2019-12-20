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

"""Train verifiable robust models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import interval_bound_propagation as ibp
import numpy as np
import six
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from tensorflow.contrib import lookup as contrib_lookup
import models
import utils


EmbeddedDataset = collections.namedtuple(
    'EmbeddedDataset',
    ['embedded_inputs', 'length', 'input_tokens', 'sentiment'])

Dataset = collections.namedtuple(
    'Dataset',
    ['tokens', 'num_tokens', 'sentiment'])

Perturbation = collections.namedtuple(
    'Perturbation',
    ['positions', 'tokens'])


def _pad_fixed(x, axis, padded_length):
  """Pads a tensor to a fixed size (rather than batch-specific)."""
  pad_shape = x.shape.as_list()
  pad_shape[axis] = tf.maximum(padded_length - tf.shape(x)[axis], 0)
  # Pad zero as in utils.get_padded_indexes.
  padded = tf.concat([x, tf.zeros(dtype=x.dtype, shape=pad_shape)], axis=axis)
  assert axis == 1
  padded = padded[:, :padded_length]

  padded_shape = padded.shape.as_list()
  padded_shape[axis] = padded_length
  padded.set_shape(padded_shape)
  return padded


class GeneratedDataset(snt.AbstractModule):
  """A dataset wrapper for data_gen such that it behaves like sst_binary."""

  def __init__(self, data_gen, batch_size, mode='train',
               num_examples=0,
               dataset_name='glue/sst2',
               name='generated_dataset'):
    super(GeneratedDataset, self).__init__(name=name)
    self._data_gen = data_gen
    self._batch_size = batch_size
    self._mode = mode
    self._shuffle = True if mode == 'train' else False
    self._num_examples = num_examples
    self._dataset_name = dataset_name

  def get_row_lengths(self, sparse_tensor_input):
    # sparse_tensor_input is a tf.SparseTensor
    # In RaggedTensor, row_lengths is a vector with shape `[nrows]`,
    # which specifies the length of each row.
    rt = tf.RaggedTensor.from_sparse(sparse_tensor_input)
    return rt.row_lengths()

  def _build(self):
    dataset = tfds.load(name=self._dataset_name, split=self._mode)
    minibatch = dataset.map(parse).repeat()

    if self._shuffle:
      minibatch = minibatch.shuffle(self._batch_size*100)
    minibatch = minibatch.batch(
        self._batch_size).make_one_shot_iterator().get_next()
    minibatch['sentiment'].set_shape([self._batch_size])
    minibatch['sentence'] = tf.SparseTensor(
        indices=minibatch['sentence'].indices,
        values=minibatch['sentence'].values,
        dense_shape=[self._batch_size, minibatch['sentence'].dense_shape[1]])
    # minibatch.sentence sparse tensor with dense shape
    # [batch_size x seq_length], length: [batch_size]
    return Dataset(
        tokens=minibatch['sentence'],
        num_tokens=self.get_row_lengths(minibatch['sentence']),
        sentiment=minibatch['sentiment'],
    )

  @property
  def num_examples(self):
    return self._num_examples


def parse(data_dict):
  """Parse dataset from _data_gen into the same format as sst_binary."""
  sentiment = data_dict['label']
  sentence = data_dict['sentence']
  dense_chars = tf.decode_raw(sentence, tf.uint8)
  dense_chars.set_shape((None,))
  chars = tfp.math.dense_to_sparse(dense_chars)
  if six.PY3:
    safe_chr = lambda c: '?' if c >= 128 else chr(c)
  else:
    safe_chr = chr
  to_char = np.vectorize(safe_chr)
  chars = tf.SparseTensor(indices=chars.indices,
                          values=tf.py_func(to_char, [chars.values], tf.string),
                          dense_shape=chars.dense_shape)
  return {'sentiment': sentiment,
          'sentence': chars}


class RobustModel(snt.AbstractModule):
  """Model for applying sentence representations for different tasks."""

  def __init__(self,
               task,
               batch_size,
               pooling,
               learning_rate,
               config,
               embedding_dim,
               fine_tune_embeddings=False,
               num_oov_buckets=1000,
               max_grad_norm=5.0,
               name='robust_model'):
    super(RobustModel, self).__init__(name=name)
    self.config = config
    self.task = task
    self.batch_size = batch_size
    self.pooling = pooling
    self.learning_rate = learning_rate
    self.embedding_dim = embedding_dim
    self.fine_tune_embeddings = fine_tune_embeddings
    self.num_oov_buckets = num_oov_buckets
    self.max_grad_norm = max_grad_norm
    self.linear_classifier = None

  def add_representer(self, vocab_filename, padded_token=None):
    """Add sentence representer to the computation graph.

    Args:
      vocab_filename: the name of vocabulary files.
      padded_token: padded_token to the vocabulary.
    """

    self.embed_pad = utils.EmbedAndPad(
        self.batch_size,
        [self._lines_from_file(vocab_filename)],
        embedding_dim=self.embedding_dim,
        num_oov_buckets=self.num_oov_buckets,
        fine_tune_embeddings=self.fine_tune_embeddings,
        padded_token=padded_token)

    self.keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_prob')

    # Model to get a sentence representation from embeddings.
    self.sentence_representer = models.SentenceRepresenterConv(
        self.config, keep_prob=self.keep_prob, pooling=self.pooling)

  def add_dataset(self):
    """Add datasets.

    Returns:
      train_data, dev_data, test_data, num_classes
    """
    if self.config.get('dataset', '') == 'sst':
      train_data = GeneratedDataset(None, self.batch_size, mode='train',
                                    num_examples=67349)
      dev_data = GeneratedDataset(None, self.batch_size, mode='validation',
                                  num_examples=872)
      test_data = GeneratedDataset(None, self.batch_size, mode='validation',
                                   num_examples=872)
      num_classes = 2
      return train_data, dev_data, test_data, num_classes
    else:
      raise ValueError('Not supported dataset')

  def get_representation(self, tokens, num_tokens):
    if tokens.dtype == tf.float32:
      return self.sentence_representer(tokens, num_tokens)
    else:  # dtype == tf.string
      return self.sentence_representer(self.embed_pad(tokens), num_tokens)

  def add_representation(self, minibatch):
    """Compute sentence representations.

    Args:
      minibatch: a minibatch of sequences of embeddings.
    Returns:
      joint_rep: representation of sentences or concatenation of
        sentence vectors.
    """

    joint_rep = self.get_representation(minibatch.tokens, minibatch.num_tokens)
    result = {'representation1': joint_rep}
    return joint_rep, result

  def add_train_ops(self,
                    num_classes,
                    joint_rep,
                    minibatch):
    """Add ops for training in the computation graph.

    Args:
      num_classes: number of classes to predict in the task.
      joint_rep: the joint sentence representation if the input is sentence
        pairs or the representation for the sentence if the input is a single
        sentence.
      minibatch: a minibatch of sequences of embeddings.
    Returns:
      train_accuracy: the accuracy on the training dataset
      loss: training loss.
      opt_step: training op.
    """
    if self.linear_classifier is None:
      classifier_layers = []
      classifier_layers.append(snt.Linear(num_classes))
      self.linear_classifier = snt.Sequential(classifier_layers)
    logits = self.linear_classifier(joint_rep)
    # Losses and optimizer.
    def get_loss(logits, labels):
      return tf.reduce_mean(
          tf.nn.sparse_softmax_cross_entropy_with_logits(
              labels=labels, logits=logits))

    loss = get_loss(logits, minibatch.sentiment)
    train_accuracy = utils.get_accuracy(logits, minibatch.sentiment)
    opt_step = self._add_optimize_op(loss)
    return train_accuracy, loss, opt_step

  def create_perturbation_ops(self, minibatch, synonym_values, vocab_table):
    """Perturb data_batch using synonym_values."""
    data_batch = _pad_fixed(
        utils.get_padded_indexes(vocab_table, minibatch.tokens,
                                 self.batch_size), axis=1,
        padded_length=self.config['max_padded_length'])

    # synonym_values: [vocab_size x max_num_synonyms]
    # data_batch: [batch_size x seq_length]
    # [batch_size x seq_length x max_num_synonyms] - synonyms for each token.
    # Defaults to same word in case of no other synonyms.
    synonym_ids = tf.gather(synonym_values, data_batch, axis=0)

    # Split along batchsize. Elements shape: [seq_length x max_num_synonyms].
    synonym_ids_per_example = tf.unstack(synonym_ids, axis=0)

    # Loop across batch.
    # synonym_ids_this_example shape: [seq_length x max_num_synonyms]
    sequence_positions_across_batch, values_across_batch = [], []
    for i_sample, synonym_ids_this_example in enumerate(
        synonym_ids_per_example):
      # [num_nonzero, 2]. The rows are pairs of (t,s), where t is an index for
      # a time step, and s is an index into the max_num_synonyms dimension.
      nonzero_indices = tf.where(synonym_ids_this_example)

      # shape [num_nonzero]. Corresponding to the entries at nonzero_indices
      synonym_tokens = tf.gather_nd(params=synonym_ids_this_example,
                                    indices=nonzero_indices)

      # [num_nonzero] - Of the (t,s) pairs in nonzero_indices, pick only the
      # time dimension (t), corresponding to perturbation positions in the
      # sequence.
      perturbation_positions_this_example = nonzero_indices[:, 0]

      # The main logic is done. Now follows padding to a fixed length of
      # num_perturbations. However, this cannot be done with 0-padding, as it
      # would introduce a new (zero) vertex. Instead, we duplicate existing
      # tokens as perturbations (which have no effect), until we have reached a
      # total of num_perturbations perturbations. In this case, the padded
      # tokens are the original tokens from the data_batch. The padded positions
      # are all the positions (using range) corresponding to the padded tokens.

      # How often seq-length fits into maximum num perturbations
      padding_multiplier = tf.floordiv(self.config['num_perturbations'],
                                       tf.cast(minibatch.num_tokens[i_sample],
                                               tf.int32)) + 1

      # original tokens  # [seq_length]
      original_tokens = data_batch[i_sample, :minibatch.num_tokens[i_sample]]
      # [padding_multiplier * seq_length]. Repeat several times, use as padding.
      padding_tokens = tf.tile(original_tokens, multiples=[padding_multiplier])
      synonym_tokens_padded = tf.concat([synonym_tokens, tf.cast(padding_tokens,
                                                                 dtype=tf.int64)
                                        ], axis=0)
      # Crop at exact num_perturbations size.
      synonym_tokens_padded = synonym_tokens_padded[
          :self.config['num_perturbations']]

      # [seq_length] padding sequence positions with tiles of range()
      pad_positions = tf.range(minibatch.num_tokens[i_sample], delta=1)
      # [padding_multiplier*seq_length]
      padding_positions = tf.tile(pad_positions, multiples=[padding_multiplier])
      perturbation_positions_this_example_padded = tf.concat(
          [perturbation_positions_this_example, tf.cast(padding_positions,
                                                        dtype=tf.int64)],
          axis=0)
      # Crop at exact size num_perturbations.
      sequence_positions_padded = perturbation_positions_this_example_padded[
          :self.config['num_perturbations']]

      # Collect across the batch for tf.stack later.
      sequence_positions_across_batch.append(sequence_positions_padded)
      values_across_batch.append(synonym_tokens_padded)

    # Both [batch_size x max_n_perturbations]
    perturbation_positions = tf.stack(sequence_positions_across_batch, axis=0)
    perturbation_tokens = tf.stack(values_across_batch, axis=0)

    # Explicitly setting the shape to self.config['num_perturbations']
    perturbation_positions_shape = perturbation_positions.shape.as_list()
    perturbation_positions_shape[1] = self.config['num_perturbations']
    perturbation_positions.set_shape(perturbation_positions_shape)
    perturbation_tokens_shape = perturbation_tokens.shape.as_list()
    perturbation_tokens_shape[1] = self.config['num_perturbations']
    perturbation_tokens.set_shape(perturbation_tokens_shape)

    return Perturbation(
        positions=perturbation_positions,
        tokens=perturbation_tokens)

  def _add_optimize_op(self, loss):
    """Add ops for training."""
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.Variable(self.learning_rate, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                                      self.max_grad_norm)
    opt = tf.train.AdamOptimizer(learning_rate)
    opt_step = opt.apply_gradients(zip(grads, tvars),
                                   global_step=global_step)
    return opt_step

  def embed_dataset(self, minibatch, vocab_table):
    return EmbeddedDataset(
        embedded_inputs=_pad_fixed(
            self.embed_pad(minibatch.tokens),
            axis=1,
            padded_length=self.config['max_padded_length']),
        input_tokens=_pad_fixed(
            utils.get_padded_indexes(vocab_table, minibatch.tokens,
                                     self.batch_size),
            axis=1,
            padded_length=self.config['max_padded_length']),
        length=tf.minimum(self.config['max_padded_length'],
                          tf.cast(minibatch.num_tokens, tf.int32)),
        sentiment=minibatch.sentiment)

  def compute_mask_vertices(self, data_batch, perturbation):
    """Compute perturbation masks and perbuted vertices.

    Args:
      data_batch: EmbeddedDataset object.
      perturbation: Perturbation object.

    Returns:
      masks: Positions where there are perturbations.
      vertices: The resulting embeddings of the perturbed inputs.
    """
    # The following are all shaped (after broadcasting) as:
    # (batch_size, num_perturbations, seq_length, embedding_size).
    embedding = self.embed_pad._embeddings  # pylint: disable=protected-access
    # (batch_size, 1, seq_length, emb_dim)
    original_vertices = tf.expand_dims(data_batch.embedded_inputs, axis=1)
    # (batch_size, num_perturbation, 1, emb_dim])
    perturbation_vertices = tf.gather(
        embedding, tf.expand_dims(perturbation.tokens, axis=2))
    # (batch_size, num_perturbations, seq_length, 1)
    mask = tf.expand_dims(
        tf.one_hot(perturbation.positions,
                   depth=self.config['max_padded_length']), axis=3)
    # (batch_size, num_perturbations, seq_length, embedding_size)
    vertices = (1 - mask) * original_vertices + mask * perturbation_vertices
    return mask, vertices

  def preprocess_databatch(self, minibatch, vocab_table, perturbation):
    data_batch = self.embed_dataset(minibatch, vocab_table)
    mask, vertices = self.compute_mask_vertices(data_batch, perturbation)
    return data_batch, mask, vertices

  def add_verifiable_objective(self,
                               minibatch,
                               vocab_table,
                               perturbation,
                               stop_gradient=False):
    # pylint: disable=g-missing-docstring
    data_batch = self.embed_dataset(minibatch, vocab_table)
    _, vertices = self.compute_mask_vertices(data_batch, perturbation)

    def classifier(embedded_inputs):
      representation = self.sentence_representer(embedded_inputs,
                                                 data_batch.length)
      return self.linear_classifier(representation)

    # Verification graph.
    network = ibp.VerifiableModelWrapper(classifier)
    network(data_batch.embedded_inputs)

    input_bounds = ibp.SimplexBounds(
        vertices=vertices,
        nominal=data_batch.embedded_inputs,
        r=(self.delta if not stop_gradient else self.config['delta']))
    network.propagate_bounds(input_bounds)

    # Calculate the verifiable objective.
    verifiable_obj = verifiable_objective(
        network, data_batch.sentiment, margin=1.)

    return verifiable_obj

  def run_classification(self, inputs, labels, length):
    prediction = self.run_prediction(inputs, length)
    correct = tf.cast(tf.equal(labels, tf.argmax(prediction, 1)),
                      dtype=tf.float32)
    return correct

  def compute_verifiable_loss(self, verifiable_obj, labels):
    """Compute verifiable training objective.

    Args:
      verifiable_obj: Verifiable training objective.
      labels: Ground truth labels.
    Returns:
      verifiable_loss: Aggregrated loss of the verifiable training objective.
    """
    # Three options: reduce max, reduce mean, and softmax.
    if self.config['verifiable_training_aggregation'] == 'mean':
      verifiable_loss = tf.reduce_mean(
          verifiable_obj)  # average across all target labels
    elif self.config['verifiable_training_aggregation'] == 'max':
      # Worst target label only.
      verifiable_loss = tf.reduce_mean(tf.reduce_max(verifiable_obj, axis=0))
    elif self.config['verifiable_training_aggregation'] == 'softmax':
      # This assumes that entries in verifiable_obj belonging to the true class
      # are set to a (large) negative value, so to not affect the softmax much.

      # [batch_size]. Compute x-entropy against one-hot distrib. for true label.
      verifiable_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=tf.transpose(verifiable_obj), labels=labels)

      verifiable_loss = tf.reduce_mean(
          verifiable_loss)  # aggregation across batch
    else:
      logging.info(self.config['verifiable_training_aggregation'])
      raise ValueError(
          'Bad input argument for verifiable_training_aggregation used.')

    return verifiable_loss

  def compute_verifiable_verified(self, verifiable_obj):
    # Overall upper bound is maximum over all incorrect target classes.
    bound = tf.reduce_max(verifiable_obj, axis=0)
    verified = tf.cast(bound <= 0, dtype=tf.float32)
    return bound, verified

  def run_prediction(self, inputs, length):
    representation = self.sentence_representer(inputs, length)
    prediction = self.linear_classifier(representation)
    return prediction

  def sentiment_accuracy_op(self, minibatch):
    """Compute accuracy of dev/test set on the task of sentiment analysis.

    Args:
      minibatch: a batch of sequences of embeddings.
    Returns:
      num_correct: the number of examples that are predicted correctly on the
        given dataset.
    """

    rep = self.get_representation(minibatch.tokens, minibatch.num_tokens)
    logits = self.linear_classifier(rep)
    num_correct = utils.get_num_correct_predictions(logits,
                                                    minibatch.sentiment)
    return num_correct

  def add_dev_eval_ops(self, minibatch):
    """Add ops for evaluating on the dev/test set.

    Args:
      minibatch: a batch of sequence of embeddings.
    Returns:
      num_correct: the number of examples that are predicted correctly.
    """
    num_correct = self.sentiment_accuracy_op(minibatch)
    return num_correct

  def _build(self):
    """Build the computation graph.

    Returns:
      graph_tensors: list of ops that are to be executed during
        training/evaluation.
    """
    train_data, dev_data, test_data, num_classes = self.add_dataset()
    train_minibatch = train_data()
    dev_minibatch = dev_data()
    test_minibatch = test_data()

    # Load the vocab without padded_token and add it to the add_representer
    # later. Otherwise, it will be sorted.
    vocab_filename = self.config['vocab_filename']
    self.add_representer(vocab_filename, padded_token=b'<PAD>')

    graph_tensors = self._build_graph_with_datasets(
        train_minibatch, dev_minibatch, test_minibatch, num_classes)
    graph_tensors['dev_num_examples'] = dev_data.num_examples
    graph_tensors['test_num_examples'] = test_data.num_examples
    return graph_tensors

  def _build_graph_with_datasets(self,
                                 train_minibatch,
                                 dev_minibatch,
                                 test_minibatch,
                                 num_classes):
    """Returns the training/evaluation ops."""
    self.keep_prob = 1.  # Using literal 1 (not placeholder) skips dropout op.
    self.sentence_representer._keep_prob = 1.  # pylint:disable=protected-access

    # Build the graph as per the base class.
    (train_joint_rep, _) = self.add_representation(train_minibatch)

    (train_accuracy,
     loss,
     opt_step) = self.add_train_ops(num_classes, train_joint_rep,
                                    train_minibatch)

    dev_num_correct = self.add_dev_eval_ops(dev_minibatch)
    test_num_correct = self.add_dev_eval_ops(test_minibatch)
    graph_tensors = {
        'loss': loss,
        'train_op': opt_step,
        'train_accuracy': train_accuracy,
        'dev_num_correct': dev_num_correct,
        'test_num_correct': test_num_correct,
        'keep_prob': self.keep_prob
    }

    vocab_table = self.embed_pad.vocab_table
    vocab_size = self.embed_pad.vocab_size

    verifiable_loss_ratio = tf.constant(
        self.config['verifiable_loss_ratio'],
        dtype=tf.float32,
        name='verifiable_loss_ratio')
    self.delta = tf.constant(self.config['delta'],
                             dtype=tf.float32, name='delta')

    lookup_token = tf.placeholder(tf.string, shape=None, name='lookup_token')
    indices = vocab_table.lookup(lookup_token)

    self.vocab_list = contrib_lookup.index_to_string_table_from_file(
        self.config['vocab_filename_pad'])
    lookup_token_index = tf.placeholder(tf.int64, shape=None,
                                        name='lookup_token_index')
    lookup_token_string = self.vocab_list.lookup(lookup_token_index)

    synonym_values = tf.placeholder(tf.int64, shape=[None, None],
                                    name='synonym_values')
    synonym_counts = tf.placeholder(tf.int64, shape=[None],
                                    name='synonym_counts')

    train_perturbation = self.create_perturbation_ops(
        train_minibatch, synonym_values, vocab_table)

    train_data_batch, _, _ = self.preprocess_databatch(
        train_minibatch, vocab_table, train_perturbation)

    train_words = self.vocab_list.lookup(train_data_batch.input_tokens)

    # [num_targets x batchsize]
    verifiable_obj = self.add_verifiable_objective(
        train_minibatch, vocab_table, train_perturbation, stop_gradient=False)

    train_nominal = self.run_classification(train_data_batch.embedded_inputs,
                                            train_data_batch.sentiment,
                                            train_data_batch.length)
    train_bound, train_verified = self.compute_verifiable_verified(
        verifiable_obj)
    verifiable_loss = self.compute_verifiable_loss(verifiable_obj,
                                                   train_minibatch.sentiment)

    if (self.config['verifiable_loss_ratio']) > 1.0:
      raise ValueError('Loss ratios sum up to more than 1.0')

    total_loss = (1 - verifiable_loss_ratio) * graph_tensors['loss']
    if self.config['verifiable_loss_ratio'] != 0:
      total_loss += verifiable_loss_ratio * verifiable_loss

    # Attack on dev/test set.
    dev_perturbation = self.create_perturbation_ops(
        dev_minibatch, synonym_values, vocab_table)
    # [num_targets x batchsize]
    dev_verifiable_obj = self.add_verifiable_objective(
        dev_minibatch, vocab_table, dev_perturbation, stop_gradient=True)
    dev_bound, dev_verified = self.compute_verifiable_verified(
        dev_verifiable_obj)

    dev_data_batch, _, _ = self.preprocess_databatch(
        dev_minibatch, vocab_table, dev_perturbation)

    test_perturbation = self.create_perturbation_ops(
        test_minibatch, synonym_values, vocab_table)
    # [num_targets x batchsize]
    test_verifiable_obj = self.add_verifiable_objective(
        test_minibatch, vocab_table, test_perturbation, stop_gradient=True)
    test_bound, test_verified = self.compute_verifiable_verified(
        test_verifiable_obj)

    test_data_batch, _, _ = self.preprocess_databatch(
        test_minibatch, vocab_table, test_perturbation)

    dev_words = self.vocab_list.lookup(dev_data_batch.input_tokens)
    test_words = self.vocab_list.lookup(test_data_batch.input_tokens)
    dev_nominal = self.run_classification(dev_data_batch.embedded_inputs,
                                          dev_data_batch.sentiment,
                                          dev_data_batch.length)
    test_nominal = self.run_classification(test_data_batch.embedded_inputs,
                                           test_data_batch.sentiment,
                                           test_data_batch.length)

    dev_predictions = self.run_prediction(dev_data_batch.embedded_inputs,
                                          dev_data_batch.length)
    test_predictions = self.run_prediction(test_data_batch.embedded_inputs,
                                           test_data_batch.length)

    with tf.control_dependencies([train_verified, test_verified, dev_verified]):
      opt_step = self._add_optimize_op(total_loss)

    graph_tensors['total_loss'] = total_loss
    graph_tensors['verifiable_loss'] = verifiable_loss
    graph_tensors['train_op'] = opt_step
    graph_tensors['indices'] = indices
    graph_tensors['lookup_token_index'] = lookup_token_index
    graph_tensors['lookup_token_string'] = lookup_token_string
    graph_tensors['lookup_token'] = lookup_token
    graph_tensors['vocab_size'] = vocab_size
    graph_tensors['synonym_values'] = synonym_values
    graph_tensors['synonym_counts'] = synonym_counts
    graph_tensors['verifiable_loss_ratio'] = verifiable_loss_ratio
    graph_tensors['delta'] = self.delta

    graph_tensors['train'] = {
        'bound': train_bound,
        'verified': train_verified,
        'words': train_words,
        'sentiment': train_minibatch.sentiment,
        'correct': train_nominal,
    }
    graph_tensors['dev'] = {
        'predictions': dev_predictions,
        'data_batch': dev_data_batch,
        'tokens': dev_minibatch.tokens,
        'num_tokens': dev_minibatch.num_tokens,
        'minibatch': dev_minibatch,
        'bound': dev_bound,
        'verified': dev_verified,
        'words': dev_words,
        'sentiment': dev_minibatch.sentiment,
        'correct': dev_nominal,
    }
    graph_tensors['test'] = {
        'predictions': test_predictions,
        'data_batch': test_data_batch,
        'tokens': test_minibatch.tokens,
        'num_tokens': test_minibatch.num_tokens,
        'minibatch': test_minibatch,
        'bound': test_bound,
        'verified': test_verified,
        'words': test_words,
        'sentiment': test_minibatch.sentiment,
        'correct': test_nominal,
    }

    return graph_tensors

  def _lines_from_file(self, filename):
    with open(filename, 'rb') as f:
      return f.read().splitlines()


def verifiable_objective(network, labels, margin=0.):
  """Computes the verifiable objective.

  Args:
    network: `ibp.VerifiableModelWrapper` for the network to verify.
    labels: 1D integer tensor of shape (batch_size) of labels for each
      input example.
    margin: Verifiable objective values for correct class will be forced to
      `-margin`, thus disregarding large negative bounds when maximising. By
      default this is set to 0.

  Returns:
    2D tensor of shape (num_classes, batch_size) containing verifiable objective
      for each target class, for each example.
  """
  last_layer = network.output_module

  # Objective, elided with final linear layer.
  obj_w, obj_b = targeted_objective(
      last_layer.module.w, last_layer.module.b, labels)

  # Relative bounds on the objective.
  per_neuron_objective = tf.maximum(
      obj_w * last_layer.input_bounds.lower_offset,
      obj_w * last_layer.input_bounds.upper_offset)
  verifiable_obj = tf.reduce_sum(
      per_neuron_objective,
      axis=list(range(2, per_neuron_objective.shape.ndims)))

  # Constant term (objective layer bias).
  verifiable_obj += tf.reduce_sum(
      obj_w * last_layer.input_bounds.nominal,
      axis=list(range(2, obj_w.shape.ndims)))
  verifiable_obj += obj_b

  # Filter out cases in which the target class is the correct class.
  # Using `margin` makes the irrelevant cases of target=correct return
  # a large negative value, which will be ignored by the reduce_max.
  num_classes = last_layer.output_bounds.shape[-1]
  verifiable_obj = filter_correct_class(
      verifiable_obj, num_classes, labels, margin=margin)

  return verifiable_obj


def targeted_objective(final_w, final_b, labels):
  """Determines final layer weights for attacks targeting each class.

  Args:
    final_w: 2D tensor of shape (last_hidden_layer_size, num_classes)
      containing the weights for the final linear layer.
    final_b: 1D tensor of shape (num_classes) containing the biases for the
      final hidden layer.
    labels: 1D integer tensor of shape (batch_size) of labels for each
      input example.

  Returns:
    obj_w: Tensor of shape (num_classes, batch_size, last_hidden_layer_size)
      containing weights (to use in place of final linear layer weights)
      for targeted attacks.
    obj_b: Tensor of shape (num_classes, batch_size) containing bias
      (to use in place of final linear layer biases) for targeted attacks.
  """
  # Elide objective with final linear layer.
  final_wt = tf.transpose(final_w)
  obj_w = tf.expand_dims(final_wt, axis=1) - tf.gather(final_wt, labels, axis=0)
  obj_b = tf.expand_dims(final_b, axis=1) - tf.gather(final_b, labels, axis=0)
  return obj_w, obj_b


def filter_correct_class(verifiable_obj, num_classes, labels, margin):
  """Filters out the objective when the target class contains the true label.

  Args:
    verifiable_obj: 2D tensor of shape (num_classes, batch_size) containing
      verifiable objectives.
    num_classes: number of target classes.
    labels: 1D tensor of shape (batch_size) containing the labels for each
      example in the batch.
    margin: Verifiable objective values for correct class will be forced to
      `-margin`, thus disregarding large negative bounds when maximising.

  Returns:
   2D tensor of shape (num_classes, batch_size) containing the corrected
   verifiable objective values for each (class, example).
  """
  targets_to_filter = tf.expand_dims(
      tf.range(num_classes, dtype=labels.dtype), axis=1)
  neq = tf.not_equal(targets_to_filter, labels)
  verifiable_obj = tf.where(neq, verifiable_obj, -margin *
                            tf.ones_like(verifiable_obj))
  return verifiable_obj
