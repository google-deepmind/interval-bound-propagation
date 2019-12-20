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

"""Minimum code to interact with a pretrained Stanford Sentiment Treebank model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf

import robust_model


SparseTensorValue = collections.namedtuple(
    'SparseTensorValue', ['indices', 'values', 'dense_shape'])


class InteractiveSentimentPredictor(object):
  """Can be used to interact with a trained sentiment analysis model."""

  def __init__(self, config_dict, model_location, max_padded_length=0,
               num_perturbations=0):
    self.graph_tensor_producer = robust_model.RobustModel(**config_dict)

    self.batch_size = self.graph_tensor_producer.batch_size
    if max_padded_length:
      self.graph_tensor_producer.config.max_padded_length = max_padded_length
    if num_perturbations:
      self.graph_tensor_producer.config.num_perturbations = num_perturbations
    self.graph_tensors = self.graph_tensor_producer()

    network_saver = tf.train.Saver(self.graph_tensor_producer.variables)
    self.open_session = tf.Session()
    self.open_session.run(tf.tables_initializer())
    network_saver.restore(self.open_session, model_location)

  def batch_predict_sentiment(self, list_of_sentences, is_tokenised=True):
    """Computes sentiment predictions for a batch of sentences.

    Note: the model batch size is usually hard-coded in the model (e.g. at 64).
    We require that len(list_of_sentences)==self.batch_size.
    If padding is necessary to reach as many sentences, this should happen
    outside of this function.

    Important: we assume that each sentence has the same number of tokens.
    Args:
      list_of_sentences: List[str] in case is_tokenised is False, or
        List[List[str]] in case is_tokenised is True. Holds inputs whose
        sentiment is to be classified.
      is_tokenised: bool. Whether sentences are already tokenised. If not,
        naive whitespace splitting tokenisation is applied.
    Returns:
      batch_label_predictions: np.array of shape [self.batch_size] holding
        integers, representing model predictions for each input.
    """

    # Prepare inputs.
    tokenised_sentence_list = []
    for sentence in list_of_sentences:
      if not is_tokenised:
        tokenised_sentence = sentence.lower().split(' ')
      else:
        tokenised_sentence = sentence
      tokenised_sentence_list.append(tokenised_sentence)
    length = len(tokenised_sentence_list[0])
    assert all([len(x) == length for x in tokenised_sentence_list])
    assert len(tokenised_sentence_list) == self.batch_size

    # Construct sparse tensor holding token information.
    indices = np.zeros([self.batch_size*length, 2])
    dense_shape = [self.batch_size, length]
    # Loop over words. All sentences have the same length.
    for j, _ in enumerate(tokenised_sentence_list[0]):
      for i in range(self.batch_size):  # Loop over samples.
        offset = i*length + j
        indices[offset, 0] = i
        indices[offset, 1] = j

    # Define sparse tensor values.
    tokenised_sentence_list = [word for sentence in tokenised_sentence_list  # pylint:disable=g-complex-comprehension
                               for word in sentence]
    values = np.array(tokenised_sentence_list)
    mb_tokens = SparseTensorValue(indices=indices, values=values,
                                  dense_shape=dense_shape)
    mb_num_tokens = np.array([length]*self.batch_size)

    # Fill feed_dict with input token information.
    feed_dict = {}
    feed_dict[self.graph_tensors['dev']['tokens']] = mb_tokens
    feed_dict[self.graph_tensors['dev']['num_tokens']] = mb_num_tokens

    # Generate model predictions [batch_size x n_labels].
    logits = self.open_session.run(self.graph_tensors['dev']['predictions'],
                                   feed_dict)
    batch_label_predictions = np.argmax(logits, axis=1)

    return batch_label_predictions, logits

  def predict_sentiment(self, sentence, tokenised=False):
    """Computes sentiment of a sentence."""
    # Create inputs to tensorflow graph.
    if tokenised:
      inputstring_tokenised = sentence
    else:
      assert isinstance(sentence, str)
      # Simple tokenisation.
      inputstring_tokenised = sentence.lower().split(' ')
    length = len(inputstring_tokenised)

    # Construct inputs to sparse tensor holding token information.
    indices = np.zeros([self.batch_size*length, 2])
    dense_shape = [self.batch_size, length]
    for j, _ in enumerate(inputstring_tokenised):
      for i in range(self.batch_size):
        offset = i*length + j
        indices[offset, 0] = i
        indices[offset, 1] = j
    values = inputstring_tokenised*self.batch_size
    mb_tokens = SparseTensorValue(indices=indices, values=np.array(values),
                                  dense_shape=dense_shape)
    mb_num_tokens = np.array([length]*self.batch_size)

    # Fill feeddict with input token information.
    feed_dict = {}
    feed_dict[self.graph_tensors['dev']['tokens']] = mb_tokens
    feed_dict[self.graph_tensors['dev']['num_tokens']] = mb_num_tokens
    # Generate predictions.
    logits = self.open_session.run(self.graph_tensors['dev']['predictions'],
                                   feed_dict)
    predicted_label = np.argmax(logits, axis=1)
    final_prediction = predicted_label[0]
    # Check that prediction same everywhere (had batch of identical inputs).
    assert np.all(predicted_label == final_prediction)
    return final_prediction, logits
