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

"""Configuration parameters for sentence representation models."""


def get_config():
  """Returns the default configuration as a dict."""

  config = {}

  config['dataset'] = 'sst'
  # Convolutional architecture.
  # Format: Tuple/List for a Conv layer (filters, kernel_size, pooling_size)
  # Otherwise, nonlinearity.
  config['conv_architecture'] = ((100, 5, 1), 'relu')

  # Fully connected layer 1 hidden sizes (0 means no layer).
  config['conv_fc1'] = 0

  # Fully connected layer 2 hidden sizes (0 means no layer).
  config['conv_fc2'] = 0

  # Number of allowable perturbations.
  # (delta specifies the budget, i.e., how many may be used at once.)
  config['delta'] = 3.0

  # Allow each character to be changed to another character.
  config['synonym_filepath'] = 'data/character_substitution_enkey_sub1.json'
  config['max_padded_length'] = 268
  # (~1*268) Max num_perturbations.
  # seqlen * max_number_synonyms (total number of elementary perturbations)
  config['num_perturbations'] = 268

  config['vocab_filename'] = 'data/sst_binary_character_vocabulary_sorted.txt'
  # Need to add pad for analysis (which is what is used after
  # utils.get_merged_vocabulary_file).
  config['vocab_filename_pad'] = (
      'data/sst_binary_character_vocabulary_sorted_pad.txt')

  config['embedding_dim'] = 150

  config['delta_schedule'] = True
  config['verifiable_loss_schedule'] = True

  # Ratio between the task loss and verifiable loss.
  config['verifiable_loss_ratio'] = 0.75

  # Aggregrated loss of the verifiable training objective
  # (among softmax, mean, max).
  config['verifiable_training_aggregation'] = 'softmax'

  config['data_id'] = 1

  config['model_location'] = '/tmp/robust_model/checkpoint/final'

  return config
