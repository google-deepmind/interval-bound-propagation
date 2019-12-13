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
  """Returns the default configuration as an instance of ConfigDict."""

  config = {}

  config['dataset'] = 'sst'
  # convolutional architecture
  config['conv_architecture'] = ((100, 5, 1), 'relu')

  # full connected layer 1
  config['conv_fc1'] = 0

  # (optional) fully connected layer 2
  config['conv_fc2'] = 0

  # Number of _available_ perturbations.
  # (delta specifies the _budget_, i.e. how many may be used at once.)
  config['delta'] = 3.0  # hotflip attack attack strength.

  # Allow each char to be changed to other 26 characters.
  # config['synonym_filepath'] = \
  # './data/character_substitution_enkey.json'
  # config['max_padded_length'] = 268
  # # (~26*268) Max num_perturbations.
  # # seqlen * max_number_synonyms (total number of elementary perturbations)
  # config['num_perturbations'] = 7300

  # Allow each char to be changed to another character.
  config['synonym_filepath'] = './data/character_substitution_enkey_sub1.json'
  config['max_padded_length'] = 268
  # (~1*268) Max num_perturbations.
  # seqlen * max_number_synonyms (total number of elementary perturbations)
  config['num_perturbations'] = 268

  config['vocab_filename'] = './data/sst_binary_character_vocabulary_sorted.txt'
  # Need to add pad for analysis (which is what is used after
  # utils.get_merged_vocabulary_file)
  config['vocab_filename_pad'] = (
      './data/sst_binary_character_vocabulary_sorted_pad.txt')

  config['embedding_dim'] = 150

  config['delta_schedule'] = True
  config['dual_loss_schedule'] = True

  # Ratio between the task loss and autoencoder loss.
  config['dual_loss_ratio'] = 0.75

  # verification
  config['verifiable_training_aggregation'] = 'softmax'

  # config.scenario_size is total number of batches to run
  # If scenario_size==0, evaluate the whole data split.
  # config.scenario_size * config.scenario_offset * batch_size
  # is roughly the test set size.
  config['scenario_size'] = 0
  config['scenario_offset'] = 0

  config['data_id'] = 1

  config['model_location'] = ''

  return config
