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

"""Functionality for exhaustive adversarial attacks on synonym perturbations.

Models restored from checkpoint can be tested w.r.t their robustness to
exhaustive-search adversaries, which have a fixed perturbation budget with which
they can flip words to synonyms.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import imp
import json
import pprint

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tqdm

import interactive_example


flags.DEFINE_boolean('character_level', True, 'Character level model.')
flags.DEFINE_boolean('debug_mode', False, 'Debug mode.')
flags.DEFINE_string('checkpoint_path', '/tmp/robust_model/checkpoint/final',
                    'Checkpoint path.')
flags.DEFINE_string('dataset', 'sst', 'Dataset name. train, dev, or test.')
flags.DEFINE_string('mode', 'validation', 'Dataset part. train, dev, or test.')
flags.DEFINE_string('config_path', './config.py',
                    'Path to training configuration file.')
flags.DEFINE_string('task', 'sst', 'One of snli, mnli, sick, sst.')
flags.DEFINE_integer('batch_size', 30, 'Batch size.')
flags.DEFINE_string('pooling', 'average', 'One of averge, sum, max, last.')
flags.DEFINE_boolean('fine_tune_embeddings', True, 'Finetune embeddings.')
flags.DEFINE_integer('num_oov_buckets', 1, 'Number of out-of-vocab buckets.')

flags.DEFINE_integer('delta', 1, 'Maximum perturbation radius')
flags.DEFINE_integer('skip_batches', 0, 'Skip this number of batches'
                     ' for analysis.')
flags.DEFINE_integer('num_examples', 100, 'Analyze this number of examples. '
                     ' 0 suggest the whole dataset.')
flags.DEFINE_integer('truncated_len', 0, 'truncated sentence length. '
                     ' 0 suggest the whole sentence.')
flags.DEFINE_integer('max_padded_length', 0, 'max_padded_length. '
                     ' 0 suggest no change.')
flags.DEFINE_integer('num_perturbations', 0, 'num_perturbations. '
                     ' 0 suggest no change.')

FLAGS = flags.FLAGS


def load_synonyms(synonym_filepath=None):
  """Loads synonym dictionary. Returns as defaultdict(list)."""
  with tf.gfile.Open(synonym_filepath) as f:
    synonyms = json.load(f)
  synonyms_ = collections.defaultdict(list)
  synonyms_.update(synonyms)
  return synonyms_


def load_dataset(mode='validation', character_level=False):
  """Loads SST dataset.

  Takes data from disk/cns if it exists, otherwise out of tensorflow graph.
  Args:
    mode: string. Either train, dev, or test.
    character_level: bool. Whether to return character-level, or token level
      inputs.
  Returns:
    List of (input, output) pairs, where input is a list of strings (tokens),
      and output is an integer (categorical label in [0,1]).
  """
  message = 'Loading SST {}, character_level {}'.format(mode,
                                                        str(character_level))
  logging.info(message)
  dataset = tfds.load(name='glue/sst2', split=mode)
  minibatch = dataset.batch(1).make_one_shot_iterator().get_next()
  label_list, input_list = [], []
  with tf.train.SingularMonitoredSession() as session:
    while True:
      output_nodes = (minibatch['label'], minibatch['sentence'])
      label, sentence = session.run(output_nodes)
      label_list.append(label[0])
      input_list.append([chr(i) for i in sentence[0]])

  # zip together.
  dataset = [(in_, out_) for (in_, out_) in zip(input_list, label_list)]
  return dataset


def expand_by_one_perturbation(original_tokenized_sentence,
                               tokenized_sentence, synonym_dict):
  """Expands given sentence by all possible synonyms.

  Note that only a single synonym replacement is applied, and it is applied
  everywhere, i.e. for every mention of the word with the synonym.
  Args:
    original_tokenized_sentence: List[str]. List of tokens.
    tokenized_sentence: List[str]. List of tokens.
    synonym_dict: dict, mapping words (str) to lists of synonyms (list of str)
  Returns:
    new_sentences_list: List[List[str]]. Outer list is across different synonym
      replacements. Inner list is over (str) tokens.
  """
  new_sentences_list = []
  for i_outer, (original_token, _) in enumerate(zip(
      original_tokenized_sentence, tokenized_sentence)):
    synonyms = synonym_dict[original_token]
    for synonym in synonyms:  # replace only one particular mention
      new_sentence = copy.copy(tokenized_sentence)
      new_sentence[i_outer] = synonym
      new_sentences_list.append(new_sentence)

  return new_sentences_list


def find_up_to_depth_k_perturbations(
    original_tokenized_sentence, tokenized_sentence, synonym_dict, k):
  """Takes sentence, finds all sentences reachable using k token perturbations.

  Args:
    original_tokenized_sentence: List[str]. List of tokens.
    tokenized_sentence: List[str]. List of tokens.
    synonym_dict: dict, mapping words (str) to lists of synonyms (list of str)
    k: int. perturbation depth parameter.
  Returns:
    output_sentences: List[List[str]]. List of tokenised sentences.
  """
  # Case: recursion ends - no further perturbations.
  if k == 0:
    return [tokenized_sentence]
  else:
    # Expand by one level.
    expanded_sentences = expand_by_one_perturbation(original_tokenized_sentence,
                                                    tokenized_sentence,
                                                    synonym_dict)

    # Call recursive function one level deeper for each expanded sentence.
    expanded_sentences_deeper = []
    for sentence in expanded_sentences:
      new_sentences = find_up_to_depth_k_perturbations(
          original_tokenized_sentence, sentence, synonym_dict, k-1)
      expanded_sentences_deeper.extend(new_sentences)

  output_sentences = expanded_sentences + expanded_sentences_deeper
  output_sentences = remove_duplicates(output_sentences)
  return output_sentences


def remove_duplicates(list_of_list_of_tokens):
  # Convert list of str to str.
  sentences = ['|'.join(s) for s in list_of_list_of_tokens]
  sentences = set(sentences)  # Now hashable -> remove duplicates.
  sentences = [s.split('|') for s in sentences]  # Convert to original format.
  return sentences


def verify_exhaustively(sample, synonym_dict, sst_model, delta,
                        truncated_len=0):
  """Returns True if a sample can be verified, False otherwise.

  Args:
    sample: a 2-tuple (x,y), where x is a tokenised sentence (List[str]), and y
      is a label (int).
    synonym_dict: str -> List[str]. Keys are words, values are word lists with
      synonyms for the key word.
    sst_model: InteractiveSentimentPredictor instance. Used to make predictions.
    delta: int. How many synonym perturbations to maximally allow.
    truncated_len: int. Truncate sentence to truncated_len. 0 for unchanged.
  Returns:
    verified: bool. Whether all possible perturbed version of input sentence x
      up to perturbation radius delta have the correct prediction.
  """
  (x, y) = sample
  counter_example = None
  counter_prediction = None
  # Create (potentially long) list of perturbed sentences from x.
  if truncated_len > 0:
    x = x[: truncated_len]
  # Add original sentence.
  altered_sentences = find_up_to_depth_k_perturbations(x, x, synonym_dict,
                                                       delta)
  altered_sentences = altered_sentences + [x]
  # Form batches of these altered sentences.
  batch = []
  num_forward_passes = len(altered_sentences)

  for sentence in altered_sentences:
    any_prediction_wrong = False
    batch.append(sentence)
    # When batch_size is reached, make predictions, break if any label flip
    if len(batch) == sst_model.batch_size:
      # np array of size [batch_size]
      predictions, _ = sst_model.batch_predict_sentiment(
          batch, is_tokenised=True)

      # Check any prediction that is different from the true label.
      any_prediction_wrong = np.any(predictions != y)
      if any_prediction_wrong:
        wrong_index = np.where(predictions != y)[0].tolist()[0]
        counter_example = ' '.join([str(c) for c in batch[wrong_index]])
        if FLAGS.debug_mode:
          logging.info('\nOriginal example: %s, prediction: %d',
                       ' '.join([str(c) for c in sentence]), y)
          logging.info('\ncounter example:  %s, prediction: %s',
                       counter_example, predictions[wrong_index].tolist())
        counter_prediction = predictions[wrong_index]
        # Break. No need to evaluate further.
        return False, counter_example, counter_prediction, num_forward_passes

      # Start filling up the next batch.
      batch = []

  if not batch:
    # No remainder, not previously broken the loop.
    return True, None, None, num_forward_passes
  else:
    # Remainder -- what didn't fit into a full batch of size batch_size.
    # We use the first altered_sentence to pad.
    batch += [altered_sentences[0]]*(sst_model.batch_size-len(batch))
    assert len(batch) == sst_model.batch_size
    predictions, _ = sst_model.batch_predict_sentiment(batch, is_tokenised=True)
    any_prediction_wrong = np.any(predictions != y)
    if any_prediction_wrong:
      wrong_index = np.where(predictions != y)[0].tolist()[0]
      counter_example = ' '.join([str(c) for c in batch[wrong_index]])
      if FLAGS.debug_mode:
        logging.info('\nOriginal example: %s, prediction: %d',
                     ' '.join([str(c) for c in sentence]), y)  # pylint: disable=undefined-loop-variable
        logging.info('\ncounter example:  %s, prediction: %s', counter_example,
                     predictions[wrong_index].tolist())
      counter_prediction = predictions[wrong_index]
    return (not any_prediction_wrong, counter_example,
            counter_prediction, num_forward_passes)


def verify_dataset(dataset, config_dict, model_location, synonym_dict, delta):
  """Tries to verify against perturbation attacks up to delta."""
  sst_model = interactive_example.InteractiveSentimentPredictor(
      config_dict, model_location,
      max_padded_length=FLAGS.max_padded_length,
      num_perturbations=FLAGS.num_perturbations)
  verified_list = []  # Holds boolean entries, across dataset.
  samples = []
  labels = []
  counter_examples = []
  counter_predictions = []
  total_num_forward_passes = []
  logging.info('dataset size: %d', len(dataset))
  num_examples = FLAGS.num_examples if FLAGS.num_examples else len(dataset)
  logging.info('skip_batches: %d', FLAGS.skip_batches)
  logging.info('num_examples: %d', num_examples)
  logging.info('new dataset size: %d',
               len(dataset[FLAGS.skip_batches:FLAGS.skip_batches+num_examples]))
  for i, sample in tqdm.tqdm(enumerate(
      dataset[FLAGS.skip_batches:FLAGS.skip_batches+num_examples])):
    if FLAGS.debug_mode:
      logging.info('index: %d', i)
      (verified_bool, counter_example, counter_prediction, num_forward_passes
      ) = verify_exhaustively(
          sample, synonym_dict, sst_model, delta, FLAGS.truncated_len)
      samples.append(''.join(sample[0]))
      labels.append(sample[1])
      counter_examples.append(counter_example)
      counter_predictions.append(counter_prediction)
      total_num_forward_passes.append(num_forward_passes)
    else:
      verified_bool, _, _, num_forward_passes = verify_exhaustively(
          sample, synonym_dict, sst_model, delta, FLAGS.truncated_len)
    verified_list.append(verified_bool)

  verified_proportion = np.mean(verified_list)
  assert len(verified_list) == len(
      dataset[FLAGS.skip_batches:FLAGS.skip_batches+num_examples])
  return (verified_proportion, verified_list, samples, counter_examples,
          counter_predictions, total_num_forward_passes)


def example(synonym_dict, dataset, k=2):
  """Example usage of functions above."""

  # The below example x has these synonyms.
  # 'decree' --> [edict, order],
  # 'tubes' --> 'pipes';
  # 'refrigerated' --> ['cooled', 'chilled']
  x = ['the', 'refrigerated', 'decree', 'tubes']

  # Example: 1 perturbation.
  new_x = expand_by_one_perturbation(x, x, synonym_dict)
  pprint.pprint(sorted(new_x))

  # Example: up to k perturbations.
  new_x = find_up_to_depth_k_perturbations(x, x, synonym_dict, k)
  pprint.pprint(sorted(new_x))

  # Statistics: how large is the combinatorial space of perturbations?
  total_x = []
  size_counter = collections.Counter()
  for (x, _) in tqdm.tqdm(dataset):
    new_x = find_up_to_depth_k_perturbations(x, x, synonym_dict, k)
    size_counter[len(new_x)] += 1
    total_x.extend(new_x)

  # Histogram for perturbation space size, computed across dataset.
  pprint.pprint([x for x in sorted(size_counter.items(), key=lambda xx: xx[0])])

  # Total number of inputs for forward pass if comprehensively evaluated.
  pprint.pprint(len(total_x))


def main(args):
  del args

  # Read the config file into a new ad-hoc module.
  with open(FLAGS.config_path, 'r') as config_file:
    config_code = config_file.read()
    config_module = imp.new_module('config')
    exec(config_code, config_module.__dict__)  # pylint: disable=exec-used
  config = config_module.get_config()

  config_dict = {'task': FLAGS.task,
                 'batch_size': FLAGS.batch_size,
                 'pooling': FLAGS.pooling,
                 'learning_rate': 0.,
                 'config': config,
                 'embedding_dim': config['embedding_dim'],
                 'fine_tune_embeddings': FLAGS.fine_tune_embeddings,
                 'num_oov_buckets': FLAGS.num_oov_buckets,
                 'max_grad_norm': 0.}

  # Maximum verification range.
  delta = FLAGS.delta
  character_level = FLAGS.character_level
  mode = FLAGS.mode
  model_location = FLAGS.checkpoint_path

  # Load synonyms.
  synonym_filepath = config['synonym_filepath']
  synonym_dict = load_synonyms(synonym_filepath)

  # Load data.
  dataset = load_dataset(mode, character_level)

  # Compute verifiable accuracy on dataset.
  (verified_proportion, _, _, _, _, _) = verify_dataset(dataset, config_dict,
                                                        model_location,
                                                        synonym_dict, delta)
  logging.info('verified_proportion:')
  logging.info(str(verified_proportion))
  logging.info({
      'delta': FLAGS.delta,
      'character_level': FLAGS.character_level,
      'mode': FLAGS.mode,
      'checkpoint_path': FLAGS.checkpoint_path,
      'verified_proportion': verified_proportion
  })


if __name__ == '__main__':
  logging.set_stderrthreshold('info')
  app.run(main)
