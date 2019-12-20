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

"""Evaluates a verifiable model on Mnist or CIFAR-10."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import interval_bound_propagation as ibp
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', 'auto', ['auto', 'mnist', 'cifar10'], 'Dataset '
                  '("auto", "mnist" or "cifar10"). When set to "auto", '
                  'the dataset is inferred from the model directory path.')
flags.DEFINE_enum('model', 'auto', ['auto', 'tiny', 'small', 'medium',
                                    'large_200', 'large'], 'Model size. '
                  'When set to "auto", the model name is inferred from the '
                  'model directory path.')
flags.DEFINE_string('model_dir', None, 'Model checkpoint directory.')
flags.DEFINE_enum('bound_method', 'ibp', ['ibp', 'crown-ibp'],
                  'Bound progataion method. For models trained with CROWN-IBP '
                  'and beta_final=1 (e.g., CIFAR 2/255), use "crown-ibp". '
                  'Otherwise use "ibp".')
flags.DEFINE_integer('batch_size', 200, 'Batch size.')
flags.DEFINE_float('epsilon', .3, 'Target epsilon.')


def layers(model_size):
  """Returns the layer specification for a given model name."""
  if model_size == 'tiny':
    return (
        ('linear', 100),
        ('activation', 'relu'))
  elif model_size == 'small':
    return (
        ('conv2d', (4, 4), 16, 'VALID', 2),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 32, 'VALID', 1),
        ('activation', 'relu'),
        ('linear', 100),
        ('activation', 'relu'))
  elif model_size == 'medium':
    return (
        ('conv2d', (3, 3), 32, 'VALID', 1),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 32, 'VALID', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'VALID', 1),
        ('activation', 'relu'),
        ('conv2d', (4, 4), 64, 'VALID', 2),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'))
  elif model_size == 'large_200':
    # Some old large checkpoints have 200 hidden neurons in the last linear
    # layer.
    return (
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('linear', 200),
        ('activation', 'relu'))
  elif model_size == 'large':
    return (
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 64, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 2),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('conv2d', (3, 3), 128, 'SAME', 1),
        ('activation', 'relu'),
        ('linear', 512),
        ('activation', 'relu'))
  else:
    raise ValueError('Unknown model: "{}"'.format(model_size))


def show_metrics(metric_values, bound_method='ibp'):
  if bound_method == 'crown-ibp':
    verified_accuracy = metric_values.crown_ibp_verified_accuracy
  else:
    verified_accuracy = metric_values.verified_accuracy
  print('nominal accuracy = {:.2f}%, '
        'verified accuracy = {:.2f}%, '
        'accuracy under PGD attack = {:.2f}%'.format(
            metric_values.nominal_accuracy * 100.,
            verified_accuracy* 100.,
            metric_values.attack_accuracy * 100.))


def main(unused_args):
  dataset = FLAGS.dataset
  if FLAGS.dataset == 'auto':
    if 'mnist' in FLAGS.model_dir:
      dataset = 'mnist'
    elif 'cifar' in FLAGS.model_dir:
      dataset = 'cifar10'
    else:
      raise ValueError('Cannot guess the dataset name. Please specify '
                       '--dataset manually.')

  model_name = FLAGS.model
  if FLAGS.model == 'auto':
    model_names = ['large_200', 'large', 'medium', 'small', 'tiny']
    for name in model_names:
      if name in FLAGS.model_dir:
        model_name = name
        logging.info('Using guessed model name "%s".', model_name)
        break
    if model_name == 'auto':
      raise ValueError('Cannot guess the model name. Please specify --model '
                       'manually.')

  checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)
  if checkpoint_path is None:
    raise OSError('Cannot find a valid checkpoint in {}.'.format(
        FLAGS.model_dir))

  # Dataset.
  input_bounds = (0., 1.)
  num_classes = 10
  if dataset == 'mnist':
    data_train, data_test = tf.keras.datasets.mnist.load_data()
  else:
    assert dataset == 'cifar10', (
        'Unknown dataset "{}"'.format(dataset))
    data_train, data_test = tf.keras.datasets.cifar10.load_data()
    data_train = (data_train[0], data_train[1].flatten())
    data_test = (data_test[0], data_test[1].flatten())

  # Base predictor network.
  original_predictor = ibp.DNN(num_classes, layers(model_name))
  predictor = original_predictor
  if dataset == 'cifar10':
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    predictor = ibp.add_image_normalization(original_predictor, mean, std)
  if FLAGS.bound_method == 'crown-ibp':
    predictor = ibp.crown.VerifiableModelWrapper(predictor)
  else:
    predictor = ibp.VerifiableModelWrapper(predictor)

  # Test using while loop.
  def get_test_metrics(batch_size, attack_builder=ibp.UntargetedPGDAttack):
    """Returns the test metrics."""
    num_test_batches = len(data_test[0]) // batch_size
    assert len(data_test[0]) % batch_size == 0, (
        'Test data is not a multiple of batch size.')

    def cond(i, *unused_args):
      return i < num_test_batches

    def body(i, metrics):
      """Compute the sum of all metrics."""
      test_data = ibp.build_dataset(data_test, batch_size=batch_size,
                                    sequential=True)
      predictor(test_data.image, override=True, is_training=False)
      input_interval_bounds = ibp.IntervalBounds(
          tf.maximum(test_data.image - FLAGS.epsilon, input_bounds[0]),
          tf.minimum(test_data.image + FLAGS.epsilon, input_bounds[1]))
      predictor.propagate_bounds(input_interval_bounds)
      test_specification = ibp.ClassificationSpecification(
          test_data.label, num_classes)
      test_attack = attack_builder(predictor, test_specification, FLAGS.epsilon,
                                   input_bounds=input_bounds,
                                   optimizer_builder=ibp.UnrolledAdam)

      # Use CROWN-IBP bound or IBP bound.
      if FLAGS.bound_method == 'crown-ibp':
        test_losses = ibp.crown.Losses(predictor, test_specification,
                                       test_attack, use_crown_ibp=True,
                                       crown_bound_schedule=tf.constant(1.))
      else:
        test_losses = ibp.Losses(predictor, test_specification, test_attack)

      test_losses(test_data.label)
      new_metrics = []
      for m, n in zip(metrics, test_losses.scalar_metrics):
        new_metrics.append(m + n)
      return i + 1, new_metrics

    if FLAGS.bound_method == 'crown-ibp':
      metrics = ibp.crown.ScalarMetrics
    else:
      metrics = ibp.ScalarMetrics
    total_count = tf.constant(0, dtype=tf.int32)
    total_metrics = [tf.constant(0, dtype=tf.float32)
                     for _ in range(len(metrics._fields))]
    total_count, total_metrics = tf.while_loop(
        cond,
        body,
        loop_vars=[total_count, total_metrics],
        back_prop=False,
        parallel_iterations=1)
    total_count = tf.cast(total_count, tf.float32)
    test_metrics = []
    for m in total_metrics:
      test_metrics.append(m / total_count)
    return metrics(*test_metrics)

  test_metrics = get_test_metrics(
      FLAGS.batch_size, ibp.UntargetedPGDAttack)

  # Prepare to load the pretrained-model.
  saver = tf.compat.v1.train.Saver(original_predictor.get_variables())

  # Run everything.
  tf_config = tf.ConfigProto()
  tf_config.gpu_options.allow_growth = True
  with tf.train.SingularMonitoredSession(config=tf_config) as sess:
    logging.info('Restoring from checkpoint "%s".', checkpoint_path)
    saver.restore(sess, checkpoint_path)
    logging.info('Evaluating at epsilon = %f.', FLAGS.epsilon)
    metric_values = sess.run(test_metrics)
    show_metrics(metric_values, FLAGS.bound_method)


if __name__ == '__main__':
  flags.mark_flag_as_required('model_dir')
  app.run(main)
