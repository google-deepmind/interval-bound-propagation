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

"""Helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
from interval_bound_propagation.src import attacks
from interval_bound_propagation.src import bounds
from interval_bound_propagation.src import layers
from interval_bound_propagation.src import loss
from interval_bound_propagation.src import specification
import numpy as np
import tensorflow as tf


# Defines a dataset sample."""
Sample = collections.namedtuple('Sample', ['image', 'label'])


def build_dataset(raw_data, batch_size=50, sequential=True):
  """Builds a dataset from raw NumPy tensors."""
  images, labels = raw_data
  # We need width, height and channel.
  if len(images.shape) == 3:
    images = np.expand_dims(images, -1)
  samples = Sample(images.astype(np.float32) / 255., labels.astype(np.int64))
  data = tf.data.Dataset.from_tensor_slices(samples)
  if not sequential:
    data = data.shuffle(1000)
  return data.repeat().batch(batch_size).make_one_shot_iterator().get_next()


def linear_schedule(step, init_step, final_step, init_value, final_value):
  """Linear schedule."""
  assert final_step >= init_step
  if init_step == final_step:
    return final_value
  rate = tf.cast(step - init_step, tf.float32) / float(final_step - init_step)
  linear_value = rate * (final_value - init_value) + init_value
  return tf.clip_by_value(linear_value, min(init_value, final_value),
                          max(init_value, final_value))


def smooth_schedule(step, init_step, final_step, init_value, final_value,
                    mid_point=.25, beta=4.):
  """Smooth schedule that slowly morphs into a linear schedule."""
  assert final_value > init_value
  assert final_step >= init_step
  assert beta >= 2.
  assert mid_point >= 0. and mid_point <= 1.
  mid_step = int((final_step - init_step) * mid_point) + init_step
  if mid_step <= init_step:
    alpha = 1.
  else:
    t = (mid_step - init_step) ** (beta - 1.)
    alpha = (final_value - init_value) / ((final_step - mid_step) * beta * t +
                                          (mid_step - init_step) * t)
  mid_value = alpha * (mid_step - init_step) ** beta + init_value
  # Tensorflow operation.
  is_ramp = tf.cast(step > init_step, tf.float32)
  is_linear = tf.cast(step >= mid_step, tf.float32)
  return (is_ramp * (
      (1. - is_linear) * (
          init_value +
          alpha * tf.pow(tf.cast(step - init_step, tf.float32), beta)) +
      is_linear * linear_schedule(
          step, mid_step, final_step, mid_value, final_value)) +
          (1. - is_ramp) * init_value)


def add_image_normalization(model, mean, std):
  def _model(x, *args, **kwargs):
    return model(layers.ImageNorm(mean, std)(x), *args, **kwargs)
  return _model


def create_classification_losses(
    global_step,
    inputs,
    label,
    predictor_network,
    epsilon,
    loss_weights,
    warmup_steps=0,
    rampup_steps=-1,
    input_bounds=(0., 1.),
    options=None):
  """Create the training loss."""
  elide = True
  loss_type = 'xent'
  loss_margin = 10.
  is_training_off_after_warmup = False
  smooth_epsilon_schedule = False
  if options is not None:
    elide = options.get('elide_last_layer', elide)
    loss_type = options.get('verified_loss_type', loss_type)
    loss_margin = options.get('verified_loss_margin', loss_type)
    is_training_off_after_warmup = options.get(
        'is_training_off_after_warmup', is_training_off_after_warmup)
    smooth_epsilon_schedule = options.get(
        'smooth_epsilon_schedule', smooth_epsilon_schedule)

  # Loss weights.
  def _get_schedule(init, final):
    if init == final:
      return init
    return linear_schedule(
        global_step, warmup_steps, warmup_steps + rampup_steps, init, final)
  def _is_active(init, final):
    return init > 0. or final > 0.
  nominal_xent = _get_schedule(**loss_weights.get('nominal'))
  attack_xent = _get_schedule(**loss_weights.get('attack'))
  use_attack = _is_active(**loss_weights.get('attack'))
  verified_loss = _get_schedule(**loss_weights.get('verified'))
  use_verification = _is_active(**loss_weights.get('verified'))
  weight_mixture = loss.ScalarLosses(
      nominal_cross_entropy=nominal_xent,
      attack_cross_entropy=attack_xent,
      verified_loss=verified_loss)

  if rampup_steps < 0:
    train_epsilon = tf.constant(epsilon)
    is_training = not is_training_off_after_warmup
  else:
    if smooth_epsilon_schedule:
      train_epsilon = smooth_schedule(
          global_step, warmup_steps, warmup_steps + rampup_steps, 0., epsilon)
    else:
      train_epsilon = linear_schedule(
          global_step, warmup_steps, warmup_steps + rampup_steps, 0., epsilon)
    if is_training_off_after_warmup:
      is_training = global_step < warmup_steps
    else:
      is_training = True

  predictor_network(inputs, is_training=is_training)
  num_classes = predictor_network.output_size
  if use_verification:
    logging.info('Verification active.')
    input_interval_bounds = bounds.IntervalBounds(
        tf.maximum(inputs - train_epsilon, input_bounds[0]),
        tf.minimum(inputs + train_epsilon, input_bounds[1]))
    predictor_network.propagate_bounds(input_interval_bounds)
    spec = specification.ClassificationSpecification(label, num_classes)
    spec_builder = lambda *args, **kwargs: spec(*args, collapse=elide, **kwargs)  # pylint: disable=unnecessary-lambda
  else:
    logging.info('Verification disabled.')
    spec = None
    spec_builder = None
  if use_attack:
    logging.info('Attack active.')
    s = spec
    if s is None:
      s = specification.ClassificationSpecification(label, num_classes)
    pgd_attack = attacks.UntargetedPGDAttack(
        predictor_network, s, epsilon, num_steps=7, input_bounds=input_bounds,
        optimizer_builder=attacks.UnrolledAdam)
  else:
    logging.info('Attack disabled.')
    pgd_attack = None
  losses = loss.Losses(predictor_network, spec_builder, pgd_attack,
                       interval_bounds_loss_type=loss_type,
                       interval_bounds_hinge_margin=loss_margin)
  losses(label)
  train_loss = sum(l * w for l, w in zip(losses.scalar_losses,
                                         weight_mixture))
  # Add a regularization loss.
  regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  train_loss = train_loss + tf.reduce_sum(regularizers)
  return losses, train_loss, train_epsilon
