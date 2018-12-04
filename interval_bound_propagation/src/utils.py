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
  rate = tf.cast(step - init_step, tf.float32) / float(final_step - init_step)
  linear_value = rate * (final_value - init_value) + init_value
  return tf.clip_by_value(linear_value, min(init_value, final_value),
                          max(init_value, final_value))


def build_classification_specification(label, num_classes):
  label_one_hot = tf.one_hot(label, depth=num_classes)
  c = tf.expand_dims(tf.eye(num_classes), 0) - tf.expand_dims(label_one_hot, 2)
  # Re-order to have batch, specs, outputs order.
  c = tf.transpose(c, [0, 2, 1])
  return specification.LinearSpecification(c)


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
  if options is not None:
    elide = options.get('elide_last_layer', elide)
    loss_type = options.get('verified_loss_type', loss_type)
    loss_margin = options.get('verified_loss_margin', loss_type)

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
    train_epsilon = epsilon
  else:
    train_epsilon = linear_schedule(
        global_step, warmup_steps, warmup_steps + rampup_steps, 0., epsilon)

  predictor_network(inputs, is_training=True)
  num_classes = predictor_network.output_size
  if use_verification:
    logging.info('Verification active.')
    input_interval_bounds = bounds.IntervalBounds(
        tf.maximum(inputs - train_epsilon, input_bounds[0]),
        tf.minimum(inputs + train_epsilon, input_bounds[1]))
    predictor_network.propagate_bounds(input_interval_bounds)
    spec = build_classification_specification(label, num_classes)
    spec_builder = lambda *args, **kwargs: spec(*args, collapse=elide, **kwargs)  # pylint: disable=unnecessary-lambda
  else:
    logging.info('Verification disabled.')
    spec = None
    spec_builder = None
  if use_attack:
    logging.info('Attack active.')
    s = spec
    if s is None:
      s = build_classification_specification(label, num_classes)
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
  return losses, train_loss
