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

"""Helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

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


def randomize(images, init_shape, expand_shape=None, crop_shape=None,
              vertical_flip=False):
  """Returns a function that randomly translates and flips images."""
  def random_image(image):
    """Randmly translates and flips images."""
    image = tf.reshape(image, init_shape)
    current_shape = init_shape
    if expand_shape is not None and expand_shape != current_shape:
      if expand_shape[-1] != current_shape[-1]:
        raise ValueError('Number channels is not specified correctly.')
      image = tf.image.resize_image_with_crop_or_pad(
          image, expand_shape[0], expand_shape[1])
      current_shape = expand_shape
    if crop_shape is not None and crop_shape != current_shape:
      image = tf.random_crop(image, crop_shape)
    if vertical_flip:
      image = tf.image.random_flip_left_right(image)
    return image
  return tf.map_fn(random_image, images)


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


def create_specification(label, num_classes, logits,
                         specification_type='one_vs_all'):
  """Creates a specification of the desired type."""
  def _num_targets(name):
    tokens = name.rsplit('_', 1)
    return int(tokens[1]) if len(tokens) > 1 else 1
  if specification_type == 'one_vs_all':
    return specification.ClassificationSpecification(label, num_classes)
  elif specification_type.startswith('random'):
    return specification.RandomClassificationSpecification(
        label, num_classes, _num_targets(specification_type))
  elif specification_type.startswith('least_likely'):
    return specification.LeastLikelyClassificationSpecification(
        label, num_classes, logits, _num_targets(specification_type))
  else:
    raise ValueError('Unknown specification type: "{}"'.format(
        specification_type))


def create_classification_losses(
    global_step, inputs, label, predictor_network, epsilon, loss_weights,
    warmup_steps=0, rampup_steps=-1, input_bounds=(0., 1.), options=None):
  """Create the training loss."""
  # Whether to elide the last linear layer with the specification.
  elide = True
  # Which loss to use for the IBP loss.
  loss_type = 'xent'
  # If the loss_type is 'hinge', which margin to use.
  loss_margin = 10.
  # Amount of label smoothing.
  label_smoothing = 0.
  # If True, batch normalization stops training after warm-up.
  is_training_off_after_warmup = False
  # If True, epsilon changes more smoothly.
  smooth_epsilon_schedule = False
  # Either 'one_vs_all', 'random_n', 'least_likely_n' or 'none'.
  verified_specification = 'one_vs_all'
  # Attack options.
  attack_specification = 'UntargetedPGDAttack_7x1x1_UnrolledAdam_.1'
  attack_scheduled = False
  attack_random_init = 1.
  # Whether the final loss from the attack should be standard cross-entropy
  # or the TRADES loss (https://arxiv.org/abs/1901.08573).
  pgd_attack_use_trades = False
  if options is not None:
    elide = options.get('elide_last_layer', elide)
    loss_type = options.get('verified_loss_type', loss_type)
    loss_margin = options.get('verified_loss_margin', loss_type)
    label_smoothing = options.get('label_smoothing', label_smoothing)
    is_training_off_after_warmup = options.get(
        'is_training_off_after_warmup', is_training_off_after_warmup)
    smooth_epsilon_schedule = options.get(
        'smooth_epsilon_schedule', smooth_epsilon_schedule)
    verified_specification = options.get(
        'verified_specification', verified_specification)
    attack_specification = options.get(
        'attack_specification', attack_specification)
    attack_scheduled = options.get('attack_scheduled', attack_scheduled)
    attack_random_init = options.get('attack_random_init', attack_random_init)
    pgd_attack_use_trades = options.get(
        'pgd_attack_use_trades', pgd_attack_use_trades)

  # Loss weights.
  def _get_schedule(init, final):
    if init == final:
      return init
    if rampup_steps < 0:
      return final
    return linear_schedule(
        global_step, warmup_steps, warmup_steps + rampup_steps, init, final)
  def _is_active(init, final):
    return init > 0. or final > 0.
  nominal_xent = _get_schedule(**loss_weights.get('nominal'))
  attack_xent = _get_schedule(**loss_weights.get('attack'))
  use_attack = _is_active(**loss_weights.get('attack'))
  verified_loss = _get_schedule(**loss_weights.get('verified'))
  use_verification = _is_active(**loss_weights.get('verified'))
  if verified_specification == 'none':
    use_verification = False
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

  logits = predictor_network(inputs, is_training=is_training)
  num_classes = predictor_network.output_size
  if use_verification:
    logging.info('Verification active.')
    input_interval_bounds = bounds.IntervalBounds(
        tf.maximum(inputs - train_epsilon, input_bounds[0]),
        tf.minimum(inputs + train_epsilon, input_bounds[1]))
    predictor_network.propagate_bounds(input_interval_bounds)
    spec = create_specification(label, num_classes, logits,
                                verified_specification)
    spec_builder = lambda *args, **kwargs: spec(*args, collapse=elide, **kwargs)  # pylint: disable=unnecessary-lambda
  else:
    logging.info('Verification disabled.')
    spec_builder = None
  if use_attack:
    logging.info('Attack active.')
    pgd_attack = create_attack(
        attack_specification, predictor_network, label,
        train_epsilon if attack_scheduled else epsilon,
        input_bounds=input_bounds, random_init=attack_random_init)
  else:
    logging.info('Attack disabled.')
    pgd_attack = None
  losses = loss.Losses(predictor_network, spec_builder, pgd_attack,
                       interval_bounds_loss_type=loss_type,
                       interval_bounds_hinge_margin=loss_margin,
                       label_smoothing=label_smoothing,
                       pgd_attack_use_trades=pgd_attack_use_trades)
  losses(label)
  train_loss = sum(l * w for l, w in zip(losses.scalar_losses,
                                         weight_mixture))
  # Add a regularization loss.
  regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  train_loss = train_loss + tf.reduce_sum(regularizers)
  return losses, train_loss, train_epsilon


# Additional helper code to build specific PGD attacks.
def get_attack_builder(logits, label, name='UntargetedPGDAttack',
                       random_seed=None, manual_target_class=None):
  """Returns a callable with the same arguments as PGDAttack.

  In addition to the callable, this function also returns the targeted class
  indices as a Tensor with the same shape as label.

  Usage is as follows:
    logits = model(inputs)
    attack_cls, specification, target_class = get_attack_builder(logits, labels)
    # target_class is None, if attack_cls is not a targeted attack.
    attack_instance = attack_cls(model, specification, epsilon)
    perturbed_inputs = attack_instance(inputs, labels)

  Args:
    logits: Tensor of nominal logits of shape [batch_size, num_classes].
    label: Tensor of labels of shape [batch_size].
    name: Name of a PGDAttack class or any of "RandomMoreLikelyPGDAttack",
      "RandomMostLikelyPGDAttack", "LeastLikelyMoreLikelyPGDAttack",
      "LeastLikelyMostLikelyPGDAttack", "ManualMoreLikelyPGDAttack",
      "ManualMostLikelyPGDAttack". Any attack name can be postfixed by
      "Xent" to use the cross-entropy loss rather than margin loss.
    random_seed: Sets the random seed for "Random*" attacks.
    manual_target_class: For "Manual*" attacks, Tensor of target class indices
      of shape [batch_size].

  Returns:
    A callable, a Specification and a Tensor of target label (or None if the
    attack is not targeted).
  """
  if name.endswith('Xent'):
    use_xent = True
    name = name[:-4]
  else:
    use_xent = False
  if name.endswith('Linf'):
    use_l2 = False
    name = name[:-4]  # Just for syntactic sugar.
  elif name.endswith('L2'):
    use_l2 = True
    name = name[:-2]
  else:
    use_l2 = False
  num_classes = logits.shape[1].value
  if num_classes is None:
    raise ValueError('Cannot determine the number of classes from logits.')

  # Special case for multi-targeted attacks.
  m = re.match(r'((?:MemoryEfficient)?MultiTargetedPGDAttack)'
               r'(?:(Top|Random)(\d)*)?', name)
  if m is not None:
    # Request for a multi-targeted attack.
    is_multitargeted = True
    name = m.group(1)
    is_random = (m.group(2) == 'Random')
    max_specs = m.group(3)
    max_specs = int(max_specs) if max_specs is not None else 0
  else:
    is_multitargeted = False

  # Any of the readily available attack classes use the standard classification
  # specification (one-vs-all) and are untargeted.
  if hasattr(attacks, name):
    attack_cls = getattr(attacks, name)
    parameters = {}
    if use_xent:
      parameters['objective_fn'] = _maximize_cross_entropy
    if use_l2:
      parameters['project_perturbation'] = _get_projection(2)
    if is_multitargeted:
      parameters['max_specifications'] = max_specs
      parameters['random_specifications'] = is_random
    if parameters:
      attack_cls = _change_parameters(attack_cls, **parameters)
    attack_specification = specification.ClassificationSpecification(
        label, num_classes)
    return attack_cls, attack_specification, None

  # Attacks can use an adaptive scheme.
  if name.endswith('AdaptivePGDAttack'):
    name = name[:-len('AdaptivePGDAttack')] + 'PGDAttack'
    is_adaptive = True
  else:
    is_adaptive = False

  # Attacks can be preceded by a number to indicate the number of target
  # classes. For efficiency, this is only available for *MoreLikely attacks.
  m = re.match(r'(\d*)(.*MoreLikelyPGDAttack)', name)
  if m is not None:
    num_targets = int(m.group(1))
    name = m.group(2)
  else:
    num_targets = 1

  # All attacks that are not directly listed in the attacks library are
  # targeted attacks that need to be manually constructed.
  if name not in ('RandomMoreLikelyPGDAttack', 'RandomMostLikelyPGDAttack',
                  'LeastLikelyMoreLikelyPGDAttack',
                  'LeastLikelyMostLikelyPGDAttack',
                  'ManualMoreLikelyPGDAttack', 'ManualMostLikelyPGDAttack'):
    raise ValueError('Unknown attack "{}".'.format(name))

  base_attack_cls = (attacks.AdaptiveUntargetedPGDAttack if is_adaptive else
                     attacks.UntargetedPGDAttack)
  if 'More' in name:
    if use_xent:
      raise ValueError('Using cross-entropy is not supported by '
                       '"*MoreLikelyPGDAttack".')
    attack_cls = base_attack_cls
  else:
    # We need to reverse the attack direction w.r.t. the specifications.
    attack_cls = _change_parameters(
        base_attack_cls,
        objective_fn=(_minimize_cross_entropy if use_xent else
                      _minimize_margin),
        success_fn=_all_smaller)
  if use_l2:
    attack_cls = _change_parameters(
        attack_cls, project_perturbation=_get_projection(2))

  # Set attack specification and target class.
  if name == 'RandomMoreLikelyPGDAttack':
    # A random target class should become more likely than the true class.
    attack_specification = specification.RandomClassificationSpecification(
        label, num_classes, num_targets=num_targets, seed=random_seed)
    target_class = (tf.squeeze(attack_specification.target_class, 1)
                    if num_targets == 1 else None)

  elif name == 'LeastLikelyMoreLikelyPGDAttack':
    attack_specification = specification.LeastLikelyClassificationSpecification(
        label, num_classes, logits, num_targets=num_targets)
    target_class = (tf.squeeze(attack_specification.target_class, 1)
                    if num_targets == 1 else None)

  elif name == 'ManualMoreLikelyPGDAttack':
    attack_specification = specification.TargetedClassificationSpecification(
        label, num_classes, manual_target_class)
    target_class = (tf.squeeze(attack_specification.target_class, 1)
                    if num_targets == 1 else None)

  elif name == 'RandomMostLikelyPGDAttack':
    # This attack needs to make the random target the highest logits for
    # it is be successful.
    target_class = _get_random_class(label, num_classes, seed=random_seed)
    attack_specification = specification.ClassificationSpecification(
        target_class, num_classes)

  elif name == 'LeastLikelyMostLikelyPGDAttack':
    # This attack needs to make the least likely target the highest logits
    # for it is be successful.
    target_class = _get_least_likely_class(label, num_classes, logits)
    attack_specification = specification.ClassificationSpecification(
        target_class, num_classes)

  else:
    assert name == 'ManualMostLikelyPGDAttack'
    target_class = manual_target_class
    attack_specification = specification.ClassificationSpecification(
        target_class, num_classes)

  return attack_cls, attack_specification, target_class


def create_attack(attack_config, predictor, label, epsilon,
                  input_bounds=(0., 1.), random_init=1.):
  """Creates an attack from a textual configuration.

  Args:
    attack_config: String with format "[AttackClass]_[steps]x
      [inner_restarts]x[outer_restarts]_[OptimizerClass]_[step_size]". Inner
      restarts involve tiling the input (they are more runtime efficient but
      use more memory), while outer restarts use a tf.while_loop.
    predictor: A VerifiableModelWrapper or StandardModelWrapper instance.
    label: A Tensor of labels.
    epsilon: Perturbation radius.
    input_bounds: Tuple with minimum and maximum value allowed on inputs.
    random_init: Probability of starting from random location rather than
      nominal input image.

  Returns:
    An Attack instance.
  """
  if attack_config:
    name, steps_and_restarts, optimizer, step_size = attack_config.split('_', 3)
    optimizer = getattr(attacks, optimizer)
    num_steps, inner_restarts, outer_restarts = (
        int(i) for i in steps_and_restarts.split('x', 3))
    step_size = step_size.replace(':', ',')
  else:
    name = 'UntargetedPGDAttack'
    num_steps = 200
    inner_restarts = 1
    outer_restarts = 1
    optimizer = attacks.UnrolledAdam
    step_size = .1

  def attack_learning_rate_fn(t):
    return parse_learning_rate(t, step_size)

  attack_cls, attack_specification, _ = get_attack_builder(
      predictor.logits, label, name=name)
  attack_strategy = attack_cls(
      predictor, attack_specification, epsilon, num_steps=num_steps,
      num_restarts=inner_restarts, input_bounds=input_bounds,
      optimizer_builder=optimizer, lr_fn=attack_learning_rate_fn,
      random_init=random_init)
  if outer_restarts > 1:
    attack_strategy = attacks.RestartedAttack(
        attack_strategy, num_restarts=outer_restarts)
  return attack_strategy


def parse_learning_rate(step, learning_rate):
  """Returns the learning rate as a tensor."""
  if isinstance(learning_rate, float):
    return learning_rate
  # Learning rate schedule of the form:
  # initial_learning_rate[,learning@steps]*. E.g., "1e-3" or
  # "1e-3,1e-4@15000,1e-5@25000". We use eval to allow learning specified as
  # fractions (e.g., 2/255).
  tokens = learning_rate.split(',')
  first_lr = float(eval(tokens[0]))  # pylint: disable=eval-used
  if len(tokens) == 1:
    return tf.constant(first_lr, dtype=tf.float32)
  # Parse steps.
  init_values = [first_lr]
  final_values = []
  init_step = [0]
  final_step = []
  for t in tokens[1:]:
    if '@' in t:
      lr, boundary = t.split('@', 1)
      is_linear = False
    elif 'S' in t:  # Syntactic sugar to indicate a step.
      lr, boundary = t.split('S', 1)
      is_linear = False
    elif 'L' in t:
      lr, boundary = t.split('L', 1)
      is_linear = True
    else:
      raise ValueError('Unknown specification.')
    lr = float(eval(lr))  # pylint: disable=eval-used
    init_values.append(lr)
    if is_linear:
      final_values.append(lr)
    else:
      final_values.append(init_values[-2])
    boundary = int(boundary)
    init_step.append(boundary)
    final_step.append(boundary)
  large_step = max(final_step) + 1
  final_step.append(large_step)
  final_values.append(lr)

  # Find current index.
  boundaries = list(final_step) + [large_step + 2]
  boundaries = tf.convert_to_tensor(boundaries, dtype=tf.int64)
  b = boundaries - tf.minimum(step + 1, large_step + 1)
  large_step = tf.constant(
      large_step, shape=boundaries.shape, dtype=step.dtype)
  b = tf.where(b < 0, large_step, b)
  idx = tf.minimum(tf.argmin(b), len(init_values) - 1)

  init_step = tf.convert_to_tensor(init_step, dtype=tf.float32)
  final_step = tf.convert_to_tensor(final_step, dtype=tf.float32)
  init_values = tf.convert_to_tensor(init_values, dtype=tf.float32)
  final_values = tf.convert_to_tensor(final_values, dtype=tf.float32)
  x1 = tf.gather(init_step, idx)
  x2 = tf.gather(final_step, idx)
  y1 = tf.gather(init_values, idx)
  y2 = tf.gather(final_values, idx)
  return (tf.cast(step, tf.float32) - x1) / (x2 - x1) * (y2 - y1) + y1


def _change_parameters(attack_cls, **updated_kwargs):
  def _build_new_attack(*args, **kwargs):
    kwargs.update(updated_kwargs)
    return attack_cls(*args, **kwargs)
  return _build_new_attack


def _get_random_class(label, num_classes, seed=None):
  batch_size = tf.shape(label)[0]
  target_label = tf.random.uniform(
      shape=(batch_size,), minval=1, maxval=num_classes, dtype=tf.int64,
      seed=seed)
  return tf.mod(tf.cast(label, tf.int64) + target_label, num_classes)


def _get_least_likely_class(label, num_classes, logits):
  target_label = tf.argmin(logits, axis=1, output_type=tf.int64)
  # In the off-chance that the least likely class is the true class, the target
  # class is changed to the be the next index.
  return tf.mod(target_label + tf.cast(
      tf.equal(target_label, tf.cast(label, tf.int64)), tf.int64), num_classes)


def _maximize_cross_entropy(specification_bounds):
  """Used to maximize the cross entropy loss."""
  # Bounds has shape [num_restarts, batch_size, num_specs].
  shape = tf.shape(specification_bounds)
  added_shape = [shape[0], shape[1], 1]
  v = tf.concat([
      specification_bounds,
      tf.zeros(added_shape, dtype=specification_bounds.dtype)], axis=2)
  l = tf.concat([
      tf.zeros_like(specification_bounds),
      tf.ones(added_shape, dtype=specification_bounds.dtype)], axis=2)
  # Minimize the cross-entropy loss w.r.t. target.
  return tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=tf.stop_gradient(l), logits=v)


def _minimize_cross_entropy(specification_bounds):
  return -_maximize_cross_entropy(specification_bounds)


def _maximize_margin(specification_bounds):
  # Bounds has shape [num_restarts, batch_size, num_specs].
  return tf.reduce_max(specification_bounds, axis=-1)


def _minimize_margin(specification_bounds):
  return -_maximize_margin(specification_bounds)


def _all_smaller(specification_bounds):
  specification_bounds = tf.reduce_max(specification_bounds, axis=-1)
  return specification_bounds < 0


def _get_projection(p):
  """Returns a projection function."""
  if p == np.inf:
    def _projection(perturbation, epsilon, input_image, image_bounds):
      clipped_perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)
      new_image = tf.clip_by_value(input_image + clipped_perturbation,
                                   image_bounds[0], image_bounds[1])
      return new_image - input_image
    return _projection

  elif p == 2:
    # avoid_zero_div must go inside sqrt to avoid a divide by zero
    # in the gradient through this operation
    def _projection(perturbation, epsilon, input_image, image_bounds):
      axes = list(range(1, len(perturbation.get_shape())))
      clipped_perturbation = tf.clip_by_norm(perturbation, epsilon, axes=axes)
      new_image = tf.clip_by_value(input_image + clipped_perturbation,
                                   image_bounds[0], image_bounds[1])
      return new_image - input_image
    return _projection

  else:
    raise ValueError('p must be np.inf or 2.')
