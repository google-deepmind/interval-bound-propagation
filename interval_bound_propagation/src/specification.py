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

"""Defines the output specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from absl import logging

from interval_bound_propagation.src import bounds as bounds_lib
from interval_bound_propagation.src import verifiable_wrapper
import six
import sonnet as snt
import tensorflow.compat.v1 as tf


@six.add_metaclass(abc.ABCMeta)
class Specification(snt.AbstractModule):
  """Defines a specification."""

  def __init__(self, name, collapse=True):
    super(Specification, self).__init__(name=name)
    self._collapse = collapse

  @abc.abstractmethod
  def _build(self, modules):
    """Computes the worst-case specification value."""

  @abc.abstractmethod
  def evaluate(self, logits):
    """Computes the specification value.

    Args:
      logits: The logits Tensor can have different shapes, i.e.,
        [batch_size, num_classes]: The output should be [batch_size, num_specs].
        [num_restarts, batch_size, num_classes]: The output should be
          [num_restarts, batch_size, num_specs]. Used by UntargetedPGDAttack.
        [num_restarts, num_specs, batch_size, num_classes]: The output should
          be [num_restarts, batch_size, num_specs]. For this case, the
          specifications must be evaluated individually for each column
          (axis = 1). Used by MultiTargetedPGDAttack.

    Returns:
      The specification values evaluated at the network output.
    """

  @abc.abstractproperty
  def num_specifications(self):
    """Returns the number of specifications."""

  @property
  def collapse(self):
    return self._collapse


class LinearSpecification(Specification):
  """Linear specifications: c^T * z_K + d <= 0."""

  def __init__(self, c, d=None, prune_irrelevant=True, collapse=True):
    """Builds a linear specification module."""
    super(LinearSpecification, self).__init__(name='specs', collapse=collapse)
    # c has shape [batch_size, num_specifications, num_outputs]
    # d has shape [batch_size, num_specifications]
    # Some specifications may be irrelevant (not a function of the output).
    # We automatically remove them for clarity. We expect the number of
    # irrelevant specs to be equal for all elements of a batch.
    # Shape is [batch_size, num_specifications]
    if prune_irrelevant:
      irrelevant = tf.equal(tf.reduce_sum(
          tf.cast(tf.abs(c) > 1e-6, tf.int32), axis=-1, keepdims=True), 0)
      batch_size = tf.shape(c)[0]
      num_outputs = tf.shape(c)[2]
      irrelevant = tf.tile(irrelevant, [1, 1, num_outputs])
      self._c = tf.reshape(
          tf.boolean_mask(c, tf.logical_not(irrelevant)),
          [batch_size, -1, num_outputs])
    else:
      self._c = c
    self._d = d

  def _build(self, modules):
    """Outputs specification value."""
    # inputs have shape [batch_size, num_outputs].
    if not (self.collapse and
            isinstance(modules[-1], verifiable_wrapper.LinearFCWrapper)):
      logging.info('Elision of last layer disabled.')
      bounds = modules[-1].output_bounds
      w = self._c
      b = self._d
    else:
      logging.info('Elision of last layer active.')
      # Collapse the last layer.
      bounds = modules[-1].input_bounds
      w = modules[-1].module.w
      b = modules[-1].module.b
      w = tf.einsum('ijk,lk->ijl', self._c, w)
      b = tf.einsum('ijk,k->ij', self._c, b)
      if self._d is not None:
        b += self._d

    # Maximize z * w + b s.t. lower <= z <= upper.
    bounds = bounds_lib.IntervalBounds.convert(bounds)
    c = (bounds.lower + bounds.upper) / 2.
    r = (bounds.upper - bounds.lower) / 2.
    c = tf.einsum('ij,ikj->ik', c, w)
    if b is not None:
      c += b
    r = tf.einsum('ij,ikj->ik', r, tf.abs(w))

    # output has shape [batch_size, num_specifications].
    return c + r

  def evaluate(self, logits):
    if len(logits.shape) == 2:
      output = tf.einsum('ij,ikj->ik', logits, self._c)
    elif len(logits.shape) == 3:
      output = tf.einsum('rij,ikj->rik', logits, self._c)
    else:
      assert len(logits.shape) == 4
      output = tf.einsum('rsbo,bso->rbs', logits, self._c)
    if self._d is not None:
      output += self._d
    return output

  @property
  def num_specifications(self):
    return tf.shape(self._c)[1]

  @property
  def c(self):
    return self._c

  @property
  def d(self):
    return self._d


class ClassificationSpecification(Specification):
  """Creates a linear specification that corresponds to a classification.

  This class is not a standard LinearSpecification as it does not materialize
  the c and d tensors.
  """

  def __init__(self, label, num_classes, collapse=True):
    super(ClassificationSpecification, self).__init__(name='specs',
                                                      collapse=collapse)
    self._label = label
    self._num_classes = num_classes
    # Precompute indices.
    with self._enter_variable_scope():
      indices = []
      for i in range(self._num_classes):
        indices.append(list(range(i)) + list(range(i + 1, self._num_classes)))
      indices = tf.constant(indices, dtype=tf.int32)
      self._correct_idx, self._wrong_idx = self._build_indices(label, indices)

  def _build(self, modules):
    if not (self.collapse and
            isinstance(modules[-1], verifiable_wrapper.LinearFCWrapper)):
      logging.info('Elision of last layer disabled.')
      bounds = modules[-1].output_bounds
      bounds = bounds_lib.IntervalBounds.convert(bounds)
      correct_class_logit = tf.gather_nd(bounds.lower, self._correct_idx)
      wrong_class_logits = tf.gather_nd(bounds.upper, self._wrong_idx)
      return wrong_class_logits - tf.expand_dims(correct_class_logit, 1)

    logging.info('Elision of last layer active.')
    bounds = modules[-1].input_bounds
    bounds = bounds_lib.IntervalBounds.convert(bounds)
    batch_size = tf.shape(bounds.lower)[0]
    w = modules[-1].module.w
    b = modules[-1].module.b
    w_t = tf.tile(tf.expand_dims(tf.transpose(w), 0), [batch_size, 1, 1])
    b_t = tf.tile(tf.expand_dims(b, 0), [batch_size, 1])
    w_correct = tf.expand_dims(tf.gather_nd(w_t, self._correct_idx), -1)
    b_correct = tf.expand_dims(tf.gather_nd(b_t, self._correct_idx), 1)
    w_wrong = tf.transpose(tf.gather_nd(w_t, self._wrong_idx), [0, 2, 1])
    b_wrong = tf.gather_nd(b_t, self._wrong_idx)
    w = w_wrong - w_correct
    b = b_wrong - b_correct
    # Maximize z * w + b s.t. lower <= z <= upper.
    c = (bounds.lower + bounds.upper) / 2.
    r = (bounds.upper - bounds.lower) / 2.
    c = tf.einsum('ij,ijk->ik', c, w)
    if b is not None:
      c += b
    r = tf.einsum('ij,ijk->ik', r, tf.abs(w))
    return c + r

  def evaluate(self, logits):
    if len(logits.shape) == 2:
      correct_class_logit = tf.gather_nd(logits, self._correct_idx)
      correct_class_logit = tf.expand_dims(correct_class_logit, -1)
      wrong_class_logits = tf.gather_nd(logits, self._wrong_idx)
    elif len(logits.shape) == 3:
      # [num_restarts, batch_size, num_classes] to
      # [num_restarts, batch_size, num_specs]
      logits = tf.transpose(logits, [1, 2, 0])  # Put restart dimension last.
      correct_class_logit = tf.gather_nd(logits, self._correct_idx)
      correct_class_logit = tf.transpose(correct_class_logit)
      correct_class_logit = tf.expand_dims(correct_class_logit, -1)
      wrong_class_logits = tf.gather_nd(logits, self._wrong_idx)
      wrong_class_logits = tf.transpose(wrong_class_logits, [2, 0, 1])
    else:
      assert len(logits.shape) == 4
      # [num_restarts, num_specs, batch_size, num_classes] to
      # [num_restarts, batch_size, num_specs].
      logits = tf.transpose(logits, [2, 3, 1, 0])
      correct_class_logit = tf.gather_nd(logits, self._correct_idx)
      correct_class_logit = tf.transpose(correct_class_logit, [2, 0, 1])
      batch_size = tf.shape(logits)[0]
      wrong_idx = tf.concat([
          self._wrong_idx,
          tf.tile(tf.reshape(tf.range(self.num_specifications, dtype=tf.int32),
                             [1, self.num_specifications, 1]),
                  [batch_size, 1, 1])], axis=-1)
      wrong_class_logits = tf.gather_nd(logits, wrong_idx)
      wrong_class_logits = tf.transpose(wrong_class_logits, [2, 0, 1])
    return wrong_class_logits - correct_class_logit

  @property
  def num_specifications(self):
    return self._num_classes - 1

  @property
  def correct_idx(self):
    return self._correct_idx

  @property
  def wrong_idx(self):
    return self._wrong_idx

  def _build_indices(self, label, indices):
    batch_size = tf.shape(label)[0]
    i = tf.range(batch_size, dtype=tf.int32)
    correct_idx = tf.stack([i, tf.cast(label, tf.int32)], axis=1)
    wrong_idx = tf.stack([
        tf.tile(tf.reshape(i, [batch_size, 1]), [1, self._num_classes - 1]),
        tf.gather(indices, label),
    ], axis=2)
    return correct_idx, wrong_idx


class TargetedClassificationSpecification(ClassificationSpecification):
  """Defines a specification that compares the true class with another."""

  def __init__(self, label, num_classes, target_class, collapse=True):
    super(TargetedClassificationSpecification, self).__init__(
        label, num_classes, collapse=collapse)
    batch_size = tf.shape(label)[0]
    if len(target_class.shape) == 1:
      target_class = tf.reshape(target_class, [batch_size, 1])
    self._num_specifications = target_class.shape[1].value
    if self._num_specifications is None:
      raise ValueError('Cannot retrieve the number of target classes')
    self._target_class = target_class
    i = tf.range(batch_size, dtype=tf.int32)
    self._wrong_idx = tf.stack([
        tf.tile(tf.reshape(i, [batch_size, 1]), [1, self.num_specifications]),
        target_class
    ], axis=2)

  @property
  def target_class(self):
    """Returns the target class index."""
    return self._target_class

  @property
  def num_specifications(self):
    return self._num_specifications


class RandomClassificationSpecification(TargetedClassificationSpecification):
  """Creates a single random specification that targets a random class."""

  def __init__(self, label, num_classes, num_targets=1, seed=None,
               collapse=True):
    # Overwrite the target indices. Each session.run() call gets new target
    # indices, the indices should remain the same across restarts.
    batch_size = tf.shape(label)[0]
    j = tf.random.uniform(shape=(batch_size, num_targets), minval=1,
                          maxval=num_classes, dtype=tf.int32, seed=seed)
    target_class = tf.mod(tf.cast(tf.expand_dims(label, -1), tf.int32) + j,
                          num_classes)
    super(RandomClassificationSpecification, self).__init__(
        label, num_classes, target_class, collapse=collapse)


class LeastLikelyClassificationSpecification(
    TargetedClassificationSpecification):
  """Creates a single specification that targets the least likely class."""

  def __init__(self, label, num_classes, logits, num_targets=1, collapse=True):
    # Do not target the true class. If the true class is the least likely to
    # be predicted, it is fine to target any other class as the attack will
    # be successful anyways.
    j = tf.nn.top_k(-logits, k=num_targets, sorted=False).indices
    l = tf.expand_dims(label, 1)
    target_class = tf.mod(
        j + tf.cast(tf.equal(j, tf.cast(l, tf.int32)), tf.int32), num_classes)
    super(LeastLikelyClassificationSpecification, self).__init__(
        label, num_classes, target_class, collapse=collapse)
