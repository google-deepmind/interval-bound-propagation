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

"""Library to train verifiably robust neural networks.

For more details see paper: On the Effectiveness of Interval Bound Propagation
for Training Verifiably Robust Models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from interval_bound_propagation.src.attacks import TargetedPGDAttack
from interval_bound_propagation.src.attacks import UnrolledAdam
from interval_bound_propagation.src.attacks import UnrolledGradientDescent
from interval_bound_propagation.src.attacks import UntargetedPGDAttack
from interval_bound_propagation.src.bounds import AbstractBounds
from interval_bound_propagation.src.bounds import IntervalBounds
from interval_bound_propagation.src.layers import BatchNorm
from interval_bound_propagation.src.layers import ImageNorm
from interval_bound_propagation.src.loss import Losses
from interval_bound_propagation.src.loss import ScalarLosses
from interval_bound_propagation.src.loss import ScalarMetrics
from interval_bound_propagation.src.model import DNN
from interval_bound_propagation.src.model import VerifiableModelWrapper
from interval_bound_propagation.src.specification import ClassificationSpecification
from interval_bound_propagation.src.specification import LinearSpecification
from interval_bound_propagation.src.utils import add_image_normalization
from interval_bound_propagation.src.utils import build_dataset
from interval_bound_propagation.src.utils import create_classification_losses
from interval_bound_propagation.src.utils import linear_schedule
from interval_bound_propagation.src.verifiable_wrapper import BatchFlattenWrapper
from interval_bound_propagation.src.verifiable_wrapper import BatchNormWrapper
from interval_bound_propagation.src.verifiable_wrapper import ImageNormWrapper
from interval_bound_propagation.src.verifiable_wrapper import LinearConv2dWrapper
from interval_bound_propagation.src.verifiable_wrapper import LinearFCWrapper
from interval_bound_propagation.src.verifiable_wrapper import MonotonicWrapper


__version__ = '1.00'
