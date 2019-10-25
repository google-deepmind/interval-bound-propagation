# Copyright 2018 The Interval Bound Propagation Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['six', 'absl-py', 'numpy']
EXTRA_PACKAGES = {
    'tensorflow': ['tensorflow>=1.8.0'],
    'tensorflow with gpu': ['tensorflow-gpu>=1.8.0'],
    'sonnet': ['dm-sonnet>=1.26'],
    'sonnet with gpu': ['dm-sonnet-gpu>=1.26'],
}


def ibp_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('interval_bound_propagation/tests',
                                    pattern='*_test.py')
  return test_suite

setup(
    name='interval_bound_propagation',
    version='1.1',
    description='A library to train verifiably robust neural networks.',
    url='https://github.com/deepmind/interval_bound_propagation',
    author='DeepMind',
    author_email='no-reply@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.ibp_test_suite',
)
