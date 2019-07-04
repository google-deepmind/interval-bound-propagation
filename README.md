# On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models

This repository contains a simple implementation of Interval Bound Propagation
(IBP) using TensorFlow: https://arxiv.org/abs/1810.12715

This is not an official Google product

## Installation

IBP can be installed with the following command:

```bash
pip install git+https://github.com/deepmind/interval-bound-propagation`
```

IBP will work with both the CPU and GPU version of tensorflow and dm-sonnet, but
to allow for that it does not list Tensorflow as a requirement, so you need to
install Tensorflow and Sonnet separately if you haven't already done so.

## Usage

This following command trains a small model on MNIST with epsilon set to 0.3:

```bash
cd examples
python train.py --model=small --output_dir=/tmp/small_model
```

## Giving credit

If you use this code in your work, we ask that you cite this paper:

Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin,
Jonathan Uesato, Relja Arandjelovic, Timothy Mann, and Pushmeet Kohli.
"On the Effectiveness of Interval Bound Propagation for Training Verifiably
Robust Models." _arXiv preprint arXiv:1810.12715 (2018)_.

## Acknowledgements

In addition to the people involved in the original publication, we would like
to thank Sumanth Dathathri and Johannes Welbl for their contributions.

