# Interval Bound Propagation for Training Verifiably Robust Models

This repository contains a simple implementation of Interval Bound Propagation
(IBP) using TensorFlow:
[https://arxiv.org/abs/1810.12715](https://arxiv.org/abs/1810.12715).
It also contains an implementation of CROWN-IBP:
[https://arxiv.org/abs/1906.06316](https://arxiv.org/abs/1906.06316).
It also contains a sentiment analysis example under [`examples/language`](https://github.com/deepmind/interval-bound-propagation/tree/master/examples/language) 
for [https://arxiv.org/abs/1909.01492](https://arxiv.org/abs/1909.01492).

This is not an official Google product

## Installation

IBP can be installed with the following command:

```bash
pip install git+https://github.com/deepmind/interval-bound-propagation
```

IBP will work with both the CPU and GPU version of tensorflow and dm-sonnet, but
to allow for that it does not list Tensorflow as a requirement, so you need to
install Tensorflow and Sonnet separately if you haven't already done so.

## Usage

The following command trains a small model on MNIST with epsilon set to 0.3:

```bash
cd examples
python train.py --model=small --output_dir=/tmp/small_model
```

## Pretrained Models

Models trained using IBP and CROWN-IBP can be downloaded
[here](https://drive.google.com/open?id=1lovI-fUabgs3swMgIe7MLRvHB9KtjzNT).

### IBP models:

| Dataset  | Test epsilon | Model path                 | Clean accuracy | Verified accuracy | Accuracy under attack |
|----------|--------------|----------------------------|----------------|-------------------|-----------------------|
| MNIST    | 0.1          | ibp/mnist_0.2_medium       | 98.94%         | 97.08%            | 97.99%                |
| MNIST    | 0.2          | ibp/mnist_0.4_large_200    | 98.34%         | 95.47%            | 97.06%                |
| MNIST    | 0.3          | ibp/mnist_0.4_large_200    | 98.34%         | 91.79%            | 96.03%                |
| MNIST    | 0.4          | ibp/mnist_0.4_large_200    | 98.34%         | 84.99%            | 94.56%                |
| CIFAR-10 | 2/255        | ibp/cifar_2-255_large_200  | 70.21%         | 44.12%            | 56.53%                |
| CIFAR-10 | 8/255        | ibp/cifar_8-255_large      | 49.49%         | 31.56%            | 39.53%                |

### CROWN-IBP models:

| Dataset  | Test epsilon | Model path                   | Clean accuracy | Verified accuracy | Accuracy under attack |
|----------|--------------|------------------------------|----------------|-------------------|-----------------------|
| MNIST    | 0.1          | crown-ibp/mnist_0.2_large    | 99.03%         | 97.75%            | 98.34%                |
| MNIST    | 0.2          | crown-ibp/mnist_0.4_large    | 98.38%         | 96.13%            | 97.28%                |
| MNIST    | 0.3          | crown-ibp/mnist_0.4_large    | 98.38%         | 93.32%            | 96.38%                |
| MNIST    | 0.4          | crown-ibp/mnist_0.4_large    | 98.38%         | 87.51%            | 94.95%                |
| CIFAR-10 | 2/255        | crown-ibp/cifar_2-255_large  | 71.52%         | 53.97%            | 59.72%                |
| CIFAR-10 | 8/255        | crown-ibp/cifar_8-255_large  | 47.14%         | 33.30%            | 36.81%                |
| CIFAR-10 | 16/255       | crown-ibp/cifar_16-255_large | 34.19%         | 23.08%            | 26.55%                |

In these tables, we evaluated the verified accuracy using IBP only.
We evaluted the accuracy under attack using a 20-step untargeted PGD attack.
You can evaluate these models yourself using `eval.py`, for example:

```bash
cd examples
python eval.py --model_dir pretrained_models/ibp/mnist_0.4_large_200/ \
  --epsilon 0.3
```

Note that we evaluated the CIFAR-10 2/255 CROWN-IBP model using CROWN-IBP
(instead of pure IBP). You can do so yourself by setting the flag
`--bound_method=crown-ibp`:

```bash
python eval.py --model_dir pretrained_models/crown-ibp/cifar_2-255_large/ \
  --epsilon 0.00784313725490196 --bound_method=crown-ibp
```

## Giving credit

If you use this code in your work, we ask that you cite this paper:

Sven Gowal, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Chongli Qin,
Jonathan Uesato, Relja Arandjelovic, Timothy Mann, and Pushmeet Kohli.
"On the Effectiveness of Interval Bound Propagation for Training Verifiably
Robust Models." _arXiv preprint arXiv:1810.12715 (2018)_.

If you use CROWN-IBP, we also ask that you cite:

Huan Zhang, Hongge Chen, Chaowei Xiao, Sven Gowal, Robert Stanforth, Bo Li,
Duane Boning, Cho-Jui Hsieh.
"Towards Stable and Efficient Training of Verifiably Robust Neural Networks."
_arXiv preprint arXiv:1906.06316 (2019)_.

If you use the sentiment analysis example, please cite:

Po-Sen Huang, Robert Stanforth, Johannes Welbl, Chris Dyer, Dani Yogatama, Sven Gowal, Krishnamurthy Dvijotham, Pushmeet Kohli.
"Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation."
_EMNLP 2019_.


## Acknowledgements

In addition to the people involved in the original IBP publication, we would
like to thank Huan Zhang, Sumanth Dathathri and Johannes Welbl for their
contributions.

