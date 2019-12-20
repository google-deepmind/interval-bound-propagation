# Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation

Here contains an implementation of
[Achieving Verified Robustness to Symbol Substitutions via Interval Bound 
Propagation](https://arxiv.org/abs/1909.01492).

## Installation

The installation can be done with the following commands:

```bash
pip3 install "tensorflow-gpu<2" "dm-sonnet<2" "tensorflow-probability==0.7.0" "tensorflow-datasets" "absl-py"
pip3 install git+https://github.com/deepmind/interval-bound-propagation
```


## Usage

The following command reproduces the [SST](https://nlp.stanford.edu/sentiment/) 
character level experiments using perturbation radius of 3:

```bash
cd examples/language
python3 robust_train.py
```

You should expect to see the following at the end of training
(note we only use SST dev set only for evaluation here).

```bash
step: 149900, train loss: 0.392112, verifiable train loss: 0.826042,
train accuracy: 0.850000, dev accuracy: 0.747619, test accuracy: 0.747619,
Train Bound = -0.42432, train verified: 0.800,
dev verified: 0.695, test verified: 0.695
best dev acc 0.780952        best test acc   0.780952
best verified dev acc        0.716667        best verified test acc  0.716667
```

We can verify the model in 
`config['model_location']='/tmp/robust_model/checkpoint/final'` using IBP.

For example, after changing `config['delta']=1.`, we can evaluate the IBP 
verified accuracy with perturbation radius of 1:

```bash
python3 robust_train.py --analysis --batch_size=1
```

We expect to see results like the following:

```bash
test final correct: 0.748, verified: 0.722
{'datasplit': 'test', 'nominal': 0.7477064220183486,
'verify': 0.7224770642201835, 'delta': 1.0, 
'num_perturbations': 268,
'model_location': '/tmp/robust_model/checkpoint/final', 'final': True}
```

We can also exhaustively search all valid perturbations to exhaustively verify
the models.

```bash
python3 exhaustive_verification.py --num_examples=0
```

We should expect the following results

```bash
verified_proportion: 0.7350917431192661
{'delta': 1, 'character_level': True, 'mode': 'validation', 'checkpoint_path': '/tmp/robust_model/checkpoint/final', 'verified_proportion': 0.7350917431192661}
```

The IBP verified accuracy ` 0.7224770642201835` is a lower bound of the
exhaustive verification results, `0.7350917431192661`.

Furthermore, we can also align the predictions between the IBP verification 
and exhaustive verification. There should not be cases where IBP can verify 
(no attack can change the predictions) and exhaustive verification cannot 
verify (there exist an attack that can change the predictions), since IBP 
provides a lower bound on the true robustness accuracy (via exhaustive search).


## Reference

If you use this code in your work, please cite the accompanying paper:

```
@inproceedings{huang-2019-achieving,
    title = "Achieving Verified Robustness to Symbol Substitutions via Interval Bound Propagation",
    author = "Po-Sen Huang and
      Robert Stanforth and
      Johannes Welbl and
      Chris Dyer and
      Dani Yogatama and
      Sven Gowal  and
      Krishnamurthy Dvijotham and
      Pushmeet Kohli",
    booktitle = "Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2019",
    pages = "4081--4091",
}
```

## Disclaimer

This is not an official Google product.
