# `pyhcrf`

*A hidden (state) conditional random field (__HCRF__) written in Python and Cython.*

[![TravisCI](https://img.shields.io/travis/althonos/pyhcrf/master.svg?logo=travis&maxAge=600&style=flat-square)](https://travis-ci.com/althonos/pyhcrf/branches)
[![AppVeyor](https://img.shields.io/appveyor/ci/althonos/pyhcrf/master?logo=appveyor&style=flat-square&maxAge=600)](https://ci.appveyor.com/project/althonos/pyhcrf)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/pyhcrf?style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/pyhcrf/)
[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/bsd-2-clause/)
[![PyPI](https://img.shields.io/pypi/v/pyhcrf.svg?style=flat-square&maxAge=600)](https://pypi.org/project/pyhcrf)
[![Wheel](https://img.shields.io/pypi/wheel/pyhcrf.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/pyhcrf/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyhcrf.svg?style=flat-square&maxAge=600)](https://pypi.org/project/pyhcrf/#files)
[![Python Implementations](https://img.shields.io/pypi/implementation/pyhcrf.svg?style=flat-square&maxAge=600)](https://pypi.org/project/pyhcrf/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/pyhcrf/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/pyhcrf.svg?style=flat-square&maxAge=600)](https://github.com/althonos/pyhcrf/issues)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/pyhcrf.py/blob/master/CHANGELOG.md)

This package is a fork of the original `pyhcrf`, written by
[Dirko Coetsee](https://github.com/dirko), featuring Python 3 and Windows support
with a cleaner code base maintained by [Martin Larralde](https://github.com/althonos).

## Overview

`pyhcrf` implements a HCRF model with a [`sklearn`](https://scikit-learn.org/)-inspired
interface. The model classifies sequences according to a latent state sequence.
This package provides methods to learn parameters from example sequences and to
classify new sequences. See the [paper](http://people.csail.mit.edu/sybor/cvpr06_wang.pdf)
by Wang *et al.* and the [report](https://api.semanticscholar.org/CorpusID:61776334)
by Dirko Coetsee.

### States
Each state is numbered `0, 1, ..., num_states - 1`. The state machine starts
in `state 0` and ends in `num_states - 1`. Currently the state transitions are
constrained so that, on each element in the input sequence, the state machine
either stays in the current state or advances to a state represented by the
next number.  This default can be changed by setting the `transitions` and
corresponding `transition_parameters` properties.

### Dependencies
`pyhcrf` depends on `numpy`, `scipy` (for the LM-BFGS optimiser and `scipy.sparse`),
and also needs `cython` for building from source.

## Example

```python
X = [array([[ 1. , -0.82683403,  2.48881337],
            [ 1. , -1.07491808,  1.55848197],
            [ 1. ,  6.7814359 ,  4.01074595]]),
     array([[ 1. , -3.01165932, -2.15972362],
            [ 1. , -3.41449473, -2.2668825 ]]),
     array([[ 1. , -2.64921323, -1.20159641],
            [ 1. ,  0.31139394,  1.58841159]]),
     array([[ 1. ,  5.85226017,  2.43317499],
            [ 1. , -1.57598266, -2.07585778]]),
     array([[ 1. , -0.32999744, -2.70695361],
            [ 1. ,  0.44311988,  0.36400733]]),
     array([[ 1. , -0.05301562,  3.95424435],
            [ 1. ,  3.04540498, -3.25040276]]),
     array([[ 1. , -4.29117715,  0.9167861 ],
            [ 1. , -3.22775884,  1.83277224]]),
     array([[ 1. , -2.80856491,  1.95630489],
            [ 1. ,  1.62290542, -0.7457237 ]]),
     array([[ 1. , -2.32682366,  2.60844469],
            [ 1. ,  2.12320609,  1.04483217]]),
     array([[ 1. , -4.17616178,  4.09969658],
            [ 1. ,  0.67287935, -5.67652159]])]

y = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0]
```

![Training examples](https://raw.githubusercontent.com/althonos/pyhcrf/master/static/training_examples.png)

```python
>>> from pyhcrf import HCRF
>>> from sklearn.metrics import confusion_matrix

>>> model = HCRF(num_states=3, random_seed=3, optimizer_kwargs={'maxfun':200})
>>> model.fit(X, y)
>>> pred = model.predict(X)
>>> confusion_matrix(y, pred)
array([[12,  0],
      [ 0,  8]])
```


## Installation
Download/clone and run

```
python setup.py build_ext --inplace
python setup.py install
```


## License

The original code, and all contributions made subsequently in this fork, are
licensed with BSD-2-Clause.
