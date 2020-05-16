# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
[Unreleased]: https://github.com/althonos/pyhcrf/compare/v0.1.0...HEAD

## [v0.1.0] - 2020-05-16
[v0.1.0]: https://github.com/althonos/pyhcrf/compare/3441ac4...v0.1.0
### Added
- Python 3 support.
- Proper `setuptools` configuration file listing all dependencies.
- Proper legal disclaimer about this project being GPLv3.
- Implementation of L1 regularization in the cost function.
### Changed
- Renamed `Hcrf` class to `HCRF` to follow Python class naming conventions.
- Renamed `HCRF.predict_proba` method to `HCRF.predict_marginals` to follow
  `sklearn` conventions.
- Made `HCRF.predict_marginals` return a list of dictionaries like
  `sklearn-crfsuite`, where keys are classes, and values are predicted
  probabilities.
### Fixed
- Linking issues with NumPy preventing import in Python 3
  ([dirko/pyhcrf#7](https://github.com/dirko/pyhcrf/issues/7)).
- Jacobian gradient vector being ignored in minimization step and having to
  be approximated by SciPy, causing longer training time.
- Gradient state and transition parameter buffers being allocated within
  the training loop.
