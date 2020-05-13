# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]
[Unreleased]: https://github.com/althonos/pyrodigal/compare/3441ac4...HEAD
### Added
- Python 3 support.
- Proper `setuptools` configuration file listing all dependencies.
- Proper legal disclaimer about this project being GPLv3.
### Changed
- Renamed `Hcrf` class to `HCRF` to follow Python class naming conventions.
- Renamed `HCRF.predict_proba` method to `HCRF.predict_marginals` to follow
  `sklearn` conventions.
### Fixed
- Linking issues with NumPy preventing import in Python 3
  ([dirko/pyhcrf#7](https://github.com/dirko/pyhcrf/issues/7)).