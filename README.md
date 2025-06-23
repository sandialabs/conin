[![GitHub Actions Status](https://github.com/sandialabs/conin/actions/workflows/conda.yml/badge.svg?branch=main&event=push)](https://github.com/sandialabs/conin/actions/workflows/conda.yml?query=branch%3Amain+event%3Apush)
[![codecov](https://codecov.io/gh/sandialabs/conin/branch/main/graph/badge.svg)](https://codecov.io/gh/sandialabs/conin)
[![Documentation Status](https://readthedocs.org/projects/conin/badge/?version=latest)](http://conin.readthedocs.org/en/latest/)
[![GitHub contributors](https://img.shields.io/github/contributors/sandialabs/conin.svg)](https://github.com/sandialabs/conin/graphs/contributors)
[![Merged PRs](https://img.shields.io/github/issues-pr-closed-raw/sandialabs/conin.svg?label=merged+PRs)](https://github.com/sandialabs/conin/pulls?q=is:pr+is:merged)

# conin

A python library that supports the constrained analysis of probabilistic graphical models

## Overview

Conin supports constrained inference and learning for hidden Markov models, Bayesian networks, dynamic Bayesian networks and Markov networks. Conin interfaces with the pgmpy python library for the specification of general probabilistic graphical models. Additionally, it interfaces with a variety of optimization solvers to support learning and inference.

## Testing

Conin tests can be executed using pytest:

```
cd conin
pytest .
```

If the pytest-cov package is installed, pytest can provide coverage statistics:

```
cd conin
pytest --cov=conin .
```

The following options list the lines that are missing from coverage tests:
```
cd conin
pytest --cov=conin --cov-report term-missing .
```

Note that pytest coverage includes coverage of test files themselves.  This gives a somewhat skewed sense of coverage for the code base, but it helps identify tests that are omitted or not executed completely.
