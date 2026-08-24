---
author: Shuhei Watanabe
title: Real-World Multi-Objective Optimization Benchmark Problems (RE Problem Suite)
description: A collection of real-world (single- and multi-objective, constrained and unconstrained) optimization benchmark problems from the RE problem suite. This package is a wrapper of a re-implementation of the reproblems repository.
tags: [benchmark, multi-objective, real-world, constrained optimization, RE]
optuna_versions: [5.0.0]
license: MIT License
---

## Abstract

This package provides the real-world multi-objective optimization benchmark problems (the RE problem suite) introduced in [An Easy-to-use Real-world Multi-objective Problem Suite](https://arxiv.org/abs/2009.12867).
The original benchmark implementation is available [here](https://github.com/ryojitanabe/reproblems).
This package serves as a wrapper for a re-implementation of the original benchmark, ported to Python from the reference C source code (`reproblem.c`).

Note that `ConstrainedProblem` relies on `optuna.trial.Trial.set_constraint`, which requires Optuna v5.0.0 or newer.

## APIs

### class `Problem(problem_name: str)`

- `problem_name`: The name of an unconstrained benchmark problem. Available names are `RE21`, `RE22`, `RE23`, `RE24`, `RE25`, `RE31`, `RE32`, `RE33`, `RE34`, `RE35`, `RE36`, `RE37`, `RE41`, `RE42`, `RE61`, and `RE91`. Note that some of these problems internally reformulate their original constraints into an additional penalty objective, so they remain unconstrained from Optuna's perspective.

#### Method and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: A dictionary representing decision variables like `{"x0": x0_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `self.n_variables`.
  - Returns: List of length `self.n_objectives`.

### class `ConstrainedProblem(problem_name: str)`

- `problem_name`: The name of a constrained benchmark problem. Available names are `CRE21`, `CRE22`, `CRE23`, `CRE24`, `CRE25`, `CRE31`, `CRE32`, and `CRE51`.

#### Method and Properties

- `search_space`: Return the search space.
  - Returns: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Returns: `list[optuna.study.StudyDirection]`
- `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: A dictionary representing decision variables, with the same format as in `Problem.evaluate`.
  - Returns: List of length `self.n_objectives`.
- `evaluate_constraints(params: dict[str, float])`: Evaluate the constraint functions and return the constraint values keyed by their names. A trial is considered feasible when all the values are zero or less.
  - Args:
    - `params`: A dictionary representing the decision variables, with the same format and value range as in `evaluate`.
  - Returns: `dict[str, float]` of length `self.n_constraints`.

The properties and functions of classes in [`reproblem.reproblem_original`](./reproblem_original) are also available, such as `lbound` and `ubound`.

## Installation

```shell
pip install numpy optunahub
```

Or you can install the required packages from optunahub as well.

```shell
pip install -r https://hub.optuna.org/benchmarks/reproblem/requirements.txt
```

## Example

```python
from __future__ import annotations

import optuna
import optunahub


reproblem = optunahub.load_module("benchmarks/reproblem")
problem = reproblem.ConstrainedProblem("CRE21")
study = optuna.create_study(directions=problem.directions)
study.optimize(problem, n_trials=10)

if len(problem.directions) == 1:
    print(study.best_trial)
else:
    print(study.best_trials)
```

## Reference

```bibtex
@article{tanabe2020easy,
  title={An easy-to-use real-world multi-objective optimization problem suite},
  author={Tanabe, Ryoji and Ishibuchi, Hisao},
  journal={Applied Soft Computing},
  volume={89},
  pages={106078},
  year={2020},
  publisher={Elsevier}
}
```
