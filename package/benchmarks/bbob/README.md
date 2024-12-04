---
author: Optuna team
title: The blackbox optimization benchmarking (bbob) test suite
description: The blackbox optimization benchmarking (bbob) test suite consists of 24 noiseless single-objective test functions including Sphere, Ellipsoidal, Rastrigin, Rosenbrock, etc. This package is a wrapper of the COCO (COmparing Continuous Optimizers) experiments library.
tags: [benchmarks, continuous optimization, BBOB, COCO]
optuna_versions: [4.1.0]
license: MIT License
---

## Abstract

The blackbox optimization benchmarking (bbob) test suite comprises 24 noiseless single-objective test functions. BBOB is one of the most widely used test suites to evaluate and compare the performance of blackbox optimization algorithms. Each benchmark function is provided in dimensions \[2, 3, 5, 10, 20, 40\] with 15 instances.

## APIs

### class `Problem(function_id: int, dimension: int, instance_id: int = 1)`

- `function_id`: [ID of the bbob benchmark function](https://numbbo.github.io/coco/testsuites/bbob) to use. It must be in the range of `[1, 24]`.
- `dimension`: Dimension of the benchmark function. It must be in `[2, 3, 5, 10, 20, 40]`.
- `instance_id`: ID of the instance of the benchmark function. It must be in the range of `[1, 110]`.

#### Methods and Properties

- `search_space`: Return the search space.
  - Return type: `dict[str, optuna.distributions.BaseDistribution]`
- `directions`: Return the optimization directions.
  - Return type: `list[optuna.study.StudyDirection]`
- `__call__(trial: optuna.Trial)`: Evaluate the objective function and return the objective value.
  - Parameters:
    - `trial`: Optuna trial object.
  - Return type: `float`
- `evaluate(params: dict[str, float])`: Evaluate the objective function.
  - Parameters:
    - `params`: Decision variable like `{"x0": x1_value, "x1": x1_value, ..., "xn": xn_value}`. The number of parameters must be equal to `dimension`.
  - Return type: `float`

It is also available that properties defiend in [cocoex.Problem](https://numbbo.github.io/coco-doc/apidocs/cocoex/cocoex.Problem.html) such as `number_of_objectives`.

## Installation

Please install the [coco-experiment](https://github.com/numbbo/coco-experiment/tree/main/build/python) package.

```shell
pip install -U coco-experiment
```

## Example

```python
import optuna
import optunahub


bbob = optunahub.load_module("benchmarks/bbob")
sphere2d = bbob.Problem(function_id=1, dimension=2, instance_id=1)

study = optuna.create_study(directions=sphere2d.directions)
study.optimize(sphere2d, n_trials=20)

print(study.best_trial.params, study.best_trial.value)
```

## List of Benchmark Functions

Please refer to [the paper](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf) for details about each benchmark function.

**Category**

1. Separable Functions
1. Functions with low or moderate conditioning
1. Functions with high conditioning and unimodal
1. Multi-modal functions with adequate global structure
1. Multi-modal functions with weak global structure

| Category  | Function ID | Function Name                                                                                         |
|-----------|-------------|-------------------------------------------------------------------------------------------------------|
| 1         | 1           | [Sphere Function](https://numbbo.it/bbob/functions/f01.html)                                          |
| 1         | 2           | [Separable Ellipsoidal Function](https://numbbo.it/bbob/functions/f02.html)                           |
| 1         | 3           | [Rastrigin Function](https://numbbo.it/bbob/functions/f03.html)                                       |
| 1         | 4           | [Büche-Rastrigin Function](https://numbbo.it/bbob/functions/f04.html)                                 |
| 1         | 5           | [Linear Slope](https://numbbo.it/bbob/functions/f05.html)                                             |
| 2         | 6           | [Attractive Sector Function](https://numbbo.it/bbob/functions/f06.html)                               |
| 2         | 7           | [Step Ellipsoidal Function](https://numbbo.it/bbob/functions/f07.html)                                |
| 2         | 8           | [Rosenbrock Function, original](https://numbbo.it/bbob/functions/f08.html)                            |
| 2         | 9           | [Rosenbrock Function, rotated](https://numbbo.it/bbob/functions/f09.html)                             |
| 3         | 10          | [Ellipsoidal Function](https://numbbo.it/bbob/functions/f10.html)                                     |
| 3         | 11          | [Discus Function](https://numbbo.it/bbob/functions/f11.html)                                          |
| 3         | 12          | [Bent Cigar Function](https://numbbo.it/bbob/functions/f12.html)                                      |
| 3         | 13          | [Sharp Ridge Function](https://numbbo.it/bbob/functions/f13.html)                                     |
| 3         | 14          | [Different Powers Function](https://numbbo.it/bbob/functions/f14.html)                                |
| 4         | 15          | [Rastrigin Function](https://numbbo.it/bbob/functions/f15.html)                                       |
| 4         | 16          | [Weierstrass Function](https://numbbo.it/bbob/functions/f16.html)                                     |
| 4         | 17          | [Schaffer's F7 Function](https://numbbo.it/bbob/functions/f17.html)                                   |
| 4         | 18          | [Schaffer's F7 Function, moderately ill-conditioned](https://numbbo.it/bbob/functions/f18.html)       |
| 4         | 19          | [Composite Griewank-Rosenbrock Function F8F2](https://numbbo.it/bbob/functions/f19.html)              |
| 5         | 20          | [Schwefel Function](https://numbbo.it/bbob/functions/f20.html)                                        |
| 5         | 21          | [Gallagher's Gaussian 101-me Peaks Function](https://numbbo.it/bbob/functions/f21.html)               |
| 5         | 22          | [Gallagher's Gaussian 21-hi Peaks Function](https://numbbo.it/bbob/functions/f22.html)                |
| 5         | 23          | [Katsuura Function](https://numbbo.it/bbob/functions/f23.html)                                        |
| 5         | 24          | [Lunacek bi-Rastrigin Function](https://numbbo.it/bbob/functions/f24.html)                            |

![BBOB Plots](images/bbob.png)

## Reference

Finck, S., Hansen, N., Ros, R., & Auger, A. [Real-Parameter Black-Box Optimization Benchmarking 2009: Presentation of the Noiseless Functions](https://numbbo.github.io/gforge/downloads/download16.00/bbobdocfunctions.pdf).
