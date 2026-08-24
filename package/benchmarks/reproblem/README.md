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
- `metric_names`: Return the objective names in the order returned by `evaluate`.
  - Returns: `list[str]` of length `self.n_objectives`.
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
- `metric_names`: Return the objective names in the order returned by `evaluate`.
  - Returns: `list[str]` of length `self.n_objectives`.
- `constraint_names`: Return the constraint names used as the keys of `evaluate_constraints`. This property is only available in `ConstrainedProblem`.
  - Returns: `list[str]` of length `self.n_constraints`.
- `evaluate(params: dict[str, float])`: Evaluate the objective function given a dictionary of parameters.
  - Args:
    - `params`: A dictionary representing decision variables, with the same format as in `Problem.evaluate`.
  - Returns: List of length `self.n_objectives`.
- `evaluate_constraints(params: dict[str, float])`: Evaluate the constraint functions and return the constraint values keyed by their names. A trial is considered feasible when all the values are zero or less.
  - Args:
    - `params`: A dictionary representing the decision variables, with the same format and value range as in `evaluate`.
  - Returns: `dict[str, float]` of length `self.n_constraints`.

The properties and functions of classes in [`reproblem.reproblem_original`](./reproblem_original) are also available, such as `lbound` and `ubound`.

## Objective and Constraint Names

Each RE problem models a different engineering problem, so unlike a synthetic suite, every problem has its own objectives.
The names below carry the `f_i` and `g_j` indices used by the [supplementary file](https://github.com/ryojitanabe/reproblems/blob/master/doc/re-supplementary_file.pdf), which defines each problem, so `metric_names` and `constraint_names` can be read directly against it.
Which names a problem uses can also be inspected at runtime via `problem.metric_names` and `problem.constraint_names`.

As Table 1 of the paper shows, an unconstrained problem and its constrained counterpart are derived from the same original problem, e.g. `RE31` and `CRE21` are both the two bar truss design problem, and the pair shares the original objectives listed below.
The two variants differ in how they treat the original constraints: `Problem` folds them into an extra aggregated objective, whereas `ConstrainedProblem` exposes them through `evaluate_constraints`.

### Objectives (all minimized)

Since the paper maximizes the annual cargo transport capacity of the conceptual marine design problem, the corresponding objective is negated and prefixed with `negative_`.

| Original problem         | `Problem` | `ConstrainedProblem` | Objective names shared by the pair                                                                                                                              |
| ------------------------ | --------- | -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Four bar truss           | `RE21`    | -                    | `f1_structural_volume`, `f2_joint_displacement`                                                                                                                 |
| Reinforced concrete beam | `RE22`    | -                    | `f1_total_cost`                                                                                                                                                 |
| Pressure vessel          | `RE23`    | -                    | `f1_total_cost`                                                                                                                                                 |
| Hatch cover              | `RE24`    | -                    | `f1_weight`                                                                                                                                                     |
| Coil compression spring  | `RE25`    | -                    | `f1_volume`                                                                                                                                                     |
| Two bar truss            | `RE31`    | `CRE21`              | `f1_structural_weight`, `f2_resultant_joint_displacement`                                                                                                       |
| Welded beam              | `RE32`    | `CRE22`              | `f1_cost`, `f2_end_deflection`                                                                                                                                  |
| Disc brake               | `RE33`    | `CRE23`              | `f1_brake_mass`, `f2_minimum_stopping_time`                                                                                                                     |
| Vehicle crashworthiness  | `RE34`    | -                    | `f1_weight`, `f2_acceleration_characteristics`, `f3_toe_board_intrusion`                                                                                        |
| Speed reducer            | `RE35`    | `CRE24`              | `f1_volume`, `f2_gear_shaft_stress`                                                                                                                             |
| Gear train               | `RE36`    | `CRE25`              | `f1_gear_ratio_error`, `f2_max_gear_size`                                                                                                                       |
| Rocket injector          | `RE37`    | -                    | `f1_max_injector_face_temperature`, `f2_inlet_distance`, `f3_max_post_tip_temperature`                                                                          |
| Car side impact          | `RE41`    | `CRE31`              | `f1_car_weight`, `f2_pubic_force`, `f3_v_pillar_average_velocity`                                                                                               |
| Conceptual marine        | `RE42`    | `CRE32`              | `f1_transportation_cost`, `f2_light_ship_weight`, `f3_negative_annual_cargo_transport_capacity`                                                                 |
| Water resource planning  | `RE61`    | `CRE51`              | `f1_drainage_network_cost`, `f2_storage_facility_cost`, `f3_treatment_facility_cost`, `f4_expected_flood_damage_cost`, `f5_expected_economic_loss_due_to_flood` |
| Car cab                  | `RE91`    | -                    | `f1_car_weight`, `f2_g1_violation`, `f3_g2_violation`, ..., `f9_g8_violation`                                                                                   |

In `Problem`, the original constraints are reformulated into one extra objective that sums their violations, named `f{n}_total_constraint_violation` where `n` is `self.n_objectives`.
For example, `Problem("RE31").metric_names` is `["f1_structural_weight", "f2_resultant_joint_displacement", "f3_total_constraint_violation"]`.
The four problems whose original formulation has no constraint, namely `RE21`, `RE34`, `RE37`, and `RE91`, expose only the objectives listed above.
`RE91` is a special case: instead of aggregating the folded constraints, it keeps each of them as its own objective, so its violation objectives are named individually after the constraint they come from.

### Constraints (feasible when zero or less)

The supplementary file defines each constraint only by its formula and does not name it, so the constraint names are the `g_j` indices themselves.
Refer to the section of the supplementary file for the corresponding original problem to look up what a constraint means.

| `ConstrainedProblem` | `constraint_names`                                                 |
| -------------------- | ------------------------------------------------------------------ |
| `CRE21`              | `g1`, `g2`, `g3`                                                   |
| `CRE22`              | `g1`, `g2`, `g3`, `g4`                                             |
| `CRE23`              | `g1`, `g2`, `g3`, `g4`                                             |
| `CRE24`              | `g1`, `g2`, `g3`, `g4`, `g5`, `g6`, `g7`, `g8`, `g9`, `g10`, `g11` |
| `CRE25`              | `g1`                                                               |
| `CRE31`              | `g1`, `g2`, `g3`, `g4`, `g5`, `g6`, `g7`, `g8`, `g9`, `g10`        |
| `CRE32`              | `g1`, `g2`, `g3`, `g4`, `g5`, `g7`, `g6`, `g8`, `g9`               |
| `CRE51`              | `g1`, `g2`, `g3`, `g4`, `g5`, `g6`, `g7`                           |

Note that `evaluate_constraints` preserves the order of the original implementation, which is not sorted by the constraint index for `CRE32`, as the table above shows.

Note also that the original implementation returns the *violation* of each constraint rather than the constraint function value itself.
The supplementary file writes every constraint as `g_j(x) >= 0`, and the implementation converts it into `max(-g_j(x), 0)`, which is zero exactly when the constraint is satisfied and positive otherwise.
The returned values are therefore always zero or greater, and a trial is feasible when every value is zero, which is consistent with Optuna's convention that a constraint is satisfied when its value is zero or less.

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
