---
author: Yoshihiko Ozaki
title: Plot Hypervolume History for Multiple Studies
description: Plot hypervolume history for multiple studies to compare their optimization performance.
tags: [visualization, multi-objective optimization, hypervolume]
optuna_versions: [3.6.1]
license: MIT License
---

## Class or Function Names

- plot_optimization_history

## Example

```python
import optuna
import optunahub


def objective(trial: optuna.Trial) -> tuple[float, float]:
    x = trial.suggest_float("x", 0, 5)
    y = trial.suggest_float("y", 0, 3)

    v0 = 4 * x**2 + 4 * y**2
    v1 = (x - 5) ** 2 + (y - 5) ** 2
    return v0, v1


samplers = [
    optuna.samplers.RandomSampler(),
    optuna.samplers.TPESampler(),
    optuna.samplers.NSGAIISampler(),
]
studies = []
for sampler in samplers:
    study = optuna.create_study(
        sampler=sampler,
        study_name=f"{sampler.__class__.__name__}",
        directions=["minimize", "minimize"],
    )
    study.optimize(objective, n_trials=50)
    studies.append(study)

reference_point = [100.0, 50.0]

m = optunahub.load_module(
    "visualization/plot_hypervolume_history_multi"
)
fig = m.plot_hypervolume_history(studies, reference_point)
fig.show()
```

![Example](images/example.png "Example")
