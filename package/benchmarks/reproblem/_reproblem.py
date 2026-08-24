from __future__ import annotations

from typing import Any
from typing import Protocol
from typing import TYPE_CHECKING

import numpy as np
import optuna
import optunahub

from .reproblem_original import problem


class BaseUnconstrainedREBenchmark(Protocol):
    problem_name: str
    n_objectives: int  # Number of objectives
    n_variables: int  # Number of parameters
    lbound: np.ndarray  # Lower bound of each parameter
    ubound: np.ndarray  # Upper bound of each parameter

    def __init__(self) -> None:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BaseConstrainedREBenchmark(Protocol):
    problem_name: str
    n_objectives: int  # Number of objectives
    n_variables: int  # Number of parameters
    n_constraints: int  # Number of constraints (always positive)
    lbound: np.ndarray  # Lower bound of each parameter
    ubound: np.ndarray  # Upper bound of each parameter

    def __init__(self) -> None:
        raise NotImplementedError

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


unconstrained_problem_names = [
    "RE21",
    "RE22",
    "RE23",
    "RE24",
    "RE25",
    "RE31",
    "RE32",
    "RE33",
    "RE34",
    "RE35",
    "RE36",
    "RE37",
    "RE41",
    "RE42",
    "RE61",
    "RE91",
]

constrained_problem_names = [
    "CRE21",
    "CRE22",
    "CRE23",
    "CRE24",
    "CRE25",
    "CRE31",
    "CRE32",
    "CRE51",
]

unconstrained_problems: dict[str, type[BaseUnconstrainedREBenchmark]] = {
    name: getattr(problem, name) for name in unconstrained_problem_names
}

constrained_problems: dict[str, type[BaseConstrainedREBenchmark]] = {
    name: getattr(problem, name) for name in constrained_problem_names
}


class Problem(optunahub.benchmarks.BaseProblem):
    def __init__(self, problem_name: str) -> None:
        """Initialize the problem.
        Args:
            problem_name: Name of problem.

        Please refer to the reproblem repository for the details.
        https://github.com/ryojitanabe/reproblems
        """
        if problem_name not in unconstrained_problems:
            raise ValueError(
                f"problem_name must be in {list(unconstrained_problems.keys())}, "
                f"but got {problem_name}."
            )

        self.problem_name = problem_name

        self._problem = unconstrained_problems[problem_name]()

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(low, high)
            for i, (low, high) in enumerate(zip(self._problem.lbound, self._problem.ubound))
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.n_objectives

    def evaluate(self, params: dict[str, float]) -> list[float]:
        x = np.array([params[name] for name in self._search_space])
        return self._problem.evaluate(x).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)


class ConstrainedProblem(optunahub.benchmarks.BaseProblem):
    def __init__(self, problem_name: str) -> None:
        """Initialize the problem.
        Args:
            problem_name: Name of problem.

        Please refer to the reproblem repository for the details.
        https://github.com/ryojitanabe/reproblems
        """
        if problem_name not in constrained_problems:
            raise ValueError(
                f"problem_name must be in {list(constrained_problems.keys())}, "
                f"but got {problem_name}."
            )

        self.problem_name = problem_name

        self._problem = constrained_problems[problem_name]()

        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(low, high)
            for i, (low, high) in enumerate(zip(self._problem.lbound, self._problem.ubound))
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.n_objectives

    def evaluate(self, params: dict[str, float]) -> list[float]:
        x = np.array([params[name] for name in self._search_space])
        f, _ = self._problem.evaluate(x)
        return f.tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def evaluate_constraints(self, params: dict[str, float]) -> dict[str, float]:
        x = np.array([params[name] for name in self._search_space])
        _, g = self._problem.evaluate(x)
        return {f"g{i}": value for i, value in enumerate(g)}


if TYPE_CHECKING:
    for constrained_problem_cls in constrained_problems.values():
        _0: BaseConstrainedREBenchmark = constrained_problem_cls()
    for unconstrained_problem_cls in unconstrained_problems.values():
        _1: BaseUnconstrainedREBenchmark = unconstrained_problem_cls()
