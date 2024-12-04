from __future__ import annotations

from typing import Any
from typing import Sequence

import cocoex as ex
import optuna
import optunahub


class Problem(optunahub.benchmarks.ConstrainedMixIn, optunahub.benchmarks.BaseProblem):
    """Wrapper class for COCO bbob-constrained test suite.
    https://numbbo.it/bbob/overview.html

    The detail is described in the following paper.
    Paul Dufossé, Nikolaus Hansen, Dimo Brockhoff, Phillipe R. Sampaio, Asma Atamna, and Anne Auger.
    Building scalable test problems for benchmarking constrained optimizers. 2022. To be submitted to the SIAM Journal of Optimization.
    https://numbbo.github.io/coco-doc/bbob-constrained/functions.pdf
    """

    def __init__(self, function_id: int, dimension: int, instance_id: int = 1):
        """Initialize the problem.
        Args:
            function_id: Function index in [1, 54].
            dimension: Dimension of the problem in [2, 3, 5, 10, 20, 40].
            instance_id: Instance index in [1, 15].

        Please refer to the COCO documentation for the details of the available properties.
        https://numbbo.github.io/coco-doc/apidocs/cocoex/cocoex.Problem.html
        """

        assert 1 <= function_id <= 54, "function_id must be in [1, 54]"
        assert dimension in [2, 3, 5, 10, 20, 40], "dimension must be in [2, 3, 5, 10, 20, 40]"
        assert 1 <= instance_id <= 15, "instance_id must be in [1, 15]"

        self._problem = ex.Suite(
            "bbob-constrained", "", ""
        ).get_problem_by_function_dimension_instance(
            function=function_id, dimension=dimension, instance=instance_id
        )
        self._search_space = {
            f"x{i}": optuna.distributions.FloatDistribution(
                low=self._problem.lower_bounds[i],
                high=self._problem.upper_bounds[i],
            )
            for i in range(self._problem.dimension)
        }

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space

    @property
    def directions(self) -> Sequence[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        """Evaluate the objective function.
        Args:
            params:
                Decision variable, e.g., evaluate({"x0": 1.0, "x1": 2.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The objective value.

        """
        return self._problem(list(params.values()))

    def evaluate_constraints(self, params: dict[str, float]) -> Sequence[float]:
        """Evaluate the constraint functions.
        Args:
            params:
                Decision variable, e.g., evaluate_constraints({"x0": 1.0, "x1": 2.0}).
                The number of parameters must be equal to the dimension of the problem.
        Returns:
            The constraint functions values.

        """
        return self._problem.constraint(list(params.values())).tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def __del__(self) -> None:
        self._problem.free()
