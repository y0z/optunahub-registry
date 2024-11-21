from __future__ import annotations

from typing import Any
from typing import Sequence

import cocoex as ex
import optuna
import optunahub


class Problem(optunahub.benchmarks.BaseProblem):
    """Wrapper class for COCO bbob test suite.
    https://numbbo.github.io/coco/testsuites/bbob

    1 Separable Functions
        f1: Sphere Function
        f2: Separable Ellipsoidal Function
        f3: Rastrigin Function
        f4: Büche-Rastrigin Function
        f5: Linear Slope
    2 Functions with low or moderate conditioning
        f6: Attractive Sector Function
        f7: Step Ellipsoidal Function
        f8: Rosenbrock Function, original
        f9: Rosenbrock Function, rotated
    3 Functions with high conditioning and unimodal
        f10: Ellipsoidal Function
        f11: Discus Function
        f12: Bent Cigar Function
        f13: Sharp Ridge Function
        f14: Different Powers Function
    4 Multi-modal functions with adequate global structure
        f15: Rastrigin Function
        f16: Weierstrass Function
        f17: Schaffer's F7 Function
        f18: Schaffer's F7 Function, moderately ill-conditioned
        f19: Composite Griewank-Rosenbrock Function F8F2
    5 Multi-modal functions with weak global structure
        f20: Schwefel Function
        f21: Gallagher's Gaussian 101-me Peaks Function
        f22: Gallagher's Gaussian 21-hi Peaks Function
        f23: Katsuura Function
        f24: Lunacek bi-Rastrigin Function
    """

    def __init__(self, function_id: int, dimension: int, instance_id: int = 1):
        """Initialize the problem.
        Args:
            function_id: Function index in [1, 24].
            dimension: Dimension of the problem in [2, 3, 5, 10, 20, 40].
            instance_id: Instance index in [1, 15].

        Please refer to the COCO documentation for the details of the available properties.
        https://numbbo.github.io/coco-doc/apidocs/cocoex/cocoex.Problem.html
        """

        assert 1 <= function_id <= 24, "function_id must be in [1, 24]"
        assert dimension in [2, 3, 5, 10, 20, 40], "dimension must be in [2, 3, 5, 10, 20, 40]"
        assert 1 <= instance_id <= 15, "instance_id must be in [1, 15]"

        self._problem = ex.Suite(
            "bbob",
            "",
            f"function_indices:{function_id} dimensions:{dimension} instance_indices:{instance_id}",
        )[0]

    # BaseProblem properties
    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return {
            f"x{i}": optuna.distributions.FloatDistribution(
                low=self._problem.lower_bounds[i],
                high=self._problem.upper_bounds[i],
            )
            for i in range(self._problem.dimension)
        }

    @property
    def directions(self) -> Sequence[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE]

    def evaluate(self, params: dict[str, float]) -> float:
        """Evaluate the objective function.
        Args:
            params: Decision variable.
        Returns:
            The objective value.
        """
        return self._problem(list(params.values()))

    # Additional problem-specific properties
    @property
    def coco_problem(self) -> ex.Problem:
        """Return the COCO problem instance."""
        return self._problem

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)
