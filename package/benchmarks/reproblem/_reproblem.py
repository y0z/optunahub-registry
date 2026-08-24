from __future__ import annotations

from typing import Any
from typing import NamedTuple
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


# The objective of an unconstrained problem that aggregates the constraints folded into it.
_TOTAL_CONSTRAINT_VIOLATION = "total_constraint_violation"


class _ProblemInfo(NamedTuple):
    original_name: str  # Name of the original problem, as in Table 1 of the paper.
    unconstrained_name: str
    constrained_name: str | None  # `None` when the suite has no constrained counterpart.
    objective_names: tuple[str, ...]  # The original objectives shared by both problems.
    constraint_indices: tuple[int, ...]  # The `g_j` indices in the order `evaluate` returns them.


# Table 1 of the paper (https://arxiv.org/abs/2009.12867) derives an unconstrained problem and its
# constrained counterpart from the same original engineering problem, e.g. both RE31 and CRE21 are
# the two bar truss design problem, so each row below covers such a pair at once. A pair shares the
# original problem name and the original objectives, and the unconstrained problem additionally
# folds the constraints into an aggregated violation objective, while the constrained problem
# exposes them as constraints.
# The objective names carry the `f_i` index used by the problem definitions in the supplementary
# file (https://github.com/ryojitanabe/reproblems/blob/master/doc/re-supplementary_file.pdf), and
# `negative_` is prefixed to the quantities that the paper maximizes, because the original
# implementation negates them so that every objective is minimized.
_PROBLEM_INFO_TABLE: list[_ProblemInfo] = [
    _ProblemInfo(
        "FourBarTruss",
        "RE21",
        None,
        ("f1_structural_volume", "f2_joint_displacement"),
        (),
    ),
    _ProblemInfo("ReinforcedConcreteBeam", "RE22", None, ("f1_total_cost",), (1, 2)),
    _ProblemInfo("PressureVessel", "RE23", None, ("f1_total_cost",), (1, 2, 3)),
    _ProblemInfo("HatchCover", "RE24", None, ("f1_weight",), (1, 2, 3, 4)),
    _ProblemInfo("CoilCompressionSpring", "RE25", None, ("f1_volume",), (1, 2, 3, 4, 5, 6)),
    _ProblemInfo(
        "TwoBarTruss",
        "RE31",
        "CRE21",
        ("f1_structural_weight", "f2_resultant_joint_displacement"),
        (1, 2, 3),
    ),
    _ProblemInfo(
        "WeldedBeam",
        "RE32",
        "CRE22",
        ("f1_cost", "f2_end_deflection"),
        (1, 2, 3, 4),
    ),
    _ProblemInfo(
        "DiscBrake",
        "RE33",
        "CRE23",
        ("f1_brake_mass", "f2_minimum_stopping_time"),
        (1, 2, 3, 4),
    ),
    _ProblemInfo(
        "VehicleCrashworthiness",
        "RE34",
        None,
        ("f1_weight", "f2_acceleration_characteristics", "f3_toe_board_intrusion"),
        (),
    ),
    _ProblemInfo(
        "SpeedReducer",
        "RE35",
        "CRE24",
        ("f1_volume", "f2_gear_shaft_stress"),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
    ),
    _ProblemInfo(
        "GearTrain",
        "RE36",
        "CRE25",
        ("f1_gear_ratio_error", "f2_max_gear_size"),
        (1,),
    ),
    _ProblemInfo(
        "RocketInjector",
        "RE37",
        None,
        (
            "f1_max_injector_face_temperature",
            "f2_inlet_distance",
            "f3_max_post_tip_temperature",
        ),
        (),
    ),
    _ProblemInfo(
        "CarSideImpact",
        "RE41",
        "CRE31",
        ("f1_car_weight", "f2_pubic_force", "f3_v_pillar_average_velocity"),
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ),
    # The original implementation of the conceptual marine design problem returns the seventh
    # constraint before the sixth one, so `constraint_indices` is not sorted for this pair.
    _ProblemInfo(
        "ConceptualMarine",
        "RE42",
        "CRE32",
        (
            "f1_transportation_cost",
            "f2_light_ship_weight",
            "f3_negative_annual_cargo_transport_capacity",
        ),
        (1, 2, 3, 4, 5, 7, 6, 8, 9),
    ),
    _ProblemInfo(
        "WaterResourcePlanning",
        "RE61",
        "CRE51",
        (
            "f1_drainage_network_cost",
            "f2_storage_facility_cost",
            "f3_treatment_facility_cost",
            "f4_expected_flood_damage_cost",
            "f5_expected_economic_loss_due_to_flood",
        ),
        (1, 2, 3, 4, 5, 6, 7),
    ),
    # Unlike the other problems, the car cab design problem keeps each folded constraint as its own
    # objective instead of aggregating them, so its violation objectives are named individually.
    _ProblemInfo(
        "CarCab",
        "RE91",
        None,
        ("f1_car_weight",) + tuple(f"f{i + 1}_g{i}_violation" for i in range(1, 9)),
        (),
    ),
]

_ORIGINAL_NAMES: dict[str, str] = {
    problem_name: info.original_name
    for info in _PROBLEM_INFO_TABLE
    for problem_name in (info.unconstrained_name, info.constrained_name)
    if problem_name is not None
}


def _build_name_tables() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Map every problem name to its objective names and, if constrained, its constraint names."""
    metric_names: dict[str, tuple[str, ...]] = {}
    constraint_names: dict[str, tuple[str, ...]] = {}
    for info in _PROBLEM_INFO_TABLE:
        objective_names = info.objective_names
        if info.constraint_indices:
            index = len(objective_names) + 1
            objective_names += (f"f{index}_{_TOTAL_CONSTRAINT_VIOLATION}",)

        metric_names[info.unconstrained_name] = objective_names
        if info.constrained_name is None:
            continue

        # The constrained counterpart exposes the constraints instead of folding them in, so it
        # keeps only the original objectives.
        metric_names[info.constrained_name] = info.objective_names
        constraint_names[info.constrained_name] = tuple(
            f"g{index}" for index in info.constraint_indices
        )

    return metric_names, constraint_names


_METRIC_NAMES, _CONSTRAINT_NAMES = _build_name_tables()


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
        self._metric_names = list(_METRIC_NAMES[problem_name])
        n_objectives = self._problem.n_objectives
        assert len(self._metric_names) == n_objectives, (
            f"{problem_name} must have {n_objectives} objectives."
        )

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.n_objectives

    @property
    def metric_names(self) -> list[str]:
        """Return the objective names in the order returned by ``evaluate``."""
        return self._metric_names.copy()

    @property
    def original_problem_name(self) -> str:
        """Return the name of the original problem, e.g. ``FourBarTruss`` for ``RE21``."""
        return _ORIGINAL_NAMES[self.problem_name]

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
        self._metric_names = list(_METRIC_NAMES[problem_name])
        n_objectives = self._problem.n_objectives
        assert len(self._metric_names) == n_objectives, (
            f"{problem_name} must have {n_objectives} objectives."
        )

        self._constraint_names = list(_CONSTRAINT_NAMES[problem_name])
        n_constraints = self._problem.n_constraints
        assert len(self._constraint_names) == n_constraints, (
            f"{problem_name} must have {n_constraints} constraints."
        )

    @property
    def search_space(self) -> dict[str, optuna.distributions.BaseDistribution]:
        """Return the search space."""
        return self._search_space.copy()

    @property
    def directions(self) -> list[optuna.study.StudyDirection]:
        """Return the optimization directions."""
        return [optuna.study.StudyDirection.MINIMIZE] * self._problem.n_objectives

    @property
    def metric_names(self) -> list[str]:
        """Return the objective names in the order returned by ``evaluate``."""
        return self._metric_names.copy()

    @property
    def constraint_names(self) -> list[str]:
        """Return the constraint names used as the keys of ``evaluate_constraints``."""
        return self._constraint_names.copy()

    @property
    def original_problem_name(self) -> str:
        """Return the name of the original problem, e.g. ``TwoBarTruss`` for ``CRE21``."""
        return _ORIGINAL_NAMES[self.problem_name]

    def evaluate(self, params: dict[str, float]) -> list[float]:
        x = np.array([params[name] for name in self._search_space])
        f, _ = self._problem.evaluate(x)
        return f.tolist()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._problem, name)

    def evaluate_constraints(self, params: dict[str, float]) -> dict[str, float]:
        x = np.array([params[name] for name in self._search_space])
        _, g = self._problem.evaluate(x)
        return dict(zip(self._constraint_names, g.tolist()))


if TYPE_CHECKING:
    for constrained_problem_cls in constrained_problems.values():
        _0: BaseConstrainedREBenchmark = constrained_problem_cls()
    for unconstrained_problem_cls in unconstrained_problems.values():
        _1: BaseUnconstrainedREBenchmark = unconstrained_problem_cls()
