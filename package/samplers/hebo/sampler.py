from __future__ import annotations

from collections.abc import Sequence
from typing import Any
import warnings

import numpy as np
import optuna
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.samplers import BaseSampler
from optuna.search_space import IntersectionSearchSpace
from optuna.study import Study
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
import optunahub
import pandas as pd

from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


class HEBOSampler(optunahub.samplers.SimpleBaseSampler):
    """A sampler using `HEBO <https://github.com/huawei-noah/HEBO/tree/master/HEBO>__` as the backend.

    For further information about HEBO algorithm, please refer to the following paper:
    - `HEBO Pushing The Limits of Sample-Efficient Hyperparameter Optimisation <https://arxiv.org/abs/2012.03826>__`

    Args:
        search_space:
            By specifying search_space, the sampling speed at each iteration becomes slightly quicker, but this argument is not necessary to run this sampler. Default is :obj:`None`.

        seed:
            A seed for ``HEBOSampler``. Default is :obj:`None`.

        constant_liar:
            If :obj:`True`, penalize running trials to avoid suggesting parameter configurations
            nearby. Default is :obj:`False`.

            .. note::
                Abnormally terminated trials often leave behind a record with a state of
                ``RUNNING`` in the storage.
                Such "zombie" trial parameters will be avoided by the constant liar algorithm
                during subsequent sampling.
                When using an :class:`~optuna.storages.RDBStorage`, it is possible to enable the
                ``heartbeat_interval`` to change the records for abnormally terminated trials to
                ``FAIL``.
                (This note is quoted from `TPESampler <https://github.com/optuna/optuna/blob/v4.1.0/optuna/samplers/_tpe/sampler.py#L215-L222>__`.)

            .. note::
                It is recommended to set this value to :obj:`True` during distributed
                optimization to avoid having multiple workers evaluating similar parameter
                configurations. In particular, if each objective function evaluation is costly
                and the durations of the running states are significant, and/or the number of
                workers is high.
                (This note is quoted from `TPESampler <https://github.com/optuna/optuna/blob/v4.1.0/optuna/samplers/_tpe/sampler.py#L224-L229>__`.)

            .. note::
                HEBO algorithm involves multi-objective optimization of multiple acquisition functions.
                While `constant_liar` is a simple way to get diverse params for parallel optimization,
                it may not be the best approach for HEBO.

        independent_sampler:
            A :class:`~optuna.samplers.BaseSampler` instance that is used for independent
            sampling. The parameters not contained in the relative search space are sampled
            by this sampler. If :obj:`None` is specified, :class:`~optuna.samplers.RandomSampler`
            is used as the default.
    """  # NOQA

    def __init__(
        self,
        search_space: dict[str, BaseDistribution] | None = None,
        *,
        seed: int | None = None,
        constant_liar: bool = False,
        independent_sampler: BaseSampler | None = None,
    ) -> None:
        super().__init__(search_space, seed)
        if search_space is not None and not constant_liar:
            self._hebo = HEBO(self._convert_to_hebo_design_space(search_space), scramble_seed=seed)
        else:
            self._hebo = None
        self._intersection_search_space = IntersectionSearchSpace()
        self._independent_sampler = independent_sampler or optuna.samplers.RandomSampler(seed=seed)
        self._is_independent_sample_necessary = False
        self._constant_liar = constant_liar
        self._rng = np.random.default_rng(seed)

    def _sample_relative_define_and_run(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        return {
            name: row.iloc[0]
            for name, row in self._hebo.suggest().items()
            if name in search_space.keys()
        }

    def _sample_relative_stateless(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        if self._constant_liar:
            target_states = [TrialState.COMPLETE, TrialState.RUNNING]
        else:
            target_states = [TrialState.COMPLETE]

        use_cache = not self._constant_liar
        trials = study._get_trials(deepcopy=False, states=target_states, use_cache=use_cache)
        is_complete = np.array([t.state == TrialState.COMPLETE for t in trials])
        if not np.any(is_complete):
            # note: The backend HEBO implementation uses Sobol sampling here.
            # This sampler does not call `hebo.suggest()` here because
            # Optuna needs to know search space by running the first trial in Define-by-Run.
            self._is_independent_sample_necessary = True
            return {}
        else:
            self._is_independent_sample_necessary = False
        trials = [t for t in trials if set(search_space.keys()) <= set(t.params.keys())]

        # Assume that the back-end HEBO implementation aims to minimize.
        values = np.array([t.value if t.state == TrialState.COMPLETE else np.nan for t in trials])
        worst_value = (
            np.nanmax(values) if study.direction == StudyDirection.MINIMIZE else np.nanmin(values)
        )
        sign = 1 if study.direction == StudyDirection.MINIMIZE else -1

        seed = int(self._rng.integers(low=1, high=(1 << 31)))
        hebo = HEBO(self._convert_to_hebo_design_space(search_space), scramble_seed=seed)
        params = pd.DataFrame([t.params for t in trials])
        values[np.isnan(values)] = worst_value
        values *= sign
        hebo.observe(params, values)
        return {
            name: row.iloc[0]
            for name, row in hebo.suggest().items()
            if name in search_space.keys()
        }

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, float]:
        if study._is_multi_objective():
            raise ValueError(
                f"{self.__class__.__name__} has not supported multi-objective optimization."
            )
        if self._hebo is None or self._constant_liar is True:
            return self._sample_relative_stateless(study, trial, search_space)
        else:
            return self._sample_relative_define_and_run(study, trial, search_space)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        if self._hebo is not None and values is not None:
            # Assume that the back-end HEBO implementation aims to minimize.
            if study.direction == StudyDirection.MAXIMIZE:
                values = [-x for x in values]
            self._hebo.observe(pd.DataFrame([trial.params]), np.asarray([values]))

    def _convert_to_hebo_design_space(
        self, search_space: dict[str, BaseDistribution]
    ) -> DesignSpace:
        design_space = []
        for name, distribution in search_space.items():
            if isinstance(distribution, FloatDistribution) and not distribution.log:
                design_space.append(
                    {
                        "name": name,
                        "type": "num",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, FloatDistribution) and distribution.log:
                design_space.append(
                    {
                        "name": name,
                        "type": "pow",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, IntDistribution) and distribution.log:
                design_space.append(
                    {
                        "name": name,
                        "type": "pow_int",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, IntDistribution) and distribution.step:
                design_space.append(
                    {
                        "name": name,
                        "type": "step_int",
                        "lb": distribution.low,
                        "ub": distribution.high,
                        "step": distribution.step,
                    }
                )
            elif isinstance(distribution, IntDistribution):
                design_space.append(
                    {
                        "name": name,
                        "type": "int",
                        "lb": distribution.low,
                        "ub": distribution.high,
                    }
                )
            elif isinstance(distribution, CategoricalDistribution):
                design_space.append(
                    {
                        "name": name,
                        "type": "cat",
                        "categories": distribution.choices,
                    }
                )
            else:
                raise NotImplementedError(f"Unsupported distribution: {distribution}")
        return DesignSpace().parse(design_space)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return optuna.search_space.intersection_search_space(
            study._get_trials(deepcopy=False, use_cache=True)
        )

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        if not self._is_independent_sample_necessary:
            warnings.warn(
                "`HEBOSampler` falls back to `RandomSampler` due to dynamic search space."
            )

        return self._independent_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )
