from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
import threading
from typing import Any
from typing import TYPE_CHECKING

import cmaes as _  # NOQA
from optuna.distributions import CategoricalDistribution
from optuna.logging import get_logger
from optuna.samplers import BaseSampler
from optuna.samplers import CmaEsSampler
from optuna.samplers import GPSampler
from optuna.samplers import NSGAIISampler
from optuna.samplers import RandomSampler
from optuna.samplers import TPESampler
from optuna.samplers._base import _process_constraints_after_trial
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import TrialState
import scipy as _  # NOQA
import torch as _  # NOQA


if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_MAXINT32 = (1 << 31) - 1
_SAMPLER_KEY = "auto:sampler"
# NOTE(nabenabe): The prefix `optuna.` enables us to use the optuna logger externally.
_logger = get_logger(f"optuna.{__name__}")


class ThreadLocalSampler(threading.local):
    sampler: BaseSampler | None = None


class AutoSampler(BaseSampler):
    _N_COMPLETE_TRIALS_FOR_CMAES = 250

    """Sampler automatically choosing an appropriate sampler based on search space.

    This sampler is convenient when you are unsure what sampler to use.

    Example:

        .. testcode::

            import optuna
            import optunahub


            def objective(trial):
                x = trial.suggest_float("x", -5, 5)
                y = trial.suggest_float("y", -5, 5)
                return x**2 + y**2

            module = optunahub.load_module("samplers/auto_sampler")
            study = optuna.create_study(sampler=module.AutoSampler())
            study.optimize(objective, n_trials=300)

    .. note::
        This sampler requires optional dependencies of Optuna.
        You can install them with ``pip install optunahub scipy torch cmaes``.
        Alternatively, you can install them with ``pip install -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt``.

    Args:
        seed: Seed for random number generator.
        constraints_func:
            An optional function that computes the objective constraints. It must take a
            :class:`~optuna.trial.FrozenTrial` and return the constraints. The return value must
            be a sequence of :obj:`float` s. A value strictly larger than 0 means that a
            constraints is violated. A value equal to or smaller than 0 is considered feasible.
            If ``constraints_func`` returns more than one value for a trial, that trial is
            considered feasible if and only if all values are equal to 0 or smaller.

            The ``constraints_func`` will be evaluated after each successful trial.
            The function won't be called when trials fail or they are pruned, but this behavior is
            subject to change in the future releases.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
    ) -> None:
        self._rng = LazyRandomState(seed)
        self._thread_local_sampler = ThreadLocalSampler()
        self._constraints_func = constraints_func

    @property
    def _sampler(self) -> BaseSampler:
        if self._thread_local_sampler.sampler is None:
            # NOTE(nabenabe): Do not do this process in the __init__ method because the
            # substitution at the init does not update attributes in self._thread_local_sampler
            # in each thread.
            seed_for_random_sampler = self._rng.rng.randint(_MAXINT32)
            self._sampler = RandomSampler(seed=seed_for_random_sampler)

        return self._thread_local_sampler.sampler

    @_sampler.setter
    def _sampler(self, sampler: BaseSampler) -> None:
        self._thread_local_sampler.sampler = sampler

    def reseed_rng(self) -> None:
        self._rng.rng.seed()
        self._sampler.reseed_rng()

    def _include_conditional_param(self, study: Study) -> bool:
        trials = study._get_trials(
            deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED), use_cache=True
        )
        if len(trials) == 0:
            return False

        param_key = set(trials[0].params)
        return any(param_key != set(t.params) for t in trials)

    def _determine_multi_objective_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> BaseSampler:
        # TODO(nabenabe): Add more efficient heuristic for MO.
        if isinstance(self._sampler, NSGAIISampler):
            return self._sampler

        seed = self._rng.rng.randint(_MAXINT32)
        return NSGAIISampler(constraints_func=self._constraints_func, seed=seed)

    def _determine_single_objective_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> BaseSampler:
        if isinstance(self._sampler, TPESampler):
            return self._sampler

        seed = self._rng.rng.randint(_MAXINT32)
        if (
            self._constraints_func is not None
            or any(isinstance(d, CategoricalDistribution) for d in search_space.values())
            or self._include_conditional_param(study)
        ):
            # Use ``TPESampler`` if search space includes conditional or categorical parameters.
            # TBD: group=True?
            return TPESampler(
                seed=seed,
                multivariate=True,
                warn_independent_sampling=False,
                constraints_func=self._constraints_func,
                constant_liar=True,
            )

        complete_trials = study._get_trials(
            deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True
        )
        if len(complete_trials) < self._N_COMPLETE_TRIALS_FOR_CMAES:
            # Use ``GPSampler`` if search space is numerical and
            # len(complete_trials) < _N_COMPLETE_TRIALS_FOR_CMAES.
            if not isinstance(self._sampler, GPSampler):
                return GPSampler(seed=seed)
        elif len(search_space) > 1:
            if not isinstance(self._sampler, CmaEsSampler):
                # Use ``CmaEsSampler`` if search space is numerical and
                # len(complete_trials) > _N_COMPLETE_TRIALS_FOR_CMAES.
                # Warm start CMA-ES with the first _N_COMPLETE_TRIALS_FOR_CMAES complete trials.
                complete_trials.sort(key=lambda trial: trial.datetime_complete)
                warm_start_trials = complete_trials[: self._N_COMPLETE_TRIALS_FOR_CMAES]
                return CmaEsSampler(
                    seed=seed, source_trials=warm_start_trials, warn_independent_sampling=True
                )
        else:
            return TPESampler(
                seed=seed,
                multivariate=True,
                warn_independent_sampling=False,
                constraints_func=self._constraints_func,
                constant_liar=True,
            )

        return self._sampler  # No update happens to self._sampler.

    def _determine_sampler(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> BaseSampler:
        if len(study.directions) == 1:
            return self._determine_single_objective_sampler(study, trial, search_space)
        else:
            return self._determine_multi_objective_sampler(study, trial, search_space)

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        return self._sampler.infer_relative_search_space(study, trial)

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: dict[str, BaseDistribution]
    ) -> dict[str, Any]:
        return self._sampler.sample_relative(study, trial, search_space)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        return self._sampler.sample_independent(study, trial, param_name, param_distribution)

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        # NOTE(nabenabe): Sampler must be updated in this method. If, for example, it is updated in
        # infer_relative_search_space, the sampler for before_trial and that for sample_relative,
        # after_trial might be different, meaning that the sampling routine could be incompatible.
        if (
            len(study._get_trials(deepcopy=False, states=(TrialState.COMPLETE,), use_cache=True))
            != 0
        ):
            search_space = IntersectionSearchSpace().calculate(study)
            self._sampler = self._determine_sampler(study, trial, search_space)

        sampler_name = self._sampler.__class__.__name__
        _logger.debug(f"Sample trial#{trial.number} with {sampler_name}.")
        study._storage.set_trial_system_attr(trial._trial_id, _SAMPLER_KEY, sampler_name)
        self._sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        if isinstance(self._sampler, RandomSampler) and self._constraints_func is not None:
            # NOTE(nabenabe): Since RandomSampler does not handle constraints, we need to
            # separately set the constraints here.
            _process_constraints_after_trial(self._constraints_func, study, trial, state)

        self._sampler.after_trial(study, trial, state, values)
