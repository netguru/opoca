# Copyright 2020 (c) Netguru S.A.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import warnings
from typing import Optional, List, Callable, Tuple, Union

import optuna
import skopt
from optuna import Study
from optuna.integration import SkoptSampler
from optuna.samplers import RandomSampler, TPESampler, BaseSampler

from opoca.hyperparam_search.objective import Objective
from opoca.hyperparam_search.utils import OPTUNA_DB


class POCStudy(Study):
    """ Performs parameters optimization and takes care of logging.

    Parameters
    ----------
    objective: Objective
        Objective object containing the function that should be optimized.
    direction: str, optional
        One of ['maximize', 'minimize']. Use 'maximize' for accuracy, f1 etc,
         use 'minimize' for error rates, loss etc.
    algorithm: str or skopt.Optimizer, optional
        Algorithm used to find best hyperparameters.
        There are following options:
        * 'TPE' - Tree Parzen Estimator, default and decent one. It handles
        well conditional hyperparameter spaces.
        * 'GP' - Gaussian Process, most famous and sophisticated, but cannot
        handle conditional hyperparameter spaces. It falls back to the random
        search for parameters that are conditioned on other parameters.
        * 'RF' - Use Random Forest Regressor as a surrogate model. Rarely used.
        * 'random' - simple and can be parallelized without any synchronization
            between instances.
        * If more flexibility is needed, provide directly a skopt.Optimizer
        instance.

    pruner: subclass of BasePruner or None, optional
        When using a k-fold cross validation, after each fold, the score is
        passed to the pruner. If it looks unpromising, the trial is stopped
        without evaluation of other folds. By default, a median pruner is used.
        It prunes if the trial's best intermediate result is worse than median
        of intermediate results of previous trials at the same step.
        If None, a dummy pruner that never prunes will be used.

    Notes
    -----
    TPE, GP, RF should be rather used with num_jobs=1.

    References
    ----------
    __init__ was inspired by `optuna.create_study`.
    """

    # A sentinel that allows specifying meaningfully None as the pruner.
    DEFAULT_PRUNER = object()

    def __init__(self,
                 objective: Objective,
                 direction: str = 'maximize',
                 algorithm: Union[str, skopt.Optimizer] = 'TPE',
                 pruner: Optional[optuna.pruners.BasePruner] = DEFAULT_PRUNER):

        storage = os.environ.get(OPTUNA_DB, None)
        if storage is not None:
            print('Using RDB for storing trial results.')
            storage = optuna.storages.RDBStorage(url=storage)
            version = storage.get_current_version()
            print(f'Successfully connected to the database (version: {version}).')

        study_name = objective.name
        assert study_name is not None
        storage = optuna.storages.get_storage(storage)
        try:
            study_id = storage.create_new_study(study_name)
            print(f"Created a new study with name '{study_name}'")
        except optuna.exceptions.DuplicatedStudyError:
            study_id = storage.get_study_id_from_name(study_name)
            print(f"Using an existing study with name '{study_name}' instead of creating a new one.")

        self.sampler_name, sampler = self.__create_sampler(algorithm)

        if pruner is None:
            # In optuna, pruner=None leads to usage of MedianPruner, which is
            # counter-intuitive. None is None. NopPruner never prunes.
            pruner = optuna.pruners.NopPruner()
        elif pruner is self.DEFAULT_PRUNER:
            # We use here value 15 instead of default 5, because for TPE we
            # need gather some initial knowledge about the search space.
            # You can provide directly a pruner instance if you need another value.
            pruner = optuna.pruners.MedianPruner(n_startup_trials=15)

        super().__init__(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)

        if direction == "minimize":
            _direction = optuna.structs.StudyDirection.MINIMIZE
        elif direction == "maximize":
            _direction = optuna.structs.StudyDirection.MAXIMIZE
        else:
            raise ValueError("Please set either 'minimize' or 'maximize' to direction.")

        self._storage.set_study_direction(study_id, _direction)

        self.objective = objective

    def __create_sampler(self, algorithm: Union[str, skopt.Optimizer]) -> (str, BaseSampler):
        """ Create sampler from Optimizer instance or string

        Parameters
        ----------
        sampler_name: skopt.Optimizer or str
            Passed from constructor - see class doc.

        Returns
        -------
        Tuple: sampler_name and an instance of optuna BaseSampler subclass
        """
        if isinstance(algorithm, skopt.Optimizer):
            sampler = algorithm
            sampler_name = algorithm.__class__.__name__
        elif algorithm == 'TPE':
            sampler = TPESampler()
            sampler_name = algorithm
        elif algorithm == 'GP':
            sampler = SkoptSampler()
            sampler_name = algorithm
            # `n_startup_trials` of optuna SkoptSampler should not be changed.
            # The internal scikit-optimize sampler has `n_initial_points=10`.
            # If needed you can change it using `skopt_kwargs`.
        elif algorithm == 'RF':
            skopt_kwargs = {'base_estimator': 'RF'}
            sampler = SkoptSampler(skopt_kwargs=skopt_kwargs)
            sampler_name = algorithm
        elif algorithm == 'random':
            sampler = RandomSampler()
            sampler_name = algorithm
        else:
            raise ValueError(f'Unknown hyperparameter search algorithm: {algorithm}. '
                             f'Use one of those: "TPE", "GP", "RF" or "random".')

        return sampler_name, sampler

    def optimize(  # pylint: disable=arguments-differ
            self,
            func: None = None,
            num_trials: Optional[int] = 50,
            timeout_seconds: Optional[float] = None,
            num_jobs: int = 1,
            catch: Tuple = (),
            callbacks: Optional[List[Callable]] = None,
            gc_after_trial: bool = True,
            show_progress_bar: bool = False):
        """ Optimize the objective function

        Parameters
        ----------
        func: None
            Ignored, only for compatibility with base class.
        other parameters:
             -> see optuna.Study.optimize for the parameters description
        """
        if func is not None:
            warnings.warn('`func` was provided but will ignored. The objective '
                          'that was provided in the constructor will be used. '
                          'The argument is present only to provide compatibility '
                          'with the  base class.')

        if self.sampler_name in ["TPE", "GP", "RF"] and num_jobs != 1:
            warnings.warn(f'The chosen algorithm ({self.sampler_name}) is not '
                          f'parallelizable and may behave strange if run in '
                          f'multiple instances. For example, it may visit the '
                          f'same point multiple times.')

        return super().optimize(
            self.objective,
            n_trials=num_trials,
            timeout=timeout_seconds,
            n_jobs=num_jobs,
            catch=catch,
            callbacks=callbacks,
            gc_after_trial=gc_after_trial,
            show_progress_bar=show_progress_bar)
