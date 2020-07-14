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


import contextlib
import os
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Union, Callable, List, Dict, Tuple, Optional

import mlflow
import numpy as np
import optuna
from sklearn.metrics import SCORERS
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from opoca.data.dataset import Dataset
from opoca.hyperparam_search.utils import check_mlflow, get_current_git_hash, \
    are_uncommitted_changes, are_untracked_files, mlflow_push_pickled, \
    MLFLOW_TRACKING_URI

NamedScorer = namedtuple('NamedScorer', ['fn', 'name'])


def _check_scorer(scorer: Union[str, Callable]) -> NamedScorer:
    """ Check and standardize a scorer """
    if isinstance(scorer, str):
        scorer_fn = SCORERS[scorer]
        scorer_name = scorer
    elif isinstance(scorer, Callable):
        scorer_fn = scorer
        scorer_name = scorer.__name__
    else:
        raise ValueError('Scorer must be either a str or a callable.')

    return NamedScorer(scorer_fn, scorer_name)


class Objective(ABC):
    """ Callable base class for implementing objective function.

    You should subclass this class and implement `create_model`. It was
    designed to provide decent hyperparameter optimization out of the box.

    Parameters
    ----------
    dataset: Dataset
        POC-native dataset with training data.
    num_cv_folds: int, optional
        Number of folds to use in CV. If set, `val_split_ratio` must be None.
    val_split_ratio: float, optional
        Percentage of the data to be used for validation. If set,
        `num_cv_folds` must be None.
    scorer: str or sklearn scorer or list of those
        Check possible scores in  sklearn.metrics.SCORERS.
        If list, the first scorer will be used as the optimization metric, the
        rest just for monitoring.
    unique_name: str, optional
        Name used to distinguish consecutive experiments in mlflow and studies
        in optuna. If you use existing name, optuna will continue study (if it
        was recorded in database) and mlflow will log under the same name.
        If None, the name will be generated based on the objective name and git
        status, see the `_get_name` doc for details.
    params_to_mlflow: bool, optional
        If True,  parameters will be logged in mlflow server.
    models_to_mlflow: bool, optional
        If True, pickled models will be pushed in mlflow server.
    fit_kwargs: dict, optional
        Kwargs that will be passed to fit method.

    If neither num_cv_folds nor val_split_ratio is set, a default
    value num_cv_folds=5 will be used.

    If any of mlflow flags are True, you must specify following variables in
    your environment:
    - MLFLOW_TRACKING_URI
    - MLFLOW_TRACKING_USERNAME
    - MLFLOW_TRACKING_PASSWORD
    """

    def __init__(self, dataset: Dataset,
                 num_cv_folds: int = None,
                 val_split_ratio: float = None,
                 scorer: Union[str, Callable, list] = 'f1_weighted',
                 unique_name: Optional[str] = None,
                 params_to_mlflow: bool = False,
                 models_to_mlflow: bool = False,
                 fit_kwargs: Optional[dict] = None):

        if num_cv_folds is not None and val_split_ratio is not None:
            raise TypeError('Both `num_cv_folds` and `val_split_ratio` were '
                            'given. Specify at most one of them.')

        if val_split_ratio is not None:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split_ratio)
            self.split_indices = [next(sss.split(dataset.x, dataset.y))]
        else:
            num_cv_folds = num_cv_folds or 5
            cross_validation = StratifiedKFold(n_splits=num_cv_folds, shuffle=True)
            self.split_indices = list(cross_validation.split(dataset.x, dataset.y))

        self.dataset = dataset

        if isinstance(scorer, list):
            scorer, *other_scorers = scorer
        else:
            other_scorers = []
        self.scorer, self.scorer_name = _check_scorer(scorer)
        self.additional_named_scorers = [_check_scorer(s) for s in other_scorers]
        if len(self.additional_named_scorers) > 0:
            print(f'Only {self.scorer_name} will be used for optimization. '
                  f'Other scores will be calculated and logged, but will not '
                  f'influence the hyperparameter search')

        self.name = self._get_name(unique_name)

        self.params_to_mlflow = params_to_mlflow
        self.models_to_mlflow = models_to_mlflow
        if self.use_mlflow:
            if not check_mlflow():
                raise ValueError('MLflow environment variables not set, cannot use it.')
            print(f'Trials will be logged to MLflow at {os.environ[MLFLOW_TRACKING_URI]}.')
            mlflow.set_experiment(self.name)

        self.fit_kwargs = fit_kwargs if fit_kwargs is not None else {}

    @property
    def use_mlflow(self) -> bool:
        """ Property indicating if mlflow is used at all """
        return any((self.params_to_mlflow, self.models_to_mlflow))

    def _get_name(self, unique_name):
        """ Assure decent name for an experiment (aka study)

        It generates name of the study based on:
        - the name of objective class (one that inherits from this one)
        - current git hash (7 characters)
        - git status by adding:
            - 'dirty' if there are some uncommitted changes,
            - 'with_untracked' if there are some untracked files.

        In other words, if no suffix is present, you can be sure that the code
        used to train model is exactly as in the commit.

        Parameters
        ----------
        unique_name: str or None
            If provided, it is returned immediately.
            If None, it is generated as described above.

        Returns
        -------
        name: str
            Provided of generated name.
        """
        if unique_name is not None:
            return unique_name
        git_hash = get_current_git_hash()
        if git_hash is None:
            return f'{self.__class__.__name__}/not-in-git'
        name = f"{self.__class__.__name__}/{git_hash}" \
               f"{'_dirty' if are_uncommitted_changes() else ''}" \
               f"{'_with_untracked' if are_untracked_files() else ''}"

        return name

    def __call__(self, trial: optuna.Trial) -> float:
        """ Make the class Callable as required by optuna.Study

        It handles most of mlflow reporting.

        Parameters
        ----------
        trial: optuna.Trial
            Object that controls how parameters are drawn.

        Returns
        -------
        score: float
            Score, averaged over folds when applicable.
        """
        if self.use_mlflow:
            context = mlflow.start_run(run_name=f'trial-{trial.number:03d}')
        else:
            context = contextlib.nullcontext()

        with context:
            model = self.create_model(trial)

            if self.params_to_mlflow:
                mlflow.log_params(trial.params)
                mlflow.log_param('dataset', self.dataset.name)

            scores, additional_scores = self._cross_validate(model, trial)
            mean_score = float(np.mean(scores))
            mean_additional_scores = {f'mean_{name}': float(np.mean(s))
                                      for name, s in additional_scores.items()}

            if self.use_mlflow and len(scores) > 1:
                mlflow.log_metrics({'MEAN_' + self.scorer_name: mean_score})
                mlflow.log_metrics(mean_additional_scores)
            if len(mean_additional_scores) > 0:
                print(*[f'{name}: {v:.3g}  ' for name, v in mean_additional_scores.items()])

            return mean_score

    def _cross_validate(self, model, trial: optuna.Trial) \
            -> Tuple[List[float], Dict[str, List[float]]]:
        """ Validate model using validation scheme defined in `__init__`.

        It handles pruning and some mlflow reporting.

        Parameters
        ----------
        model: sklearn estimator
            Model returned by `create_model` specific for the subclass.
        trial: optuna.Trial
            Object that controls how parameters are drawn.

        Returns
        -------
        List of scores and dict of additional scores.

        Raises
        ------
        optuna.exceptions.TrialPruned
            when trial is not promising and should be pruned. The exception is
            further handled by optuna.
        """
        x, y = self.dataset.x, self.dataset.y
        scores = []
        additional_scores = {ans.name: [] for ans in self.additional_named_scorers}
        for i, (train_index, val_index) in enumerate(self.split_indices, start=1):
            x_train, x_val = x.iloc[train_index], x.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            model.fit(x_train, y_train, **self.fit_kwargs)

            score = self.scorer(model, x_val, y_val)
            scores.append(score)
            if self.use_mlflow:
                mlflow.log_metrics({self.scorer_name: score}, step=i)

            for fn, name in self.additional_named_scorers:
                additional_score = fn(model, x_val, y_val)
                additional_scores[name].append(additional_score)
                if self.use_mlflow:
                    mlflow.log_metrics({name: additional_score}, step=i)

            if self.models_to_mlflow:
                mlflow_push_pickled(model, f'model_cv{i:02d}_')

            trial.report(score, i)
            if trial.should_prune():
                if self.use_mlflow:
                    mlflow.set_tag('PRUNED_FOLD', str(i + 1))
                raise optuna.exceptions.TrialPruned()

        return scores, additional_scores

    @abstractmethod
    def create_model(self, trial: optuna.Trial):
        """ Return a model with parameters drawn using trial.

        This method should be implemented in the subclass. See `examples.py`.

        Parameters
        ----------
        trial: optuna.Trial
            Object that should be used in the implementation to get parameters
            using `suggest_*` methods, e.g.:
            ```
            C = trial.suggest_loguniform('C', 1e-6, 1e6)
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            model = SVC(C=C, kernel=kernel)
            ```

        Returns
        -------
        model: estimator
        """
