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


import optuna
from horology import timed
from optuna import Trial
from optuna.trial import FixedTrial
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

from opoca.data.dataset import Dataset
from opoca.data.splitter import StratifiedSplitter
from opoca.hyperparam_search.objective import Objective
from opoca.hyperparam_search.pocstudy import POCStudy


class SimpleSVMObjective(Objective):
    """ Very simple objective that depends only on two parameters."""

    def create_model(self, trial: Trial):
        """
        In class that inherits from Objective, this method must be implemented
        and return a model. Parameters of the model should be obtained with the
        trial object.
        """
        C = trial.suggest_loguniform('C', 1e-6, 1e6)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

        model = SVC(C=C, kernel=kernel)

        return model


class MultiLevelSVMObjective(Objective):
    """ More sophisticated example with non-constant dimension of search space"""

    def create_model(self, trial: Trial):
        """
        All optimized model parameters should be obtained form trial object
        using one of `suggest_*` methods. There is no requirement that always
        the same parameters are used. Here, depending on kernel, only
        applicable parameters are used. E.g. 'degree' make sense only for the
        polynomial kernel.
        """
        C = trial.suggest_loguniform('C', 1e-6, 1e6)
        dfs = trial.suggest_categorical('decision_function_shape', ['ovo', 'ovr'])
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        max_iter = 1000

        if kernel == 'poly':
            degree = trial.suggest_int('degree', 2, 4)
            return SVC(C=C, kernel=kernel,
                       degree=degree,
                       decision_function_shape=dfs, max_iter=max_iter)

        elif kernel in ['rbf', 'sigmoid']:
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            return SVC(C=C, kernel=kernel,
                       gamma=gamma,
                       decision_function_shape=dfs, max_iter=max_iter)

        # kernel == 'linear'
        return SVC(C=C, kernel=kernel,
                   decision_function_shape=dfs, max_iter=max_iter)


@timed
def example():
    """
    In this example we search for best hyperparameters for a Support Vector
    Machine Classifier on the breast cancer dataset.

    We split the data for training and testing (we do not need a separate
    validation set, because POCStudy internally does a cross validation).

    We run optimization for 60 seconds. You may see some ConvergenceWarning
    from SVC. Don't worry - those trials are usually pruned anyway. Number of
    explored points depends on how fast is your machine.

    We use the best params to train the model on the whole train set and score
    it on the test set.

    Finally we plot the optimization history. For first 10 trials, optuna will
    perform random search to warm up TPE, then it will focus on exploring most
    promising areas. Observe that after 15th iteration, multiple trials are
    pruned.
    """

    # Remove next 3 lines if you want to see some SVC warnings.
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

    # Load and optionally split data.
    x, y = load_breast_cancer(return_X_y=True)
    dataset = Dataset(x, y, 'sklearn_breast_cancer')
    splitter = StratifiedSplitter(train_ratio=0.7, val_ratio=0.0, random_state=2020)
    split = splitter.split(dataset)

    # Instantiate objective with train data.
    # You can pass a list of scorers - the first one wll be used for
    # optimization and the rest just for monitoring
    objective = MultiLevelSVMObjective(split.train, scorer=['accuracy',
                                                            'f1',
                                                            'recall',
                                                            'precision'])

    # Create study and run optimization (internally it will use 5-fold CV)
    study = POCStudy(objective)
    study.optimize(timeout_seconds=60, num_trials=None)

    # Now you can use best trial params to create a model...
    best_params = study.best_trial.params
    print('Creating model with best params:', best_params)
    best_trial = FixedTrial(best_params)
    best_model = objective.create_model(best_trial)

    # ... which you can fit and score.
    best_model.fit(X=split.train.x, y=split.train.y)
    test_score = best_model.score(X=split.test.x, y=split.test.y)
    print(f'test_score: {test_score:.3g}')

    # Plot optimization history.
    print('Plot will hopefully open in a browser.')
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    print('Done.')


if __name__ == '__main__':
    example()
