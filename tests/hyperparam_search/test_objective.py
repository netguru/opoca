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
import shutil
from contextlib import contextmanager

from opoca.hyperparam_search.examples import SimpleSVMObjective
from opoca.hyperparam_search.pocstudy import POCStudy
from opoca.hyperparam_search.utils import MLFLOW_TRACKING_URI, MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD

NUM_TRIALS = 8
BEST_VALUE = 0.8


@contextmanager
def mlflow_context():
    os.environ[MLFLOW_TRACKING_URI] = "mlruns"
    os.environ[MLFLOW_TRACKING_USERNAME] = ""
    os.environ[MLFLOW_TRACKING_PASSWORD] = ""
    try:
        yield
    finally:
        if os.path.exists(os.environ[MLFLOW_TRACKING_URI]):
            shutil.rmtree(os.environ[MLFLOW_TRACKING_URI])


def test_default(iris_dataset):
    objective = SimpleSVMObjective(iris_dataset)
    study = POCStudy(objective)
    study.optimize()

    assert study.best_value > BEST_VALUE


def test_floating_split(iris_dataset):
    objective = SimpleSVMObjective(iris_dataset, val_split_ratio=0.2)
    study = POCStudy(objective, algorithm='random', pruner=None)
    study.optimize(num_trials=NUM_TRIALS, num_jobs=4)

    assert study.best_value > BEST_VALUE


def test_with_mlflow(iris_dataset):
    with mlflow_context():
        objective = SimpleSVMObjective(iris_dataset, num_cv_folds=3, params_to_mlflow=True)
        study = POCStudy(objective)
        study.optimize(num_trials=NUM_TRIALS)

        assert study.best_value > BEST_VALUE


def test_gp(iris_dataset):
    objective = SimpleSVMObjective(iris_dataset, num_cv_folds=3, scorer=['accuracy'])
    study = POCStudy(objective, algorithm='GP')
    study.optimize(num_trials=NUM_TRIALS)

    assert study.best_value > BEST_VALUE


def test_with_multiple_scorers(iris_dataset):
    with mlflow_context():
        objective = SimpleSVMObjective(iris_dataset, num_cv_folds=3, params_to_mlflow=True,
                                       scorer=['f1_weighted', 'recall_weighted', 'precision_weighted'])
        study = POCStudy(objective, algorithm='GP')
        study.optimize(num_trials=NUM_TRIALS)

        assert study.best_value > BEST_VALUE
