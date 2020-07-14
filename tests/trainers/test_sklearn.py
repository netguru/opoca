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


import pytest
from imblearn.under_sampling import NearMiss
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

from opoca.trainers.sklearn import SupervisedTrainer


def test_train_clf_without_eval_set_parameter(iris_split, features_pipeline, clf):
    trainer = SupervisedTrainer(features_pipeline, clf)

    with pytest.warns(UserWarning):
        trainer.train(iris_split.train, val_dataset=iris_split.val, fit_data=iris_split.test.x)


def test_train_clf_default_path(iris_split, features_pipeline, clf):
    trainer = SupervisedTrainer(features_pipeline, clf)

    trainer.train(iris_split.train)

    check_is_fitted(clf)


def test_train_clf_with_sampler(iris_split, features_pipeline, clf):
    trainer = SupervisedTrainer(features_pipeline, clf, sampler=NearMiss())

    trainer.train(iris_split.train)

    check_is_fitted(clf)


def test_train_clf_with_early_stopping(iris_split, features_pipeline):
    clf = XGBClassifier()

    clf_fit_params = {
        "early_stopping_rounds": 10,
        "eval_metric": ["mlogloss"]
    }

    trainer = SupervisedTrainer(features_pipeline, clf, clf_fit_params=clf_fit_params)

    trainer.train(iris_split.train, val_dataset=iris_split.val)

    check_is_fitted(clf)


def test_features_pipeline_callback(iris_split, clf):
    class DummyPipeline(BaseEstimator):
        REF = 5

        def fit(self, _):
            return self

        def transform(self, x):
            return x - self.REF

    def callback(pipeline):
        pipeline.REF = 3
        return pipeline

    features_pipeline = DummyPipeline()

    trainer = SupervisedTrainer(features_pipeline, clf, features_pipeline_callback=callback)

    trainer.train(iris_split.train)

    assert features_pipeline.REF == 3
