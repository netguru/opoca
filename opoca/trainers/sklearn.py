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


"""
Sklearn flavor module that collects trainers implementations that supports sklearn-like models
"""
import inspect
import warnings
from abc import ABC
from typing import Dict, Callable, Union

from imblearn.base import BaseSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline

from opoca.data.targets import ToNumpy


class BaseTrainer(ABC):
    """
    Abstract class for sklearn-like trainer
    """

    def __init__(self, features_pipeline: Union[BaseEstimator, TransformerMixin],
                 clf: BaseEstimator):
        self.features_pipeline = features_pipeline
        self.clf = clf


class SupervisedTrainer(BaseTrainer):
    """
    Class that manages training models in supervised manner

    Parameters
    ----------
    features_pipeline: [BaseEstimator, TransformerMixin]
        Pipeline for transforming input data to features

    clf: BaseEstimator
        Classifier sklearn-like

    targets_transformer: default acts like identity function
        Transforms targets vector if needed (for example to one-hot encoding)

    sampler: BaseSampler, default returns unchanged dataset
        Resamples data, supports various oversampling and undersampling techniques from imblearn

    clf_fit_params: Dict, optional
        Dictionary with keywords args for classifier fit method

    features_pipeline_callback: Callable, optional
        Might be used for updating pipeline between fit and transform invokes
    """

    def __init__(self, features_pipeline: Union[BaseEstimator, TransformerMixin, Pipeline],
                 clf: BaseEstimator,
                 targets_transformer: Union[TransformerMixin, BaseEstimator] = None,
                 sampler: BaseSampler = None,
                 clf_fit_params: Dict = None,
                 features_pipeline_callback: Callable = None
                 ):
        super().__init__(features_pipeline, clf)

        self.features_pipeline_callback = features_pipeline_callback

        if clf_fit_params is None:
            clf_fit_params = {}
        self.clf_fit_params = clf_fit_params

        if targets_transformer is None:
            targets_transformer = ToNumpy()
        self.targets_transformer = targets_transformer

        self.sampler = sampler

    def train(self, train_dataset, val_dataset=None, fit_data=None):
        """
        Runs complete training. Currently val_dataset is used only in case of early stopping for XGBoost.

        Parameters
        ----------
        train_dataset: Dataset
            Instance of Dataset class that represents training set

        val_dataset: Dataset, optional
            Instance of Dataset class that represents validation set

        fit_data: pd.DataFrame, optional
            DataFrame that is to be used for fitting features pipeline

        Returns
        -------
        pipeline: Pipeline
            Complete fitted pipeline
        """

        fit_data = fit_data if fit_data is not None else train_dataset.x

        # first we fit features pipeline
        self.features_pipeline.fit(fit_data)

        self.targets_transformer.fit(train_dataset.y)

        # if needed, we update features pipeline (typical case when you fit features on test time-series-data
        if self.features_pipeline_callback is not None:
            self.features_pipeline_callback(self.features_pipeline)

        if self.sampler is not None:
            x_train, y_train = self.sampler.fit_resample(train_dataset.x, train_dataset.y)
        else:
            x_train, y_train = train_dataset.x, train_dataset.y

        x_features = self.features_pipeline.transform(x_train)
        y_targets = self.targets_transformer.transform(y_train)

        if val_dataset is not None:
            self.__prepare_early_stop_params(val_dataset)

        self.clf.fit(x_features, y_targets, **self.clf_fit_params)

        pipeline = make_pipeline(self.features_pipeline,
                                 self.clf)

        return pipeline

    def __prepare_early_stop_params(self, val_dataset):
        """

        Parameters
        ----------
        val_dataset: Dataset
            Validation dataset used for early stopping
        """

        if "eval_set" in inspect.signature(self.clf.fit).parameters:
            val_y = self.targets_transformer.fit_transform(val_dataset.y)
            val_x = self.features_pipeline.transform(val_dataset.x)

            self.clf_fit_params["eval_set"] = [(val_x, val_y)]

            self.clf_fit_params["eval_metric"] = self.clf_fit_params["eval_metric"] or ["logloss"]

            self.clf_fit_params["early_stopping_rounds"] = self.clf_fit_params["early_stopping_rounds"] or 10
        else:
            warnings.warn(f"There's no 'eval_set' parameter in fit method of instance "
                          f"of {self.clf.__class__}. Providing validation set will have no effect on "
                          f"training", UserWarning)
