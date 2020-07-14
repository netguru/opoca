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


import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.multiclass import OneVsRestClassifier, _ConstantPredictor, clone
from sklearn.preprocessing import LabelBinarizer


def _fit_binary(estimator, x, y, classes=None, sample_weight=None, eval_set=None, eval_metric=None,
                early_stopping_rounds=None, verbose=True, xgb_model=None, sample_weight_eval_set=None, callbacks=None):
    """Fit a single binary estimator.
    This function is ported from sklearn multiclass module."""
    unique_y = np.unique(y)
    if len(unique_y) == 1:
        if classes is not None:
            if y[0] == -1:
                c = 0
            else:
                c = y[0]
            warnings.warn(f"Label {str(classes[c])} is present in all training examples.")
        estimator = _ConstantPredictor().fit(x, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(x, y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,
                      early_stopping_rounds=early_stopping_rounds, verbose=verbose,
                      xgb_model=xgb_model, sample_weight_eval_set=sample_weight_eval_set, callbacks=callbacks)
    return estimator


class XGBoostOneVsRest(OneVsRestClassifier):

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_metric=None, early_stopping_rounds=None,
            verbose=True, xgb_model=None, sample_weight_eval_set=None, callbacks=None):
        """Fit underlying estimators.
        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.
        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.
        Other Parameters: see XGBoostClassifer documentation.
        Returns
        -------
        self
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in overall
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        jobs = []
        for i, column in enumerate(columns):
            class_name = self.label_binarizer_.classes_[i]
            classes = [f"not {class_name}", class_name]
            job_eval_set = [(x_set, y_set[:, i]) for (x_set, y_set) in eval_set]

            delayed_job = delayed(_fit_binary)(self.estimator, X, column, classes, sample_weight, job_eval_set,
                                               eval_metric, early_stopping_rounds, verbose, xgb_model,
                                               sample_weight_eval_set, callbacks)
            jobs.append(delayed_job)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(jobs)
        return self
