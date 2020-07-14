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
Scoring machine learning models
"""
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from opoca.data.dataset import Dataset
from opoca.evaluation.plotter import Plotter
from opoca.evaluation.utils import convert_list_probas_to_array, calculate_roc_auc, check_task


class Scorer:
    """Class for scoring models and plotting metrics."""
    def __init__(self, metrics: List[str] = None, threshold: float = 0.5, report: bool = False):
        """
        Parameters
        ----------
        metrics: List of strings, optional
            List of metrics' names as strings. If None, all supported metrics get calculated.
            Supported metrics: ['precision', 'recall', 'f1', 'accuracy', 'auc']
        threshold: float, optional
            Threshold for rounding prediction in case of binary classification
        report: bool, optional
            Determines whether to print classification report
        """
        self.report = report
        self.threshold = threshold
        self.metrics = ['precision', 'recall', 'f1', 'accuracy', 'auc'] if not metrics else metrics
        self.n_classes = None
        self.task = None
        self.fpr = None
        self.tpr = None
        self.roc_auc_dict = None
        self.class_names = None
        self.scores_dict = defaultdict(dict)

    def score(self, pipeline: Pipeline, dataset: Dataset, class_names: List[str] = None):
        """
        Computes scores for metrics provided in Scorer constructor. If y_true is multi class then scorer does
        macro average mode for precision/recall/f1. If y_true is multilabel the scores performs macro averaging.

        Parameters
        ----------
        pipeline: Pipeline
            Complete pipeline including features pipeline and classifier.
        dataset: Dataset
            Dataset containing x and y pd.DataFrames
            For Multiclass the shape of y_true should be 1-D (NOT one-hot encoded).
            For Multilabel the shape should be n-dimensional (where n is number of classes).
        class_names: List of strings, optional
            If given, the scores for separate classes will be displayed with appropriate names.

        Returns
        -------
        metrics: Dict
            Dictionary with metrics' names as keys and scores as values
        """

        x, y_true = dataset.x, dataset.y.to_numpy()

        if len(y_true.shape) == 1:
            y_true = y_true.reshape(-1, 1)

        # run inference for probabilities
        probabilities = pipeline.predict_proba(x)

        # check if the output of inference is a list (sklearn models often output probabilities in this form
        # in case of multilabel task). If yes, convert to a single array.
        if isinstance(probabilities, list):
            probabilities = convert_list_probas_to_array(probabilities)

        # check task type based on the true and predicted arrays.
        self.task = check_task(y_true, probabilities)

        # turn probabilities into predictions with chosen threshold
        if self.task in ['binary', 'multiclass']:
            predictions = np.argmax(probabilities, axis=-1)
        else:
            predictions = np.where(probabilities >= self.threshold, 1, 0)

        # assign number of classes based on given array
        self.n_classes = probabilities.shape[-1]

        # assign names of classes
        self.class_names = class_names if class_names else [f'class_{i}' for i in range(self.n_classes)]

        # check if any of ['precision', 'recall', 'f1', 'accuracy'] are in the metrics.
        # if yes generate classification report - it calculates all of these metrics.
        # it does not calculate accuracy for multilabel problem so additional check is done in such case.
        if [metric for metric in ['precision', 'recall', 'f1', 'accuracy'] if metric in self.metrics]:
            self.scores_dict.update(classification_report(y_true, predictions,
                                                          target_names=self.class_names, output_dict=True))
            if self.task == 'multilabel':
                self.scores_dict['accuracy'] = accuracy_score(y_true, predictions)

        if 'auc' in self.metrics:
            self.fpr, self.tpr, self.roc_auc_dict = calculate_roc_auc(y_true, probabilities,
                                                                      self.class_names, self.task)
            for key, value in self.roc_auc_dict.items():
                self.scores_dict[key]['auc'] = value

        if self.report:
            print(pd.DataFrame(self.scores_dict).transpose())

        return self._get_metrics()

    def _get_metrics(self, avg_type: str = 'macro avg') -> dict:
        """
        Convenience method extracting proper metrics from the score_dict
        and outputting them in the right format.

        Parameters
        ----------
        avg_type: str, default = 'macro avg'
            Type of average to use as the output value for chosen metrics.
            Possible averages are: ['macro avg', 'micro avg', 'weighted avg']

        Returns
        -------
        out: dict
            Dictionary, in which keys are names of metrics, and values are calculated metrics scores.
        """
        scores = dict()
        for metric in self.metrics:
            if metric == 'accuracy':
                scores.update({'accuracy': self.scores_dict['accuracy']})
            elif metric == 'f1':
                scores.update({'f1': self.scores_dict[avg_type]['f1-score']})
            else:
                scores.update({metric: self.scores_dict[avg_type][metric]})
        return scores

    def plot_roc_auc(self, classes_to_plot: List[str] = None, show: bool = True):
        """
        Plots Receiver Operating Characteristic Area Under Curve.
        The plot is partially based on previous calculations performed in the `score` method.

        IMPORTANT: Please run the `score` method with 'auc' metric inside of `metrics` argument before
        using this method.
        Parameters
        ----------
        classes_to_plot: List of strings, optional
            If provided, only the given classes get plotted.
            Otherwise all classes are plotted.
        show: bool, optional, default = True
            If True, the plot gets displayed.
        """
        if not self.fpr:
            raise RuntimeError("plot_roc_auc method called before running the `score` method with 'auc' in metrics.")
        plotter = Plotter()
        plotter._plot_roc_auc(self.fpr, self.tpr, self.roc_auc_dict, self.class_names, classes_to_plot, show=show)
