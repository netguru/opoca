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


import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.tree import DecisionTreeClassifier

from opoca.data.dataset import Dataset
from opoca.data.split import Split
from opoca.data.splitter import RandomSplitter
from opoca.evaluation.plotter import Plotter
from opoca.evaluation.scorer import Scorer
from opoca.evaluation.utils import convert_list_probas_to_array


def get_classification_data(task: str) -> Split:
    """
    Creates dataset for classification task according to given task.
    Parameters
    ----------
    task: str
        Classification task. Possible tasks are: ['binary', 'multiclass', 'multilabel'].

    Returns
    -------
    out: Split
        Data split into train, val and test datasets.
    """
    if task == 'binary':
        x_data, y_data = make_classification(random_state=2020)
    elif task == 'multiclass':
        x_data, y_data = make_classification(
            n_features=40, n_classes=4, n_informative=8, random_state=2020)
    elif task == 'multilabel':
        x_data, y_data = make_multilabel_classification(random_state=2020)
    x_data, y_data = pd.DataFrame(x_data), pd.DataFrame(y_data)
    dataset = Dataset(x=x_data, y=y_data)
    splitter = RandomSplitter(0.6, 0.3, random_state=42)
    split = splitter.split(dataset)
    return split


@pytest.mark.parametrize("tasks,metrics", [(['binary', 'multiclass', 'multilabel'],
                                            ['accuracy', 'precision', 'recall', 'f1', 'auc']),
                                           (['binary', 'multiclass',
                                             'multilabel'], ['auc']),
                                           (['binary', 'multiclass', 'multilabel'], [
                                            'precision', 'recall']),
                                           (['binary', 'multiclass', 'multilabel'], ['f1'])])
def test_scorer(tasks: list, metrics: list):
    """Tests Scorer class."""
    for task in tasks:
        split = get_classification_data(task)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(split.train.x, split.train.y)
        scorer = Scorer(metrics=metrics, report=True)
        scores = scorer.score(clf, split.val)
        if 'auc' not in metrics:
            with pytest.raises(RuntimeError):
                assert scorer.plot_roc_auc(show=False)
        else:
            scorer.plot_roc_auc(show=False)
        for metric in metrics:
            assert metric in scores
            assert isinstance(scores[metric], float)
            assert all([0. <= score <= 1. for score in scores.values()])
            assert set(metrics) == set(scores.keys())


@pytest.mark.parametrize("tasks", [['binary', 'multiclass', 'multilabel']])
def test_plotter(tasks: list):
    """Tests Plotter class."""
    for task in tasks:
        split = get_classification_data(task)
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(split.train.x, split.train.y)
        y_prob = clf.predict_proba(split.val.x)
        if isinstance(y_prob, list):
            y_prob = convert_list_probas_to_array(y_prob)
        class_names = [f'class_{i}' for i in range(y_prob.shape[-1])]
        plotter = Plotter()
        plotter.plot_roc_auc(split.val.y.to_numpy(), y_prob, show=False)
        plotter.plot_prauc(split.val.y.to_numpy(), y_prob, show=False)
        plotter.plot_roc_auc(split.val.y.to_numpy(),
                             y_prob, class_names, show=False)
        plotter.plot_prauc(split.val.y.to_numpy(), y_prob,
                           class_names, show=False)
