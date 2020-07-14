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
from typing import List, Tuple

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import OneHotEncoder


def convert_list_probas_to_array(probabilities_list: list) -> np.ndarray:
    """
    Converts list of probability arrays to a single array of probabilities.

    In multilabel tasks the model's predict_proba method might output the probabilities of a class being
    a 0 or a 1 as separate arrays - the output is list of length of number of classes.
    The elements of the list are 2 dimensional arrays of probabilities.

    Parameters
    ----------
    probabilities_list: List of numpy ndarrays
        List of length of number of classes. The elements of the list are 2 dimensional arrays of probabilities.

    Returns
    -------
    out: np.ndarray
        Single array of probabilities.
        The last shape of the output array is equal to number of classes.
    """
    assert isinstance(probabilities_list, list)

    probabilities = np.empty(shape=(probabilities_list[0].shape[0], len(probabilities_list)))
    for i, array in enumerate(probabilities_list):
        if isinstance(array, np.ndarray):
            probabilities[:, i] = array[:, 1]
    return probabilities


def calculate_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str],
                      task: str = None) -> Tuple[dict, dict, dict]:
    """
    Calculates Receiver Operating Characteristic Area Under Curve.
    Apart from ROC scores the function outputs the False Positives Ratios,
    and True Positive Ratios, which can be used to plot ROC curves.
    Parameters
    ----------
    y_true: np.ndarray
        Array of true labels (last shape of array should equal to number of classes).
    y_prob: np.ndarray
        Array of predicted probabilities (last shape of array should equal to number of classes).
    class_names: List of strings
        Names of classes.
    task: str, optional
        Type of classification task. Types of task supported: ['binary', 'multiclass', 'multilabel'].
    Returns
    -------
    out: Tuple of 3 dictionaries
        The outputs are 3 dictionaries: False Positives Ratios and True Positive Ratios which are dictionaries
        in which keys are class names and values are numpy arrays, and a dictionary of ROC AUC scores.
    """
    fpr = dict()
    tpr = dict()
    roc_auc_dict = dict()

    if not task:
        task = check_task(y_true, y_prob)

    if task in ['binary', 'multiclass']:
        encoder = OneHotEncoder()
        y_true = encoder.fit_transform(y_true).toarray()

    for i, name in enumerate(class_names):
        fpr[name], tpr[name], _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc_dict[name] = auc(fpr[name], tpr[name])

    # Compute micro-average ROC curve and ROC area
    fpr["micro avg"], tpr["micro avg"], _ = roc_curve(y_true.ravel(), y_prob.ravel())
    if task == 'binary':
        # if task is binary avg roc auc is the same as roc auc of both classes
        # therefore we use the calculated value of one of the classes to fill it
        roc_auc_dict["micro avg"] = roc_auc_dict[class_names[0]]
    else:
        roc_auc_dict["micro avg"] = auc(fpr["micro avg"], tpr["micro avg"])

    fpr, tpr, roc_auc_dict = calculate_macro_auc(fpr, tpr, roc_auc_dict, class_names)

    return fpr, tpr, roc_auc_dict


def calculate_macro_auc(fpr: dict, tpr: dict, roc_auc_dict: dict, class_names: List[str]):
    """
    Calculates Receiver Operating Characteristic Area Under Curve macro average.
    Parameters
    ----------
    fpr: dict
        Dictionary in which keys are names of classes and values are False Positives Ratio arrays.
    tpr: dict
        Dictionary in which keys are names of classes and values are True Positives Ratio arrays.
    roc_auc_dict: dict
        Dictionary in which keys are names of classes and values are calculated ROC area under curve.
    class_names: List of strings
        Names of classes.

    Returns
    -------
    out: Tuple of 3 dictionaries
        The outputs are 3 dictionaries: False Positives Ratios and True Positive Ratios which are dictionaries
        in which keys are names and values are numpy arrays, and a dictionary of ROC AUC scores with 'macro avg' added.
    """
    nans_list = [name for name in class_names if any(np.isnan(fpr[name])) or any(np.isnan(tpr[name]))]

    # If some classes contain NaNs remove them from further calculations.
    if nans_list:
        warnings.warn(f'Classes {nans_list} contain NaN values. Ignoring them in macro average calculations.')

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[name] for name in class_names if name not in nans_list]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for name in class_names:
        if name not in nans_list:
            mean_tpr += np.interp(all_fpr, fpr[name], tpr[name])

    # Finally average it and compute AUC
    mean_tpr /= (len(class_names) - len(nans_list))

    fpr["macro avg"] = all_fpr
    tpr["macro avg"] = mean_tpr
    roc_auc_dict["macro avg"] = auc(fpr["macro avg"], tpr["macro avg"])

    return fpr, tpr, roc_auc_dict


def check_task(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    """
    Checks classification task type based on shapes of true and predicted probabilities arrays.
    Parameters
    ----------
    y_true: np.ndarray
        Array of true labels (last shape of array should equal to number of classes).
    y_prob: np.ndarray
        Array of predicted probabilities (last shape of array should equal to number of classes).

    Returns
    -------
    out: str
        Type of task as string. Possible outputs: ['binary', 'multiclass', 'multilabel'].
    """
    if y_prob.shape[-1] == 2 and len(np.unique(y_true)) == 2:
        task = 'binary'
    elif len(np.unique(y_true)) > 2 and y_prob.shape[-1] > 2:
        task = 'multiclass'
    else:
        task = 'multilabel'
    return task


def calculate_prauc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[dict, dict, dict]:
    """
    Calculates Precision-Recall Area Under Curve.
    Parameters
    ----------
    y_true: np.ndarray
        Array of true labels (last shape of array should equal to number of classes).
    y_prob: np.ndarray
        Array of predicted probabilities (last shape of array should equal to number of classes).

    Returns
    -------
    out: Tuple of 3 dictionaries
        The outputs are 3 dictionaries. Precision and Recall curves and Average Precision scores.
    """
    precision = dict()
    recall = dict()
    average_precision = dict()

    task = check_task(y_true, y_prob)

    if task in ['binary', 'multiclass']:
        encoder = OneHotEncoder()
        y_true = encoder.fit_transform(y_true).toarray()

    for i in range(y_prob.shape[-1]):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_prob[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_prob.ravel())
    average_precision["micro"] = average_precision_score(y_true, y_prob, average="micro")

    return precision, recall, average_precision
