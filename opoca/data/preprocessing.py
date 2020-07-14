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
Module that collects functions and classes for inputs and targets preprocessing.

Available classes:
* Dataset
* Split
* Preprocessor
* TargetTransformer
* IdentityTransformer

"""
from __future__ import annotations

from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Callable, Tuple, List
from typing import Dict, Union

import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder

from opoca.features.transformers import NaNColumnDropper, SingleValueColumnDropper, NanReplacer


def find_correlated_with_target(dataset: pd.DataFrame, target_name: str = "Target", threshold: float = 0.8,
                                cv: int = 3, scoring: Union[Callable, str] = 'f1') -> List[Tuple[str, float]]:
    """
    Finds variables that have `scoring` (default to f1-score) higher than `threshold` when logistic regression is
    trained with them as single input variable.

    Parameters
    ----------
    dataset:
        Input dataset
    target_name:
        Name of column that contain target variable
    threshold:
        Determines if variables is too much correlated with target or not
    cv:
        Number of folds in cross-validation
    scoring:
        Metric to determine whether variable is correlated with target or not

    Returns
    -------
    List of correlated variables along with scores
    """
    correlated_columns = []
    y = dataset[target_name]

    for col in dataset.columns:
        if col == target_name:
            continue
        if dataset[col].dtype == "object":
            one_hot_encoder = OneHotEncoder()
            x = one_hot_encoder.fit_transform(dataset[col])
        else:
            x = dataset[col].values.reshape(-1, 1)
        clf = LogisticRegression()
        score = cross_val_score(clf, x, y, cv=cv, scoring=scoring).mean()
        if score > threshold:
            correlated_columns.append((col, score))

    return correlated_columns


def find_correlated_variables(data_frame: pd.DataFrame, kind: str = "categorical", threshold: float = 0.9) -> Dict:
    """
    Computes which variables are correlated more than threshold. Works for categorical-categorical and
    continuous-continuous variables.

    Parameters
    ----------
    data_frame:
        input data frame
    kind:
        categorical or continuous
    threshold:
        Returns variables that have correlation coefficient higher or equal to threshold

    Returns
    -------
    Dictionary where key is variable name and value is list of variables correlated with key variable more than
    threshold.
    """
    correlated_var2var_dict = defaultdict(list)

    def pearson_coeff(x, y):
        result, _ = ss.pearsonr(x, y)
        return result

    if kind == "categorical":
        compute_correlation_coeff = cramers_corrected_stat
        types = ["object", "b"]
    elif kind == "continuous":
        types = ["float", "int64"]
        compute_correlation_coeff = pearson_coeff
    else:
        raise ValueError("Incorrect kind of variable")

    df = data_frame.select_dtypes(types)

    for i in range(len(df.columns)):
        for j in range(i + 1, len(df.columns)):
            col_1 = df.columns[i]
            col_2 = df.columns[j]
            try:
                res = compute_correlation_coeff(df[col_1], df[col_2])
                if abs(res) >= threshold:
                    correlated_var2var_dict[col_1].append(col_2)
                    correlated_var2var_dict[col_2].append(col_1)
            except ValueError:
                continue

    return correlated_var2var_dict


def cramers_corrected_stat(x: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]) -> float:
    """
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    eps = 1.e-6
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] == 2:
        correct = False
    else:
        correct = True
    chi2 = ss.chi2_contingency(confusion_matrix, correction=correct)[0]

    n = sum(confusion_matrix.sum())
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1 + eps))
    rcorr = r - ((r - 1) ** 2) / (n - 1 + eps)
    kcorr = k - ((k - 1) ** 2) / (n - 1 + eps)
    result = np.sqrt(phi2corr / (eps + min((kcorr - 1), (rcorr - 1))))
    return result


class Preprocessor(TransformerMixin, BaseEstimator, ABC):
    """
    Abstract class for all preprocessors
    """

    @abstractmethod
    def fit(self, df):
        pass

    @abstractmethod
    def transform(self, df):
        pass


class BasicPreprocessor(Preprocessor):
    """
    Basic preprocessor that includes:
    * dropping predefined columns
    * dropping columns with too many NaNs
    * replacing various representations of NaNs with np.nan
    * dropping single value columns

    Parameters
    ----------
    rules:
        dictionary of preprocessing rules. Under key `DROP` there is list of columns that shall be dropped, under key
        `STAY` there is list of columns that shall stay regardless of other partial preprocessors
    drop_nan_threshold
        drop columns that have more NaNs fraction that this threshold
    to_replace
        list of NaNs representations, if None then take a look into `NanReplacer` docs to find default representations
    """

    def __init__(self, rules: Dict, drop_nan_threshold: float = 0.8, to_replace: List[str] = None):
        self.rules = rules
        self.ncd = NaNColumnDropper(threshold=drop_nan_threshold)
        self.svcd = SingleValueColumnDropper()
        self.replace_nan = NanReplacer(to_replace=to_replace)
        self.drop_columns_ = None

    def fit(self, df: pd.DataFrame) -> BasicPreprocessor:
        self.ncd.fit(df)
        self.drop_columns_ = set(self.rules["DROP"]).union(set(self.ncd.columns_to_drop_)) - set(self.rules["STAY"])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=self.drop_columns_, inplace=True)

        df = self.replace_nan.fit_transform(df)
        df = self.svcd.fit_transform(df)

        return df
