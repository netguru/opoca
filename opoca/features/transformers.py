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


from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class BaseTransformer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """
    Base Feature Transformer class.

    Use this class when using scikit-learn trainers.
    It supports the built-in get_feature_names method of these trainers.

    IMPORTANT: When inheriting from this class remember to update self.feature_names with names of created features
               in the fit method of the inheriting Transformer.
    """
    features_names_: List[str]

    @abstractmethod
    def __init__(self):
        self.feature_names_ = None

    def fit(self, df: pd.DataFrame, *args) -> BaseTransformer:
        self.feature_names_ = df.columns
        return self

    def transform(self, x) -> pd.DataFrame:
        x = x[self.feature_names_]
        return x

    def get_feature_names(self) -> List[str]:
        return self.feature_names_


class ColumnsSelector(BaseTransformer):
    """
    Selects columns provided as constructor argument
    """

    def __init__(self, columns_names: List[str]):
        super().__init__()
        self.columns_names = columns_names

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> ColumnsSelector:
        self.feature_names_ = self.columns_names
        return self


class ColumnDropper(BaseTransformer):
    """
    Drops columns provided as argument to constructor

    Parameters
        ----------
        columns_names
            List of columns names that shall be dropped
        inplace
            Determines whether dropping shall be done inplace
    """

    def __init__(self, columns_names: List[str], inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.columns_to_drop_ = columns_names

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> ColumnDropper:
        self.feature_names_ = list(set(x.columns) - set(self.columns_to_drop_))
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.inplace:
            x.drop(columns=self.columns_to_drop_, inplace=True, axis=1)
        else:
            x = x.drop(columns=self.columns_to_drop_, axis=1)
        return x


class NaNColumnDropper(ColumnDropper):
    """
    Drops columns that have more NaNs than threshold * number of elements in column.

    Parameters
    ----------
    threshold
        Determines what maximal fraction of NaNs we allow
    inplace
        Determines whether columns drop should be done inplace
    """

    def __init__(self, threshold: float = 0.3, inplace: bool = False):
        super().__init__([], inplace=inplace)
        self.threshold = threshold

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> NaNColumnDropper:
        columns_nan_ratio = x.isnull().sum(axis=0) / len(x)
        self.columns_to_drop_ = x.columns[(columns_nan_ratio >= self.threshold)].to_list()
        self.features_names_ = list(set(x.columns) - set(self.columns_to_drop_))
        return self


class SingleValueColumnDropper(ColumnDropper):
    """
    Drops column that have only 1 value

    Parameters
    ----------
    inplace
        Determines whether columns drop should be done inplace
    """

    def __init__(self, inplace: bool = False):
        super().__init__([], inplace=inplace)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> SingleValueColumnDropper:
        for col in x.columns:
            if len(x[col].unique()) == 1:
                self.columns_to_drop_.append(col)
        self.feature_names_ = list(set(x.columns) - set(self.columns_to_drop_))
        return self


class ExtremelyDominantColumnDropper(ColumnDropper):
    """
    Drops columns that have value that extremely dominants other values within that column

    Parameters
    ----------
    threshold
        Determines maximal fraction of mode of given column
    inplace
        Determines whether columns drop should be done inplace
    """

    def __init__(self, threshold: float = 0.99, inplace: bool = False):

        super().__init__([], inplace=inplace)
        if threshold <= 0. or threshold >= 1.0:
            raise ValueError("threshold has to be within (0, 1)")
        self.threshold = threshold

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> ExtremelyDominantColumnDropper:
        for c in x.columns:
            mode = x[c].mode()[0]
            ratio = x[c].value_counts()[mode] / len(x[c])
            if ratio > self.threshold:
                self.columns_to_drop_.append(c)
        self.feature_names_ = list(set(x.columns) - set(self.columns_to_drop_))
        return self


class NanReplacer(BaseTransformer):
    """
    Replaces predefined values with np.nan.
    By default NaN representations is `["nan", "None", "none", "NaN"]`
    """
    def __init__(self, to_replace: List[str] = None):
        super().__init__()
        if to_replace is None:
            self.to_replace = ["nan", "None", "none", "NaN", '']
        else:
            self.to_replace = to_replace

    def transform(self, x):
        return x.replace(to_replace=self.to_replace, value=np.nan)


class TypeSelector(BaseTransformer):
    """
    Selects columns which types belong to `types_list`

    Parameters
    ----------
    types_list
        List of types to be selected
    """

    def __init__(self, types_list: List[str]):
        super().__init__()
        self.types_list = types_list

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> TypeSelector:
        self.feature_names_ = x.select_dtypes(include=self.types_list).columns
        return self


class NumericalFeatureSelector(TypeSelector):
    """
    Selects numerical features
    """
    NUMERICAL_TYPES = [
        "float",
        "float16",
        "float32",
        "float64",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Uint8",
        "Uint16",
        "Uint32",
        "Uint64"
    ]

    def __init__(self):
        super().__init__(self.NUMERICAL_TYPES)


class IntegerFeatureSelector(TypeSelector):
    """
    Selects integer columns
    """
    INTEGER_TYPES = [
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64"
    ]

    def __init__(self):
        super().__init__(self.INTEGER_TYPES)


class FloatFeaturesSelector(TypeSelector):
    """
    Selects float columns
    """
    FLOAT_TYPES = [
        "float",
        "float16",
        "float32",
        "float64"
    ]

    def __init__(self):
        super().__init__(self.FLOAT_TYPES)


class CategoricalFeatureSelector(TypeSelector):
    """
    Selects categorical columns
    """
    CATEGORY_TYPES = [
        "category",
        "object"
    ]

    def __init__(self):
        super().__init__(self.CATEGORY_TYPES)


class BooleanFeatureSelector(TypeSelector):
    """
    Selects boolean columns
    """

    def __init__(self):
        super().__init__(["bool"])


class TopKEncoder(BaseTransformer):
    """
    Encodes top k values in columns and rest as `OTHER_REPRESENTATION`. As a result there are k + 1 unique values in
    encoded columns.

    Important: works in place!!!

    Parameters
    ----------
    column_names
        List of columns to be encoded
    top_k
        Number of values to be left
    """
    OTHER_REPRESENTATION = "other"

    def __init__(self, column_names: List[str], top_k: int = 5):
        super().__init__()

        if top_k < 1:
            raise ValueError("top_k has to be greater than 0")

        self.column_names = column_names
        self.top_k = top_k
        self.values = {}
        self.cols_transformed = []

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> TopKEncoder:
        if "float" in x[self.column_names].dtypes.unique():
            warn("Input data frame has floating columns. Those probably have only unique values and shouldn't "
                 "be encoded through this encoder")

        super(TopKEncoder, self).fit(x)

        for col in self.column_names:
            values = x[col].value_counts().index.tolist()
            if len(values) <= self.top_k:
                warn(f"top_k is greater than number of unique values in column {col}")
            if x[col].value_counts().values[self.top_k] == 1:
                warn("top k-th has only 1 element. Consider lower k")
            self.values[col] = values[:self.top_k]

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        to_be_transformed = list(set(self.column_names) - set(self.cols_transformed))
        for col in to_be_transformed:
            other_indices = ~x[col].isin(self.values[col])
            nan_indices = x[col].isna()
            x.loc[other_indices, col] = self.OTHER_REPRESENTATION
            x.loc[nan_indices, col] = self.OTHER_REPRESENTATION
        return x


class UpperQuantileEncoder(BaseTransformer):
    """
    Encodes upper quantile values in column. This transformer computes cumulative sum of desceding sorted value counts
    for particular columns. Then, it divides them by total number of non empty values. Then it leaves (as they are )
    values that are lower than input quantile threshold and encodes rest to 'Other'.

    IMPORTANT: works in place !!!

    Example:
    >> df = pd.DataFrame({"col": [1, 1, 2, 1, 2, 0, 3, 7]})
    >> uqe = UpperQuantileEncoder(["col"], quantile_threshold=0.625)
    >> uqe.fit_transform(df)
         col
    0      1
    1      1
    2      2
    3      1
    4      2
    5  other
    6  other
    7  other

    Parameters
    ----------
    column_names
        Names of columns that shall be encoded
    quantile_threshold
        Determines which values to leave as they are (lower than quantile_threshold)
    """
    OTHER_REPRESENTATION = "other"

    def __init__(self, column_names: List[str], quantile_threshold: float = 0.5):
        super().__init__()
        if quantile_threshold <= 0 or quantile_threshold >= 1.0:
            raise ValueError("quantile_threshold has to be within (0, 1)")
        self.quantile_threshold = quantile_threshold
        self.column_names = column_names
        self.column_to_staying_values = {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> UpperQuantileEncoder:
        if "float" in x[self.column_names].dtypes.unique():
            warn("Input data frame has floating columns. Those probably have only unique values and shouldn't "
                 "be encoded through this encoder")

        super(UpperQuantileEncoder, self).fit(x)

        for col in self.column_names:
            top_percentile_values = self.__get_top_percentile_values(x[col])
            if len(top_percentile_values) == 0:
                warn(f"quantile_threshold is too low. {col} won't be changed")
            else:
                self.column_to_staying_values[col] = top_percentile_values

        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        for col, staying_values in self.column_to_staying_values.items():
            other_indices = ~x[col].isin(staying_values)
            nan_indices = x[col].isna()
            x.loc[other_indices, col] = self.OTHER_REPRESENTATION
            x.loc[nan_indices, col] = self.OTHER_REPRESENTATION
        return x

    def __get_top_percentile_values(self, ps: pd.Series):
        total = ps.value_counts().sum()
        cumsum = ps.value_counts().sort_values(ascending=False).cumsum() / total
        return set(cumsum.loc[cumsum <= self.quantile_threshold].index)


class CurrencyTransformer(BaseTransformer):
    """
    Transforms columns with amount with respect to `CURRENCY_EXCHANGE_MULTIPLIER` dictionary.

    Parameters
    ----------
    amount_cols:
        List of amount columns to be transformed
    currency_col:
        Column that determines currency
    """
    CURRENCY_EXCHANGE_MULTIPLIER = {
        'PLN': 1,
        'EUR': 4,
        'USD': 4,
        'GBP': 6
    }

    def __init__(self, amount_cols: List[str], currency_col: str):
        super().__init__()
        self.currency_col = currency_col
        self.amount_cols = amount_cols

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        multiplier = x[self.currency_col].map(self.CURRENCY_EXCHANGE_MULTIPLIER)
        for c in self.amount_cols:
            x[c] = x[c].multiply(multiplier, axis="index")
        return x


class LogTransformer(BaseTransformer):
    """
    Applies log to columns provided in constructor. It uses log(x+1) in order to make sure it does not explode for small
    values.

    Parameters
    ----------
    cols:
        list of columns that log transformation shall be applied to
    inplace:
        Determines whether log transformation is done inplace. If not, then separate columns are added for log
        transformation with suffixes `_log`
    """

    def __init__(self, cols: List = None, inplace: bool = True):
        super().__init__()
        self.in_place = inplace
        self.cols = cols

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> LogTransformer:
        if self.cols is None:
            self.cols = x.columns
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if self.in_place:
            cols = self.cols
        else:
            cols = [f'{col}_log' for col in self.cols]
        x[cols] = np.log1p(x[self.cols])
        return x


class Scaler(BaseTransformer):
    """
    Wraps around `StandardScaler` so it provides `get_feature_names`, needed in complex pipelines

    Parameters
    ----------
    copy:
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean:
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std:
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    """

    def __init__(self, copy: bool = True, with_mean: bool = True, with_std: bool = True):
        super().__init__()
        self.scaler = StandardScaler(copy=copy, with_mean=with_mean, with_std=with_std)

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> Scaler:
        self.scaler.fit(x)
        self.feature_names_ = x.columns.tolist()
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(x)


class ScaleByMean(BaseTransformer):
    """
    Scales feature `numerical_feature` by mean according to grouping by categorical or boolean feature `group_feature`
    """

    def __init__(self, numerical_feature: str, group_feature: str):
        """

        Parameters
        ----------
        numerical_feature
            Numerical feature name that transformation is applied to
        group_feature
            Categorical or boolean feature name that is used for aggregation
        """

        super().__init__()
        self.numerical_feature = numerical_feature
        self.group_feature = group_feature
        self.name = f'{self.numerical_feature}_{self.group_feature}_scale_by_mean'

    def fit(self, x: pd.DataFrame, y: pd.DataFrame = None) -> ScaleByMean:
        self.feature_names_ = x.columns.tolist()
        self.feature_names_.append(self.name)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.name] = x.groupby(self.group_feature)[self.numerical_feature].apply(lambda x: x / (np.mean(x) + 1.e-6))
        return x
