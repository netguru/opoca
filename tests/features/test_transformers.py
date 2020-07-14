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
from math import ceil
from random import randint, sample
from typing import List

import numpy as np
import pandas as pd
import pytest

from opoca.features.transformers import BaseTransformer, ColumnsSelector, ColumnDropper, NaNColumnDropper, \
    SingleValueColumnDropper, ExtremelyDominantColumnDropper, TopKEncoder, UpperQuantileEncoder
from tests.fixtures.data import NB_EXAMPLES, NB_NUMERICAL_FEATURES


def test_base_transformer_raises_exception():
    with pytest.raises(TypeError):
        _ = BaseTransformer()


@pytest.mark.parametrize("columns", [["numerical_5", "cat_3", "bool_4"], [], ["cat_1"]])
def test_columns_selector(x: pd.DataFrame, y: pd.Series, columns: List[str]):
    columns_selector = ColumnsSelector(columns)
    x = columns_selector.fit_transform(x, y)

    assert len(x.columns) == len(columns)
    assert set(x.columns) == set(columns)


@pytest.mark.parametrize("columns", [["numerical_5", "cat_3", "bool_4"], [], ["cat_1"]])
def test_columns_dropper(x: pd.DataFrame, y: pd.Series, columns: List[str]):
    columns_dropper = ColumnDropper(columns)

    cols_before_drop = x.columns

    x = columns_dropper.fit_transform(x, y)

    assert len(cols_before_drop) - len(columns) == len(x.columns)
    assert set(cols_before_drop) == set(x.columns).union(set(columns))


def test_drop_all_columns(x: pd.DataFrame, y: pd.Series):
    columns = x.columns
    columns_dropper = ColumnDropper(columns)
    x = columns_dropper.fit_transform(x, y)

    assert len(x.columns) == 0


@pytest.mark.parametrize("cols_count", [1, 5, 10])
@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.2])
def test_nan_columns_dropper(x: pd.DataFrame, y: pd.Series, cols_count: int, threshold: float):
    nan_col_dropper = NaNColumnDropper(threshold=threshold)
    cols_to_fill = sample(x.columns.tolist(), cols_count)

    to_fill_count = randint(ceil(threshold * NB_EXAMPLES + 1), NB_EXAMPLES)

    indexes = list(range(NB_EXAMPLES))

    for col in cols_to_fill:
        fill_indexes = sample(indexes, to_fill_count)
        x.loc[fill_indexes, col] = np.nan

    x_wo = nan_col_dropper.fit_transform(x, y)

    assert len(x_wo.columns) + len(cols_to_fill) == len(x.columns)


@pytest.mark.parametrize("single_cols_count", [0, 1, 2, 4, 10])
def test_single_value_column_dropper(x: pd.DataFrame, y: pd.Series, single_cols_count: int):
    cols_to_fill_with_single_val = sample(x.columns.tolist(), single_cols_count)

    x.loc[:, cols_to_fill_with_single_val] = 0

    svcd = SingleValueColumnDropper(inplace=False)

    x_wo = svcd.fit_transform(x, y)

    assert len(x_wo.columns) + single_cols_count == len(x.columns)

    svcd = SingleValueColumnDropper(inplace=True)

    svcd.fit_transform(x, y)

    assert len(x.columns) == len(x_wo.columns)


@pytest.mark.parametrize("threshold", [0.2, 0.9, 0.99])
@pytest.mark.parametrize("extremely_dominant_cols_count", [0, 1, 5, 20, 30])
def test_extremely_dominant_column_dropper(x: pd.DataFrame, y: pd.Series, threshold: float,
                                           extremely_dominant_cols_count: int):
    numerical_cols = x.columns[:NB_NUMERICAL_FEATURES]

    x = x[numerical_cols].copy(deep=True)

    cols_to_fill = sample(x.columns.tolist(), extremely_dominant_cols_count)

    to_fill_count = randint(ceil(threshold * NB_EXAMPLES + 1), NB_EXAMPLES)

    indexes = list(range(NB_EXAMPLES))

    for col in cols_to_fill:
        fill_indexes = sample(indexes, to_fill_count)
        x.loc[fill_indexes, col] = -10000

    edcd = ExtremelyDominantColumnDropper(threshold=threshold, inplace=False)

    x_wo = edcd.fit_transform(x, y)

    assert len(x_wo.columns) + len(cols_to_fill) == len(x.columns)


@pytest.mark.parametrize("threshold", [-5, 0, 1, 3.3])
def test_extremely_dominant_threshold_out_of_scope(threshold: float):
    with pytest.raises(ValueError):
        _ = ExtremelyDominantColumnDropper(threshold=threshold, inplace=False)


@pytest.mark.parametrize("top_k", [2, 4, 6])
def test_top_k_encoder_default_path(x: pd.DataFrame, y: pd.Series, top_k: int):
    cols = sample(x.select_dtypes(["object"]).columns.tolist(), randint(1, 10))
    top_k_encoder = TopKEncoder(cols, top_k=top_k)

    with warnings.catch_warnings(record=True) as warnings_:
        x = top_k_encoder.fit_transform(x, y)
        for col in cols:
            assert len(x[col].unique()) == top_k + 1
        assert 1 >= len(warnings_) >= 0


def test_top_k_encoder_raises_value_error(x: pd.DataFrame, y: pd.Series):
    with pytest.raises(ValueError):
        _ = TopKEncoder([], top_k=0)


def test_top_k_encoder_warnings(x: pd.DataFrame, y: pd.Series):
    cols = sample(x.select_dtypes(["float"]).columns.tolist(), randint(1, 10))
    with warnings.catch_warnings(record=True) as warnings_:
        top_k_encoder = TopKEncoder(cols, top_k=5)

        _ = top_k_encoder.fit_transform(x, y)

        # should be 2 because: 1 warning for floating point, 1 warning for top k-th value having just 1 element
        assert len(warnings_) == 2


@pytest.mark.parametrize("quantile_threshold", [-5, 0, 1.0])
def test_upper_quantile_raises_value_error(x: pd.DataFrame, y: pd.Series, quantile_threshold: float):
    cols = sample(x.columns.tolist(), randint(1, 10))
    with pytest.raises(ValueError):
        _ = UpperQuantileEncoder(cols, quantile_threshold=quantile_threshold)


@pytest.mark.parametrize("quantile_threshold", [0.1, 0.3, 0.5])
def test_upper_quantile(x: pd.DataFrame, y: pd.Series, quantile_threshold: float):
    cols = sample(x.select_dtypes(["object"]).columns.tolist(), randint(1, 10))
    upper_quantile_encoder = UpperQuantileEncoder(cols, quantile_threshold=quantile_threshold)

    original_x = x.copy(deep=True)

    with warnings.catch_warnings(record=True) as warnings_:

        x = upper_quantile_encoder.fit_transform(x, y)

        for warning in warnings_:
            assert issubclass(warning.category, UserWarning)

    for col in cols:
        cumsum = original_x[col].value_counts().sort_values(ascending=False).cumsum()
        unique_values = x[col].unique()

        if UpperQuantileEncoder.OTHER_REPRESENTATION in unique_values:
            cumsum_index = len(unique_values)
        else:
            cumsum_index = len(unique_values) - 1

        count = original_x[col].loc[original_x[col].isin(unique_values)].count()
        assert (count + cumsum[cumsum_index]) / NB_EXAMPLES >= quantile_threshold
