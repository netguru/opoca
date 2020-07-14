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


from typing import List, Dict, Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd


def get_col_stat(df: pd.DataFrame, column: str, stat: str) -> float:
    """
    Calculates statistics of a given column from given DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing given column.
    column : str
        Column name given as string.
    stat: str
        Name of the statistic to calculate.
        Available statistics are: ['min', 'max', 'mean', 'mode', 'median']

    Returns
    -------
    Depending of the data type in the column the result might be a float, int, or a string.

    """
    assert stat in ['min', 'max', 'mean', 'mode', 'median'], 'The stat is not supported'
    if stat == 'mode':
        result = df[column].value_counts().idxmax()
    else:
        result = df[column].apply(stat)
    return result


class MissingDataHandler(BaseEstimator, TransformerMixin):
    """
    Stores names of columns with missing data in objective specific lists or dictionaries.
    """

    def __init__(self, to_drop: List[str] = None, to_fill_with_value: Dict[str, Any] = None,
                 to_fill_with_stat: Dict[str, List[str]] = None):
        """
        Parameters
        ----------
        to_drop: list
            List of column names to drop from DataFrame
        to_fill_with_value: dict
            Dictionary whose keys are column names, and values are values to fill missing data with in that column.
            Examples
            >>> to_fill_with_value={'income': 11000, 'sex': 'F'}
        to_fill_with_stat: dict
            Dictionary whose keys are stat to calculate, and values are lists of columns (str) to fill with that stat.
            Supported stats are: ['min', 'max', 'mean', 'mode', 'median']
            Examples
            >>> to_fill_with_stat={'mean': ['revenue', 'income'], 'median': ['age']}
        """

        self.to_drop = to_drop if to_drop else []
        self.to_fill_with_value = to_fill_with_value if to_fill_with_value else {}
        self.to_fill_with_stat = to_fill_with_stat if to_fill_with_stat else {}

    def fit(self, df: pd.DataFrame):
        for stat, columns in self.to_fill_with_stat.items():
            for column in columns:
                calculated = get_col_stat(df=df, column=column, stat=stat)
                self.to_fill_with_value[column] = calculated
        self.fitted_ = True  # fitted_ flag - for check_is_fitted function in transform method
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        df.drop(self.to_drop, axis=1, inplace=True)
        for column in set(df.columns).intersection(self.to_fill_with_value):
            df[column].fillna(self.to_fill_with_value[column], inplace=True)
        return df
