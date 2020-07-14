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

from dataclasses import dataclass
from typing import Optional, List
from warnings import warn

import pandas as pd


@dataclass
class Dataset:
    x: pd.DataFrame
    y: pd.DataFrame
    name: str = "unknown"
    meta_data: Optional[dict] = None

    def __repr__(self):
        return f' Dataset {self.name}  x.shape = {self.x.shape}  y.shape = {self.y.shape}  meta_data = {self.meta_data}'

    def __getitem__(self, is_in: pd.Series):
        """ Subset of the dataset

        Parameters
        ----------
        is_in: pd.Series
            Pandas Series with bool values indicating if a row should be
            selected.

        Returns
        -------
        dataset: Dataset
            New Dataset where x and y are chosen using `is_in`. Other
            attributes are the same.

        Notes
        -----
        A deep copy is not performed.
        """
        is_in = is_in.reindex_like(self.x)
        if is_in.hasnans:
            warn('It seems that `is_in` index is not consistent with dataset '
                 'index. Rows that do not have corresponding values in `is_in` '
                 'are dropped.')
            is_in.fillna(False, inplace=True)
        return Dataset(x=self.x[is_in], y=self.y[is_in],
                       name=self.name, meta_data=self.meta_data)

    @staticmethod
    def from_target_columns(df: pd.DataFrame, target_columns: List[str],
                            name: str = "unknown",
                            meta_data: Optional[dict] = None,
                            allow_empty_y: bool = False) -> Dataset:
        """ Factory method for creating a Dataset from list of targets.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame that contains all the data: features and targets
        target_columns: list of strings
            Columns that should become targets
        name: str, optional
            Name of the dataset
        meta_data: dict, optional
            Optional metadata
        allow_empty_y: bool, optional
            If True, will not raise an error when names from target column
            lists are not in df or if the `target_columns` is an empty list.
            In such cases, it will create a dataset with empty targets and same
            features as in df.

        Returns
        -------
        dataset: Dataset
            An object with features and targets separated.
        """
        data = df.copy()

        try:
            y = pd.concat([data.pop(c) for c in target_columns], 1)
        except (KeyError, ValueError):
            if allow_empty_y:
                return Dataset(x=df.copy(), y=None, name=name, meta_data=meta_data)
            raise

        return Dataset(x=data, y=y, name=name, meta_data=meta_data)
