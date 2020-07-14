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


from __future__ import annotations  # for correct annotation of forward reference in Split

from dataclasses import dataclass
from enum import IntEnum

import pandas as pd

from opoca.data.dataset import Dataset

SPLIT_FLAG_NAME = 'split_flag'


class SplitFlag(IntEnum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    # IGNORE  can be used to mark rows that should be  omitted completely,
    # e.g. with wrong labels
    IGNORE = -1


@dataclass
class Split:
    """ A simple dataclass used for handling train/val/test splits

    `to_split_table` allows exporting the current split to a human-readable
    format and further saving it as csv. If a file name corresponds to data set
    file name, the split can be automatically reloaded with `DataHandler.load`
    method with `return_as='split'`.
    """
    train: Dataset
    val: Dataset
    test: Dataset

    def __iter__(self):
        return iter([self.train, self.val, self.test])

    def __str__(self):
        return f'Split\n  * train: {self.train}\n  * val:   {self.val}\n  ' \
               f'* test:  {self.test}\n'

    @classmethod
    def from_split_table(cls, dataset: Dataset, split_table: pd.Series) -> Split:
        """ Factory method to create a Split object from split table

        Parameters
        ----------
        dataset: Dataset
            Dataset that contains all the data or its subsample.
        split_table: pd.Series
            Series with int values indicating to which dataset a given row
            belongs. See `SplitFlag` enum for details. Any value that is not in
            SplitFlag is ignored.

        Notes
        -----
        Every element of the index in `dataset` should have a corresponding
        element in the index of `split_table`. Opposite is not necessary, i.e.
        `split_table` can have elements that are absent in `dataset` (for
        example when subsampling).

        Returns
        -------
        split: Split
            Split object with dataset split according to the `split_table`.

        """
        train_dataset = dataset[split_table == SplitFlag.TRAIN]
        train_dataset.name += '/train'
        val_dataset = dataset[split_table == SplitFlag.VAL]
        val_dataset.name += '/val'
        test_dataset = dataset[split_table == SplitFlag.TEST]
        test_dataset.name += '/test'

        return cls(train=train_dataset, val=val_dataset, test=test_dataset)

    def to_split_table(self, sort_index: bool = True) -> pd.Series:
        """ Generate a split table from a Split object

        Parameters
        ----------
        sort_index: bool, optional
            If True (default) the index is be sorted before returning the split
            table. Consider setting to False for huge data sets.

        Returns
        -------
        split_table: pd.Series
            Series with int flags indicating to which dataset each row belongs.
            See `SplitFlag` enum for exact meanings of the flags.
        """
        ind_train = pd.Series(SplitFlag.TRAIN, index=self.train.x.index)
        ind_val = pd.Series(SplitFlag.VAL, index=self.val.x.index)
        ind_test = pd.Series(SplitFlag.TEST, index=self.test.x.index)

        split_table = pd.concat([ind_train, ind_test, ind_val])

        if sort_index:
            split_table.sort_index(inplace=True)

        return split_table
