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

import os

import pandas as pd
import pytest
from pytest import approx

from opoca.data.handler import SPLIT_TABLE_SUFFIX
from opoca.data.split import Split
from opoca.data.splitter import RandomSplitter
from tests.fixtures.playground_data_loader import PlaygroundDataHandler
from tests.fixtures.utils import is_recently_changed


def assert_splits_equal(split_a: Split, split_b: Split):
    for dataset_a, dataset_b in zip(split_a, split_b):
        pd.testing.assert_frame_equal(dataset_a.x.sort_index(), dataset_b.x.sort_index())
        pd.testing.assert_frame_equal(dataset_a.y.sort_index(), dataset_b.y.sort_index())


def test_splitter_consistency(diabetes_data_handler):
    dataset = diabetes_data_handler.load(return_as='dataset')

    rs = RandomSplitter(random_state=42)
    split = rs.split(dataset)
    split_table = split.to_split_table()

    split2 = Split.from_split_table(dataset, split_table)

    assert_splits_equal(split, split2)


@pytest.mark.slow
def test_save_split():
    data_handler = PlaygroundDataHandler()
    dataset = data_handler.load(return_as='dataset')
    df = data_handler.load(return_as='pandas')

    rs = RandomSplitter(random_state=42)
    split = rs.split(dataset)
    split_table = split.to_split_table()

    data_handler.save(df, split_table=split_table)

    # Checks
    path = os.path.join(data_handler.local_data_dir, 'diabetes.ftr' + SPLIT_TABLE_SUFFIX)
    assert os.path.isfile(path), f'{path} should be a regular local file.'
    assert is_recently_changed(path), f'{path} was not changed as expected.'


def test_load_as_split(diabetes_data_handler):
    split = diabetes_data_handler.load(return_as='split', num_samples=500)

    assert isinstance(split, Split), 'Wrong object returned.'

    assert split.train.x.shape[0] == approx(300, rel=0.2)
    assert split.val.x.shape[0] == approx(100, rel=0.2)
    assert split.test.x.shape[0] == approx(100, rel=0.2)

    assert 'Outcome' not in split.train.x.columns
    assert 'Outcome' in split.train.y.columns

    assert 'Age' in split.val.x.columns
    assert 'Age' not in split.val.y.columns
