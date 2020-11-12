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


import pytest

from opoca.data.handler import DataHandler
from opoca.data.splitter import ProportionalTimeSplitter, CutOffDateSplitter


@pytest.mark.xfail(reason="Quilt is not available. This will be fixed with MLBM-130")
def test_proportional_time_splitter():
    data_handler = DataHandler('room_occupancy', 'clean')
    dataset_full = data_handler.load(return_as='dataset')
    dataset_sampled = data_handler.load(return_as='dataset', num_samples=100)

    for dataset in [dataset_full, dataset_sampled]:
        sts = ProportionalTimeSplitter()
        split = sts.split(dataset)

        assert split.train.x.index.equals(split.train.y.index), 'Inconsistent index.'
        assert split.val.x.index.equals(split.val.y.index), 'Inconsistent index.'
        assert split.test.x.index.equals(split.test.y.index), 'Inconsistent index.'

        assert split.train.x.index.max() < split.val.x.index.min(), 'Overlapping train and val.'
        assert split.val.x.index.max() < split.test.x.index.min(), 'Overlapping val and test.'


@pytest.mark.xfail(reason="Quilt is not available. This will be fixed with MLBM-130")
def test_loading_time_split():
    data_handler = DataHandler('room_occupancy', 'split')
    split = data_handler.load(return_as='split')

    assert split.train.x.index.equals(split.train.y.index), 'Inconsistent index.'
    assert split.val.x.index.equals(split.val.y.index), 'Inconsistent index.'
    assert split.test.x.index.equals(split.test.y.index), 'Inconsistent index.'

    assert split.train.x.index.max() < split.val.x.index.min(), 'Overlapping train and val.'
    assert split.val.x.index.max() < split.test.x.index.min(), 'Overlapping val and test.'


@pytest.mark.xfail(reason="Quilt is not available. This will be fixed with MLBM-130")
def test_cut_off_date_splitter():
    train_cutoff_date = '2015-02-08 09:00:05'
    val_cutoff_date = '2015-02-09 09:00:00'

    data_handler = DataHandler('room_occupancy', 'clean')
    dataset_full = data_handler.load(return_as='dataset')
    dataset_sampled = data_handler.load(return_as='dataset', num_samples=100)

    for dataset in [dataset_full, dataset_sampled]:
        sts = CutOffDateSplitter(train_cutoff_date, val_cutoff_date)
        split = sts.split(dataset)

        assert split.train.x.index.equals(split.train.y.index), 'Inconsistent index.'
        assert split.val.x.index.equals(split.val.y.index), 'Inconsistent index.'
        assert split.test.x.index.equals(split.test.y.index), 'Inconsistent index.'

        assert split.train.x.index.max() < split.val.x.index.min(), 'Overlapping train and val.'
        assert split.val.x.index.max() < split.test.x.index.min(), 'Overlapping val and test.'
