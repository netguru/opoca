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
from horology import Timing

from opoca.data.dataset import Dataset
from tests.fixtures.playground_data_loader import PlaygroundDataHandler


@pytest.mark.slow
def test_default_usecase():
    # Create instance of a data loader
    # One instance can load data from one form, e.g. raw, preprocessed, normalized, etc.
    # Default data form is set in implementation of the concrete class,
    # here it is normalized form:
    data_loader = PlaygroundDataHandler()

    # Now let's load default file with data:
    df = data_loader.load()

    # Simple, isn't it? What about other files from dataset?
    # They can be loaded by supplying their names explicitly
    _ = data_loader.load('diabetes.csv')

    # Check that we actually loaded sth:
    df.describe()

    # And it is indeed normalized:
    assert df['BMI'].mean() == pytest.approx(0, abs=0.01)


@pytest.mark.slow
def test_load_raw():
    data_loader = PlaygroundDataHandler('raw')
    df = data_loader.load()

    assert isinstance(df, pd.DataFrame), 'Wrong format loaded.'
    assert 'Glucose' in df.columns, 'Missing column.'
    assert df['BMI'].mean() > 15, 'Probably normalized data was loaded instead of raw data.'


@pytest.mark.slow
def test_return_as_dataset():
    data_loader = PlaygroundDataHandler()
    dataset = data_loader.load(return_as='dataset')

    assert data_loader.package_name.count('/') == 1, 'Wrong package names.'
    assert isinstance(dataset, Dataset), 'Wrong object returned.'
    assert len(data_loader.package.meta['target_column_names']) == 1, 'Wrong meta data.'

    assert 'Glucose' in dataset.x.columns, 'Missing feature column.'
    assert 'Glucose' not in dataset.y.columns, 'Feature column in targets.'

    assert 'Outcome' in dataset.y.columns, 'Missing target column.'
    assert 'Outcome' not in dataset.x.columns, 'Target column in features.'


@pytest.mark.slow
def test_timings():
    with Timing('Create loader ') as t:
        data_loader = PlaygroundDataHandler()
    assert t.interval < 1, 'Creating a class should not load anything and should be quick.'

    with Timing('Load first time ') as t:
        _ = data_loader.load()
    assert t.interval > 0.01, f'{t.interval} seconds seems to short to load this dataset.'

    with Timing('Load second time ') as t:
        _ = data_loader.load()
    assert t.interval < 0.01, f'Result should be cached, but loading took {t.interval} seconds.'
