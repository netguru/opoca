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


import random
import string

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from opoca.data.targets import ToNumpy


@pytest.fixture
def df():
    data = np.random.normal(size=(10, 8))

    columns = [''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) for _ in range(8)]

    df = pd.DataFrame(data=data.copy(), columns=columns)
    return df


def test_to_numpy_sorted_index(df):
    to_numpy = ToNumpy()

    df.sort_values(by=df.columns[0], inplace=True)

    x = to_numpy.fit_transform(df)

    df_reconstructed = to_numpy.inverse_transform(x)

    pd.testing.assert_frame_equal(df_reconstructed, df)


def test_to_numpy_trivial(df):
    to_numpy = ToNumpy()

    x = to_numpy.fit_transform(df)

    df_reconstructed = to_numpy.inverse_transform(x)

    pd.testing.assert_frame_equal(df_reconstructed, df)


def test_raise_no_fitted_error(df):
    to_numpy = ToNumpy()

    with pytest.raises(NotFittedError):
        _ = to_numpy.transform(df)


def test_empty_data_frame():
    to_numpy = ToNumpy()
    df = pd.DataFrame(data=[], columns=[])

    x = to_numpy.fit_transform(df)

    assert x.size == 0
