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


import numpy as np
import pandas as pd
import pytest

from opoca.data.adversarial_validation import create_adversarial_validation_data, find_most_divergent_features


def test_create_adversarial_validation_data():
    train = pd.DataFrame({'numeric_col': pd.Series([1, 2, 3], dtype='float32'),
                          'category_col': pd.Series(['cat', 'dog', None], dtype='category')})

    test = pd.DataFrame({'numeric_col': pd.Series([2, None], dtype='float32'),
                         'category_col': pd.Series(['cat', 'cat'], dtype='category')})

    with pytest.warns(UserWarning, match='Missing values are filled with medians.'):
        x, y = create_adversarial_validation_data(train, test)

    assert np.array_equal(y.values, [0, 0, 0, 1, 1]), 'Wrong train/test labels.'
    assert x.iloc[4, 0] == 2, 'Wrong <NA> fill value.'
    assert np.array_equal(x.iloc[1, 1:].values, [0, 1]), 'Wrong encoding of dog.'
    assert np.array_equal(x.iloc[2, 1:].values, [0, 0]), 'Wrong encoding of <NA> category.'


def test_find_most_divergent_features_on_same_data(diabetes_data_handler):
    data = diabetes_data_handler.load()
    train_df = data.iloc[:500, :]
    test_df = data.iloc[500:, :]

    dropped_features = find_most_divergent_features(train_df, test_df,
                                                    num_features_to_drop=3,
                                                    show_plots=False)

    assert len(dropped_features) == 3, 'Wrong number of features dropped'
    assert 'DiabetesPedigreeFunction' in dropped_features
    assert 'Outcome' not in dropped_features


def test_find_most_divergent_features_on_biased_age(diabetes_data_handler):
    data = diabetes_data_handler.load()

    split_age = data['Age'].median()
    train_df = data[data['Age'] <= split_age]
    test_df = data[data['Age'] > split_age]

    dropped_features = find_most_divergent_features(train_df, test_df,
                                                    num_features_to_drop=1,
                                                    show_plots=False)

    assert dropped_features[0] == 'Age', 'Wrong feature was found.'


def test_find_most_divergent_features_on_biased_glucose_and_insulin(diabetes_data_handler):
    data = diabetes_data_handler.load()

    split_glucose = data['Glucose'].median()
    split_insulin = data['Insulin'].median()
    test_indices = (data['Glucose'] > split_glucose) & (data['Insulin'] > split_insulin)
    train_df = data[~ test_indices]
    test_df = data[test_indices]

    dropped_features = find_most_divergent_features(train_df, test_df,
                                                    num_features_to_drop=2,
                                                    show_plots=False)

    assert set(dropped_features) == {'Glucose', 'Insulin'}, 'Wrong features were found.'
