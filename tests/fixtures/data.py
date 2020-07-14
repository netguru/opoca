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

from random import randint
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
from pytest_cases import fixture_plus
from sklearn.datasets import load_iris

from opoca.data.dataset import Dataset
from opoca.data.split import Split
from opoca.data.splitter import RandomSplitter
from tests.sample_data import DiabetesDataHandler

NB_EXAMPLES = 100
NB_NUMERICAL_FEATURES = 50
NB_BOOL_FEATURES = 30
MIN_CATEGORIES = 10
MAX_CATEGORIES = 30
NB_CATEGORICAL_FEATURES = 10
RANDOM_STATE = 42


def create_data_frame(data: np.ndarray, prefix: str) -> pd.DataFrame:
    cols = [f"{prefix}_{i}" for i in range(data.shape[1])]
    return pd.DataFrame(data=data, columns=cols)


def get_iris_dataset():
    iris = load_iris(as_frame=True)
    return Dataset(x=iris.data, y=iris.target.to_frame(), name='iris')


@pytest.fixture(scope="session")
def iris_dataset() -> Dataset:
    return get_iris_dataset()


@pytest.fixture(scope="session")
def iris_split() -> Split:
    dataset = get_iris_dataset()

    rs = RandomSplitter(random_state=RANDOM_STATE)
    split = rs.split(dataset)

    return split


@fixture_plus(unpack_into="x,y")
def heterogeneous_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    numerical_data = np.random.normal(size=(NB_EXAMPLES, NB_NUMERICAL_FEATURES))
    bool_data = np.random.binomial(1, 0.3, size=(NB_EXAMPLES, NB_BOOL_FEATURES)).astype(np.bool)

    categorical_data = []

    for i in range(NB_CATEGORICAL_FEATURES):
        categories_count = randint(MIN_CATEGORIES, MAX_CATEGORIES)
        population = [f"cat_{i}" for i in range(categories_count)]
        column = np.random.choice(population, size=NB_EXAMPLES, replace=True).tolist()
        categorical_data.append(column)

    categorical_data = np.column_stack(categorical_data)

    numerical_df = create_data_frame(numerical_data, "numerical")

    bool_df = create_data_frame(bool_data, "bool")

    categorical_df = create_data_frame(categorical_data, "cat")

    x = pd.concat([numerical_df, bool_df, categorical_df], axis=1)

    y = pd.DataFrame(data=np.random.binomial(1, 0.3, size=NB_EXAMPLES), columns=["Target"])

    return x, y


@pytest.fixture()
def diabetes_data_handler() -> DiabetesDataHandler:
    diabetes_data_handler = DiabetesDataHandler()
    return diabetes_data_handler
