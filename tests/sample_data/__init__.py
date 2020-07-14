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
from typing import Optional, Union

import pandas as pd

from opoca.data.dataset import Dataset
from opoca.data.handler import DataHandler
from opoca.data.split import Split
from opoca.data.splitter import RandomSplitter


def create_dataset(data: pd.DataFrame) -> Dataset:
    x = data.drop(columns=["Outcome"])
    y = data["Outcome"].to_frame()
    return Dataset(x=x, y=y)


class DiabetesDataHandler(DataHandler):
    DATASET_NAME = 'diabetes'
    DATA_FORM = 'raw'
    SEED = 42

    def __init__(self):
        super().__init__(self.DATASET_NAME, self.DATA_FORM)
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")
        self.dataset = pd.read_csv(dataset_path)

    def load(self, filename: Optional[str] = None, return_as: str = 'pandas',
             num_samples: Optional[int] = None, **kwargs) -> Union[pd.Series, pd.DataFrame, Dataset, Split]:
        if num_samples is not None:
            data = self.dataset.sample(n=num_samples)
        else:
            data = self.dataset
        if return_as == "pandas":
            return data
        elif return_as == "dataset":
            return create_dataset(data)
        elif return_as == "split":
            dataset = create_dataset(data)
            splitter = RandomSplitter(random_state=self.SEED)
            return splitter.split(dataset)
        else:
            raise ValueError(f'Unrecognized `return_as` value: {return_as}.')
