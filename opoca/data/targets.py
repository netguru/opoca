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


"""
Module collects common targets transformers
"""
from typing import Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ToNumpy(TransformerMixin, BaseEstimator):
    """
    Converts pd.DataFrame targets to numpy array
    """
    def fit(self, y: Union[pd.DataFrame, pd.Series]):
        if isinstance(y, pd.DataFrame):
            self.columns_ = y.columns
        else:
            self.columns_ = [y.name]
        self.index_ = y.index
        return self

    def transform(self, y: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self)
        return y.to_numpy()

    def inverse_transform(self, y: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(y, columns=self.columns_, index=self.index_)
