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


from abc import abstractmethod, ABC
from math import sqrt
from typing import Optional
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold

from opoca.data.adversarial_validation import create_adversarial_validation_data
from opoca.data.dataset import Dataset
from opoca.data.split import Split


class Splitter(ABC):
    @abstractmethod
    def split(self, data: Dataset) -> Split:
        pass


class RandomSplitter(Splitter):
    """
    Splits data randomly (uniformly) into train, val and test datasets. If `shuffle` kwarg is set to True,
    then it shuffles data before splitting.
    """

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2,
                 shuffle: bool = True, random_state: Optional[int] = None):
        self.shuffle = shuffle
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1. - train_ratio - val_ratio
        self.random_state = random_state

        assert self.train_ratio > 0
        assert self.val_ratio >= 0
        assert self.test_ratio > 0

    def split(self, data: Dataset) -> Split:
        x_train, x_test_val, y_train, y_test_val = self._split(data.x, data.y, self.train_ratio)
        train_dataset = Dataset(x=x_train, y=y_train, name=f"{data.name}/train")

        if self.val_ratio > 0:
            test_size = self.test_ratio / (self.val_ratio + self.test_ratio)
            x_test, x_val, y_test, y_val = self._split(x_test_val, y_test_val, test_size)
            val_dataset = Dataset(x=x_val, y=y_val, name=f"{data.name}/val")
            test_dataset = Dataset(x=x_test, y=y_test, name=f"{data.name}/test")
        else:  # self.val_ratio == 0
            val_dataset = None
            test_dataset = Dataset(x=x_test_val, y=y_test_val, name=f"{data.name}/test")

        return Split(train=train_dataset, val=val_dataset, test=test_dataset)

    def _split(self, x: pd.DataFrame, y: pd.DataFrame, train_size: float):
        return train_test_split(x, y, train_size=train_size, stratify=None,
                                shuffle=self.shuffle, random_state=self.random_state)


class StratifiedSplitter(RandomSplitter):
    """
    Performs stratified (with the same distribution of target variable) split into train, val and test datasets.
    Shuffles the data before performing the split.
    """

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2, random_state: Optional[int] = None):
        super().__init__(train_ratio, val_ratio, shuffle=True, random_state=random_state)

    def _split(self, x: pd.DataFrame, y: pd.DataFrame, train_size: float):
        return train_test_split(x, y, train_size=train_size, stratify=y,
                                shuffle=self.shuffle, random_state=self.random_state)


class TimeSplitter(Splitter, ABC):
    """ Abstract base class for time splitters.

    Implements `split` method as a template method which does a few checks
    necessary to assure that the time split is reliable. In subclasses, one
    should implement `_split` method.
    """

    def split(self, dataset: Dataset) -> Split:
        """ Split dataset for three subsets: for training, validation and
        testing.

        Parameters
        ----------
        dataset: Dataset
            A time series Dataset. Its index must be a DatetimeIndex.

        Returns
        -------
        split: Split
            A split with strict time separation: the latest observation from the
            train set is guaranteed to be earlier than earliest observation
            from the validation set.
            Analogically for validation/test sets.
        """
        assert dataset.x.index.is_all_dates and dataset.y.index.is_all_dates, \
            'DataFrames index must have DatetimeIndex format.'

        x = dataset.x.sort_index()
        y = dataset.y.sort_index()

        assert x.index.equals(y.index), 'x and y must have the same index.'

        x_train, y_train, x_val, y_val, x_test, y_test = self._split(x, y)

        train_dataset = Dataset(x=x_train, y=y_train, name=f"{dataset.name}/train")
        val_dataset = Dataset(x=x_val, y=y_val, name=f"{dataset.name}/val")
        test_dataset = Dataset(x=x_test, y=y_test, name=f"{dataset.name}/test")

        return Split(train=train_dataset, val=val_dataset, test=test_dataset)

    @abstractmethod
    def _split(self, x, y) -> tuple:
        pass


class ProportionalTimeSplitter(TimeSplitter):
    """ Time split based on proportions of dataset

    Parameters
    ----------
    train_ratio: float, optional
        fraction of examples to be put in the training set
    val_ratio: float, optional
        fraction of examples to be put in the validation set

    Fraction of examples in the test set is calculated from the above ones.

    See Also
    --------
    `TimeSplitter` class docs.
    """

    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1. - train_ratio - val_ratio

        assert self.train_ratio > 0
        assert self.val_ratio > 0
        assert self.test_ratio > 0

    def _split(self, x, y):
        train_val_break_ind = int(self.train_ratio * len(y.index))
        val_test_break_ind = int((self.train_ratio + self.val_ratio) * len(y.index))

        x_train, y_train = x.iloc[:train_val_break_ind], y.iloc[:train_val_break_ind]
        x_val, y_val = x.iloc[train_val_break_ind:val_test_break_ind], y.iloc[train_val_break_ind:val_test_break_ind]
        x_test, y_test = x.iloc[val_test_break_ind:], y.iloc[val_test_break_ind:]

        return x_train, y_train, x_val, y_val, x_test, y_test


class CutOffDateSplitter(TimeSplitter):
    """ Time split based on explicit cut-off dates

    Parameters
    ----------
    train_cutoff_date: str
        when training ends and validation begins
    val_cutoff_date: str
        when validation ends and testing begins

    `train_cutoff_date` must be earlier than `val_cutoff_date`.

    See Also
    --------
    `TimeSplitter` class docs.
    """

    def __init__(self, train_cutoff_date: str, val_cutoff_date: str):
        self.train_cutoff_date = pd.Timestamp(train_cutoff_date)
        self.val_cutoff_date = pd.Timestamp(val_cutoff_date)

        assert self.train_cutoff_date < self.val_cutoff_date, \
            'Train cut-off date must be earlier than val cut-off'

    def _split(self, x, y):
        x_train = x.loc[:self.train_cutoff_date]
        y_train = y.loc[:self.train_cutoff_date]
        x_val = x.loc[self.train_cutoff_date:self.val_cutoff_date]
        y_val = y.loc[self.train_cutoff_date:self.val_cutoff_date]
        x_test = x.loc[self.val_cutoff_date:]
        y_test = y.loc[self.val_cutoff_date:]

        # If the cutoff date is exactly equal to a timestamp in the index,
        # we may end up having one row in two sets,
        # because loc indexing is inclusive on both sides.
        last_train = x_train.tail(1).index
        first_val = x_val.head(1).index
        if last_train == first_val:
            x_train = x_train.iloc[:-1]
            y_train = y_train.iloc[:-1]

        last_val = x_val.tail(1).index
        first_test = x_test.head(1).index
        if last_val == first_test:
            x_val = x_val.iloc[:-1]
            y_val = y_val.iloc[:-1]

        return x_train, y_train, x_val, y_val, x_test, y_test


class AdversarialSplitter(Splitter):
    """ Split the train set into train and validation sets in such a way that
    the validation set is as much similar to the test set as possible.

    Useful when the test set is from different distribution then the train
    set. Use the `AdversarialValidator` to check if this is the case.

    Consider two scenarios:
    1. You have a train and test sets with labels. Then you simply want to get
    a subset of the train set to validate your model frequently on data similar
    to the test set, because you do not want to overfit on the test set and use
    it only to the final evaluation.
    2. Your test set has no labels. Then you have to chose a subset of the train
    set somehow to evaluate the model. `AdversarialSplitter` picks those samples
    that look like from the test set.

    Both scenarios are supported.

    Parameters
    ----------
    train: pd.DataFrame
        Test dataset is used for adversarial split fitting and is put in the
        final Split, divided into `split.train` and `split.val`.
    test: pd.DataFrame
        Test dataset is used for adversarial split fitting and is put in the
        final Split as the `split.test`. May not contain the columns which are
        defined in `target_columns`.
    target_columns: list of strings
        Names of columns that should become targets in the final split.
    val_ratio: float or None, optional
        Defines fraction of the train set that should be put in the validation
        set. If None, number of samples in the validation set will be same as
        in the test set, but no more than 50% of the train set.
    num_cv_splits: int, optional
        number of cv splits

    Remarks
    _______
    In cross validation procedure, multiple classifiers are created. They are
    then used to predict probabilities which can be interpreted how a given
    sample is likely to come from the test set. But each classifier may have
    different calibration, so comparing probabilities between elements from
    different folds may not be correct. Read more about calibration here:
    https://scikit-learn.org/stable/modules/calibration.html
    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame,
                 target_columns: list,
                 val_ratio: Optional[float] = None, num_cv_splits=5):

        self.train = train.copy()
        self.test = test.copy()
        self.target_columns = target_columns

        self.x, self.y = create_adversarial_validation_data(train, test)

        if val_ratio is None:
            self.num_val = int(min(len(test), 0.5 * len(train)))
        else:
            self.num_val = int(val_ratio * len(train))

        self.num_cv_splits = num_cv_splits

    def split(self, data: Dataset) -> Split:
        """ Perform the adversarial split

        Parameters
        ----------
        data: Dataset
            Used only to get the name of the dataset. The data inside is not
            used, because the necessary data is provided in the constructor.
            Needed to follow the `Splitter` protocol.
            In order to avoid the warning, use it with a dummy Dataset like
            this:
            >>> split = adversarial_splitter.split(Dataset(None, None, 'pima'))

        Returns
        -------
        split: Split
            Split object where `train` is divided into `split.train` and
            `split.val`, and `test` is `split.test`.

        """
        if data.x is not None:
            warn('Actual data in `data` is not used. Only data.name is read.')

        cross_validation = StratifiedKFold(n_splits=self.num_cv_splits, shuffle=True)

        # this highly confusing heuristics limits number of trees in random
        # forest for small sets
        num_estimators = int(max(3, min(100, 0.5 * sqrt(len(self.y)))))

        testlike_probas = - np.ones_like(self.y, dtype='float')

        for cv_train_index, cv_val_index in cross_validation.split(self.x, self.y):
            x_train, x_val = self.x.iloc[cv_train_index], self.x.iloc[cv_val_index]
            y_train, _ = self.y.iloc[cv_train_index], self.y.iloc[cv_val_index]

            classifier = RandomForestClassifier(num_estimators, n_jobs=-1)
            classifier.fit(x_train, y_train)

            y_pred = classifier.predict_proba(x_val)[:, 1]

            testlike_probas[cv_val_index] = y_pred

        assert np.all(testlike_probas >= 0), 'Some samples were not classified.'

        y_with_predictions = self.y.copy().to_frame()
        y_with_predictions['testlike'] = testlike_probas

        y_with_predictions_from_train_only: pd.DataFrame = y_with_predictions.loc['train']
        # This removes the train/test level from MultiIndex, so further we deal
        # with a simple index
        assert np.all(y_with_predictions_from_train_only['is_test'] == 0), \
            'Mismatch in train/test datasets.'

        y_most_testlikes = y_with_predictions_from_train_only.sort_values('testlike')
        y_most_testlikes = y_most_testlikes.tail(self.num_val)
        # We don't really care about `y_most_testlikes` values anymore, we use
        # only its index to chose samples from primary train dataset.

        new_val = self.train[self.train.index.isin(y_most_testlikes.index)]
        new_train = self.train[~self.train.index.isin(y_most_testlikes.index)]

        train_dataset = Dataset.from_target_columns(new_train, self.target_columns,
                                                    name=data.name + '/train')
        val_dataset = Dataset.from_target_columns(new_val, self.target_columns,
                                                  name=data.name + '/val')
        test_dataset = Dataset.from_target_columns(self.test, self.target_columns,
                                                   name=data.name + '/test', allow_empty_y=True)

        return Split(train_dataset, val_dataset, test_dataset)
