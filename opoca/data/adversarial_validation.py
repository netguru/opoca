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
useful links
------------
https://towardsdatascience.com/adversarial-validation-ca69303543cd
https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms
"""

from math import sqrt
from typing import Tuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from horology import timed
from pandas import get_dummies
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_CLASSIFIER_AUROC = 0.5
DECENT_CLASSIFIER_AUROC = 0.8
SIGNIFICANT_Z_SCORE = 2
EPSILON = 1.e-6


def drop_uncommon_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Drops columns that are not in both DataFrames.

    Parameters
    ----------
    df_a, df_b: pd.DataFrame
        DataFrames that should have columns synchronised.

    Returns
    -------
    df_a, df_b: pd.DataFrame
        DataFrames with the same columns.

    """
    common_columns = set(df_a.columns).intersection(set(df_b.columns))

    return df_a.loc[:, common_columns], df_b.loc[:, common_columns]


def transform_to_numeric_form(df: pd.DataFrame) -> pd.DataFrame:
    """ Transform all data to be in numeric form

    A few steps are performed:
    1. Remove timestamp from features if any, because it can be easily used to
    distinguish the train set from the test set in case of time split.
    2. Category/object columns are transformed to one-hot encoded (dummy)
    variables. It deals well with <NA>.
    3. <NA> are filled with median.

    Those steps may not be suitable for every dataset.
    They are designed to be as general as possible and to work with every data.
    Consider more hand-crafted preprocessing if any warning appear.

    Parameters
    ----------
    df: pd.DataFrame
        Initially preprocessed data that should be transformed.

    Returns
    -------
    df: pd.DataFrame
        A DataFrame with all transformation performed.

    """
    STANDARD_WARNING = '\nKeep in mind that the dataset should be initially preprocessed.'

    # remove timestamp
    columns_before = set(df.columns)
    df = df.select_dtypes(exclude='datetime')
    columns_after = set(df.columns)
    if columns_after != columns_before:
        warn('Columns with dtype=datetime were removed. You may want to use '
             'timedelta features instead.' + STANDARD_WARNING)

    # encode all categorical features as one-hot
    num_cols_before = len(df.columns)
    df = get_dummies(df)
    num_cols_after = len(df.columns)
    if num_cols_after > 10 * num_cols_before:
        warn(f'Number of columns after dummy encoding changed from '
             f'{num_cols_before} to {num_cols_after}. It can mean that:\n'
             f'(1) you have category type columns with a lof of unique variables (this is fine), or \n'
             f'(2) a numeric column was treated as category-like and every value is one-hot encoded (this is wrong).'
             f'{STANDARD_WARNING}')

    # fill <NA>
    if df.isna().any().any():
        df.fillna(df.median(), inplace=True)
        warn('Missing values are filled with medians.' + STANDARD_WARNING)

    return df


def create_adversarial_validation_data(train: pd.DataFrame, test: pd.DataFrame) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Creates adversarial validation dataset, i.e. binary classification data
    with test set being the positive class. It performs some necessary
    preprocessing - drops uncommon columns and transforms all data to numeric
    forms.


    Parameters
    ----------
    train: pd.DataFrame
        train data set
    test: pd.DataFrame
        test data set

    Returns
    -------
    x: pd.DataFrame
        processed features. For details see doc string of `fix_everything`
    y: pd.DataFrame
        binary targets: 0 for train, 1 for test

    See also
    --------
    `drop_uncommon_columns` and `transform_to_numeric_form` docstrings

    """
    train, test = drop_uncommon_columns(train, test)
    train.loc[:, 'is_test'] = 0
    test.loc[:, 'is_test'] = 1

    data = pd.concat([train, test], keys=['train', 'test'])
    data = transform_to_numeric_form(data)
    y = data.pop('is_test').astype('uint8')

    return data, y


def interpret(roc_auc: float, z_score: float) -> str:
    """ Subjective interpretation of adversarial validation.

    Parameters
    ----------
    roc_auc: float
        roc_auc score of train vs test classifier
    z_score: float
        how far are we from random classifier, measured in standard deviations

    Returns
    -------
    Subjective opinion about the situation.

    """
    if z_score < 1 - SIGNIFICANT_Z_SCORE:
        return 'Classifier did not converge on the data.'

    if z_score < SIGNIFICANT_Z_SCORE:
        if roc_auc < DECENT_CLASSIFIER_AUROC:
            return 'The test set is probably from the same distribution as the train set.'
        else:
            # This is a strange situation: roc is high, but its variance is
            # also very high yielding small z_score
            return 'Analysis is non-conclusive. Possibly more data would help.'
    else:
        if roc_auc < DECENT_CLASSIFIER_AUROC:
            return 'Test and train set differ.'
        else:
            return 'Test and train set substantially differ.'


class AdversarialValidator:
    """ Performs automagic analysis of train/test set distributions. It tries
    to fit a classifier that can distinguish if a given example is from test or
    train set. High classifier performance means that those sets differ. See
    `do_analysis` method for more details.

    Parameters
    ----------
    train: pd.DataFrame
        train data set
    test: pd.DataFrame
        test data set

    Example
    -------
    av = AdversarialValidator(train, test)
    av.do_analysis()
    av.plot_features_importances()

    """

    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.x, self.y = create_adversarial_validation_data(train, test)
        self.classifiers_: list
        self.feat_importances_: pd.Series

    def _roc_score(self, num_cv_splits: int = 5) -> Tuple[float, float]:
        """ Calculates mean area under receiver operating curve for test/train
        classifier. Mean is calculated over `num_cv_splits` folds of
        RandomForestClassifier trainings.

        This method also populates `self.feat_importances` with mean feature
        importances from each fold. Keep in mind that feature importances may
        be misleading for unbalanced data.

        Parameters
        ----------
        num_cv_splits: int, optional
            How many cross validation splits should be done.

        Returns
        -------
        roc_mean: float
            average values of auc roc score
        roc_std: float
            standard deviation of auc roc score

        """
        self.classifiers_ = []
        rocs = []
        cross_validation = StratifiedKFold(n_splits=num_cv_splits, shuffle=True)

        # this highly confusing heuristics limits number of trees in random
        # forest for small sets
        num_estimators = int(max(3, min(100, 0.5 * sqrt(len(self.y)))))

        for train_index, val_index in cross_validation.split(self.x, self.y):
            x_train, x_val = self.x.iloc[train_index], self.x.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            classifier = RandomForestClassifier(num_estimators, n_jobs=-1)
            classifier.fit(x_train, y_train)
            y_pred = classifier.predict_proba(x_val)[:, 1]
            self.classifiers_.append(classifier)

            roc = roc_auc_score(y_val, y_pred)
            rocs.append(roc)

        roc_mean = np.mean(rocs)
        roc_std = np.std(rocs)

        cv_feat_importances = pd.concat(
            [pd.Series(cl.feature_importances_, index=self.x.columns)
             for cl in self.classifiers_])

        self.feat_importances_ = \
            cv_feat_importances.groupby(cv_feat_importances.index).mean()

        return roc_mean, roc_std

    def do_analysis(self):
        """ Checks if a classifier can distinguish between train and test sets.

        It checks two values:
        - classifier auc roc scores. It should be 0.5 when it is impossible to
        distinguish sets. Values close to 1 indicate that datasets differ
        substantially.
        - z_score, which can be interpreted as distance (measured in standard
        deviations) between random classifier expected roc and actual roc.
        High z_score mean that the difference between sets is not accidental,
        but statistically significant.

        In other words, the roc can be interpreted as a measure of how the sets
        differ and z_score tells us how sure we are about it.

        See z_score explanation here:
        https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/z-score/

        Returns
        -------
        roc_mean: float
            average values of auc roc score
        z_score: float
            z score as explained above

        """
        roc_mean, roc_std = self._roc_score()
        z_score = (roc_mean - RANDOM_CLASSIFIER_AUROC) / (roc_std + EPSILON)

        print(f'mean roc: {roc_mean:.2f} (std={roc_std:.2f}) '
              f'z_score: {z_score:.2f} - {interpret(roc_mean, z_score)}')

        return roc_mean, z_score

    def drop_most_important_feature(self):
        """ Drops a columns that contributes the most to classification accuracy.

        Remarks
        -------
        Feature importances may be misleading, especially for unbalanced data.
        It is also biased towards features with high cardinality (for example
        float vs binary). Treat feature importances rather as a hint on which
        features you should inspect further.

        Returns
        -------
        most_important: str
            name of the most significant feature
        importance: float
            its importance from sk-learn

        """
        if not hasattr(self, 'feat_importances_'):
            raise AttributeError('Cannot drop a feature before analysis is performed.')

        most_important = self.feat_importances_.idxmax()
        importance = self.feat_importances_.max()
        self.x.pop(most_important)

        del self.feat_importances_

        return most_important, importance

    def plot_features_importances(self, num_largest: int = 12):
        """ Plots `num_largest` most important features as a horizontal bar plot

        Parameters
        ----------
        num_largest: int, optional
            How many features should be on the plot

        """
        if not hasattr(self, 'feat_importances_'):
            raise AttributeError('Cannot plot feature importances before analysis is performed.')
        self.feat_importances_.nlargest(num_largest).plot(kind='barh')
        plt.show()


@timed
def find_most_divergent_features(train: pd.DataFrame, test: pd.DataFrame,
                                 num_features_to_drop: int = 3,
                                 show_plots: bool = True) -> list:
    """ Use AdversarialValidator to find features that may have different
    distributions in test and train sets

    Parameters
    ----------
    train: pd.DataFrame
        train data set
    test: pd.DataFrame
        test data set
    num_features_to_drop: int, optional
        how many most relevant features should be dropped
    show_plots: bool, optional
        if bar plots should be generated

    Returns
    -------
    dropped_features: list
        list of the most divergent features

    """
    dropped_features = []
    adv_validator = AdversarialValidator(train, test)

    initial_roc, initial_z_score = adv_validator.do_analysis()
    if show_plots:
        adv_validator.plot_features_importances()

    for _ in range(num_features_to_drop):
        most_important, importance = adv_validator.drop_most_important_feature()
        dropped_features.append(most_important)

        print(f'`{most_important}` (importance={importance:.3f}) is the most '
              f'important feature allowing distinguish the train set from the '
              f'test set. It was removed. \n')

        final_roc, final_z_score = adv_validator.do_analysis()
        if show_plots:
            adv_validator.plot_features_importances()

    print(f'\n\n\n'
          f'After dropping {dropped_features}\n'
          f'roc {initial_roc:.2f} -> {final_roc:.2f}\n'
          f'z_score {initial_z_score:.2f} -> {final_z_score:.2f}\n')

    return dropped_features
