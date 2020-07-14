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

from opoca.data.adversarial_validation import AdversarialValidator
from opoca.data.dataset import Dataset
from opoca.data.splitter import AdversarialSplitter


def test_adversarial_splitter(diabetes_data_handler):
    diabetes_frame = diabetes_data_handler.load()

    # Create highly different train and test sets.
    test_indices = (diabetes_frame['Age'] + 3 * diabetes_frame['Pregnancies'] <= 28)
    train_df = diabetes_frame[~test_indices]
    test_df = diabetes_frame[test_indices]

    print('train:')
    print(train_df.sample(5).T)

    print('test (young):')
    print(test_df.sample(5).T)

    adv_splitter = AdversarialSplitter(train_df, test_df, target_columns=['Outcome'])

    split = adv_splitter.split(Dataset(None, None, 'pima'))

    print('val:')
    print(split.val.x.sample(5).T)

    def mean_bio_age(d: pd.DataFrame):
        return d['Age'].mean() + 3 * d['Pregnancies'].mean()

    mean_bio_age_train_before_split = mean_bio_age(train_df)
    mean_bio_age_train = mean_bio_age(split.train.x)
    mean_bio_age_val = mean_bio_age(split.val.x)
    mean_bio_age_test = mean_bio_age(split.test.x)

    print('\nMean bio age:')
    print(f'train before {mean_bio_age_train_before_split:.2f}')
    print(f'train {mean_bio_age_train:.2f}')
    print(f'val {mean_bio_age_val:.2f}')
    print(f'test {mean_bio_age_test:.2f}')

    assert mean_bio_age_test < mean_bio_age_val < mean_bio_age_train, \
        'Adversarial split did not get the difference.'

    av_tt = AdversarialValidator(split.train.x, split.test.x)
    roc_tt, _ = av_tt.do_analysis()

    av_vt = AdversarialValidator(split.val.x, split.test.x)
    roc_vt, _ = av_vt.do_analysis()

    assert roc_tt > roc_vt, \
        'The validation set is not closer to the test set than the train set.'
