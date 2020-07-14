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
import random
import string

import pytest

from tests.fixtures.playground_data_loader import PlaygroundDataHandler
from tests.fixtures.utils import is_pushed_recently, is_recently_changed


@pytest.mark.slow
def test_default_usecase():
    # load
    data_handler = PlaygroundDataHandler()
    df = data_handler.load()

    # Fix sth
    df.fillna(0, inplace=True)

    # save
    path = data_handler.save(df)

    # check if worked fine
    assert os.path.isfile(path), f'{path} should be a regular local file.'
    assert is_recently_changed(path), f'{path} was not changed as expected.'


@pytest.mark.slow
def test_save_under_different_name():
    # load
    data_handler = PlaygroundDataHandler()
    df = data_handler.load()

    # save
    path = data_handler.save(df, filename='another.pickle')

    # check if worked fine
    assert os.path.isfile(path), f'{path} should be a regular local file.'
    assert is_recently_changed(path), f'{path} was not changed as expected.'


@pytest.mark.slow
def test_simple_push():
    data_handler = PlaygroundDataHandler()
    df = data_handler.load()
    df.fillna(0, inplace=True)
    data_handler.save(df)

    data_handler.push()

    assert is_pushed_recently(data_handler), 'Push did not happen.'


@pytest.mark.slow
def test_push_of_a_new_file():
    data_handler = PlaygroundDataHandler()
    df = data_handler.load()

    # limit the data pushed with each unittest run
    df = df.iloc[:5, 2:4]

    # assure that the file is really new
    random_filename = 'test_' + ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

    data_handler.save(df, f'{random_filename}.csv')
    data_handler.push()

    assert is_pushed_recently(data_handler), 'Push did not happen.'


@pytest.mark.slow
def test_list():
    data_handler = PlaygroundDataHandler()
    revisions = data_handler.list()

    assert revisions[-1][0] == 'latest', 'Format of revision list not correct.'
    assert revisions[-1][1] == revisions[-2][1], 'Hash of the latest and most recent are not the same.'


@pytest.mark.skip('Adds a new package to S3. Deleting packages is not '
                  'straightforward, so we should not create a new one with each '
                  'tests run. It is still worth to run the test when changing '
                  'DataHandler init, save and push methods.')
def test_init_scenario():
    # load some data
    source_dh = PlaygroundDataHandler()
    df = source_dh.load()

    # process the data
    df.fillna(0, inplace=True)
    # limit the data pushed with each unittest run
    df = df.iloc[:5, 2:4]

    # create a new data_form
    # here it is random so the test can be repeated
    random_data_form = 'test_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

    target_dh = PlaygroundDataHandler(data_form=random_data_form)

    # test init
    target_dh.init()

    assert os.path.isdir(target_dh.local_data_dir), 'Local directory was not created.'
    assert target_dh.package is not None, 'Package was not initialized.'

    # test save
    target_dh.save(df, 'test.parquet')

    path = os.path.join(target_dh.local_data_dir, 'test.parquet')
    assert os.path.isfile(path), f'{path} should be a regular local file.'
    assert is_recently_changed(path), f'{path} was not changed as expected.'

    # test push to a new repo
    meta = {
        'default_data_file': 'test.parquet',
    }
    target_dh.push(new_meta=meta)

    assert is_pushed_recently(target_dh), 'Push did not happen.'
    assert not is_pushed_recently(source_dh), 'Push happened to wrong repo.'
