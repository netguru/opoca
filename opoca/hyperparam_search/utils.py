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
import pickle
import subprocess
import warnings
from tempfile import NamedTemporaryFile
from typing import Optional

import mlflow

OPTUNA_DB = 'OPTUNA_DB'
MLFLOW_TRACKING_URI = 'MLFLOW_TRACKING_URI'
MLFLOW_TRACKING_USERNAME = 'MLFLOW_TRACKING_USERNAME'
MLFLOW_TRACKING_PASSWORD = 'MLFLOW_TRACKING_PASSWORD'


def check_mlflow() -> bool:
    """ Check if mlflow variables are set withing the current environment.

    It checks following variables:
    - MLFLOW_TRACKING_URI
    - MLFLOW_TRACKING_USERNAME
    - MLFLOW_TRACKING_PASSWORD

    Returns
    -------
    True, when all variables are set.
    False, otherwise

    Notes
    -----
    It does not check if the connection is actually working.

    """
    try:
        _ = os.environ[MLFLOW_TRACKING_URI]
        _ = os.environ[MLFLOW_TRACKING_USERNAME]
        _ = os.environ[MLFLOW_TRACKING_PASSWORD]
    except KeyError:
        return False

    return True


def mlflow_push_pickled(obj, name: str = '', put_in_dir: Optional[str] = None):
    """ Push a picklable object to s3 storage associated with mlflow

    It should be used within a mlflow run context.

    Parameters
    ----------
    obj:
        Any picklable object.
    name: str, optional
        File name will be constructed as: this name + 7 random chars + '.pickle'
    put_in_dir: str that is a correct directory name, optional
        If provided, the object will be placed in a folder with this name.
    """
    with NamedTemporaryFile(prefix=name, suffix='.pickle') as tempf:
        pickle.dump(obj, tempf)
        mlflow.log_artifact(tempf.name, artifact_path=put_in_dir)


def get_current_git_hash(raise_on_error: bool = False) -> Optional[str]:
    """ Return git hash of the latest commit

    Parameters
    ----------
    raise_on_error: bool, optional
        If False (default), will return None, when it fails to obtain commit hash.
        If True, will raise, when it fails to obtain commit hash.

    Returns
    -------
    Short hash of the current HEAD or None.
    """
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('utf-8').strip()
    except subprocess.CalledProcessError:
        if raise_on_error:
            raise
        warnings.warn('Probably not in a git repo.')
        git_hash = None

    return git_hash


def are_uncommitted_changes() -> bool:
    """ Check if there are any uncommitted changes

    Returns
    -------
    Bool indicating if there are any changes. False means no changes.

    Notes
    -----
    It does not look at untracked files.
    `git update-index --refresh` makes touched but not changed files to be treated correctly (as unchanged),
        ref.: https://stackoverflow.com/a/3879077/8788960
    """
    subprocess.call(['git', 'update-index', '--refresh'], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    changes = subprocess.call(['git', 'diff-index', '--quiet', 'HEAD', '--'])

    return bool(changes)


def are_untracked_files() -> bool:
    """ Check if there are any untracked files

    Returns
    -------
    Bool indicating if there are any untracked files. False means no files.
    """
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard']).decode('utf-8').strip()

    return bool(untracked)
