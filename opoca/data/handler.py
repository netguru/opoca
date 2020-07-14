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
from functools import lru_cache
from typing import Union, List, Tuple, Optional

import pandas as pd
import quilt3
from dynaconf import settings
from quilt3 import list_package_versions

from opoca.data.dataset import Dataset
from opoca.data.split import Split, SPLIT_FLAG_NAME

# Following default setting may be overridden by exporting env variables with prefix DYNACONF_, e.g.
# $ export DYNACONF_QUILT_DATA_DIR=".data/quilt"
# or by using env file (see docs of dynaconf)
QUILT_DATA_DIR = settings.get('QUILT_DATA_DIR', default='~/.quilt_data/')
S3_BUCKET = settings.get('S3_BUCKET')
QUILT_URL = settings.get('QUILT_URL')

SPLIT_TABLE_SUFFIX = '__split_table.csv'


def set_up_quilt() -> str:
    """ Set up default values in the quilt config

    Assures that QUILT_DATA_DIR folder exists and set it in quilt3 config as
    the default install location.

    Returns
    -------
    default install location: str
        Local path where packages will be downloaded
    """
    quilt3.config(QUILT_URL)
    quilt3.config(default_remote_registry=S3_BUCKET)
    default_install_location = os.path.expanduser(QUILT_DATA_DIR)
    os.makedirs(default_install_location, exist_ok=True)
    quilt3.config(default_install_location='file://' + default_install_location)

    return default_install_location


def pd_read_any(filename: str, **kwargs) -> pd.DataFrame:
    """ Wrapper around pandas read_*

    Tries to infer proper read method by file extension.

    Parameters
    ----------
    filename: str
        Path to a file with tabular data that should be read
    **kwargs:
        kwargs that are further passed to the pandas read method

    Returns
    -------
    data: pd.DataFrame
        loaded tabular data

    Raises
    ------
    ValueError: if file extensions in unknown.
    """
    data_format = filename.split('.')[-1].lower().strip()

    if data_format == 'csv':
        return pd.read_csv(filename, **kwargs)

    if data_format in ['feather', 'ftr', ]:
        return pd.read_feather(filename, **kwargs)

    if data_format == 'parquet':
        return pd.read_parquet(filename, **kwargs)

    if data_format in ['pickle', 'pkl']:
        return pd.read_pickle(filename, **kwargs)

    raise ValueError(f'Unknown data format: {data_format} of file {filename}')


def pd_save_any(df: Union[pd.Series, pd.DataFrame], filename: str, **kwargs):
    """ Wrapper around pandas to_*

    Tries to infer proper save method by file extension.

    Parameters
    ----------
    df: pd.DataFrame
        Data to be saved
    filename: str
        Path to a file where data will be saved
    **kwargs:
        kwargs that are further passed to the pandas to_ method if it accepts any

    Raises
    ------
    ValueError: if file extensions in unknown.
    """
    data_format = filename.split('.')[-1].lower().strip()

    if data_format == 'csv':
        df.to_csv(filename, **kwargs)

    elif data_format in ['feather', 'ftr', ]:
        df.to_feather(filename)

    elif data_format == 'parquet':
        df.to_parquet(filename, **kwargs)

    elif data_format in ['pickle', 'pkl']:
        df.to_pickle(filename, **kwargs)

    else:
        raise ValueError(f'Unknown data format: {data_format} of file {filename}')


class DataHandler:
    """ Base class for data loaders that are specific for each dataset

    Parameters
    ----------
    dataset_name: str
        in quilt3 refereed to as `namespace`
    data_form: str
        in quilt3 referred to as `packagename`, it can take values like:
        'zipped', 'raw', 'preprocessed' etc, depending what forms of data are
        handled in _load_* methods of subclasses.
    """

    def __init__(self, dataset_name: str, data_form: str):
        self.package_name = f'{dataset_name}/{data_form}'
        self.package: Optional[quilt3.Package] = None
        self.default_data_file = None
        self.target_column_names = None
        self.install_location = set_up_quilt()

    def __repr__(self):
        return f'{self.__class__.__name__} for {self.package_name} (local data path: {self.local_data_dir})'

    @property
    def local_package_root(self):
        return os.path.join(self.install_location, self.package_name)

    @property
    def local_data_dir(self):
        return os.path.join(self.local_package_root, 'data')

    def list(self) -> List[Tuple[str, str]]:
        """ List package revisions

        Returns
        -------
        revisions: List[Tuple[str, str]]
            A list of two-element tuples (timestamp, hash).
            The timestamp is in a form of an epoch.
            The last entry instead of a timestamp has a word 'latest' and the
            same hash as the last but one entry.
        """
        return list(list_package_versions(self.package_name, registry=S3_BUCKET))

    @lru_cache(maxsize=16)
    def load(self, filename: Optional[str] = None, return_as: str = 'pandas',
             num_samples: Optional[int] = None, **kwargs) \
            -> Union[pd.Series, pd.DataFrame, Dataset, Split]:
        """ Load data set as a pandas DataFrame

        It assures that data is downloaded and loaded only once.

        Parameters
        ----------
        filename: str, optional
            If None, default_data_file from package metadata is used. If
            given, it tries to read this file from the package. Useful when
            there are additional files with data.
        return_as: str, optional
            If 'pandas', returns loaded data as a single pandas DateFrame or
            Series.
            If 'dataset', uses target_column_names from package metadata, to
            split for X and y, and packs it in a Dataset object.
            If 'split', loads split table from a dedicated file and returns as
            a Split object.
        num_samples: int or None, optional
            In None, all samples are returned (default).
            If an int, randomly chosen subset of data will be returned. Useful
            for big data sets or for unit-testing.
        **kwargs:
            kwargs that are further passed to the pandas read method.

        Returns
        -------
        df: pd.Series, pd.DataFrame, Dataset or Split
            loaded data

        Notes
        -----
        `lru_cache` decorator caches results in case of multiple calls.

        Examples
        --------
        >>> loader = DataHandler('playground', 'normalized')
        >>> dataset = loader.load(return_as='dataset')
        ...
        >>> dataset.x.describe()
        ...
        """
        return_as = return_as.lower()
        self._install()
        local_path = self._local_path_from_filename(filename)
        data = pd_read_any(local_path, **kwargs)

        if num_samples is not None:
            data = data.sample(n=num_samples)

        if return_as == 'pandas':
            return data

        y = pd.concat([data.pop(c) for c in self.target_column_names], 1)
        dataset = Dataset(x=data, y=y,
                          name=self.package_name,
                          meta_data=self.package.meta)

        if return_as == 'dataset':
            return dataset

        if return_as == 'split':
            split_table = self._load_split_table(local_path)
            return Split.from_split_table(dataset=dataset, split_table=split_table)

        raise ValueError(f'Unrecognized `return_as` value: {return_as}.')

    def save(self, df: pd.DataFrame, filename: Optional[str] = None,
             split_table: Optional[pd.Series] = None, **kwargs) -> str:
        """ Save data set as a pandas DataFrame

        It infers a proper format from the file name extension.

        Parameters
        ----------
        df: pd.DataFrame
            Data to be saved.
        filename: str, optional
            If None, default_data_file from package metadata is used and file
            is overwritten. If given, a new file is created.
            When creating a new package, it must be not None, because no
            metadata is present.
        split_table: pd.Series, optional
            Series with flags indicating to which set each row belongs.
            If not None, a split table will be saved in the same location under
            filename constructed by appending `SPLIT_TABLE_SUFFIX` to the main
            data file. The split table is always saved as csv.
        **kwargs:
            kwargs that are further passed to the pandas to_ method.

        Returns
        -------
        local_path: str
            Path to the file where data was written.
        """
        self._install()
        local_path = self._local_path_from_filename(filename)
        pd_save_any(df, local_path, **kwargs)

        if split_table is not None:
            split_local_path = local_path + SPLIT_TABLE_SUFFIX
            pd_save_any(split_table, split_local_path, index=True, header=False)

        return local_path

    def init(self):
        """ Init a new data package

        Should be used only when creating a new dataset or data form.

        Raises
        ------
        OSError
            when package with this names already exists.

        """
        os.makedirs(self.local_data_dir, exist_ok=True)
        self.package = quilt3.Package()
        self.package.set_dir('/', self.local_package_root)

    def push(self, message='', new_meta: Optional[dict] = None):
        """ Commit and push the current content of local_package_root to the
        remote quilt server

        Parameters
        ----------
        message: str, optional
            A commit message visible in the quilt server.
        new_meta: dictionary, optional
            A dictionary with new metadata for the package.
            If None, existing metadata is used (if any).
            When pushing a new package, it is good to set at least
            `default_data_file` and `target_column_names`.
        """
        if new_meta is not None:
            meta = new_meta
        elif self.package.meta is not None:
            meta = self.package.meta.copy()
        else:
            meta = {}

        self.package.set_dir('/', self.local_package_root)
        # setting a dir removes all metadata, so we have to restore it here:
        self.package.set_meta(meta)
        self.package.push(self.package_name, message=message)
        self.package = None

    def _install(self):
        """ Download latest version of a dataset if needed

        This method is called by multiple methods. It uses quilt3 `install`
        method to download the latest version of the dataset. It avoids
        downloading data if it is already present in the default location.
        It also extracts some metadata from the package.

        """
        if self.package is not None:
            return

        quilt3.Package.install(self.package_name)
        # Following is a workaround for lack of `return` in the install
        # method in quilt 3.1.8
        self.package = quilt3.Package.browse(self.package_name)
        self.default_data_file = self.package.meta.get('default_data_file',
                                                       None)
        self.target_column_names = self.package.meta.get('target_column_names',
                                                         [])

    def _local_path_from_filename(self, filename: str):
        """ Finds an absolute local path for a given data file

        Parameters
        ----------
        filename: str or None
            If None, filename is read from package metadata.
            Keep in mind that filename may not contain '/' - nesting is not
            supported.

        Raises
        ------
        ValueError
            when filename is None and default_data_file is not specified for
            the package.
        """
        if filename is None:
            filename = self.default_data_file
            if filename is None:
                raise ValueError(f'No default_data_file for package '
                                 f'{self.package_name}. You must specify '
                                 f'filename explicitly.')

        return os.path.join(self.local_data_dir, filename)

    def _load_split_table(self, local_path: str) -> pd.Series:
        """ Load split table from a file

        Parameters
        ----------
        local_path: str
            Path to a base file with data for which a split table should be
            loaded.

        Returns
        -------
        split_table: pd.Series
            Series with int values indicating to which dataset a given row
            belongs. See `SplitFlag` enum for details.
        """
        local_path = local_path + SPLIT_TABLE_SUFFIX
        try:
            split_table = self.load(local_path, index_col=0, header=None,
                                    squeeze=True)
            return split_table.rename(SPLIT_FLAG_NAME)
        except FileNotFoundError:
            raise FileNotFoundError(f'Cannot find a split table file. It should'
                                    f' be named as the base file with data with'
                                    f' a suffix: {local_path}.')
