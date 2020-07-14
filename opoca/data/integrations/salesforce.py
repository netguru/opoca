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


import logging
import os
from typing import Dict

import pandas as pd
from dotenv import load_dotenv
from simple_salesforce import Salesforce

load_dotenv()


def get_data_frame_from_salesforce(sf_instance: Salesforce, table_name: str) -> pd.DataFrame:
    """
    Queries for data in SalesForce for a given table.
    Parameters
    ----------
    sf_instance
        Instance of SalesForce connector
    table_name
        Name of table in SalesForce
    Returns
    -------
    Table rows as DataFrame
    """
    desc = sf_instance.__getattr__(table_name).describe()

    field_names = [field['name'] for field in desc['fields']]
    soql = "SELECT {} FROM {}".format(','.join(field_names), table_name)
    results = sf_instance.query_all(soql)

    return pd.DataFrame().from_dict(results['records'])


class SFCache:
    """
    Simple cache for SalesForce connector. Assumes credentials to salesforce are stored in environment variables:
    * `SF_{prefix}_username`
    * `SF_{prefix}_password`
    * `SF_{prefix}_security_token`

    `prefix` can be `{'staging', 'production'}`.

    Parameters
    ----------
    data_root
        Path to local data storage directory
    source
        'production' or 'staging'
    """
    def __init__(self, data_root: str, source: str = "production"):
        self.cache_dir_path = os.path.join(data_root, f"sf_cache_{source}")
        os.makedirs(self.cache_dir_path, exist_ok=True)
        self.kwargs = self.__get_sf_kwargs(source)
        self.sf_instance = None

    @staticmethod
    def __get_sf_kwargs(source: str) -> Dict[str, str]:
        """
        Prepares kwargs for SalesForce connector
        """
        if source == "staging":
            kwargs = dict(domain="test")
        elif source == "production":
            kwargs = dict()
        else:
            raise ValueError("Source should be either production or staging")

        password = os.environ[f"SF_{source}_password"]
        username = os.environ[f"SF_{source}_username"]
        security_token = os.environ[f"SF_{source}_security_token"]

        credentials_dict = dict(password=password,
                                username=username,
                                security_token=security_token)

        kwargs.update(credentials_dict)

        return kwargs

    def load(self, table_name: str, refresh_cache: bool = False) -> pd.DataFrame:
        """
        Loads data from SalesForce and saves in local file. If data already exists and `force` flag is off, then it
        loads data from local file. If `force` is on, then it always loads data from SalesForce
        Parameters
        ----------
        table_name
            Name of table that should be loaded
        refresh_cache
            Enforces to not use cached local file
        Returns
        -------
        Table as data frame
        """
        logging.info(f"Loading table {table_name}")
        table_csv_path = os.path.join(self.cache_dir_path, f'{table_name}.csv')
        if os.path.exists(table_csv_path) and not refresh_cache:
            logging.debug(f"Loading data from cache located in {table_csv_path}")
            df = pd.read_csv(table_csv_path)
        else:
            logging.debug("Downloading data from Salesforce")
            if self.sf_instance is None:
                self.sf_instance = Salesforce(**self.kwargs)

            df = get_data_frame_from_salesforce(self.sf_instance, table_name)
            df.to_csv(table_csv_path)
        return df
