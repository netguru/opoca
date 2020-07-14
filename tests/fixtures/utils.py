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
import time

from opoca.data.handler import DataHandler


def is_pushed_recently(handler: DataHandler, delta_s: float = 5.) -> bool:
    """
    Returns True if the package in `handler` was pushed to within last `delta_s` seconds
    """
    now = time.time()

    revisions = handler.list()
    mtime = float(revisions[-2][0])

    return now - mtime < + delta_s


def is_recently_changed(filename: str, delta_s: float = 1.) -> bool:
    """
    Returns True if the file `filename` was changed within last `delta_s` seconds
    """
    now = time.time()
    mtime = os.stat(filename).st_mtime
    return now - mtime < + delta_s
