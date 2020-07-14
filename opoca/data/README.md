# Data handling with quilt3

## Definition
Let's have an examples to understand what is a *dataset* and what is a *dataform*.
We have a `Santander` dataset (from kaggle) which consists of three files: 
a training set, an unlabeled test set, and a sample submission. This dataset has 3 *data forms*:
- raw: files as provided in kaggle, in csv format.
- preprocessed: with NAs filled and column types properly encoded, compressed, in feather format.
- transformed: with carefully engineered features for model training, also with a defined train-val-test split, in pickle format.

Those are different *dataforms* of one dataset. See [this](https://netguru.atlassian.net/l/c/dqAZnz0b) for more inspirations.

On the other hand, in quilt we have:
- catalog - a server that can show content of a few files and allows powerful search, 
in our case it is located under `DYNACONF_QUILT_URL`
- registry, in our case it is `DYNACONF_S3_BUCKET` which corresponds to a S3 bucket on AWS.
- in the registry we have multiple users (or in general *namespaces*).
- in a namespace we have multiple packages.
- each package can have multiple files.

__Important:__ a quilt namespace corresponds to our dataset and a quilt package corresponds to our dataform.

Quilt treats a package as a whole (so downloads always the whole package). 
Thanks to the fact, that we have a separate quilt package for each dataform, 
we can only download the form of the data that we need at the current stage of work.
For example, when we are interested only in training a new model, we just load `transformed` dataform,
without a need to download raw format.

Now we can start actually using quilt.

## Setup
### Install the quilt
If you use a conda environment, activate it first.

Install quilt with `pip`:
```bash
pip install quilt3==3.1.8
```

### Other versions
Note that all the code was implemented and tested with this particular version in mind. 
Version `3.1.9` is known to introduce some bugs which were supposed to be fixed in version `3.1.10`, but we did not check it.
If you insist on using other version of quilt, first run all unit tests to catch the problems.
After upgrade from `3.1.8` to `3.1.9`, downgrade fails and env must be recreated. There may be similar problems in other versions.

### Login
Run the following command (in main `ml-poc-template` dir):
```bash
python -m poc.quilt_one_time_setup
```
A web browser will open. After you log in to quilt, copy a token. 
It will be saved on the local machine and this step will not be required anymore.


## How to create a new dataset
| | scenario | when | quilt side|
| ---: | --- | --- | --- |
| 1 | create a dataset with completely new data | on the stage of data engineering of a new project | create a new namespace and a package inside|
| 2 | add a new form of existing data | for example, after some preprocessing or feature engineering | create a package inside an existing namespace |
| 3 | modify an existing dataset | for example, when new data arrives, when wrong labels are fixed | push a new revision of a package |

Next section covers the two first cases - each form of the data is treated as 
a separate repo by quilt, so both cases look the same.
 
### Create a new dataset or a new form of data (scenarios 1 and 2)
0. Imports
```python
from opoca.data.handler import DataHandler
```

1. Prepare your data as a pandas DataFrame
```python
df: pd.DataFrame
```


2. Create a data handler and init it
```python
data_handler = DataHandler('playground', 'raw')
#                          ^             ^
#                          dataset name  |
#                                        dataform
data_handler.init()
```
It will create a quilt root directory `~/.quilt_data/` (if does not exist yet)
and directories inside.
You can override the location by setting an environment variable `DYNACONF_QUILT_DATA_DIR=/other/cool/path/`.



3. Save the data to the file or files
```python
data_handler.save(df, 'data.parquet')
```

`save` method can handle multiple data formats automatically:
- csv,
- pickle,
- parquet,
- feather.

See details in docs of `poc.data.handler.pd_save_any`.

4. Prepare the metadata (optional, but recommended)
```python
meta = dict(
    # Name of the file that should be loaded when `load` method is called
    # with default arguments. This is a path relative to the 'data' directory.
    # Optional but convenient to set.

    default_data_file='test.parquet',


    # Names of the columns that will be popped from the data and put to a
    # separate DataFrame as y when loading with `return_as='dataset'`
    # If not set, `load` cannot be called with this flag.
    # Optional but convenient to set.

    target_column_names=['ind_nomina_ult1', 'ind_cno_fin_ult1', ...],

    )
```


5. Push the package to s3
```python
data_handler.push('Add super normalized data', new_meta=meta)
```
You can supply a commit message.

That's all. Let the other know that they can use your data.

### Modify an existing dataset (scenario 3)
Follow similar way as above, but do not call `init` method.
```python
data_handler = PlaygroundDataHandler()
df = data_handler.load()
# process somehow
df.fillna(0, inplace=True)
# and simply save and push with default values
data_handler.save(df)
data_handler.push('Fix <NA>')
```

## How to read data
Follow the documentation in the source code.
