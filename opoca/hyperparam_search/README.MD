# Hyperparameters optimization guide

## Naming conventions
Terms `optimization`, `tuning` and `search` are used interchangeably. 

Optuna's *trial* corresponds to mlflow *run*. And an optuna *study* is logged under one *experiment* in mlflow:

| **optuna** | **mlflow** |
| --- | --- |
| trial | run |
| study | experiment |

## Mindest
There is no golden hyperparameters optimization algorithm that works out of the box for all scenarios and models. The best algorithms have itself some parameters (ultraparameters) that influence their performance. Human assistance is usually required. A few general tips:
- optimize only relevant parameters,
- use your knowledge to choose not too wide ranges,
- use your intuition to choose distributions (uniform vs loguniform).


## Objective
You have to create a class that inherits from `Objective` and implement there `create_model` method.
It should return a model that will be fitted and scored. The key point is, that instead of hardcoding model hyperparameters, use a `trial` object to obtain suggestions:
```python
lr = 0.001  # no
lr = trial.suggest_loguniform('lr', 1e-6, 0.5)  # yes
```
Following methods are at your disposal:
- `suggest_float`
- `suggest_uniform`
- `suggest_loguniform`
- `suggest_discrete_uniform`
- `suggest_int`
- `suggest_categorical`

Check `examples.py` to see a few such implementations.

## Optimization algorithm
First you have to decide if your space of hyperparameters is simple or conditional. It becomes conditional if existence (or possible values) of some parameters depends of another parameters, e.g.:
```python
kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
if kernel == 'poly':
    degree = trial.suggest_int('degree', 2, 4)
```
You can see in this snippet, that degree is defined only when polynomial kernel is used. For other kernels, the degree is not used.

If you have many (possibly nested) conditions like this, go with TPE (Tree Parzen Estimators) - it can handle it well, because it uses trees internally. If you don't have any conditions (or maybe just one or two), go with GP (Gaussian Process). But keep in mind, that with GP, those conditional variables (`degree` in the example) will be optimized with the random search.

### How long?
You can specify number of trials or time in `POCStudy.optimize`. 

### In parallel?
Well, it's better to train a model in parallel (in sklearn it can be done by passing `n_jobs=-1` to fit method, here you can pass it by `fit_kwargs` argument in the objective constructor.) then to optimize a few models (trained on a single core) in parallel. Only the random search is truly parallel, TPE seems rather ok with it, GP is not.


## Logging results
By default, results are only logged to stdout. Consider turning on following options when you do more serious hyper-parameter search.

### ML flow
Logging to MLflow allows:
- conveniently track and visualize results of optimization,
- save trained models and other artifacts.

When using CV, score of every fold is logged (as a separate step within the run). In order to be able to sort experiments by a score averaged over folds, a 'MEAN_' prefixed value is also logged. For example: if you use 5-fold cross validation and accuracy as the metric, you will have parameters: `accuracy` with 5 values and `MEAN_accuracy` with a single value.

#### How to enable?
1. Export variables 
```shell script
export MLFLOW_TRACKING_URI=https://<xxx> MLFLOW_TRACKING_USERNAME=<xxx> MLFLOW_TRACKING_PASSWORD=<xxx>
```

2. Set `params_to_mlflow=True` and optionally `models_to_mlflow=True` 


### SQL
Logging to SQL allows:
- resume optimization,
- synchronize multiple optuna searches.

#### How to enable?
1. Export a variable 
```shell script
export OPTUNA_DB='mysql://<user>:<password>@<IP>:3306/<db instance name>'
```

2. Add your IP in the database connection settings (or enable any external connections temporarily). If you connect from a kubernetes cluster, use the private DB IP.

That's it, it is enabled by default when the env variable is set.

### MLflow and MySQL FAQ

- Can I use both?

Sure, no problem.


- Is there a big overhead of using them?

Yes, each of them adds a few seconds to a trial. It can take even more if you push big models to mlflow storage by enabling `models_to_mlflow=True`.
So if you optimize e.g. a small random forest that trains a few seconds, there is no point to use mlflow and the database. You can simply push the best model manually using mlflow. But when training a neural network (which may take hours or days or weeks) - you should use both to be able to resume the optimization and keep track of results.

- How can I continue a stopped optimization?
It's done automatically if results were logged to a database (MLflow will not help) AND 
you use the same `unique_name` in the objective. If you did not provide one, 
it was created for you using the objective name and git hash. You can find it in logs from the previous optimization.
