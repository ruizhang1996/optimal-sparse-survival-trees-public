# OSST Documentation
Implementation of [Optimal Sparse Survival Trees (OSST)](https://arxiv.org/abs/2211.14980), an optimal decision tree algorithm for survival analysis. This is implemented based on [Generalized Optimal Sparse Decision Tree framework (GOSDT)](https://github.com/ubc-systopia/gosdt-guesses). If you need classification trees, please use GOSDT. If you need regression trees, please use [Optimal Sparse Regression Trees (OSRT)](https://github.com/ruizhang1996/optimal-sparse-regression-tree-public).

---
# Installation

You may use the following commands to install OSST along with its dependencies on macOS, Ubuntu and Windows.  
You need **Python 3.9 or later** to use the module `osst` in your project.

```bash
pip3 install attrs packaging editables pandas scikit-learn sortedcontainers gmpy2 matplotlib scikit-survival
pip3 install osst
```

You need to install `gmpy2==2.0.a1` if You are using Python 3.12
# Configuration

The configuration is a JSON object and has the following structure and default values:
```json
{ 
  "regularization": 0.01,
  "depth_budget": 5,
  "minimum_captured_points": 7,
  "bucketize": false,
  "number_of_buckets": 0,
  "warm_LB": false,
  "path_to_labels": "",
  
  "uncertainty_tolerance": 0.0,
  "upperbound": 0.0,
  "worker_limit": 1,
  "precision_limit": 0,
  "model_limit": 1,
  "time_limit": 0,

  "verbose": false,
  "diagnostics": false,
  "look_ahead": true,

  "model": "",
  "timing": "",
  "trace": "",
  "tree": "",
  "profile": ""
}
```

## Key parameters

**regularization**
- Values: Decimal within range [0,1]
- Description: Used to penalize complexity. A complexity penalty is added to the risk in the following way.
  ```
  ComplexityPenalty = # Leaves x regularization
  ```
- Default: 0.01
- **Note: We highly recommend setting the regularization to a value larger than 1/num_samples. A small regularization could lead to a longer training time and possible overfitting.**

**depth_budget**
- Values: Integers >= 1
- Description: Used to set the maximum tree depth for solutions, counting a tree with just the root node as depth 1. 0 means unlimited.
- Default: 5

**minimum_captured_points**
- Values: Integers >= 1
- Description: Minimum number of sample points each leaf node must capture
- Default: 7

**bucketize**
- Values: true or false
- Description: Enables bucketization of time threshold for training
- Default: false 

**number_of_buckets**
- Values: Integers 
- Description: The number of time thresholds to which origin data mapping to if bucktize flag is set to True
- Default: 0

**warm_LB**
- Values: true or false
- Description: Enables the reference lower bound 
- Default: false

**path_to_labels**
- Values: string representing a path to a directory.
- Description: IBS loss of reference model
- Special Case: When set to empty string, no reference IBS loss are stored.
- Default: Empty string

**time_limit**
- Values: Decimal greater than or equal to 0
- Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
- Special Cases: When set to 0, no time limit is imposed.
- Default: 0


## More parameters
### Flag
**look_ahead**
- Values: true or false
- Description: Enables the one-step look-ahead bound implemented via scopes
- Default: true

**diagnostics**
- Values: true or false
- Description: Enables printing of diagnostic trace when an error is encountered to standard output
- Default: false

**verbose**
- Values: true or false
- Description: Enables printing of configuration, progress, and results to standard output
- Default: false




### Tuners

**uncertainty_tolerance**
- Values: Decimal within range [0,1]
- Description: Used to allow early termination of the algorithm. Any models produced as a result are guaranteed to score within the lowerbound and upperbound at the time of termination. However, the algorithm does not guarantee that the optimal model is within the produced model unless the uncertainty value has reached 0.
- Default: 0.0

**upperbound**
- Values: Decimal within range [0,1]
- Description: Used to limit the risk of model search space. This can be used to ensure that no models are produced if even the optimal model exceeds a desired maximum risk. This also accelerates learning if the upperbound is taken from the risk of a nearly optimal model.
- Special Cases: When set to 0, the bound is not activated.
- Default: 0.0

### Limits

**model_limit**
- Values: Decimal greater than or equal to 0
- Description: The maximum number of models that will be extracted into the output.
- Special Cases: When set to 0, no output is produced.
- Default: 1

**precision_limit**
- Values: Decimal greater than or equal to 0
- Description: The maximum number of significant figures considered when converting ordinal features into binary features.
- Special Cases: When set to 0, no limit is imposed.
- Default: 0

**worker_limit**
- Values: Decimal greater than or equal to 1
- Description: The maximum number of threads allocated to executing th algorithm.
- Special Cases: When set to 0, a single thread is created for each core detected on the machine.
- Default: 1

### Files

**model**
- Values: string representing a path to a file.
- Description: The output models will be written to this file.
- Special Case: When set to empty string, no model will be stored.
- Default: Empty string

**profile**
- Values: string representing a path to a file.
- Description: Various analytics will be logged to this file.
- Special Case: When set to empty string, no analytics will be stored.
- Default: Empty string

**timing**
- Values: string representing a path to a file.
- Description: The training time will be appended to this file.
- Special Case: When set to empty string, no training time will be stored.
- Default: Empty string

**trace**
- Values: string representing a path to a directory.
- Description: snapshots used for trace visualization will be stored in this directory
- Special Case: When set to empty string, no snapshots are stored.
- Default: Empty string

**tree**
- Values: string representing a path to a directory.
- Description: snapshots used for trace-tree visualization will be stored in this directory
- Special Case: When set to empty string, no snapshots are stored.
- Default: Empty string

---

# Example

Example code to run OSST with lower bound guessing, and depth limit. The example python file is available in [osst/example.py](/osst/example.py).

```
import pandas as pd
import numpy as np
from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.datasets import get_x_y
import pathlib


dataset_path = "experiments/datasets/churn/churn.csv"

# read the dataset
# preprocess your data otherwise OSST will binarize continuous feature using all threshold values.
df = pd.read_csv(dataset_path)
X, event, y = df.iloc[:,:-2].values, df.iloc[:,-2].values.astype(int), df.iloc[:,-1].values
h = df.columns[:-2]
X = pd.DataFrame(X, columns=h)
event = pd.DataFrame(event)
y = pd.DataFrame(y)
_, y_sksurv = get_x_y(df, df.columns[-2:], 1)
print("X shape: ", X.shape)
# split train and test set
X_train, X_test, event_train, event_test, y_train, y_test, y_sksurv_train, y_sksurv_test \
      = train_test_split(X, event, y, y_sksurv, test_size=0.2, random_state=2024)

times_train = np.unique(y_train.values.reshape(-1))
times_test = np.unique(y_test.values.reshape(-1))
print("Train time thresholds range: ({:.1f}, {:.1f}),  Test time thresholds range: ({:.1f}, {:.1f})".format(\
    times_train[0], times_train[-1], times_test[0], times_test[-1]))

# compute reference lower bounds
ref_model = RandomSurvivalForest(n_estimators=100, max_depth=3, random_state=2024)
ref_model.fit(X_train, y_sksurv_train)
ref_S_hat = ref_model.predict_survival_function(X_train)
ref_estimates = np.array([f(times_train) for f in ref_S_hat])
ibs_loss_per_sample = compute_ibs_per_sample(event_train, y_train, event_train, y_train, ref_estimates, times_train)

labelsdir = pathlib.Path('/tmp/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)

labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)

pd.DataFrame(ibs_loss_per_sample, columns=['class_labels']).to_csv(labelpath, header='class_labels', index=None)

# fit model

config = {
    "look_ahead": True,
    "diagnostics": True,
    "verbose": False,

    "regularization": 0.01,
    "uncertainty_tolerance": 0.0,
    "upperbound": 0.0,
    "depth_budget": 5,
    "minimum_captured_points": 7,

    "model_limit": 100,
    
    "warm_LB": True,
    "path_to_labels": labelpath,
  }


model = OSST(config)
model.fit(X_train, event_train, y_train)
print("evaluate the model, extracting tree and scores", flush=True)

# evaluation
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.time
print("Model training time: {}".format(time))
print("# of leaves: {}".format(n_leaves))

print("Train IBS score: {:.6f} , Test IBS score: {:.6f}".format(\
    model.score(X_train, event_train, y_train), model.score(X_test, event_test, y_test)))

S_hat_train = model.predict_survival_function(X_train)
estimates_train = np.array([f(times_train) for f in S_hat_train])

S_hat_test = model.predict_survival_function(X_test)
estimates_test = np.array([f(times_test) for f in S_hat_test])

print("Train Harrell's c-index: {:.6f}, Test Harrell's c-index: {:.6f}".format(\
    harrell_c_index(event_train, y_train, estimates_train, times_train)[0], \
    harrell_c_index(event_test, y_test, estimates_test, times_test)[0]))

print("Train Uno's c-index: {:.6f}, Test Uno's c-index: {:.6f}".format(\
    uno_c_index(event_train, y_train, event_train, y_train, estimates_train, times_train)[0],\
    uno_c_index(event_train, y_train, event_test, y_test, estimates_test, times_test)[0]))

print("Train AUC: {:.6f}, Test AUC: {:.6f}".format(\
    cumulative_dynamic_auc(event_train, y_train, event_train, y_train, estimates_train, times_train)[0],\
    cumulative_dynamic_auc(event_train, y_train, event_test, y_test, estimates_test, times_test)[0]))

print(model.tree)
```

**Output**

```
X shape:  (2000, 42)
Train time thresholds range: (0.0, 12.0),  Test time thresholds range: (0.0, 12.0)
osst reported successful execution
training completed. 4.968 seconds.
bounds: [0.168379..0.168379] (0.000000) IBS loss = 0.118379, iterations=16920
evaluate the model, extracting tree and scores
Model training time: 4.9679999351501465
# of leaves: 5
Train IBS score: 0.118379 , Test IBS score: 0.124289
Train Harrell's c-index: 0.737871, Test Harrell's c-index: 0.734727
Train Uno's c-index: 0.689405, Test Uno's c-index: 0.706680
Train AUC: 0.800940, Test AUC: 0.806016
if product_accounting_No = 1 then:
    predicted time: 4
    normalized loss penalty: 0.0
    complexity penalty: 0.01

else if csat_score_7 = 1 and product_accounting_No != 1 then:
    predicted time: 3
    normalized loss penalty: 0.0
    complexity penalty: 0.01

else if csat_score_7 != 1 and product_accounting_No != 1 and product_payroll_No = 1 then:
    predicted time: 2
    normalized loss penalty: 0.0
    complexity penalty: 0.01

else if csat_score_7 != 1 and csat_score_8 = 1 and product_accounting_No != 1 and product_payroll_No != 1 then:
    predicted time: 1
    normalized loss penalty: 0.0
    complexity penalty: 0.01

else if csat_score_7 != 1 and csat_score_8 != 1 and product_accounting_No != 1 and product_payroll_No != 1 then:
    predicted time: 0
    normalized loss penalty: 0.0
    complexity penalty: 0.01
```


# License

This software is licensed under a 3-clause BSD license (see the LICENSE file for details).

---