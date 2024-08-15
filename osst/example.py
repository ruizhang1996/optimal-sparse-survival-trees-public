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