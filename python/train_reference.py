# an example script to produce the file that stores the IBS of reference model
from sksurv.ensemble import RandomSurvivalForest
from sksurv.datasets import get_x_y
import numpy as np
import pandas as pd


def compute_ibs_by_curves(curves, c, y):
    # sort y
    sort_order = np.argsort(y)
    sorted_c = c[sort_order]
    reverse_c = 1 - c
    sorted_y = y[sort_order]
    # normalize y
    # sorted_y = sorted_y / sorted_y[-1]
    sorted_curves = curves[sort_order]
    target_values = np.unique(sorted_y)

    ipcw = np.full(target_values.size, -1.0)
    prod = 1
    for idx, value in enumerate(target_values):
        captured_set = (y == value)
        prod *= (1 - sum(reverse_c[captured_set]) / np.where(sorted_y >= value)[0].size)
        ipcw[idx] = 1 / prod if prod else 0

    ibs = []
    for i, curve in enumerate(curves):
        # each sample
        ibs.append(0)
        for idx, threshold in enumerate(target_values[:-1]):
            if threshold < curve.x.min():
                threshold_tmp = curve.x.min()
                if threshold < y[i]:
                    ibs[i] += (curve(threshold_tmp) - 1) ** 2 * ipcw[idx] * (target_values[idx + 1] - threshold)
                elif c[i]:
                    ibs[i] += (curve(threshold_tmp)) ** 2 * ipcw[np.where(target_values >= y[i])[0][0]] * (
                                target_values[idx + 1] - threshold)
            elif threshold > curve.x.max():
                threshold_tmp = curve.x.max()
                if threshold < y[i]:
                    ibs[i] += (curve(threshold_tmp) - 1) ** 2 * ipcw[idx] * (target_values[idx + 1] - threshold)
                elif c[i]:
                    ibs[i] += (curve(threshold_tmp)) ** 2 * ipcw[np.where(target_values >= y[i])[0][0]] * (
                                target_values[idx + 1] - threshold)

            else:
                if threshold < y[i]:
                    ibs[i] += (curve(threshold) - 1) ** 2 * ipcw[idx] * (target_values[idx + 1] - threshold)
                elif c[i]:
                    ibs[i] += (curve(threshold)) ** 2 * ipcw[np.where(target_values >= y[i])[0][0]] * (
                                target_values[idx + 1] - threshold)

    ibs = np.array(ibs) / sorted_y[-1]
    return ibs / len(y)


rf = RandomSurvivalForest(n_estimators=100, max_depth=3, random_state=2023)
df = pd.read_csv("../experiments/datasets/churn/churn.csv")
X, y = get_x_y(df, df.columns[-2:], 1)
rf.fit(X, y)
curves = rf.predict_survival_function(X)
c_train = df[df.columns[-2]].to_numpy()
y_train = df[df.columns[-1]].to_numpy()

warm_labels = compute_ibs_by_curves(curves, c_train, y_train)

label_path = 'churn_reference.tmp'
pd.DataFrame(warm_labels).to_csv(label_path, header="class_labels", index=None)
