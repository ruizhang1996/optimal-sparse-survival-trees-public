import pandas as pd
import numpy as np
from osst.model.osst import OSST
from osst.model.metrics import harrell_c_index, uno_c_index, integrated_brier_score, cumulative_dynamic_auc, compute_ibs_per_sample
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.datasets import get_x_y
from sklearn.preprocessing import MinMaxScaler
import pathlib
import argparse
from osst.model.utils import BaseEstimator

def first_fit():    
# first fit 

    df_new = df_normalized[['Age', 'KPS', 'A_LOG_1.0S_FO_10Percentile', 'A_LOG_1.0S_FO_90Percentile', 'A_LOG_1.0S_FO_Energy', 'A_LOG_1.0S_FO_Entropy', 'A_LOG_1.0S_FO_IQR', 'A_LOG_1.0S_FO_Kurtosis' ]]


    df_combined = pd.concat([df_new,df_target], axis = 1)
    print(df_combined )
    X, event, y = df_combined.iloc[:,:-2].values, df.iloc[:,-2].values.astype(int), df.iloc[:,-1].values
    h = df_combined.columns[:-2]
    X = pd.DataFrame(X, columns=h)
    event = pd.DataFrame(event)
    y = pd.DataFrame(y)
    _, y_sksurv = get_x_y(df_combined, df_combined.columns[-2:], 1)

    # split train and test set
    X_train, X_test, event_train, event_test, y_train, y_test, y_sksurv_train, y_sksurv_test = train_test_split(X, event, y, y_sksurv, test_size=0.2, random_state=2024)
    times_train = np.unique(y_train.values.reshape(-1))
    times_test = np.unique(y_test.values.reshape(-1))
    # compute Kaplan meier baseline
    km = BaseEstimator(event_train.values.reshape(-1), y_train.values.reshape(-1))
    km_probs = km.predict_survival_prob(times_train)
    km_estimates = np.array([km_probs]* X_train.shape[0])
    print("ibs loss of root node: ", integrated_brier_score(event_train, y_train, event_train, y_train, km_estimates, times_train))
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
        "verbose": True,

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
        
    # evaluation
    n_leaves = model.leaves()
    n_nodes = model.nodes()
    time = model.time
    print(f"Model finished execution with {df_new.columns}")
    print("# of leaves: {}".format(n_leaves))

def second_fit():
    # second fit
    df_new = df_normalized[['Age', 'KPS', 'A_LOG_1.0S_FO_10Percentile', 'A_LOG_1.0S_FO_90Percentile', 'A_LOG_1.0S_FO_Energy', 'A_LOG_1.0S_FO_Entropy' , 'A_LOG_1.0S_FO_IQR', 'A_LOG_1.0S_FO_Kurtosis', 'A_LOG_1.0S_FO_Maximum', 'A_LOG_1.0S_FO_Mean', 'A_LOG_1.0S_FO_MAD', 'A_LOG_1.0S_FO_Median' ]]



    df_combined = pd.concat([df_new,df_target], axis = 1)
    X, event, y = df_combined.iloc[:,:-2].values, df.iloc[:,-2].values.astype(int), df.iloc[:,-1].values
    h = df_combined.columns[:-2]
    X = pd.DataFrame(X, columns=h)
    event = pd.DataFrame(event)
    y = pd.DataFrame(y)
    _, y_sksurv = get_x_y(df_combined, df_combined.columns[-2:], 1)

    # split train and test set
    X_train, X_test, event_train, event_test, y_train, y_test, y_sksurv_train, y_sksurv_test = train_test_split(X, event, y, y_sksurv, test_size=0.2, random_state=2024)
    times_train = np.unique(y_train.values.reshape(-1))
    times_test = np.unique(y_test.values.reshape(-1))

    # compute Kaplan meier baseline
    km = BaseEstimator(event_train.values.reshape(-1), y_train.values.reshape(-1))
    km_probs = km.predict_survival_prob(times_train)
    km_estimates = np.array([km_probs]* X_train.shape[0])
    print("ibs loss of root node: ", integrated_brier_score(event_train, y_train, event_train, y_train, km_estimates, times_train))
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
        "verbose": True,

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
        
    # evaluation
    n_leaves = model.leaves()
    n_nodes = model.nodes()
    time = model.time
    print(f"Model finished execution with {df_new.columns}")
    print("# of leaves: {}".format(n_leaves))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("order", type=int, help="fitting order")
    args = parser.parse_args()
    order = args.order 
    
    df = pd.read_excel('LOG_S_1.0mm.xlsx')

    df_no_target = df.drop(columns = ['Event', 'Time'])

    scaler = MinMaxScaler()

    # Apply normalization to the entire dataframe
    df_normalized = pd.DataFrame(scaler.fit_transform(df_no_target), columns=df_no_target.columns)

    # Display the normalized dataframe
    df_normalized


    # Compute the correlation matrix
    corr_matrix = df_normalized.corr().abs()

    corr_matrix_dimensions = corr_matrix.shape

    print("Dimensions of the correlation matrix:", corr_matrix_dimensions)

    # Set the threshold for considering correlation as high
    threshold = 0.9

    # Find pairs of highly correlated features
    high_corr_pairs = np.where(corr_matrix > threshold)
    high_corr_pairs = [(corr_matrix.index[x], corr_matrix.columns[y]) for x, y in zip(*high_corr_pairs) if x != y and x < y]

    feature1 = 'A_LOG_1.0S_FO_IQR'  
    feature2 = 'A_LOG_1.0S_FO_Mean'  

    correlation_value = corr_matrix.loc[feature1, feature2]
    print(f"Correlation between {feature1} and {feature2}: {correlation_value}")


    feature1 = 'A_LOG_1.0S_FO_IQR'  
    feature2 = 'A_LOG_1.0S_FO_MAD'  

    correlation_value = corr_matrix.loc[feature1, feature2]
    print(f"Correlation between {feature1} and {feature2}: {correlation_value}")

    df_target = df[['Event', 'Time']]

    if order == 0:
        first_fit()
        second_fit()
    else:
        second_fit()
        first_fit()
        