from .utils import BaseEstimator
import numpy as np

"""
This module includes code from pakcage scikit-survival: https://github.com/sebp/scikit-survival
    - Source File: sksurv/metrics.py
    - License: GPLv3
    - Modifications: Optimized for metrics computing precision, 
                     handling more general cases: e.g. y_test is outside the scope of y_train; y_train and y_test contains zero.  
"""

def _integrated_brier_score(G, event_test, y_test, estimates, times):
    """ Helper function for osst score function: compute the IBS score 
    
    """
    brier_scores = _brier_score(G, event_test, y_test, estimates, times)
    ibs = np.sum([ brier_scores[i] * (times[i+1] - times[i]) for i in range(len(times) - 1)]) / (times[-1] - times[0])
    return ibs 


def _brier_score(G, event_test, y_test, estimates, times):
    """
    Helper function for osst score function: compute the time-dependent brier scores
    """
    n_times = times.shape[0]
    ipcw_times = G.predict_ipcw(times)
    ipcw_y_test = G.predict_ipcw(y_test)


    brier_scores = np.zeros(n_times)
    for i in range(n_times):
        S_hat = estimates[:, i]
        brier_scores[i] = np.mean(np.square(S_hat) * ((y_test <= times[i]) & event_test).astype(int) * ipcw_y_test + 
                                  np.square(1.0 - S_hat) * (y_test > times[i]).astype(int) * ipcw_times[i])
        
    return brier_scores

def integrated_brier_score(event_train, y_train, event_test, y_test, estimates, times):
    """ The Integrated Brier Score (IBS)

    Parameters
    ----------
    event_train : array-like, shape = (n_train_samples,) or (n_train_samples, 1)
                  column containing event indicator for each training sample

    y_train : array-like, shape = (n_train_samples,) or (n_train_samples, 1)
              column containing the observed time for each training sample


    event_test : array-like, shape = (n_test_samples,) or (n_train_samples, 1)
                 column containing event indicator for each testing sample

    y_test : array-like, shape = (n_test_samples,) or (n_train_samples, 1)
              column containing the observed time for each testing sample

    estimate : array-like, shape = (n_test_samples, n_times)
        Estimated survival probability of remaining event-free at time points
        specified by `times`. The value of ``estimates[i][j]`` must correspond to
        the estimated probability of remaining event-free up to the time point
        the predicted survival proability of sample `i` at ``times[j]``. 
        Typically, estimated probabilities are obtained via the
        survival function returned by an estimator's ``predict_survival_function`` method.

    times : array-like, shape = (n_times,) 
        The time points for which to estimate the time-dependent brier scores.
        Values must be sorted in ascending order.


    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------

    ibs : float
        The integrated Brier score.


    """
    if len(event_train.shape) > 1:
        event_train = event_train.values.reshape(-1)
    if len(y_train.shape) > 1:
        y_train = y_train.values.reshape(-1)
    if len(event_test.shape) > 1:
        event_test = event_test.values.reshape(-1)
    if len(y_test.shape) > 1:
        y_test = y_test.values.reshape(-1)

    brier_scores = brier_score(event_train, y_train, event_test, y_test, estimates, times)
    ibs = np.sum([ brier_scores[i] * (times[i+1] - times[i]) for i in range(len(times) - 1)]) / (times[-1] - times[0])
    return ibs


def brier_score(event_train, y_train, event_test, y_test, estimates, times):
    """ Time-dependent Brier score

    Parameters
    ----------
    event_train : array-like, shape = (n_train_samples,)
                  column containing event indicator for each training sample

    y_train : array-like, shape = (n_train_samples,)
              column containing the observed time for each training sample


    event_test : array-like, shape = (n_test_samples,)
                 column containing event indicator for each testing sample

    y_test : array-like, shape = (n_test_samples,)
              column containing the observed time for each testing sample

    Estimated survival probability of remaining event-free at time points
        specified by `times`. The value of ``estimates[i][j]`` must correspond to
        the estimated probability of remaining event-free up to the time point
        the predicted survival proability of sample `i` at ``times[j]``. 
        Typically, estimated probabilities are obtained via the survival function 
        returned by an estimator's ``predict_survival_function`` method.

    times : array-like, shape = (n_times,)
        The time points for which to estimate the Brier score.
        Values must be within the range of follow-up times of
        the test data `survival_test`.

    Returns
    -------
    brier_scores : array , shape = (n_times,)
        Values of the brier score.
    """
    n_times = times.shape[0]
    assert n_times == estimates.shape[1]
    G = BaseEstimator(event=event_train, y=y_train, reverse=True)
    ipcw_times = G.predict_ipcw(times)
    ipcw_y_test = G.predict_ipcw(y_test)


    brier_scores = np.zeros(n_times)
    for i in range(n_times):
        S_hat = estimates[:, i]
        brier_scores[i] = np.mean(np.square(S_hat) * ((y_test <= times[i]) & event_test).astype(int) * ipcw_y_test
                                 + np.square(1.0 - S_hat) * (y_test > times[i]).astype(int) * ipcw_times[i])
        
    return brier_scores



def _find_all_comparable_pairs(event, time, order):
    """ Helper function for _weighted_c_index: find all comparable pairs
    """
    n = len(time)
    i = 0
    while i < n - 1:
        time_i = time[order[i]]
        end = i + 1
        while end < n and event[order[end]] == time_i:
            end += 1
        
        censored_at_same_time = ~event[order[i:end]]
        for j in range(i, end):
            if event[order[j]] > 0:
                comparable_idx = np.zeros(n, dtype=bool)
                comparable_idx[end:] = True 
                comparable_idx[i:end] = censored_at_same_time
                
                yield (j, comparable_idx)
        
        i = end



def _weighted_c_index(event, time, estimates, threshold_mapping, weights, tied_tol=1e-8):
    """ Helper function for harrell_c_index and uno_c_index 
    """
    order = np.argsort(time)
    num_concordant = 0
    num_tied = 0
    num_comparable = 0
    numerator = 0.0
    denominator = 0.0
    for i, comparable_idx in _find_all_comparable_pairs(event, time, order):
        est_i = estimates[order[i]][threshold_mapping[order[i]]]
        event_i = event[order[i]]
        w_i = weights[order[i]]
        est = estimates[order[comparable_idx]][:, threshold_mapping[order[i]]]
        assert event_i > 0

        concordant = est > est_i
        ties = np.absolute(est - est_i) <= tied_tol
        n_ties = ties.sum()
        n_con = concordant[~ties].sum()
        n_comparable = comparable_idx.sum()

        numerator += w_i * n_con + 0.5 * w_i * n_ties
        denominator += w_i * n_comparable 

        num_comparable += n_comparable 
        num_concordant += n_con
        num_tied += n_ties

    
    if num_comparable < 1:
        raise Exception("No comparable pairs available.")
    
    return numerator / denominator, num_concordant, num_tied, num_comparable





def harrell_c_index(event, y, estimates, times):
    """Concordance index for right-censored data
    
    Parameters
    ----------
    event : array-like, shape = (n_samples,) or (n_samples, 1)
                  column containing event indicator for each testing sample

    y : array-like, shape = (n_samples,) or (n_samples, 1)
              column containing the observed time for each testing sample

    estimate : array-like, shape = (n_samples, n_times)
        Estimated survival probability of remaining event-free at time points
        specified by `times`. The value of ``estimates[i][j]`` must correspond to
        the estimated probability of remaining event-free up to the time point
        the predicted survival proability of sample `i` at ``times[j]``. 
        Typically, estimated probabilities are obtained via the
        survival function returned by an estimator's ``predict_survival_function`` method.

    times : array-like, shape = (n_times,) 
        The time points for which to estimate the harrell's c-index score.
        Values must be sorted in ascending order.


    Returns
    -------
    cindex : float
        Concordance index

    num_concordant : int
        Number of concordant pairs

    num_tied : int
        Number of tied pairs

    num_comparable : int
        Number of comparable pairs

        
    """
    if len(event.shape) > 1:
            event = event.values.reshape(-1)
    if len(y.shape) > 1:
        y = y.values.reshape(-1)

    if estimates.shape[0] != len(event) or estimates.shape[0] != len(y):
        raise ValueError("Estimates should have same row number as length of event and y")
    if estimates.shape[1] != len(times):
        raise ValueError("Estimates should have same column number as number of time thresholds ")
    weights = np.ones_like(y)
    time_threshold_mapping = np.searchsorted(times, y, side='left')
    assert (times[time_threshold_mapping] == y).all()

    return _weighted_c_index(event, y, estimates, time_threshold_mapping, weights)
    
def uno_c_index(event_train, y_train, event_test, y_test, estimates, times):
    """ Concordance index for right-censored data based on inverse probability of censoring weights.

    Parameters
    ----------
    event_train : array-like, shape = (n_train_samples,) or (n_train_samples, 1)
                  column containing event indicator for each training sample

    y_train : array-like, shape = (n_train_samples,) or (n_train_samples, 1)
              column containing the observed time for each training sample


    event_test : array-like, shape = (n_test_samples,) or (n_train_samples, 1)
                 column containing event indicator for each testing sample

    y_test : array-like, shape = (n_test_samples,) or (n_train_samples, 1)
              column containing the observed time for each testing sample

    estimate : array-like, shape = (n_test_samples, n_times)
        Estimated survival probability of remaining event-free at time points
        specified by `times`. The value of ``estimates[i][j]`` must correspond to
        the estimated probability of remaining event-free up to the time point
        the predicted survival proability of sample `i` at ``times[j]``. 
        Typically, estimated probabilities are obtained via the
        survival function returned by an estimator's ``predict_survival_function`` method.

    times : array-like, shape = (n_times,) 
        The time points for which to estimate the uno's c-index score.
        Values must be sorted in ascending order.


    Returns
    -------
    cindex : float
        Concordance index

    num_concordant : int
        Number of concordant pairs

    num_tied : int
        Number of tied pairs

    num_comparable : int
        Number of comparable pairs

    """
    if len(event_train.shape) > 1:
        event_train = event_train.values.reshape(-1)
    if len(y_train.shape) > 1:
        y_train = y_train.values.reshape(-1)
    if len(event_test.shape) > 1:
        event_test = event_test.values.reshape(-1)
    if len(y_test.shape) > 1:
        y_test = y_test.values.reshape(-1)

    if estimates.shape[0] != len(event_test) or estimates.shape[0] != len(y_test):
        raise ValueError("Estimates should have same row number as length of event_test and y_test")
    if estimates.shape[1] != len(times):
        raise ValueError("Estimates should have same column number as number of time thresholds ")
    
    # compute ipcw
    G = BaseEstimator(event=event_train, y=y_train, reverse=True)
    weights = np.square(G.predict_ipcw(y_test))

    time_threshold_mapping = np.searchsorted(times, y_test, side='left')
    assert (times[time_threshold_mapping] == y_test).all()
    return _weighted_c_index(event_test, y_test, estimates, time_threshold_mapping, weights)


def cumulative_dynamic_auc(event_train, y_train, event_test, y_test, estimates, times, tied_tol=1e-8):
    """ cumulative/dynamic AUC metric for right-censored time-to-event data.

    Parameters
    ----------
    event_train : array-like, shape = (n_train_samples,) or (n_train_samples, 1)
                  column containing event indicator for each training sample

    y_train : array-like, shape = (n_train_samples,) or (n_train_samples, 1)
              column containing the observed time for each training sample


    event_test : array-like, shape = (n_test_samples,) or (n_train_samples, 1)
                 column containing event indicator for each testing sample

    y_test : array-like, shape = (n_test_samples,) or (n_train_samples, 1)
              column containing the observed time for each testing sample

    estimate : array-like, shape = (n_test_samples, n_times)
        Estimated survival probability of remaining event-free at time points
        specified by `times`. The value of ``estimates[i][j]`` must correspond to
        the estimated probability of remaining event-free up to the time point
        the predicted survival proability of sample `i` at ``times[j]``. 
        Typically, estimated probabilities are obtained via the
        survival function returned by an estimator's ``predict_survival_function`` method.

    times : array-like, shape = (n_times,) 
        The time points for which to estimate the time-dependent AUC score.
        Values must be sorted in ascending order.


    tied_tol : float, optional, default: 1e-8
        The tolerance value for considering ties.
        If the absolute difference between risk scores is smaller
        or equal than `tied_tol`, risk scores are considered tied.

    Returns
    -------
    mean_auc : float
        Summary measure referring to the mean cumulative/dynamic AUC
        over the specified time range in `times`.

    auc scores : array, shape = (n_times',)
        The valid time-dependent AUC scores (evaluated at `times`). 
        n_times' <= n_times

    times': array, shape = (n_times',)
        The valid time thresholds to compute time-dependent AUC score
    

    """
    # input shape check
    if len(event_train.shape) > 1:
        event_train = event_train.values.reshape(-1)
    if len(y_train.shape) > 1:
        y_train = y_train.values.reshape(-1)
    if len(event_test.shape) > 1:
        event_test = event_test.values.reshape(-1)
    if len(y_test.shape) > 1:
        y_test = y_test.values.reshape(-1)
    
    assert len(event_test) == estimates.shape[0] and len(y_test) == estimates.shape[0]
    
    assert estimates.shape[1] == len(times)

    n_samples, n_times = estimates.shape

    # compute ipcw
    cens = BaseEstimator(event_train, y_train, True)
    ipcw_y_test = cens.predict_ipcw(y_test)

    # convert event_test, y_test, times, ipcw_y_test to (n_samples, n_times) shape
    event_test = np.broadcast_to(event_test[:, np.newaxis], (n_samples, n_times))
    y_test = np.broadcast_to(y_test[:, np.newaxis], (n_samples, n_times))
    times_2d = np.broadcast_to(times, (n_samples, n_times))
    ipcw_y_test = np.broadcast_to(ipcw_y_test[:, np.newaxis], (n_samples, n_times))

    # sort each column by estimated survival probability
    o = np.argsort(estimates, axis=0)
    estimates = np.take_along_axis(estimates, o, axis=0)
    event_test = np.take_along_axis(event_test, o, axis=0)
    y_test = np.take_along_axis(y_test, o, axis=0)
    ipcw_y_test = np.take_along_axis(ipcw_y_test, o, axis=0)

    # compute time-dependent AUC score
    is_case = (y_test <= times_2d) & event_test # each col indicates which samples are in case
    is_control = y_test > times_2d

    estimate_diff = np.concatenate((np.broadcast_to(np.inf, (1, n_times)), estimates))
    is_tied = np.absolute(np.diff(estimate_diff, axis=0)) <= tied_tol

    cumsum_tp = np.cumsum(is_case * ipcw_y_test, axis=0)
    cumsum_fp = np.cumsum(is_control, axis=0)
    # handle when no control / no case:
    valid_time_idx = ~ ((cumsum_tp[-1] == 0 ) | (cumsum_fp[-1] == 0) )
    cumsum_tp = cumsum_tp[:, valid_time_idx]
    cumsum_fp = cumsum_fp[:, valid_time_idx]
    is_tied = is_tied[:, valid_time_idx]
    times = times[valid_time_idx]

    true_pos = cumsum_tp / cumsum_tp[-1]
    false_pos = cumsum_fp / cumsum_fp[-1] 

    scores = np.empty(valid_time_idx.sum(), dtype=float)  

    for j in range(len(times)):
        tp = true_pos[:, j]
        fp = false_pos[:, j]
        mask = is_tied[:, j]

        idx = np.flatnonzero(mask) - 1
        # only keep the last estimate for tied risk scores
        if idx[0] < 0:
            # handle case when first estimate is tied
            idx = idx[1:]
        tp_no_ties = np.delete(tp, idx)
        fp_no_ties = np.delete(fp, idx)
        # Add an extra threshold position
        # to make sure that the curve starts at (0, 0)
        tp_no_ties = np.r_[0, tp_no_ties]
        fp_no_ties = np.r_[0, fp_no_ties]
    
        scores[j] = np.trapz(tp_no_ties, fp_no_ties)


    if n_times == 1:
        mean_auc = scores[0]
    else:
        surv = BaseEstimator(event_test[:, 0], y_test[:, 0])
        s_times = surv.predict_survival_prob(times)
        # compute integral of AUC over survival function
        d = -np.diff(np.r_[1.0, s_times])
        integral = (scores * d).sum()
        mean_auc = integral / (1.0 - s_times[-1])

    return mean_auc, scores, times


def compute_ibs_per_sample(event_train, y_train, event_test, y_test, estimates, times):
    """ Compute IBS score for each sample
    """
    if len(event_train.shape) > 1:
        event_train = event_train.values.reshape(-1)
    if len(y_train.shape) > 1:
        y_train = y_train.values.reshape(-1)
    if len(event_test.shape) > 1:
        event_test = event_test.values.reshape(-1)
    if len(y_test.shape) > 1:
        y_test = y_test.values.reshape(-1)

    assert len(event_test) == estimates.shape[0] and len(y_test) == estimates.shape[0]
    
    assert estimates.shape[1] == len(times) 

    n_samples, n_times = estimates.shape

    # compute ipcw
    cens = BaseEstimator(event_train, y_train, True)
    ipcw_times = cens.predict_ipcw(times)
    ipcw_y_test = cens.predict_ipcw(y_test)
    
    # compute ibs loss per sample
    ibs = np.zeros(n_samples, dtype=float)

    for i in range(n_times - 1):
        ibs += (np.square(estimates[:, i]) * ((y_test <= times[i]) & event_test).astype(int) * ipcw_y_test + \
                np.square(1.0 - estimates[:, i]) * (y_test > times[i]).astype(int) * ipcw_times[i]) * (times[i+1] - times[i])

    return ibs / (times[-1] - times[0]) / n_samples

    



    