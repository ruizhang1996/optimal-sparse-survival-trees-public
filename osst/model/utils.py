import numpy as np

class BaseEstimator:
    """
    Kaplan Meier / Nelson Aalen estimator
    """
    def __init__(self, event=None, y=None, reverse=False):
        """constructor of Kaplan Meier Estimator"""
        self.unique_times = None
        self.cumu_harzard_prob = None 
        self.survival_prob = None 
        if event is not None and y is not None:
            self.fit(event=event, y=y, reverse=reverse)

    def fit(self, event, y, reverse=False):
        """
        Parameters
        ---
        event: array-like, shape = [n_samples, ]
            an n-by-1 column of event associated with each sample
        y : array-like, shape = [n_samples, ]
            column containing the observed time for each sample in X
        reverse: bool
            fit a censoring distribution if reverse is True
        Modifies
        ---
        fit the Kaplan Meier / Nelson Aalen Estimator
        """
        n_samples = len(y)
        
        if reverse:
            event = 1 - event 
        order = np.argsort(y)

        time_threshold = np.zeros(n_samples, dtype=y.dtype)
        num_event_at_threshold = np.zeros(n_samples, dtype=int)
        cumu_num_total_at_threshold = np.zeros(n_samples, dtype=int)

        num_total = 0 
        num_event = 0 
        j = 0
        for i in range(n_samples):
            num_total += 1
            num_event += event[order[i]]
            if i < n_samples - 1 and y[order[i]] != y[order[i+1]]:
                 time_threshold[j] = y[order[i]]
                 num_event_at_threshold[j] = num_event
                 cumu_num_total_at_threshold[j] = num_total
                 num_event = 0 
                 j += 1
            elif i == n_samples - 1:
                time_threshold[j] = y[order[i]]
                num_event_at_threshold[j] = num_event
                cumu_num_total_at_threshold[j] = num_total
                j += 1
        
        time_threshold = np.resize(time_threshold, j)
        num_event_at_threshold = np.resize(num_event_at_threshold, j)
        cumu_num_total_at_threshold = np.resize(cumu_num_total_at_threshold, j)

        num_risky_at_threshold = n_samples - cumu_num_total_at_threshold
        num_risky_at_threshold = np.r_[n_samples, num_risky_at_threshold]
        num_risky_at_threshold = num_risky_at_threshold[:-1]

        ratio = num_event_at_threshold / num_risky_at_threshold # care for NaN here

        survival_prob = np.cumprod(1.0 - ratio)
        cumu_harzard_prob = np.cumprod(ratio)
        # take care before time before first time threshold 
        self.unique_times = np.r_[-np.infty, time_threshold]
        self.survival_prob = np.r_[1.0, survival_prob]
        self.cumu_harzard_prob = np.r_[0.0, cumu_harzard_prob]

        return self
    
    def predict_survival_prob(self, times):
        """
        Parameters
        ---
        times: array-like, shape = [n_times, ]
            an vector of time thresholds at which to predict survival probability
        ---
        Returns
        ---
            survival probabilities: array-like, shape = [n_times, ]
        """
        indices = np.searchsorted(self.unique_times, times, side="right") - 1

        return self.survival_prob[indices] 
    
    def predict_cumu_harzard_prob(self, times):
        """
        Parameters
        ---
        times: array-like, shape = [n_times, ]
            an vector of time thresholds at which to predict survival probability
        ---
        Returns
        ---
            cumulative harzard probability: array-like, shape = [n_times, ]
        """

        indices = np.searchsorted(self.unique_times, times, side="right") - 1

        return self.cumu_harzard_prob[indices] 


    def predict_ipcw(self, times):
        """
        Parameters
        ---
        times: array-like, shape = [n_times, ]
            an vector of time thresholds at which to predict survival probability
        ---
        Returns
        ---
            inverse probability of censoring weights: array-like, shape = [n_times, ]
        """
        probs = self.predict_survival_prob(times)
        probs[probs == 0] = np.inf
        return 1 / probs
        
    


            

            
            
