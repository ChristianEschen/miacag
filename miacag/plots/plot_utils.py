import numpy as np
from sklearn.metrics import roc_auc_score

def get_mean_lower_upper(y_pred, y_true, score_type):
    bootstrapped_scores = compute_bootstrapped_scores(y_pred, y_true, score_type)
    mean, upper, lower = compute_mean_lower_upper(bootstrapped_scores)
    return mean, upper, lower

def compute_bootstrapped_scores(y_pred, y_true, score_type):
    n_bootstraps = 1000
    rng_seed = 42 # control reproducibility

    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range (n_bootstraps): 
        #bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if score_type == 'roc_auc_score':
            if len(np.unique(y_true[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # # to be defined: reject the sample
                continue
        
        scores = get_score_type(y_true, y_pred, indices, score_type)
        bootstrapped_scores.append(scores)
        # print ("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    return bootstrapped_scores

def compute_mean_lower_upper(bootstrapped_scores):
    mean = np.mean (bootstrapped_scores)
    std = np.std(bootstrapped_scores)
    upper = mean + 2*std
    lower = mean - 2*std
    return mean, upper, lower

def get_score_type(y_true, y_pred, indices, score_type):
    if score_type == 'roc_auc_score':
        scores = roc_auc_score(
            y_true[indices],
            y_pred[indices],
            multi_class="ovr",
            average="micro")
    elif score_type == 'mse_score':
        # implement MSE with numpy
        scores = np.mean((y_true[indices] - y_pred[indices])**2)
    
    else:
        raise ValueError('this score is not implemented:', score_type)
    return scores