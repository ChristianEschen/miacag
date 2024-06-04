import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix

def get_mean_lower_upper(y_pred, y_true, score_type):
    bootstrapped_scores = compute_bootstrapped_scores(y_pred, y_true, score_type)
    mean, lower, upper = compute_mean_lower_upper(bootstrapped_scores)
    return mean, lower, upper

def compute_bootstrapped_scores(y_pred, y_true, score_type):
    n_bootstraps = 1000
    rng_seed = 42 # control reproducibility

    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range (n_bootstraps): 
        #bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if score_type in ['roc_auc_score', 'f1', 'pr_auc_score', 'classification_report']:
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
    # upper = mean + 2*std
    # lower = mean - 2*std
    lower = np.percentile(bootstrapped_scores, 100 * (1 - 0.95) / 2)
    upper = np.percentile(bootstrapped_scores, 100 * (1 + 0.95) / 2)
    
    return mean, lower, upper

def get_score_type(y_true, y_pred, indices, score_type):
    if score_type == 'roc_auc_score':
        scores = roc_auc_score(
            y_true[indices],
            y_pred[indices],
            multi_class="ovr",
            average="micro")
    elif score_type == 'pr_auc_score':
        # precision-recall curve auc
        scores = average_precision_score(
            y_true[indices],
            y_pred[indices],
            average='micro')

    elif score_type == 'f1':
        #y_true_bin = transform_confidences_to_by_label_type(y_true, plot_type)
        scores = f1_score(
            y_true[indices],
            y_pred[indices])
    elif score_type == 'mse_score':
        # implement MSE with numpy
        scores = np.mean((y_true[indices] - y_pred[indices])**2)
    
    elif score_type == 'mae_score':
        # implement MSE with numpy
        scores = np.mean(np.absolute((y_true[indices] - y_pred[indices])))
        
    elif score_type == 'std_score':
        scores = np.std(y_true[indices] - y_pred[indices])
        
    elif score_type == 'pearson_score':
        scores = np.corrcoef(y_true[indices], y_pred[indices])[0,1]

   # elif score_type == "specificity":

    #     tn, fp, fn, tp = confusion_matrix(y_true[indices], y_pred[indices]).ravel()
    #     scores = tn / (tn+fp)
        
    
    
    # elif score_type == 'classification_report':
        
    else:
        raise ValueError('this score is not implemented:', score_type)
    return scores

def transform_confidences_to_by_label_type(confidences, name):
    if name.startswith('ffr'):
        confidences = 1 - confidences
    elif name.startswith('sten'):
        pass
    elif name.startswith('timi'):
        pass
    else:
        raise ValueError('name is not ffr or sten')
    return confidences
