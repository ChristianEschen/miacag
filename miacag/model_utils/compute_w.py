from collections import Counter
from scipy.ndimage import convolve1d
from miacag.model_utils.utils_regression import get_lds_kernel_window
import numpy as np

def get_bin_idx(x):
    return min(int(x * np.float32(1)), 0.99)
def get_lds():
    # preds, labels: [Ns,], "Ns" is the number of total samples
    preds, labels = [0, 0, 0.5, 0.7, 0.7, 1, 0, 0.8, 0, 0, 0], [0, 0, 0.5, 0.7, 0.7, 1, 0, 0.8, 0, 0, 0]
    # assign each label to its corresponding bin (start from 0)
    # with your defined get_bin_idx(), return bin_index_per_label: [Ns,] 
    bin_index_per_label = [get_bin_idx(label) for label in labels]

    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label) + 1
    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    from loss import weighted_mse_loss

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]

    # calculate loss
    loss = weighted_mse_loss(preds, labels, weights=weights)
    
    return loss

if __name__ == '__main__':
    get_lds()