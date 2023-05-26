
from miacag.utils.sql_utils import getDataFromDatabase
from miacag.plots.plotter import plot_results, plotRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
import numpy as np
import os
import yaml
from sklearn import metrics
from miacag.utils.script_utils import mkFolder
from miacag.plots.plotter import rename_columns, mkFolder
#from sklearn.metrics import fl_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils import check_matplotlib_support



def run_plotter_ruc_multi_class(y_score, y_onehot_test,
                                type_outcome, model_name,
                                save_name, output_path):
    mean_auc, upper_auc, lower_auc = get_confidence_auc(y_score, y_onehot_test)
    plot_roc_multi_class(y_score, y_onehot_test, mean_auc, lower_auc,
                         upper_auc, type_outcome, model_name,
                         save_name, output_path)
def plot_roc_multi_class(y_score, y_onehot_test, mean_auc, lower_auc,
                         upper_auc, type_outcome, model_name,
                         save_name, output_path):
    fig, ax = plt.subplots()
    plot_micro_roc_mult_wrap(y_score, y_onehot_test, mean_auc, lower_auc,
                             upper_auc, type_outcome, model_name, ax)
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Prediction of " +
              type_outcome +
              " One-vs-Rest \nReceiver Operating Characteristic")
    plt.legend(prop={'size': 6})
    ax.legend(loc="lower right")
    plt.show()
    plt.savefig(
        os.path.join(output_path,
                     save_name +"roc_curve.png"))
    ax.legend(bbox_to_anchor=(1.0, 0.0), loc='upper left', bbox_transform=ax.transAxes)

    plt.close()


def plot_micro_roc_mult_wrap(y_score, y_onehot_test, mean_auc, lower_auc,
                             upper_auc, type_outcome,
                             model_name, ax):
    return from_predictions(
        y_onehot_test.ravel(), y_score.ravel(),
        mean_auc, lower_auc, upper_auc, type_outcome,
        model_name, name="micro-average OvR",
        color="darkorange", ax=ax)

def from_predictions(y_true, y_pred,
                     mean_auc, lower_auc, upper_auc, type_outcome,
                     model_name,
                     *,
                     sample_weight=None, drop_intermediate=True,
                     pos_label=None, name=None, ax=None, **kwargs,):
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=pos_label,
                            sample_weight=sample_weight, drop_intermediate=drop_intermediate)
    name = "Classifier" if name is None else name
    return plot_wrap(name, ax, fpr, tpr, mean_auc, lower_auc, upper_auc, type_outcome, model_name)

def plot_wrap(name, ax, fpr, tpr, mean_auc, lower_auc, upper_auc, type_outcome, model_name, **kwargs):
    line_kwargs = {}
    line_kwargs["label"] = \
        f"{model_name}, AUC={mean_auc:0.3f} \
        ({upper_auc:0.3f}-{lower_auc:0.3f})"
    pos_label = None
    (line_,) = ax.plot(fpr, tpr, **line_kwargs)
    info_pos_label = f" (Positive label: {pos_label})" \
        if pos_label is not None else ""

    xlabel = "False Positive Rate" + info_pos_label
    ylabel = "True Positive Rate" + info_pos_label
    ax.set(xlabel=xlabel, ylabel=ylabel)
   # if "label" in line_kwargs:
    ax.legend(loc="lower right")
    ax_ = ax
    figure_ = ax.figure
    return


def get_confidence_auc(y_pred, y_true):
    bootstrapped_scores = compute_bootstrapped_scores(y_pred, y_true)
    mean, upper, lower = compute_confidence_auc(bootstrapped_scores)
    return mean, upper, lower

def compute_bootstrapped_scores(y_pred, y_true):
    n_bootstraps = 1000
    rng_seed = 42 # control reproducibility

    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range (n_bootstraps): 
        #bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # # to be defined: reject the sample
            continue
        micro_roc_auc_ovr = roc_auc_score(
            y_true[indices],
            y_pred[indices],
            multi_class="ovr",
            average="micro")
        bootstrapped_scores.append(micro_roc_auc_ovr)
        # print ("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    return bootstrapped_scores

def compute_confidence_auc(bootstrapped_scores):
    mean = np.mean (bootstrapped_scores)
    std = np.std(bootstrapped_scores)
    upper = mean + 2*std
    lower = mean - 2*std
    return mean, upper, lower