
from miacag.utils.sql_utils import getDataFromDatabase
from miacag.plots.plotter import plot_results, plotRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
from sklearn.metrics import fl_score, \
accuracy_score, confusion_matrix, plot_confusion_matrix
import numpy as np
import os
import yaml
from sklearn import metrics
from miacag.utils.script_utils import mkFolder
from miacag.plots.plotter import rename_columns, mkFolder
from sklearn.metrics import fl_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import LabelBinarizer from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.utils import check_matplotlib_support
#from sklearn.base import _check_pos_label_consistency

def rename_columns(df, label_name): 
    if '_1 _prox' in label_name:
        value = '1: Proximal RCA'
    elif '_2 mi' in label_name:
        value = '2: Mid RCA'
    elif '_3_dist' in label_name:
        value = '3: Distale RCA'
    elif '_4 pda' in label_name:
        value = '4: PDA'
    elif '_5_1m' in label_name:
        value = '5: LM'
    elif '_6_prox' in label_name:
        value = '6: Proximal LAD'
    elif '_7_mi' in label_name:
        value = '7: Mid LAD'
    elif '_8 _dist' in label_name:
        value = '8: Distale LAD'
    elif '_9 dl' in label_name:
        value = '93: Diagonal 1'
    elif '_10_d2' in label_name:
        value = '10: Diagonal 2'
    elif '_11l_prox' in label_name:
        value = '1ll: Proximal LCX'
    elif '_12_om' in label_name:
        value = '12: Marginal 1'
    elif '_13_midt' in label_name:
        value = '13: Mid LCX'
    elif '_14 om' in label_name:
        value = '14: Marginal 2'
    elif '_15_dist' in label_name:
        value = '15: Distale LCX'
    elif '_16_pla' in label_name:
        value = '16: PLA'
    key = label_name
    dictionary = {key: value}
    df = df.rename(columns=dictionary)
    return df, value

def getNormConfMat(df, labels_col, preds_col, fl, plot_name, output, num_classes, support, c):
    num_classes_for_plot = num_classes[c]
    if num_classes_for_plot == 1:
        num_classes_for_plot = 2
    labels = [i for i in range(0, num_classes_for_plot) ]
    df[labels_col] = df[labels_col].astype(int)
    df[preds_col] = df[preds_col].astype (int)
    conf_arr = confusion_matrix(df[labels_col], df[preds_col], labels=labels)
    sum = conf_arr.sum()
    df_cm = pd.DataFrame(
        conf_arr, index=[str(i) for i in range(0, num_classes_for_plot)],
        columns=[str(i) for i in range(0, num_classes_for_plot})]} fig = plt.figure() plt.clft({) fig.add_subplot (111) ax.set_aspect (1) cmap = sns.cubehelix_palette(light=1, as_cmap=True) res = sns.heatmap(df_cm, annot=True, vmin=0.0, fmt='g', #fmt='.2f"', square=True, linewidths=0.1, annot_kws={"size": 8),
ax
cmap=cmap)} res.invert_yaxis() plt.title( Plot_name + ': Confusion Matrix, Fl-macro:' + str(fl}) + ', support (N)=" + str(support)) plt.xlabel("Predicted") # not sure plt.ylabel ("Actual") # not sure maybe reverse plt.savefig(os.path.join(output, plot_name + ‘_cmat.png'}), dpi=100, bbox_inches='tight'")} plt.close() plt.title( | plot_name + ': Confusion Matrix, Fl-macro:' + str(fl)} + ', support (N)=" + str(support))} 1t.xlabel ("Predicted") # not sure plt.ylabel("Actual") # not sure maybe reverse plt.savefig(os.path.join(output, plot_name + '_cmat_support.pdf"'}), dpi=100, bbox_inches='"tight")} plt.close() return None
Pplot_roc_curve(labels, confidences, output_plots,
plot_name, support, num_classes, config, loss_name): #try: if len(labels.index) !=0:
fpr, tpr, thresholds = metrics.roc_curve(labels, confidences, pos_label=1)




def plot_coronar_pat(path):
    labels = "koronarpatologi_nativekar_udfyldesforallept_"
    type_outcome = "degree of coronary pathology"
    labels = ""
    return None
if __name__ == '__main__':
    pathologies= [
        "1 gebet",
        "2 gebeter",
        "3 gebeter",
        "Ateromatose uden signifikante stenoser",
        "Ingen vegforandringer"]
    
    plot_coronar_pat(path)   
    
    
    
fig, ax = plt.subplots()
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Prediction of " + type_outcome +" One-vs-Rest \nReceiver Operating Characteristic")
plt.legend(prop=[{'size': 6}) plt.show() if type_outcome == "treatment": Save_name="treatment" else: save_name = "cad _type_”" plt.savefig(os.path.join(output_path, save_name +"roc_curve.png"))} plt.close()