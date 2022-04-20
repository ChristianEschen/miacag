import os
import numpy as np
from mia.preprocessing.utils.sql_utils import getDataFromDatabase
from sklearn.metrics import f1_score, \
     accuracy_score, confusion_matrix, plot_confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
matplotlib.use('Agg')


def map_1abels_to_0neTohree():
    labels_dict = {
        0: 0,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        14: 2,
        15: 2,
        16: 2,
        17: 2,
        18: 2,
        19: 2,
        20: 2}
    return labels_dict


def create_empty_csv():
    df = {'Experiment name': [],
          'Test F1 score on data labels transformed': [],
          'Test F1 score on three class labels': [],
          'Test acc on three class labels': []}
    return df


def getNormConfMat(df, labels_col, preds_col, plot_name, f1, output):
    conf_arr = confusion_matrix(df[labels_col], df[preds_col])
    sum = conf_arr.sum()
    conf_arr = conf_arr * 100.0 / (1.0 * sum)
    df_cm = pd.DataFrame(
        conf_arr,
        index=[
            str(i) for i in range(0, len(df.drop_duplicates([labels_col])))],
        columns=[
            str(i) for i in range(0, len(df.drop_duplicates([labels_col])))])
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    res = sns.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0, fmt='.2f',
                      cmap=cmap)
    res.invert_yaxis()
    f1 = np.round(f1, 3)
    plt.title(plot_name + ': Confusion Matrix, F1-macro:' + str(f1))
    plt.savefig(os.path.join(output, plot_name + '_cmat.png'), dpi=100,
                bbox_inches='tight')
    plt.close()
    return None


def plot_results(sql_config, output_plots):
    df, _ = getDataFromDatabase(sql_config)
    df = df[df[sql_config['labels_names']].notna()]
    df = df[df['labels'].notna()]
    df = df.dropna(subset=sql_config["labels_names"], how='any')
    df = df[df['predictions'].notna()]
    df['predictions'] = df['predictions'].astype(float).astype(int)
    df['labels_transformed'] = df['labels_transformed'] \
        .astype(float).astype(int)
    df['labels'] = df['labels'].astype(float).astype(int)
    f1_transformed = f1_score(df['labels_transformed'],
                              df['predictions'],
                              average='macro')

    getNormConfMat(
        df,
        'labels_transformed',
        'predictions',
        'transformed_labels',
        f1_transformed,
        output_plots)
    for label_name in sql_config["labels_names"]:
        df = df.replace(
            {label_name: map_1abels_to_0neTohree()})
        df = df.replace(
            {'predictions': map_1abels_to_0neTohree()})
        f1 = f1_score(df[label_name],
                      df['predictions'], average='macro')
        getNormConfMat(df, label_name, 'predictions',
                       'labels_3_classes', f1, output_plots)
    return
