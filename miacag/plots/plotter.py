import os
import numpy as np
from miacag.utils.sql_utils import getDataFromDatabase
from sklearn.metrics import f1_score, \
     accuracy_score, confusion_matrix, plot_confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
from sklearn import metrics
import scipy
from sklearn.metrics import r2_score
import statsmodels.api as sm
from miacag.dataloader.Classification._3D.dataloader_monai_classification_3D_mil \
    import reorder_rows
from decimal import Decimal


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


def convertConfFloats(confidences, loss_name):
    confidences_conv = []
    for conf in confidences:
        if loss_name.startswith('CE'):
            confidences_conv.append(float(conf.split(";1:")[-1][:-1]))

        elif loss_name in ['MSE', 'L1', 'L1smooth', 'BCE_multilabel']:
            confidences_conv.append(float(conf.split("0:")[-1][:-1]))
        else:
            raise ValueError('not implemented')
    return np.array(confidences_conv)


def create_empty_csv():
    df = {
          'Experiment name': [],
          'Test F1 score on data labels transformed': [],
          'Test F1 score on three class labels': [],
          'Test acc on three class labels': []
          }
    return df

def getNormConfMat_3class(df, labels_col, preds_col,
                   plot_name, f1, output, num_classes, support, c):
    labels = [i for i in range(0, num_classes[c])]
    conf_arr = confusion_matrix(df[labels_col], df[preds_col], labels=labels)
    sum = conf_arr.sum()
    conf_arr = conf_arr * 100.0 / (1.0 * sum)
    df_cm = pd.DataFrame(
        conf_arr,
        index=[
            str(i) for i in range(0, num_classes[c])],
        columns=[
            str(i) for i in range(0, num_classes[c])])
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    res = sns.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0, fmt='.2f',
                      square=True, linewidths=0.1, annot_kws={"size": 8},
                      cmap=cmap)
    res.invert_yaxis()
    f1 = np.round(f1, 3)
    plt.title(
        plot_name + ': Confusion Matrix, F1-macro:' + str(f1))
    plt.savefig(os.path.join(output, plot_name + '_cmat.png'), dpi=100,
                bbox_inches='tight')
    plt.close()

    plt.title(
        plot_name + ': Confusion Matrix, F1-macro:' + str(f1) +
        ',support(N)=' + str(support))
    plt.savefig(os.path.join(output, plot_name + '_cmat_support.png'), dpi=100,
                bbox_inches='tight')
    plt.close()
    return None



def getNormConfMat(df, labels_col, preds_col,
                   plot_name, f1, output, num_classes, support, c):
    num_classes_for_plot = num_classes[c]
    if num_classes_for_plot == 1:
        num_classes_for_plot = 2
    labels = [i for i in range(0, num_classes_for_plot)]
    df[labels_col] = df[labels_col].astype(int)
    df[preds_col] = df[preds_col].astype(int)
    conf_arr = confusion_matrix(df[labels_col], df[preds_col], labels=labels)
    sum = conf_arr.sum()
    # Normalized confusion matrix???
    #conf_arr = conf_arr * 100.0 / (1.0 * sum)
    df_cm = pd.DataFrame(
        conf_arr,
        index=[
            str(i) for i in range(0, num_classes_for_plot)],
        columns=[
            str(i) for i in range(0, num_classes_for_plot)])
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    res = sns.heatmap(df_cm, annot=True, vmin=0.0, fmt='g',
                      square=True, linewidths=0.1, annot_kws={"size": 8},
                      cmap=cmap)
    res.invert_yaxis()
    f1 = np.round(f1, 3)
    plt.title(
        plot_name + ': Confusion Matrix, F1-macro:' + str(f1))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(os.path.join(output, plot_name + '_cmat.png'), dpi=100,
                bbox_inches='tight')
    plt.close()

    plt.title(
        plot_name + ': Confusion Matrix, F1-macro:' + str(f1) +
        ',support(N)=' + str(support))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(os.path.join(output, plot_name + '_cmat_support.png'), dpi=100,
                bbox_inches='tight')
    plt.close()
    return None


def threshold_continuois(labels, config, plot_name):
    labels_copy = labels
    labels_copy = labels_copy.to_numpy()
    if 'ffr' in plot_name:
        thres = config['loaders']['val_method']['threshold_ffr']
        labels_copy[labels_copy >= thres] = 1
        labels_copy[labels_copy < thres] = 0
        labels_copy = np.logical_not(labels_copy).astype(int)
    elif 'sten' in plot_name:
        thres = config['loaders']['val_method']['threshold_sten']
        if 'lm' in plot_name:
            thres = 0.5
        labels_copy[labels_copy >= thres] = 1
        labels_copy[labels_copy < thres] = 0
    else:
        pass
    return labels_copy


def plot_roc_curve(labels, confidences, output_plots,
                   plot_name, support, num_classes, config, loss_name):
    fpr, tpr, thresholds = metrics.roc_curve(labels, confidences, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.clf()
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', lw=2, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.show()
    plt.savefig(os.path.join(output_plots, plot_name + '_roc.png'), dpi=100,
                bbox_inches='tight')
    plt.close()

    plt.clf()
    plt.figure()
    plt.title('Receiver Operating Characteristic, support(N):' + str(support))
    plt.plot(fpr, tpr, 'b', lw=2, label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.show()
    plt.savefig(os.path.join(
        output_plots, plot_name + '_roc_support.png'), dpi=100,
                bbox_inches='tight')
    plt.close()


def plot_results_regression(df_label, confidence_name,
                            label_name, config, c, support,
                            output_plots, group_aggregated):
    if not group_aggregated:
        df_label[confidence_name] = convertConfFloats(
            df_label[confidence_name], config['loss']['name'][c])
    df_label[label_name] = threshold_continuois(
        df_label[label_name],
        config,
        label_name)
    df_label[confidence_name] = np.clip(
        df_label[confidence_name], a_min=0, a_max=1)
    plot_roc_curve(
        df_label[label_name], df_label[confidence_name],
        output_plots, label_name, support,
        config['model']['num_classes'][c], config,
        config['loss']['name'][c])
    df_label[confidence_name] = threshold_continuois(
        df_label[confidence_name],
        config,
        confidence_name)
    f1_transformed = f1_score(
        df_label[label_name],
        df_label[confidence_name],
        average='macro')

    getNormConfMat(
        df_label,
        label_name,
        confidence_name,
        label_name,
        f1_transformed,
        output_plots,
        config['model']['num_classes'],
        support,
        c)
    return None


def plot_results_classification(df_label,
                                label_name,
                                prediction_name,
                                confidence_name,
                                output_plots,
                                config,
                                support,
                                c,
                                group_aggregated):
    f1_transformed = f1_score(
        df_label[label_name],
        df_label[prediction_name],
        average='macro')

    getNormConfMat(
        df_label,
        label_name,
        prediction_name,
        label_name,
        f1_transformed,
        output_plots,
        config['model']['num_classes'],
        support,
        c)

    if not group_aggregated:
        df_label[confidence_name] = convertConfFloats(
            df_label[confidence_name], config['loss']['name'][c])
    df_label[confidence_name] = np.clip(
        df_label[confidence_name], a_min=0, a_max=1)
    plot_roc_curve(
        df_label[label_name], df_label[confidence_name],
        output_plots, label_name, support,
        config['model']['num_classes'][c], config,
        config['loss']['name'][c])

    if config['loss']['name'][c].startswith('CE'):
        df_label = df_label.replace(
            {label_name: map_1abels_to_0neTohree()})
        df_label = df_label.replace(
            {prediction_name: map_1abels_to_0neTohree()})
        f1 = f1_score(df_label[label_name],
                      df_label[prediction_name], average='macro')
        getNormConfMat(df_label, label_name, prediction_name,
                       'labels_3_classes', f1, output_plots, [3], support, 0)

    return None


def plot_results(sql_config, label_names, prediction_names, output_plots,
                 num_classes, config, confidence_names,
                 group_aggregated=False):
    df, _ = getDataFromDatabase(sql_config)
    if group_aggregated:
        confidence_names = [c + '_aggregated' for c in confidence_names]
    for c, label_name in enumerate(label_names):
        confidence_name = confidence_names[c]
        prediction_name = prediction_names[c]
        df_label = df[df[label_name].notna()]
        df_label = df_label[df_label[confidence_name].notna()]
        df_label = df_label[df_label[prediction_name].notna()]
        if group_aggregated:
            df_label = df_label.drop_duplicates(
                subset=['StudyInstanceUID', "PatientID"])
        support = len(df_label)
        if config['loss']['name'][c] in ['MSE', 'L1', 'L1smooth']:
            plot_results_regression(df_label, confidence_name,
                                    label_name, config, c, support,
                                    output_plots,
                                    group_aggregated)

        elif config['loss']['name'][c].startswith('CE') or \
                config['loss']['name'][c] == 'BCE_multilabel':
            plot_results_classification(df_label,
                                        label_name,
                                        prediction_name,
                                        confidence_name,
                                        output_plots,
                                        config,
                                        support,
                                        c,
                                        group_aggregated)
    return None


def annotate(data, label_name, prediction_name, **kws):
    #r, p = scipy.stats.pearsonr(data[label_name], data[prediction_name])
    #r2_score(df[label_name], df[prediction_name])
    X2 = sm.add_constant(data[label_name])
    est = sm.OLS(data[prediction_name], X2)
    est2 = est.fit()
    ax = plt.gca()
    ax.text(.05, .8, 'r-squared={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)


def plotStenoserTrueVsPred(sql_config, label_names,
                           prediction_names, output_folder):
    df, _ = getDataFromDatabase(sql_config)
    df = df.drop_duplicates(
            ['PatientID',
             'StudyInstanceUID'])

    for c, label_name in enumerate(label_names):
        df = df.dropna(
            subset=[label_name],
            how='any')
        df = df.astype({label_name: int})
        prediction_name = prediction_names[c]
        df = df.astype({prediction_name: int})
        g = sns.lmplot(x=label_name, y=prediction_name, data=df)
        X2 = sm.add_constant(df[label_name])
        est = sm.OLS(df[prediction_name], X2)
        est2 = est.fit()
        r = est2.rsquared
        p = est2.pvalues[label_name]
        p = '%.2E' % Decimal(p)


        for ax, title in zip(g.axes.flat, [label_name]):
            ax.set_title(title)
            #ax.ticklabel_format(useOffset=False)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylim(bottom=0.)
            ax.text(0.05, 0.85,
                    f'R-squared = {r:.3f}',
                    fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.9,
                    "p-value = " + p,
                    fontsize=9,
                    transform=ax.transAxes)
            plt.show()
        plt.title('Number of reported significant stenoses vs predicted')
        plt.savefig(
            os.path.join(output_folder, label_name + '_scatter.png'), dpi=100,
            bbox_inches='tight')
        plt.close()
        return None


def plotRegression(sql_config, label_names,
                   prediction_names, output_folder, group_aggregated=False):
    df, _ = getDataFromDatabase(sql_config)

    for c, label_name in enumerate(label_names):
        label_name_ori = label_name
        prediction_name = prediction_names[c]
        df_plot = df.dropna(
            subset=[label_name, prediction_name],
            how='any')
        if group_aggregated:
            df_plot = df_plot.drop_duplicates(
                    ['PatientID',
                    'StudyInstanceUID'])
        df_plot, label_name = rename_columns(df_plot, label_name)
        df_plot = df_plot.astype({label_name: float})

        if group_aggregated is False:
            df_plot[prediction_name] = \
                convertConfFloats(
                    df_plot[prediction_name],
                    sql_config['loss_name'][c])

        df_plot[prediction_name] = np.clip(
            df_plot[prediction_name], a_min=0, a_max=1)

        df_plot = df_plot.astype({prediction_name: float})
        g = sns.lmplot(x=label_name, y=prediction_name, data=df_plot)
        X2 = sm.add_constant(df_plot[label_name])
        est = sm.OLS(df_plot[prediction_name], X2)
        est2 = est.fit()
        r = est2.rsquared
        p = est2.pvalues[label_name]
        p = '%.2E' % Decimal(p)


        for ax, title in zip(g.axes.flat, [label_name]):
            ax.set_title(title)
            ax.set_ylim(bottom=0.)
            ax.text(0.05, 0.85,
                    f'R-squared = {r:.3f}',
                    fontsize=9, transform=ax.transAxes)
            ax.text(0.05, 0.9,
                    "p-value = " + p,
                    fontsize=9,
                    transform=ax.transAxes)
            plt.xlabel("True")
            plt.ylabel("Predicted")
            plt.show()

        plt.title(label_name)
        plt.savefig(
            os.path.join(
                output_folder, label_name_ori + '_scatter.png'), dpi=100,
            bbox_inches='tight')
        plt.close()
    return None


def rename_columns(df, label_name):
    if '_1_prox' in label_name:
        value = '1: Proximal RCA'
    elif '_2_mi' in label_name:
        value = '2: Mid RCA'
    elif '_3_dist' in label_name:
        value = '3: Distale RCA'
    elif '_4_pda' in label_name:
        value = '4: PDA'
    elif '_5_lm' in label_name:
        value = '5: LM'
    elif '_6_prox' in label_name:
        value = '6: Proximal LAD'
    elif '_7_mi' in label_name:
        value = '7: Mid LAD'
    elif '_8_dist' in label_name:
        value = '8: Distale LAD'
    elif '_9_d1' in label_name:
        value = '9: Diagonal 1'
    elif '_10_d2' in label_name:
        value = '10: Diagonal 2'
    elif '_11_prox' in label_name:
        value = '11: Proximal LCX'
    elif '_12_om' in label_name:
        value = '12: Marginal 1'
    elif '_13_midt' in label_name:
        value = '13: Mid LCX'
    elif '_14_om' in label_name:
        value = '14: Marginal 2'
    elif '_15_dist' in label_name:
        value = '15: Distale LCX'
    elif '_16_pla' in label_name:
        value = '16: PLA'
    key = label_name
    dictionary = {key: value}
    df = df.rename(columns=dictionary)
    return df, value
