import os
import numpy as np
from miacag.utils.sql_utils import getDataFromDatabase
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

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
import math
import scipy
from sklearn.metrics import r2_score
import statsmodels.api as sm
from decimal import Decimal
from miacag.utils.script_utils import mkFolder
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

from miacag.plots.plot_roc_auc_all import plot_roc_all
from miacag.plots.plot_roc_auc_all import select_relevant_data


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
            if conf is None:
                confidences_conv.append(np.nan)
            else:
                confidences_conv.append(float(conf.split(";1:")[-1][:-1]))

        elif loss_name in ['MSE', '_L1', 'L1smooth', 'BCE_multilabel']:
            if conf is None:
                confidences_conv.append(np.nan)
            else:
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
    
    
    if plot_name.startswith('ffr_'):
        confidences_trans = confidences.copy()
        confidences_trans = 1 - confidences_trans
    else:
        confidences_trans = confidences
    fpr, tpr, thresholds = metrics.roc_curve(labels, confidences_trans, pos_label=1)
    
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

def select_relevant_columns(label_names, label_type):
    # select relevant columns starting with "sten"
    if label_type == 'sten':
        sten_cols = [i for i in label_names if i.startswith('sten')]
        return sten_cols
    elif label_type == 'ffr':
        ffr_cols = [i for i in label_names if i.startswith('ffr')]
        return ffr_cols
    else:
        raise ValueError('label_type must be either sten or ffr')


def wrap_plot_all_sten_reg(df, label_names, confidence_names, output_plots,
                           group_aggregated, config):
    threshold_ffr = config['loaders']['val_method']['threshold_ffr']
    threshold_sten = config['loaders']['val_method']['threshold_sten']
    df_sten = df.copy()
    sten_cols_conf = select_relevant_columns(confidence_names, 'sten')
    sten_cols_true = select_relevant_columns(label_names, 'sten')
    if not group_aggregated:
        for sten_col_conf in sten_cols_conf:
            df_sten[sten_col_conf] = convertConfFloats(
                df[sten_col_conf],
                config['loss']['name'][0])
    sten_trues_concat = []      
    sten_conf_concat = []  
    #concantenating all stenosis columns
    # df_sten['stenosis'] = []
    for idx, label in enumerate(sten_cols_true):
        sten_trues_concat.append(df_sten[label])
        sten_conf_concat.append(df_sten[sten_cols_conf[idx]])
    stenosis = pd.concat(
        [pd.concat(sten_trues_concat), pd.concat(sten_conf_concat)],
        axis=1)
    if len(sten_cols_true) >= 2:
        plot_col = [0, 1]
    else:
        plot_col = [sten_cols_true[0], sten_cols_conf[0]]

    plot_regression_density(x=stenosis[plot_col[0]], y=stenosis[plot_col[1]],
                            cmap='jet', ylab='prediction', xlab='true',
                            bins=100,
                            figsize=(5, 4),
                            snsbins=60,
                            plot_type='stenosis',
                            output_folder=output_plots,
                            label_name_ori='sten_all')
    df_ffr = df.copy()
    ffr_cols_true = select_relevant_columns(label_names, 'ffr')
    if len(ffr_cols_true) > 0:
        ffr_cols_conf = select_relevant_columns(confidence_names, 'ffr')
        if not group_aggregated:
            for ffr_col_conf in ffr_cols_conf:
                df_ffr[ffr_col_conf] = convertConfFloats(
                    df_ffr[ffr_col_conf],
                    config['loss']['name'][0])
        ffr_trues_concat = []      
        ffr_conf_concat = []  
        #concantenating all stenosis columns
        # df_sten['stenosis'] = []
        for idx, label in enumerate(ffr_cols_true):
            ffr_trues_concat.append(df_ffr[label])
            ffr_conf_concat.append(df_ffr[ffr_cols_conf[idx]])
        ffr = pd.concat(
            [pd.concat(ffr_trues_concat), pd.concat(ffr_conf_concat)],
            axis=1)
        
        if len(ffr_cols_true) >= 2:
            plot_col = [0, 1]
        else:
            plot_col = [ffr_cols_true[0], ffr_cols_conf[0]]
        plot_regression_density(x=ffr[plot_col[0]], y=ffr[plot_col[1]],
                                cmap='jet', ylab='Prediction', xlab='True',
                                bins=100,
                                figsize=(5, 4),
                                snsbins=60,
                                plot_type='Stenosis',
                                output_folder=output_plots,
                                label_name_ori='ffr_all')
    return None 

def wrap_plot_all_roc(df, label_names, confidence_names, output_plots,
                      group_aggregated, config):
    #if config['loss']['name'][0] in ['MSE', '_L1', 'L1smooth']:
    threshold_ffr = config['loaders']['val_method']['threshold_ffr']
    threshold_sten = config['loaders']['val_method']['threshold_sten']
    # else:
    #     threshold_ffr = 0.5
    #     threshold_sten = 0.5

        
    df_sten = df.copy()
    sten_cols_conf = select_relevant_columns(confidence_names, 'sten')
    sten_cols_true = select_relevant_columns(label_names, 'sten')
    if not group_aggregated:
        for sten_col_conf in sten_cols_conf:
            df_sten[sten_col_conf] = convertConfFloats(
                df[sten_col_conf],
                config['loss']['name'][0])
    plot_roc_all(df_sten, sten_cols_true, sten_cols_conf, output_plots,
                 plot_type='stenosis',
                 config=config,
                 theshold=threshold_sten)
    df_ffr = df.copy()
    ffr_cols_true = select_relevant_columns(label_names, 'ffr')
    if len(ffr_cols_true) > 0:
        ffr_cols_conf = select_relevant_columns(confidence_names, 'ffr')
        if not group_aggregated:
            for ffr_col_conf in ffr_cols_conf:
                df_ffr[ffr_col_conf] = convertConfFloats(
                    df_ffr[ffr_col_conf],
                    config['loss']['name'][0])
        plot_roc_all(df_ffr, ffr_cols_true, ffr_cols_conf, output_plots,
                     plot_type='FFR',
                     config=config,
                     theshold=threshold_ffr)
    # else:
    #     print('No FFR labels found in database')
    


def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string

def plot_results(sql_config, label_names, prediction_names, output_plots,
                 num_classes, config, confidence_names,
                 group_aggregated=False):
    df, _ = getDataFromDatabase(sql_config)
    if group_aggregated:
        confidence_names = [c + '_aggregated' for c in confidence_names]
    wrap_plot_all_sten_reg(df, label_names, confidence_names, output_plots,
                           group_aggregated, config)    
    wrap_plot_all_roc(df, label_names, confidence_names, output_plots,
                      group_aggregated,
                      config=config)

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
        if config['loss']['name'][c] in ['MSE', '_L1', 'L1smooth']:
            df_to_process = df_label.copy()
            plot_results_regression(df_to_process, confidence_name,
                                    label_name, config, c, support,
                                    output_plots,
                                    group_aggregated)
            #if any(item.startswith('ffr') for item in config['labels_names']):
            if label_name.startswith('sten_'):
                try:
                    df_to_process_ffr = df_label.copy()
                    name = 'ffr' + label_name[4:]
                    name = remove_suffix(name, "_transformed")
                    ffr_thres = config[
                        'loaders']['val_method']['threshold_ffr']
                    df_to_process_ffr[label_name + '_ffr_corrected'] \
                        = (df_to_process_ffr[name] <= ffr_thres).astype(int)
                    df_to_process_ffr = df_to_process_ffr[
                        df_to_process_ffr[name].notna()]
                    output_plots_ffr = os.path.join(
                        output_plots, 'ffr_corrected')
                    mkFolder(output_plots_ffr)
                    support = len(df_to_process_ffr)
                    plot_results_regression(df_to_process_ffr, confidence_name,
                                            label_name + '_ffr_corrected',
                                            config, c, support,
                                            output_plots_ffr,
                                            group_aggregated)
                except IndexError:
                    print('No FFR labels found in database')
                except:
                    print('Error in FFR correction')


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
        if sql_config['task_type'] != 'regression':
            df_plot_rep, _ = select_relevant_data(
                df_plot, prediction_name, label_name)
        else:
            df_plot_rep = df_plot.copy()
        mask1 = df_plot_rep[prediction_name].isna()
        mask2 = df_plot_rep[label_name].isna()
        mask = mask1 | mask2
        df_plot_rep = df_plot_rep[~mask]
        
        df_plot_rep, label_name = rename_columns(df_plot_rep, label_name)
        df_plot_rep = df_plot_rep.astype({label_name: float})

        if group_aggregated is False:
            df_plot_rep[prediction_name] = \
                convertConfFloats(
                    df_plot_rep[prediction_name],
                    sql_config['loss_name'][c])

        df_plot_rep[prediction_name] = np.clip(
            df_plot_rep[prediction_name], a_min=0, a_max=1)

        df_plot_rep = df_plot_rep.astype({prediction_name: float})
        plot_regression_density(x=df_plot_rep[label_name],
                                y=df_plot_rep[prediction_name],
                                cmap='jet', ylab='prediction', xlab='true',
                                bins=100,
                                figsize=(5, 4),
                                snsbins=60,
                                plot_type='Stenosis' if label_name_ori.startswith('sten') else 'FFR',
                                output_folder=output_folder,
                                label_name_ori=label_name_ori)
    #remove nan
        g = sns.lmplot(x=label_name, y=prediction_name, data=df_plot_rep)
        X2 = sm.add_constant(df_plot_rep[label_name])
        est = sm.OLS(df_plot_rep[prediction_name], X2)
        est2 = est.fit()
        r = est2.rsquared
        p = est2.pvalues[label_name]
        p = '%.2E' % Decimal(p)
        rmse = math.sqrt(mean_squared_error(df_plot_rep[label_name], df_plot_rep[prediction_name]))
        mae = mean_absolute_error(df_plot_rep[label_name], df_plot_rep[prediction_name])


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
        
        df2 = pd.DataFrame(
            np.array([[mae, rmse, p, r]]),
            columns=['MAE', 'RMSE', 'p-value', 'R-squared'])
        df2.to_csv(
            os.path.join
            (output_folder, label_name_ori + '_regression.csv'))
    return None


def plot_regression_density(x=None, y=None, cmap='jet', ylab=None, xlab=None,
                            bins=100,
                            figsize=(5, 4),
                            snsbins=60,
                            plot_type=None,
                            output_folder=None,
                            label_name_ori=None):

    #remove nan
    mask = x.isna()
    mask = mask | y.isna()
    x = x[~mask]
    y = y[~mask]
    
    x = x.to_numpy()
    y = y.to_numpy()
    y = np.clip(
            y, a_min=0, a_max=1)
    x = np.clip(
            x, a_min=0, a_max=1)
    ax1 = sns.jointplot(x=x, y=y, marginal_kws=dict(bins=snsbins))
    ax1.fig.set_size_inches(figsize[0], figsize[1])
    ax1.ax_joint.cla()
    plt.sca(ax1.ax_joint)
    plt.hist2d(
        x, y, bins=bins,
        norm=matplotlib.colors.LogNorm(), cmap=cmap)
    #plt.title('Density plot')
    plt.xlabel(plot_type + ' ' + xlab, fontsize=12)
    plt.ylabel(plot_type + ' ' + ylab, fontsize=12)
    cbar_ax = ax1.fig.add_axes([1, 0.1, 0.03, 0.7])
    cb = plt.colorbar(cax=cbar_ax)
    cb.set_label(r'$\log_{10}$ density of points',
                 fontsize=13)
    plt.savefig(os.path.join(output_folder, label_name_ori + '_density.png'),
                dpi=100, bbox_inches='tight')
    plt.savefig(os.path.join(output_folder, label_name_ori + '_density.pdf'),
                dpi=100, bbox_inches='tight')
    plt.close()

    return None


def combine_plots(path_rca, path_lca):
    rca = pd.read_csv(path_rca)
    lca = pd.read_csv(path_lca)
    # concatenate csv files
    combined = pd.concat([rca, lca])
    return combined
    
    
def plot_combined_roc(combined_bce, combined_reg, output_path):
    from sklearn.metrics import roc_curve, roc_auc_score
    y_test = combined_bce['trues']
    yproba = combined_bce['probas']
    
    y_test_reg = combined_reg['trues']
    yprobareg = combined_reg['probas']
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    fpr_reg, tpr_reg, _ = roc_curve(y_test_reg,  yprobareg)
    auc_reg = roc_auc_score(y_test_reg, yprobareg)
    plt.figure(figsize=(16, 12))
    plt.plot(fpr_reg,
             tpr_reg,
             label="Regression model, mean AUC={:.3f}".format(auc_reg))

    plt.plot(fpr,
             tpr,
             label="Classification model, mean AUC={:.3f}".format(auc))
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_path, '_roc_all.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_path, '_roc_all.pdf'), dpi=100,
                bbox_inches='tight')
    plt.close()


def plot_combined_roc_mean_vs_max(combined_bce, combined_reg, combine_bce_max, combine_reg_max, output_path):
    from sklearn.metrics import roc_curve, roc_auc_score
    y_test = combined_bce['trues']
    yproba = combined_bce['probas']
    
    y_test_reg = combined_reg['trues']
    yprobareg = combined_reg['probas']
    
    y_test_reg_max = combine_reg_max['trues']
    yprobareg_max = combine_reg_max['probas']
    y_test_bce_max = combine_bce_max['trues']
    yprobabce_max = combine_bce_max['probas']
    
    fpr_reg_max, tpr_reg_max, _ = roc_curve(y_test_reg_max,  yprobareg_max)
    auc_reg_max = roc_auc_score(y_test_reg_max, yprobareg_max)
    
    fpr_bce_max, tpr_bce_max, _ = roc_curve(y_test_bce_max,  yprobabce_max)
    auc_bce_max = roc_auc_score(y_test_bce_max, yprobabce_max)
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    fpr_reg, tpr_reg, _ = roc_curve(y_test_reg,  yprobareg)
    auc_reg = roc_auc_score(y_test_reg, yprobareg)
    
    
    
    plt.figure(figsize=(16, 12))
    plt.plot(fpr_reg,
             tpr_reg,
             label="Regression model (mean aggregated), mean AUC={:.3f}".format(auc_reg))

    plt.plot(fpr_reg_max,
             tpr_reg_max,
             label="Regression model (max aggregated), mean AUC={:.3f}".format(auc_reg_max))

    plt.plot(fpr,
             tpr,
             label="Classification model (mean aggregated), mean AUC={:.3f}".format(auc))
    

    
    plt.plot(fpr_bce_max,
             tpr_bce_max,
             label="Classification model (max aggregated), mean AUC={:.3f}".format(auc_bce_max))
    
    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_path, '_roc_all.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_path, '_roc_all.pdf'), dpi=100,
                bbox_inches='tight')
    plt.close()



def make_plots(path_rca_bce, path_lca_bce, path_rca, path_lca, output_path):
    combined_bce = combine_plots(path_rca_bce, path_lca_bce)
    combined_reg = combine_plots(path_rca, path_lca)
    mkFolder(output_path)
    plot_combined_roc(combined_bce, combined_reg, output_path)
    return None

def make_plots_compare_max_mean(path_rca_bce,
                                path_lca_bce,
                                path_rca_bce_max,
                                path_lca_bce_max,
                                path_rca,
                                path_lca,
                                path_rca_max,
                                path_lca_max,
                                output_path):
    combined_bce = combine_plots(path_rca_bce, path_lca_bce)
    combined_reg = combine_plots(path_rca, path_lca)
    combine_plots_bce_max = combine_plots(path_rca_bce_max, path_lca_bce_max)
    combine_plots_reg_max = combine_plots(path_rca_max, path_lca_max)
    
    mkFolder(output_path)
    plot_combined_roc_mean_vs_max(combined_bce, combined_reg, combine_plots_bce_max, combine_plots_reg_max, output_path)
    return None

if __name__ == '__main__':
    path_rca_bce = "/home/alatar/miacag/output/outputs_stenosis_identi/classification_config_angio_SEP_Jan21_15-32-06/plots/train/probas_trues.csv"
    path_lca_bce = "/home/alatar/miacag/output/outputs_stenosis_identi/classification_config_angio_SEP_Jan21_15-37-00/plots/train/probas_trues.csv"
    path_lca = "/home/alatar/miacag/output/outputs_stenosis_reg/classification_config_angio_SEP_Jan21_15-19-24/plots/train/probas_trues.csv"
    path_rca = "/home/alatar/miacag/output/outputs_stenosis_reg/classification_config_angio_SEP_Jan21_15-38-35/plots/train/probas_trues.csv"
    output_path = "/home/alatar/miacag/output/outputs_stenosis_identi/classification_config_angio_SEP_Jan21_15-32-06/plots/comb"
    make_plots(path_rca_bce, path_lca_bce, path_rca, path_lca, output_path)
    