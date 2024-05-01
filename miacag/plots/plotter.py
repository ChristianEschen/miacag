import os
import numpy as np
from miacag.utils.sql_utils import getDataFromDatabase
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import f1_score, \
     accuracy_score, confusion_matrix#, plot_confusion_matrix
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
from functools import reduce
import re
import copy

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


def aggregate_pr_group(df, aggregated_cols_list, agg_type):
    cols_select = aggregated_cols_list + ["PatientID", "StudyInstanceUID", "rowid"]
def compute_aggregation(df, aggregated_cols_list, agg_type="max"):
        # return records from dataframe to update
        df_copy = copy.deepcopy(df)
        df_copy.drop(columns=aggregated_cols_list, inplace=True)
        cols_select = aggregated_cols_list + ["PatientID", "StudyInstanceUID"]
        df = df[cols_select]
        count = 0
        df_frames = []
        for field in aggregated_cols_list:
            df_field = df.copy()
            df_field = df_field[df_field[field].notna()]
            agg_field = aggregated_cols_list[count]
            confidences = df_field[field]

            df_new = df_field.copy()
           # self.df[agg_field] = confidences
            df_new[agg_field] = confidences
            if agg_type == "mean":
                df_new = df_new.groupby(
                    ['PatientID', 'StudyInstanceUID'])[agg_field].mean()
            elif agg_type == "max":
                if field.startswith('ffr'):
                    df_new = df_new.groupby(
                        ['PatientID', 'StudyInstanceUID'])[agg_field].min()
                elif field.startswith('sten'):
                    df_new = df_new.groupby(
                        ['PatientID', 'StudyInstanceUID'])[agg_field].max()
                else:
                    raise ValueError('not implemented')
            else:
                raise ValueError('agg_type must be either mean or max')
            df_new = df_new.to_frame()
            df_new.reset_index(inplace=True)
            df_field.drop(columns=aggregated_cols_list, inplace=True)

            
            df2 = df_new.merge(df_field, left_on=['PatientID', 'StudyInstanceUID'],
                    right_on=['PatientID', 'StudyInstanceUID'],
                    how='right').drop_duplicates(["StudyInstanceUID", "PatientID"])
            df_frames.append(df2)
            count += 1
        result_df = reduce(lambda  left,right: pd.merge(left,right,on=['PatientID', "StudyInstanceUID"],
                                            how='inner'), df_frames)
        result_df = result_df.merge(df_copy, on=['PatientID', 'StudyInstanceUID'],how='inner')
        return result_df
        
def convertConfFloats(confidences, loss_name, config):
    confidences_conv = []
    for conf in confidences:
        if loss_name.startswith('CE'):
            if conf is None:
                confidences_conv.append(np.nan)
            else:
                confidences_conv.append(float(conf.split(";1:")[-1][:-1]))

        elif loss_name.startswith('NNL'):
            if conf is None:
                float_converted = []
                for i, c in enumerate(range(0,config['model']['num_classes'][0])):
                    float_converted.append(np.nan)
                confidences_conv.append(float_converted)
            
            else:
                strings = ""
                float_converted = []
                for i, c in enumerate(range(0,config['model']['num_classes'][0])):
                    if i <config['model']['num_classes'][0]-1:

                        match =  re.search('{}:(.*){}:'.format(c, c+1), conf)
                        strings = conf[match.regs[0][0]+2: match.regs[0][1]-3]
                    else:
                        match =  re.search('{}:(.*)'.format(c), conf)
                        strings = conf[match.regs[0][0]+2: match.regs[0][1]-1]
                    #   strings = conf[idx[0]+2:idx[1]]
                    float_converted.append(float(strings))
                confidences_conv.append(float_converted)
        elif loss_name in ['MSE', '_L1', 'L1smooth', 'BCE_multilabel', 'wfocall1']:
            if conf is None:
                confidences_conv.append(np.nan)
            # test if conf is np.nan
            elif conf == np.nan:
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
                   plot_name, f1, output, num_classes, support, c, mcc=None):
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
        plot_name + ': Confusion Matrix, F1-macro:' + str(f1) +
        ',support(N)=' + str(support))
    if mcc is not None:
        plt.title(
            
        plot_name + ': Confusion Matrix, F1-macro:' +  str(f1) +  ' ,MCC:' + str(mcc) +
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


    df_label[label_name] = threshold_continuois(
        df_label[label_name],
        config,
        label_name)
    if label_name.startswith('ffr'):
        max_v = 1
    elif label_name.startswith('sten'):
        max_v = 1
    elif label_name.startswith('timi'):
        max_v = 3
    else:
        raise ValueError('label_name_ori must be either sten or ffr or timi')
    df_label[confidence_name] = np.clip(
        df_label[confidence_name], a_min=0, a_max=max_v)
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
            df_label[confidence_name], config['loss']['name'][c], config)
    if label_name.startswith('ffr'):
        max_v = 1
    elif label_name.startswith('sten'):
        max_v = 1
    elif label_name.startswith('timi'):
        max_v = 3
    else:
        raise ValueError('label_name_ori must be either sten or ffr or timi')
    df_label[confidence_name] = np.clip(
        df_label[confidence_name], a_min=0, a_max=max_v)
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
    if len(sten_cols_conf) > 0:
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
    if len(sten_cols_true) > 0:
        plot_roc_all(df_sten, sten_cols_true, sten_cols_conf, output_plots,
                    plot_type='stenosis',
                    config=config,
                    theshold=threshold_sten)
    df_ffr = df.copy()
    ffr_cols_true = select_relevant_columns(label_names, 'ffr')
    if len(ffr_cols_true) > 0:
        ffr_cols_conf = select_relevant_columns(confidence_names, 'ffr')
      
        plot_roc_all(df_ffr, ffr_cols_true, ffr_cols_conf, output_plots,
                     plot_type='FFR',
                     config=config,
                     theshold=threshold_ffr)
    # else:
    #     print('No FFR labels found in database')
    

def add_misssing_rows(df, label_name):
    # Forward fill the 'ffr_proc_2_midt_rca_transformed' within each group of 'PatientID' and 'StudyInstanceUID'
    df[label_name] = df.groupby(['PatientID', 'StudyInstanceUID'])[label_name].fillna(method='ffill')

    # If any NaNs are still present, backfill them within each group
    df[label_name] = df.groupby(['PatientID', 'StudyInstanceUID'])[label_name].fillna(method='bfill')
    return df

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string




def select_relevant_data_dominans(result_table, segments):
    # Make a deep copy of the DataFrame to avoid modifying the original data
    final_table = copy.deepcopy(result_table)

    # Iterate over each row in the DataFrame
    for index, row in final_table.iterrows():
        # Check the 'labels_predictions' for each row to determine the logic to apply
        if row['labels_predictions'] == 1:  # Assuming '1' is for 'right'
            for seg in segments:
                if 'pla_rca' in seg:
                    # Set values to np.nan based on 'dominans' condition
                    if row['dominans'] not in ["Højre dominans (PDA+PLA fra RCA)", None]:
                        final_table.at[index, seg] = np.nan
                elif 'pda_t' in seg:
                    # Set values to np.nan based on 'dominans' condition
                    if row['dominans'] not in ["Højre dominans (PDA+PLA fra RCA)", "Balanceret (PDA fra RCA/PLA fra LCX)", None]:
                        final_table.at[index, seg] = np.nan
                elif 'pla_lca' in seg:
                    final_table.at[index, seg] = np.nan
                elif 'pda_lca' in seg:
                    final_table.at[index, seg] = np.nan
                    

        elif row['labels_predictions'] == 0:  # Assuming '0' is for 'left'
            for seg in segments:
                if 'pla_lca' in seg:
                    # Set values to np.nan based on 'dominans' condition
                    if row['dominans'] not in ["Venstre dominans (PDA+PLA fra LCX)", "Balanceret (PDA fra RCA/PLA fra LCX)"]:
                        final_table.at[index, seg] = np.nan
                elif 'pda_lca' in seg:
                    # Set values to np.nan based on 'dominans' condition
                    if row['dominans'] not in ["Venstre dominans (PDA+PLA fra LCX)"]:
                        final_table.at[index, seg] = np.nan
                elif 'pla_rca' in seg:
                    final_table.at[index, seg] = np.nan
                elif 'pda_t' in seg:
                    final_table.at[index, seg] = np.nan

        else:
            # Raise an error if 'labels_predictions' contains an unexpected value
            raise ValueError('Unexpected value in labels_predictions')
  #  final_table['sten_proc_16_pla_lca_transformed_confidences'] = df2['combined_confidences'].fillna(df2['sten_proc_16_pla_lca_transformed_confidences'])
    pda_names = [i for i in segments if '4_pda' in i]
    agg_pda_name = [i for i in pda_names if 'lca' not in i]
    pla_names = [i for i in segments if '16_pla' in i]
    agg_pla_name = [i for i in pla_names if 'lca' not in i]
    if len(pda_names) > 1:
        final_table[agg_pda_name[0]] = final_table[pda_names[0]].fillna(final_table[pda_names[1]])
    else:
        try:
            final_table['sten_proc_4_pda_transformed_confidences'] = final_table[pda_names[0]]
        except:
            print('No pda_lca found in database')
    if len(pla_names) > 1:

        final_table[agg_pla_name[0]] = final_table[pla_names[0]].fillna(final_table[pla_names[1]])
    else:
        try:
            final_table['sten_proc_16_pla_rca_transformed_confidences'] = final_table[pla_names[0]]
        except:
            print('No pla_lca found in database')
    # only keep the elemetn without lca
    
    return final_table

def select_only_aggregates(segments):
    pda_names = [i for i in segments if '4_pda' in i]
    agg_pda_name = [i for i in pda_names if 'lca' not in i]
    pla_names = [i for i in segments if '16_pla' in i]
    agg_pla_name = [i for i in pla_names if 'lca' not in i]
    include = agg_pda_name + agg_pla_name
    union = pda_names + pla_names
    exclude = [i for i in union if i not in include]
    final_segments = [i for i in segments if i not in exclude]
    return final_segments
def rename_label_names(label_names, prediction_names, confidence_names):
    
    label_names = select_only_aggregates(label_names)
    prediction_names = select_only_aggregates(prediction_names)
    confidence_names = select_only_aggregates(confidence_names)
    return label_names, prediction_names, confidence_names

def simulte_df(label_names, prediction_names, confidence_names):
    np.random.seed(42)

    # Define the number of rows
    num_rows = 1000

    # Creating synthetic data for the DataFrame
    data = {
        confidence_names[0]: np.random.rand(num_rows),
        prediction_names[0]: np.random.rand(num_rows),
        label_names[0]: np.random.rand(num_rows),
        confidence_names[1]: np.random.rand(num_rows),
        prediction_names[1]: np.random.rand(num_rows),
        label_names[1]: np.random.rand(num_rows),
        confidence_names[2]: np.random.rand(num_rows),
        prediction_names[2]: np.random.rand(num_rows),
        label_names[2]: np.random.rand(num_rows),
        # confidence_names[3]: np.random.rand(num_rows),
        # prediction_names[3]: np.random.rand(num_rows),
        # label_names[3]: np.random.rand(num_rows),
        # confidence_names[4]: np.random.rand(num_rows),
        # prediction_names[4]: np.random.rand(num_rows),
        # label_names[4]: np.random.rand(num_rows),
        # confidence_names[5]: np.random.rand(num_rows),
        # prediction_names[5]: np.random.rand(num_rows),
        # label_names[5]: np.random.rand(num_rows),
        'labels_predictions': np.random.randint(0, 2, num_rows),
        'dominans': np.random.choice(["Venstre dominans (PDA+PLA fra LCX)", "Balanceret (PDA fra RCA/PLA fra LCX)", "Højre dominans (PDA+PLA fra RCA)", None], num_rows),
        'StudyInstanceUID': [f"1.3.12.2.1107.5.4.3.{np.random.randint(1,3)}.{np.random.randint(1,2)}{np.random.randint(1,2):02d}{np.random.randint(1,2):02d}" for _ in range(num_rows)],
        'PatientID': [f"{np.random.choice(['D97258', 'X82947', 'Z78325'])}/{np.random.randint(1,3)}" for _ in range(num_rows)]
    }

    # Create the DataFrame
    large_df = pd.DataFrame(data)
    # use index to create a column with unique values
    large_df['rowid'] = large_df.index

    # Replace random values with NaN to mimic the original pattern
    nan_fill_probability = 0.5  # 20% NaNs in each column approximately
    for column in large_df.columns[:6]:
        large_df[column].iloc[0:500] = np.nan
    # for column in large_df.columns[3:6]:
    #     large_df[column].iloc[500:] = np.nan
    return large_df
def plot_results(sql_config, label_names, prediction_names, output_plots,
                 num_classes, config, confidence_names,
                 group_aggregated=False):
    df, _ = getDataFromDatabase(sql_config)
    
    df = select_relevant_data_dominans(df, label_names)
    label_names,prediction_names, confidence_names = rename_label_names(label_names, prediction_names, confidence_names)
    df =df.dropna(subset=confidence_names, how='all')
    for conf in confidence_names:
        df[conf] = convertConfFloats(df[conf], config['loss']['name'][0], config)
    
    #df = simulte_df(label_names, prediction_names, confidence_names)

    plot_wrapper(df, label_names, prediction_names, confidence_names, output_plots, config, group_aggregated)
    
    
def plot_wrapper(df, label_names, prediction_names, confidence_names, output_plots, config, group_aggregated=False):
    # create simulated df with columns: label_names, prediction_names, confidence_names

    if group_aggregated:
        df = compute_aggregation(df, confidence_names, agg_type="max")
        # insert group_aggregation function here
    # test if a element is a list starts with a string: "sten"
    stens = select_relevant_columns(label_names, 'sten')
    # copy row values in _transfored if PatientID and StudyInstanceUID are the same
    
    wrap_plot_all_sten_reg(df, label_names, confidence_names, output_plots,
                        group_aggregated, config)    
    wrap_plot_all_roc(df, label_names, confidence_names, output_plots,
                      group_aggregated,
                      config=config)

    for c, label_name in enumerate(label_names):
        confidence_name = confidence_names[c]
        prediction_name = prediction_names[c]
      #  df = add_misssing_rows(df, label_name)
        df_label = df[df[label_name].notna()]
        df_label = df_label[df_label[confidence_name].notna()]
       # df_label = df_label[df_label[prediction_name].notna()]
        if group_aggregated:
            df_label = df_label.drop_duplicates(
                subset=['StudyInstanceUID', "PatientID"])
        support = len(df_label)
        if config['loss']['name'][c] in ['MSE', '_L1', 'L1smooth', 'wfocall1']:
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
    for label_name in label_names:
        df = add_misssing_rows(df, label_name)
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
                   confidence_names, output_folder, config, group_aggregated=False):
    df, _ = getDataFromDatabase(sql_config)
    prediction_names = [label_name+'_predictions' for label_name in label_names]
    df = select_relevant_data_dominans(df, label_names)
    label_names, prediction_names, confidences = rename_label_names(label_names, prediction_names, confidence_names)
    df =df.dropna(subset=confidence_names, how='all')
    
    for conf in confidence_names:
        df[conf] = convertConfFloats(df[conf], config['loss']['name'][0], sql_config)

   # df = simulte_df(label_names, prediction_names, confidence_names)

    wrap_plotRegression(df, label_names, prediction_names, output_folder, group_aggregated, sql_config)
    return None
def wrap_plotRegression(df, label_names, prediction_names, output_folder, group_aggregated, sql_config):
    from miacag.plots.plot_roc_auc_all import plot_regression_all
    if group_aggregated:
        df = compute_aggregation(df, prediction_names, agg_type="max")
    plot_regression_all(df,
                        label_names, prediction_names,
                        output_folder, sql_config)

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
        #if sql_config['task_type'] != 'regression':
        if sql_config['task_type'] != 'mil_classification':
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


        if label_name_ori.startswith('ffr'):
            max_v = 1
        elif label_name_ori.startswith('sten'):
            max_v = 1
        elif label_name_ori.startswith('timi'):
            max_v = 3
        else:
            raise ValueError('label_name_ori must be either sten or ffr or timi')
        df_plot_rep[prediction_name] = np.clip(
            df_plot_rep[prediction_name], a_min=0, a_max=max_v)

        df_plot_rep = df_plot_rep.astype({prediction_name: float})
        if label_name_ori.startswith('sten'):
            plot_type = 'Stenosis'
        elif label_name_ori.startswith('ffr'):
            plot_type = 'FFR'
        elif label_name_ori.startswith('timi'):
            plot_type = 'TIMI flow'
        plot_regression_density(x=df_plot_rep[label_name],
                                y=df_plot_rep[prediction_name],
                                cmap='jet', ylab='prediction', xlab='true',
                                bins=100,
                                figsize=(5, 4),
                                snsbins=60,
                                plot_type=plot_type,
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
            # ax.text(0.05, 0.9,
            #         "p-value = " + p,
            #         fontsize=9,
            #         transform=ax.transAxes)
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
    if label_name_ori.startswith('ffr'):
        max_v = 1
    elif label_name_ori.startswith('sten'):
        max_v = 1
    elif label_name_ori.startswith('timi'):
        max_v = 3
    else:
        raise ValueError('label_name_ori must be either sten or ffr or timi')
    x = x.to_numpy()
    y = y.to_numpy()
    y = np.clip(
            y, a_min=0, a_max=max_v)
    x = np.clip(
            x, a_min=0, a_max=max_v)
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
  #  make_plots(path_rca_bce, path_lca_bce, path_rca, path_lca, output_path)
    output_plots = "/home/alatar/miacag/output_plots/test"

    output_plots_Reg = "/home/alatar/miacag/output_plots/test_reg"
    config_path = '/home/alatar/miacag/my_configs/stenosis_regression/classification_config_angio.yaml'
    # for combined plots
    import yaml
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    labels = ['sten_proc_1_prox_rca_transformed', 'sten_proc_6_prox_lad_transformed', "sten_proc_16_pla_rca_transformed", "sten_proc_16_pla_lca_transformed", 'sten_proc_4_pda_transformed', 'sten_proc_4_pda_lca_transformed'] #_4_pda_lca
    predictions = ['sten_proc_1_prox_rca_transformed_predictions', 'sten_proc_6_prox_lad_transformed_predictions', "sten_proc_16_pla_rca_transformed_predictions", "sten_proc_16_pla_lca_transformed_predictions",  'sten_proc_4_pda_transformed_predictions', 'sten_proc_4_pda_lca_transformed-predictions']
    confidences = ['sten_proc_1_prox_rca_transformed_confidences', 'sten_proc_6_prox_lad_transformed_confidences',  "sten_proc_16_pla_rca_transformed_confidences", "sten_proc_16_pla_lca_transformed_confidences",  'sten_proc_4_pda_transformed_confidences', 'sten_proc_4_pda_lca_transformed_confidences']
    labels = ['sten_proc_1_prox_rca_transformed', "sten_proc_16_pla_rca_transformed", 'sten_proc_4_pda_transformed'] #_4_pda_lca
    predictions = ['sten_proc_1_prox_rca_transformed_predictions', "sten_proc_16_pla_rca_transformed_predictions",  'sten_proc_4_pda_transformed_predictions']
    confidences = ['sten_proc_1_prox_rca_transformed_confidences',  "sten_proc_16_pla_rca_transformed_confidences", 'sten_proc_4_pda_transformed_confidences']

    config['query'] = config['query_rca']

    # df, _ = getDataFromDatabase(config)
    # confidence_names = [label_name+'_transformed_confidences' for label_name in config['labels_names']]
    # for conf in confidence_names:
    #     df[conf] = convertConfFloats(df[conf], config['loss']['name'][0], config)

    df = simulte_df(labels, predictions, confidences)

    df = select_relevant_data_dominans(df,  confidences)
    labels, predictions, confidences = rename_label_names(labels, predictions, confidences)
    df = select_relevant_data_dominans(df, labels)
   # df = select_relevant_data_dominans(df, label_names)
    wrap_plotRegression(df,  labels, predictions,output_plots_Reg, True, config)
    plot_wrapper(df, labels, predictions, confidences, output_plots, config, True)
    
    # for individual plots
    # read config file
    # import yaml
    # with open(config_path, 'r') as stream:
    #     config = yaml.safe_load(stream)
    # labels = ['ffr_proc_1_prox_rca_transformed', 'ffr_proc_6_prox_lad_transformed']
    # predictions = ['ffr_proc_1_prox_rca_transformed_predictions', 'ffr_proc_6_prox_lad_transformed_predictions']
    # confidences = ['ffr_proc_1_prox_rca_transformed_confidences', 'ffr_proc_6_prox_lad_transformed_confidences']
    # config['query'] = config['query_rca']
    # df, _ = getDataFromDatabase(config)
    # confidence_names = [label_name+'_transformed_confidences' for label_name in config['labels_names']]
    # for conf in confidence_names:
    #     df[conf] = convertConfFloats(df[conf], config['loss']['name'][0], config)

    # df = simulte_df(labels, predictions, confidences)


    # wrap_plotRegression(df,  labels, predictions,output_plots_Reg, True, config)
    # plot_wrapper(df, labels, predictions, confidences, output_plots, config, True)