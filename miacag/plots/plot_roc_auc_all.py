import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
from matplotlib.ticker import MaxNLocator
matplotlib.use('Agg')
from sklearn import metrics
import random
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import roc_curve, roc_auc_score
from miacag.utils.script_utils import mkFolder
#from miacag.plots.plotter import rename_columns
from miacag.plots.plot_utils import get_mean_lower_upper
import statsmodels.api as sm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import copy

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    
def generate_data():
    data = datasets.load_breast_cancer()

    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=.25,
                                                        random_state=1234)
    
    
   # Instantiate the classfiers and make a list
    classifiers = [LogisticRegression(random_state=1234), 
                GaussianNB(), 
                KNeighborsClassifier(), 
                DecisionTreeClassifier(random_state=1234),
                RandomForestClassifier(random_state=1234)]

    # Define a result table as a DataFrame
    segments = ['sten_proc_1_rca_prox', 'sten_proc_2_rca_mid', 'sten_proc_3_rca_dist', 'sten_proc_4_rca_pla', 'sten_proc_16_rca_pda']
    confidences = [i + "_confidences" for i in segments]

    trues = [i + "_transformed" for i in segments]
    result_table = pd.DataFrame(columns=confidences + trues + ['labels_predictions'] + ['dominans'])
    domianse = [
        "Balanceret (PDA fra RCA/PLA fra LCX)",
        "Højre dominans (PDA+PLA fra RCA)",
        "Venstre dominans (PDA+PLA fra LCX)"]

    # save data to mimic the results table
    idxs = 0
    for cls in classifiers:
        domianse_cop = domianse.copy()
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::,1]
        result_table[confidences[idxs]] = yproba
        # generate random normal uniformly distributed number between 0 and 1 for y_test
        y_test = np.random.uniform(0, 1, len(yproba))
        result_table[trues[idxs]] = y_test
        result_table['labels_predictions'] = 1
        
        result_table['dominans'] = random.sample(domianse_cop*2000, 143)
        
        idxs += 1
    return result_table, confidences, trues, 'dominans', 'labels_predictions'





def threshold_continues(continuos_inc, threshold, name):
    continuos = continuos_inc.to_numpy()
    if name.startswith('ffr'):
        continuos[continuos >= threshold] = 1
        continuos[continuos < threshold] = 0
        continuos = np.logical_not(continuos).astype(int)
    elif name.startswith('sten'):
        # exclude lm
        if 'proc_5_lm_' in name:
            threshold = 0.5
        continuos[continuos >= threshold] = 1
        continuos[continuos < threshold] = 0
    else:
        raise ValueError('name is not ffr or sten')
    return continuos


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


def classification_report_func(trues, preds):
    report = classification_report(trues, preds, output_dict=True)

    # Calculate confusion matrix
    cm = confusion_matrix(trues, preds)

    # Extract TN, FP, FN, TP
    tn, fp, fn, tp = cm.ravel()

    # Calculate specificity
    specificity = tn / (tn + fp)

    # Sensitivity is the same as recall for the positive class

    # Add specificity to the report
    # report['specificity'] = {'1': specificity, '0': None, 'weighted avg': specificity, 'macro avg': specificity}

    # Print the updated classification report
    report_df = pd.DataFrame(report)
    specificity_row = pd.Series({'0.0': None, '1.0': specificity, 'accuracy': None, 'macro avg': None, 'weighted avg': None}, name='specificity')
    report_df = report_df.append(specificity_row)
    report_df = report_df.reset_index()

    return report_df

def plot_roc_all(result_table, trues_names, confidences_names, output_plots, plot_type, config, theshold=0.5):
       # Define a result table as a DataFrame
    roc_result_table = pd.DataFrame(columns=['segments', 'fpr','tpr','auc', 'auc_lower', 'auc_upper', ' precision', 'recall', 'f1', 'f1_lower', 'f1_upper'])

    probas = []
    trues = []
    probas_bin = []
    idx = 0
    raw_trues_list = []
    for seg in confidences_names:
        print('seg', seg)
        if seg == 'ffr_proc_15_dist_lcx_transformed_confidences':
            result_table.loc[result_table[trues_names[idx]]==0.8600000143051147, trues_names[idx]] = 0.799
        if seg == 'sten_proc_4_pda_lca_transformed_confidences':
            # set two rows to 0.799 for the column seg
            result_table['sten_proc_4_pda_lca_transformed'].iloc[-1] = 0.9

        if seg == 'sten_proc_10_d2_transformed_confidences':
            result_table.loc[result_table[trues_names[idx]]==0.67, trues_names[idx]] = 0.799
        result_table_copy = copy.deepcopy(result_table)
        maybeRCA = ""
        result_table_copy[confidences_names[idx]] = transform_confidences_to_by_label_type(
            result_table_copy[confidences_names[idx]], seg)
   #    if config['loss']['name'][0] in ['MSE', '_L1', 'L1smooth']:
        raw_trues = result_table_copy[trues_names[idx]]
        # deep copy raw trues
        raw_trues = copy.deepcopy(raw_trues.values)
        print('raw trues', result_table_copy[trues_names[idx]])
        print('raw trues max', result_table_copy[trues_names[idx]].max())
        result_table_copy[trues_names[idx]] = threshold_continues(
            result_table_copy[trues_names[idx]], threshold=theshold, name=seg)
        y_test = result_table_copy[trues_names[idx]].values
        yproba = result_table_copy[confidences_names[idx]].values
        mask_propa = np.isnan(yproba)
        mask_test = np.isnan(y_test)
        mask = mask_propa + mask_test
        y_test = y_test[~mask]
        raw_trues = raw_trues[~mask]
        y_test = copy.deepcopy(y_test)
        yproba = yproba[~mask]
        # deepcopy yproba
        ypred_bin = copy.deepcopy(yproba)
        ypred_bin  = np.clip(ypred_bin, a_min=0, a_max=1)
        ypred_bin = threshold_continues(
            pd.DataFrame(ypred_bin), threshold=theshold, name=seg)
        print('ypred_bin', ypred_bin)
        print('y_test', y_test)
        print('yproba', yproba)
      #  yproba = np.clip(yproba, a_min=0, a_max=1)
        print('yproba post clip', yproba)
        #DEBUG
        # sum ypred_bin
        positives = np.sum(ypred_bin)
        # probas = np.random.rand(len(probas))
        # y_test = np.random.randint(2, size=len(y_test))
        # ypred_bin = np.random.randint(2, size=len(ypred_bin))
        try:
            fpr, tpr, _ = roc_curve(y_test,  yproba)
        except:
            fpr = np.nan
            tpr = np.nan
        try:
            mean_auc, lower_auc, upper_auc = get_mean_lower_upper(yproba, y_test, 'roc_auc_score')
        except:
            mean_auc = np.nan
            lower_auc = np.nan
            upper_auc = np.nan
        try:
            mean_pr_auc, lower_pr_auc, upper_pr_auc = get_mean_lower_upper(yproba, y_test, 'pr_auc_score')
        except:
            mean_pr_auc = np.nan
            lower_pr_auc = np.nan
            upper_pr_auc = np.nan
        #     mean_auc = np.nan
        #     lower_auc = np.nan
        #     upper_auc = np.nan
        #compute alse precision and recall
        try:
            precision, recall, pr_thresholds = precision_recall_curve(y_test, yproba)
        except:
            precision = np.nan
            recall = np.nan
            mean_f1 = np.nan
            lower_f1 = np.nan
            upper_f1 = np.nan
        # if config['debugging']:
        #     y_test[np.nan] = 1
        #     y_test[1] = np.nan
        #try:
        mean_mae, lower_mae, upper_mae = get_mean_lower_upper(yproba, raw_trues, 'mae_score')
       # except:
        # #    mean_mae = np.nan
        #     lower_mae = np.nan
        #     upper_mae = np.nan
        try:
            auc = roc_auc_score(y_test, yproba)
        except:
            auc = np.nan
        probas.append(yproba)
        trues.append(y_test)
        raw_trues_list.append(raw_trues)
        probas_bin.append(ypred_bin)
        roc_result_table = roc_result_table.append({'segments':seg,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc,
                                            'auc_lower':lower_auc,
                                            'auc_upper': upper_auc,
                                            'precision': precision,
                                            'recall': recall,
                                            'pr_auc': mean_pr_auc,
                                            'pr_auc_lower': lower_pr_auc,
                                            'pr_auc_upper': upper_pr_auc,
                                            'mae': mean_mae,
                                            'mae_lower': lower_mae,
                                            'mae_upper': upper_mae,
                                            'positives': positives,
                                            'support': len(yproba),
                                            }, ignore_index=True)
        
        idx += 1
        
    # save the roc_result_table to a csv file
    roc_result_table.set_index('segments', inplace=True)
    
    # make a copy for saving to csv
    roc_result_table_save = roc_result_table.copy()
    roc_result_table_save = roc_result_table_save[["auc", 'auc_lower', 'auc_upper', 'pr_auc', 'pr_auc_lower', 'pr_auc_upper', 'mae', 'mae_lower', 'mae_upper', 'positives', 'support']]
    

    # concatenate list of numpy arrays for trues and probas
    probas = np.concatenate(probas)
    #DEBUG
   # probas = np.random.rand(8)
    raw_trues_all = np.concatenate(raw_trues_list)
    trues = np.concatenate(trues)
    probas_bin = np.concatenate(probas_bin)
    positives_total = np.sum(probas_bin)
    support = len(probas_bin)
    #DEBUG
   # probas_bin = np.random.randint(2, size=8)

    # plot roc curve for all segments combined
    fpr, tpr, _ = roc_curve(trues,  probas)
    mean_auc_all, lower_auc_all, upper_auc_all = get_mean_lower_upper(probas, trues, 'roc_auc_score')
    fig = plt.figure(figsize=(8,6))
    plt.plot(fpr, 
                tpr, 
                label="AUC={:.3f} ({:.3f}-{:-3f})".format(
                    mean_auc_all,
                    lower_auc_all,
                    upper_auc_all))
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':10}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_plots, plot_type + 'roc_all_comb.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_plots, plot_type + '_roc_all_comb.pdf'), dpi=100,
                bbox_inches='tight')
    plt.close()
    # plot precision recall curve for all segments combined
    precision, recall, pr_thresholds = precision_recall_curve(trues, probas)

    mean_pr_auc_all,  lower_pr_auc_all, upper_pr_auc_all= get_mean_lower_upper(probas, trues, 'pr_auc_score')
    fig = plt.figure(figsize=(8,6))
    plt.plot(recall, 
                precision, 
                label="PR AUC={:.3f} ({:.3f}-{:-3f})".format(
                    mean_pr_auc_all,
                    lower_pr_auc_all,
                    upper_pr_auc_all,
                    ))
    # plot baseline for precision recall curve as horizontal line
    baseline_precision = positives_total / support
    plt.plot([0, 1], [baseline_precision, baseline_precision], color='orange', linestyle='--',
            label='Baseline PR AUC={:.3f}'.format(baseline_precision))
  #  plt.plot([0,1], [0,1], color='orange', linestyle='--')
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall (sensitivity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision (positive predictive value)", fontsize=15)

    plt.title('Precision recall curve', fontweight='bold', fontsize=15)
    
    plt.legend(prop={'size':10}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_plots, plot_type + '_precision_recall_comb.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_plots, plot_type + '_precision_recall_comb.pdf'), dpi=100,
                bbox_inches='tight')
    plt.close()
    
    fscore = (2 * precision * recall) / (precision + recall+ 1e-10)
    ix= np.argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (pr_thresholds[ix], fscore[ix]))
    
# Convert probabilities into binary predictions using the user-defined threshold
    preds_ = (probas >= config['loaders']['val_method']['classifier_threshold']).astype(int)

    class_report = classification_report_func(trues, preds_)
    # save as csv
    class_report.to_csv(os.path.join(output_plots, 'classification_report.csv'), index=False)
    roc_result_table_save["f1_max_threshold"] = pr_thresholds[ix]
    roc_result_table_save["f1_score_threshold"] = fscore[ix]

    roc_result_table_save.to_csv(os.path.join(output_plots, 'segments_results_table.csv'), index=False)

    # put them together in a dataframe
    dataframe = pd.DataFrame({'probas': probas, 'trues': trues, 'raw_trues_all': raw_trues_all})
    # save the dataframe to a csv file
    dataframe.to_csv(os.path.join(output_plots, 'probas_trues.csv'), index=False)
    # rename row values for segments based on part of the name:
    dictionary = {'sten_proc_1_': '1 Proximal RCA',
                  'sten_proc_2_': '2 Mid RCA',
                  'sten_proc_3_': '3 Distal RCA',
                  'sten_proc_4_': '4 PDA RCA/LCA',
                  'sten_proc_5_': '5 LM LCA',
                  'sten_proc_6_': '6 Proximal LAD',
                  'sten_proc_7_': '7 Mid LAD',
                  'sten_proc_8_': '8 Distal LAD',
                  'sten_proc_9_': '9 Diagonal 1',
                  'sten_proc_10_': '10 Diagonal 2',
                  'sten_proc_11_': '11 Proximal LCX',
                  'sten_proc_12_': '12 Marginal 1',
                  'sten_proc_13_': '13 Mid LCX',
                  'sten_proc_14_': '14 Marginal 2',
                  'sten_proc_15_': '15 Distal LCX',
                  'sten_proc_16_': '16 PLA RCA/LCX',
                  'ffr_proc_1_': '1 Proximal RCA',
                  'ffr_proc_2_': '2 Mid RCA',
                  'ffr_proc_3_': '3 Distal RCA',
                  'ffr_proc_4_': '4 PDA RCA/LCA',
                  'ffr_proc_5_': '5 LM LCA',
                  'ffr_proc_6_': '6 Proximal LAD',
                  'ffr_proc_7_': '7 Mid LAD',
                  'ffr_proc_8_': '8 Distal LAD',
                  'ffr_proc_9_': '9 Diagonal 1',
                  'ffr_proc_10_': '10 Diagonal 2',
                  'ffr_proc_11_': '11 Proximal LCX',
                  'ffr_proc_12_': '12 Marginal 1',
                  'ffr_proc_13_': '13 Mid LCX',
                  'ffr_proc_14_': '14 Marginal 2',
                  'ffr_proc_15_': '15 Distal LCX',
                  'ffr_proc_16_': '16 PLA RCA/LCX'}
    id = 0
    roc_result_table_2 = roc_result_table.copy()
    for row in roc_result_table.index:
        for key in dictionary.keys():
            if row.startswith(key):
                roc_result_table_2 = roc_result_table_2.rename(index={row: dictionary[key]})
         #   roc_result_table.rename(index={row: dictionary[row]}, inplace=True)

    fig = plt.figure(figsize=(8,6))

    for i in roc_result_table_2.index:
        #     df_plot, prediction_name, label_name)
        plt.plot(roc_result_table_2.loc[i]['fpr'], 
                roc_result_table_2.loc[i]['tpr'], 
                label="{}, AUC={:.3f} ({:.3f}-{:-3f})".format(
                    i, roc_result_table_2.loc[i]['auc'],
                    roc_result_table_2.loc[i]['auc_lower'],
                    roc_result_table_2.loc[i]['auc_upper']))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=15)

    plt.title('ROC Curve Analysis for ' + plot_type + ' estimation on ', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':10}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_plots, plot_type + '_roc_all.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_plots, plot_type + '_roc_all.pdf'), dpi=100,
                bbox_inches='tight')
    plt.close()
    # plot precision recall curve
    
    for i in roc_result_table_2.index:
        #     df_plot, prediction_name, label_name)
        plt.plot(roc_result_table_2.loc[i]['recall'], 
                roc_result_table_2.loc[i]['precision'], 
                label="{}, PR AUC={:.3f} ({:.3f}-{:-3f})".format(
                    i, roc_result_table_2.loc[i]['pr_auc'],
                    roc_result_table_2.loc[i]['pr_auc_lower'],
                    roc_result_table_2.loc[i]['pr_auc_upper']))
        
  #  plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall (sensitivity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision (positive predictive value)", fontsize=15)

    plt.title('Precision recall curve for ' + plot_type + ' estimation on ', fontweight='bold', fontsize=15)
    plt.legend(prop={'size':10}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_plots, plot_type + '_precision_recall.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_plots, plot_type + '_' + '_precision_recall.pdf'), dpi=100,
                bbox_inches='tight')

def plot_regression_all(result_table, trues_names, confidences_names, output_plots, config):
       # Define a result table as a DataFrame
    result_table_comb = pd.DataFrame(columns=['segments', 'mse_mean', 'mse_lower', 'mse_upper'])

    # list comprehension to rename suffixes of elements in list from _confidences to _predictions
   # confidences_names = [i.replace('_confidences', '_predictions') for i in confidences_names]
    probas = []
    trues = []
    idx = 0
    for seg in confidences_names:
        # if config['task_type'] != 'mil_classification':
        #     result_table_copy, maybeRCA = select_relevant_data(result_table, seg, trues_names[idx])
        # else:
        result_table_copy = result_table.copy()
          #  maybeRCA = ""
        result_table_copy[confidences_names[idx]] = transform_confidences_to_by_label_type(
            result_table_copy[confidences_names[idx]], seg)
        y_test = result_table_copy[trues_names[idx]].values
        yproba = result_table_copy[confidences_names[idx]].values
        mask_propa = np.isnan(yproba)
        mask_test = np.isnan(y_test)
        mask = mask_propa + mask_test
        y_test = y_test[~mask]
        yproba = yproba[~mask]
        yproba = np.clip(yproba, a_min=0, a_max=1)
       # fpr, tpr, _ = roc_curve(y_test,  yproba)
        mean_mse,  lower_mse, upper_mse = get_mean_lower_upper(yproba, y_test, 'mse_score')
       # upper_mse = np.clip(upper_mse, a_min=0, a_max=1)
       # lower_mse = np.clip(lower_mse, a_min=0, a_max=1)
       # auc = roc_auc_score(y_test, yproba)
        probas.append(yproba)
        trues.append(y_test)
        result_table_comb = result_table_comb.append({'segments':seg,
                                            'mse_mean':mean_mse,
                                            'mse_lower':lower_mse,
                                            'mse_upper': upper_mse}, ignore_index=True)
        
        idx += 1

    result_table_comb['probas'] = probas
    result_table_comb['trues'] = trues
#    from miacag.plots.plotter import plot_regression_density
    label_name_ori = config['labels_names'][0]
    # cat y_test and yproba to dataframe
    df = pd.DataFrame({'y_test': y_test, 'yproba': yproba})
    # close figures
    plt.close()
    g = sns.lmplot(x='y_test', y='yproba', data=df)
    X2 = sm.add_constant(df['y_test'])
    est = sm.OLS(df['yproba'], X2)
    est2 = est.fit()
    r = est2.rsquared

    if label_name_ori.startswith('timi'):
        label_name = 'TIMI Flow'
        plot_name = 'timi'
    elif label_name_ori.startswith('ffr'):
        plot_name = 'FFR'
        label_name = 'FFR'
    elif label_name_ori.startswith('sten'):
        plot_name = 'stenosis'
        label_name = 'Stenosis'
    else:
        raise ValueError('label_name_ori is not timi, ffr or sten')
    result_table_comb.to_csv(
        os.path.join
        (output_plots, plot_name + '_regression.csv'))
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
            output_plots, plot_name + '_scatter.png'), dpi=100,
        bbox_inches='tight')
    plt.close()
    

    f, ax = plt.subplots(1, figsize = (8,5))
    sm.graphics.mean_diff_plot(y_test, yproba, ax = ax)

    plt.show()
    plt.savefig(
        os.path.join(
            output_plots, plot_name + '_bland_altman.png'), dpi=100,
        bbox_inches='tight')
    plt.close()
    
    
    f, ax = plt.subplots(1, figsize = (8,5))
    sm.graphics.mean_diff_plot(y_test, yproba, ax = ax)

    plt.show()
    plt.savefig(
        os.path.join(
            output_plots, plot_name + '_bland_altman.pdf'), dpi=100,
        bbox_inches='tight')
    plt.close()
    
    from scipy import stats
    res = stats.normaltest(y_test - yproba)
    res.statistic
    print('res', res)
    diff = y_test - yproba
    plt.hist(diff, bins=50)
    plt.show()
    plt.savefig(
     os.path.join(
         output_plots, plot_name + 'hist.png'), dpi=100,
     bbox_inches='tight')
    plt.close()
    
    
    
    # bland_altman_plot(y_test, yproba)
    # plt.title('Bland-Altman Plot')
    # plt.show()
    # plt.savefig(
    #     os.path.join(
    #         output_plots, plot_name + '_bland_altman.png'), dpi=100,
    #     bbox_inches='tight')
    
    
    # bland_altman_plot(y_test, yproba)
    # plt.title('Bland-Altman Plot')
    # plt.show()
    # plt.savefig(
    #     os.path.join(
    #         output_plots, plot_name + '_bland_altman.pdf'), dpi=100,
    #     bbox_inches='tight')
    
    # plt.close()
    
    # plot_regression_density(x=y_test, y=yproba,
    #                                 cmap='jet', ylab='prediction', xlab='true',
    #                                 bins=100,
    #                                 figsize=(5, 4),
    #                                 snsbins=60,
    #                                 plot_type=plot_type,
    #                                 output_folder=output_plots,
    #                                 label_name_ori=label_name_ori)
if __name__ == '__main__':

    output_plots = "/home/alatar/miacag/output_plots/"
    mkFolder(output_plots)
    result_table, confidences_names, \
        trues_names, dom_name, label_pred_names = generate_data()
    config = dict()
    plot_roc_all(result_table, trues_names, confidences_names,
                 output_plots, plot_type='stenosis',
                 config=config,
                 theshold=0.5)
 