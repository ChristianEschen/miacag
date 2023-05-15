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
from miacag.plots.plotter import rename_columns

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


def select_relevant_data(result_table, seg, y_true_name):
    result_table_cop = result_table.copy()
    maybeRCA = None
    if result_table_cop['labels_predictions'].iloc[0] == 1:  # right
        maybeRCA = True
        if 'pla' in seg:
            not_in_list_mask = result_table_cop['dominans'].isin(["Højre dominans (PDA+PLA fra RCA)", None])
            result_table_cop.loc[~not_in_list_mask, y_true_name] = np.nan
        elif 'pda' in seg:
            not_in_list_mask = result_table_cop['dominans'].isin(["Højre dominans (PDA+PLA fra RCA)", "Balanceret (PDA fra RCA/PLA fra LCX)", None])
            result_table_cop.loc[~not_in_list_mask, y_true_name] = np.nan
        else:
            return result_table_cop, maybeRCA
    elif result_table_cop['labels_predictions'].iloc[0] == 0:  # left
        maybeRCA = False
        if 'pla' in seg:
            not_in_list_mask = result_table_cop['dominans'].isin(["Venstre dominans (PDA+PLA fra LCX)", "Balanceret (PDA fra RCA/PLA fra LCX)"])
            result_table_cop.loc[~not_in_list_mask, y_true_name] = np.nan
        elif 'pda' in seg:
            not_in_list_mask = result_table_cop['dominans'].isin(["Venstre dominans (PDA+PLA fra LCX)"])
            result_table_cop.loc[~not_in_list_mask, y_true_name] = np.nan
        else:
            result_table_cop, maybeRCA
    else:
        raise ValueError('labels_predictions is not 0 or 1')
    return result_table_cop, maybeRCA



def threshold_continues(continuos_inc, threshold, name):
    continuos = continuos_inc.to_numpy()
    if name.startswith('ffr'):
        continuos[continuos >= threshold] = 1
        continuos[continuos < threshold] = 0
        continuos = np.logical_not(continuos).astype(int)
    elif name.startswith('sten'):
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
    else:
        raise ValueError('name is not ffr or sten')
    return confidences


def plot_roc_all(result_table, trues_names, confidences_names, output_plots, plot_type, config, theshold=0.5):
       # Define a result table as a DataFrame
    roc_result_table = pd.DataFrame(columns=['segments', 'fpr','tpr','auc'])

    probas = []
    trues = []
    idx = 0
    for seg in confidences_names:
        if config['task_type'] != 'mil_classification':
            result_table_copy, maybeRCA = select_relevant_data(result_table, seg, trues_names[idx])
        else:
            result_table_copy = result_table.copy()
            maybeRCA = ""
        result_table_copy[confidences_names[idx]] = transform_confidences_to_by_label_type(
            result_table_copy[confidences_names[idx]], seg)
   #    if config['loss']['name'][0] in ['MSE', '_L1', 'L1smooth']:
        result_table_copy[trues_names[idx]] = threshold_continues(
            result_table_copy[trues_names[idx]], threshold=theshold, name=seg)
        y_test = result_table_copy[trues_names[idx]].values
        yproba = result_table_copy[confidences_names[idx]].values
        mask_propa = np.isnan(yproba)
        mask_test = np.isnan(y_test)
        mask = mask_propa + mask_test
        y_test = y_test[~mask]
        yproba = yproba[~mask]
        yproba = np.clip(yproba, a_min=0, a_max=1)
        fpr, tpr, _ = roc_curve(y_test,  yproba)
        auc = roc_auc_score(y_test, yproba)
        probas.append(yproba)
        trues.append(y_test)
        roc_result_table = roc_result_table.append({'segments':seg,
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc}, ignore_index=True)
        
        idx += 1
    roc_result_table.set_index('segments', inplace=True)
    # concatenate list of numpy arrays for trues and probas
    probas = np.concatenate(probas)
    trues = np.concatenate(trues)
    # put them together in a dataframe
    dataframe = pd.DataFrame({'probas': probas, 'trues': trues})
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
            
    if maybeRCA:
        location = 'RCA'
    elif maybeRCA == "":
        location = ""
    else:
        location = 'LCA'
    fig = plt.figure(figsize=(8,6))

    for i in roc_result_table_2.index:
        #     df_plot, prediction_name, label_name)
        plt.plot(roc_result_table_2.loc[i]['fpr'], 
                roc_result_table_2.loc[i]['tpr'], 
                label="{}, AUC={:.3f}".format(i, roc_result_table_2.loc[i]['auc']))
        
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=15)

    plt.title('ROC Curve Analysis for ' + plot_type + ' estimation on ' + location, fontweight='bold', fontsize=15)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()
    plt.savefig(os.path.join(
        output_plots, plot_type + '_' + location + '_roc_all.png'), dpi=100,
                bbox_inches='tight')
    plt.savefig(os.path.join(
        output_plots, plot_type + '_' + location + '_roc_all.pdf'), dpi=100,
                bbox_inches='tight')

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
 