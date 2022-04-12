from mia.preprocessing.utils.sql_utils import getDataFromDatabase
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd


def map_labels_to_OneTwoThree():
    labels_dict = {0: 0,
                   1: 1,
                   2: 2,
                   3: 2,
                   4: 2,
                   6: 2,
                   7: 2,
                   8: 2,
                   9: 0,
                   10: 2,
                   11: 2,
                   12: 2,
                   13: 1,
                   14: 2,
                   15: 2,
                   16: 2,
                   17: 2,
                   18: 2,
                   19: 2,
                   20: 2}
    return labels_dict


def appendDataFrame(sql_config, df_results, experiment_name):
    df, _ = getDataFromDatabase(sql_config)
    f1_test_transformed = f1_score(
      df[sql_config['labels_names']], df['predictions'], average='macro')
    
    df = df.replace(
      {'predictions': map_labels_to_OneTwoThree()})

    for label_name in sql_config["labels_names"]:
        if len(sql_config["labels_names"]) > 1:
            raise ValueError('not implemtend')
        df = df.replace({label_name: map_labels_to_OneTwoThree()})

        f1_test = f1_score(df[sql_config["labels_names"]],
                           df['predictions'], average='macro')

        acc_test = accuracy_score(
          df[label_name],
          df['predictions'])
    # df = pd.DataFrame({'Experiment name': [experiment_name],
    #                    'Test F1 score on data['labels'] transformed': [f1_test]})
    df_results['Experiment name'].append(experiment_name)
    df_results['Test F1 score on data labels transformed'].append(
      f1_test_transformed)
    df_results['Test F1 score on three class labels'].append(
      f1_test)
    df_results['Test acc on three class labels'].append(
      acc_test)
    return df_results
  #  df.to_csv(df_results, mode='a', index=True, header=False)
# 'Experiment name': [],
#                        'Test F1 score':
