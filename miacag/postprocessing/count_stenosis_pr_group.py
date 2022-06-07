import pandas as pd
from dataclasses import dataclass
from miacag.postprocessing.aggregate_pr_group import Aggregator
from miacag.plots.plotter import convertConfFloats
from miacag.utils.sql_utils import getDataFromDatabase, \
    update_cols, add_columns
import psycopg2
import numpy
from psycopg2.extensions import register_adapter, AsIs
from miacag.plots.plotter import plotStenoserTrueVsPred
from miacag.utils.script_utils import mkFolder


def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


register_adapter(numpy.float64, addapt_numpy_float64)
register_adapter(numpy.int64, addapt_numpy_int64)


def within_threshold(x, thresh):
    key = pd.to_numeric(x[thresh.keys()]).idxmax(axis=0)
    return x[key] > thresh[key]


@dataclass
class CountSignificantStenoses(Aggregator):
    sql_config: dict
    fields_to_aggregate: list
    output_dir: str

    # def __post_init__(self):
    #     self.thresholded_cols_list = [
    #         i[:i.find('_confidences')] + "_thres"
    #         for i in self.fields_to_aggregate]

    def threshold_segments(self):
        for c, agg_col in enumerate(self.fields_to_aggregate):
           # tresh_col = self.thresholded_cols_list[c]
            thresholds = dict(zip([agg_col], [0.5]))
            self.df[agg_col] = self.df.apply(
                lambda x: within_threshold(x, thresholds), axis=1)
            self.df[agg_col] = self.df[agg_col].astype(int)
        df = self.df[self.fields_to_aggregate + ['rowid']]
        records = df.to_records()
        return records

    def compute_nr_stenosis(self, sum_col):
        # return records from dataframe to update
        sum_sten = self.df.groupby(
                ['PatientID', 'StudyInstanceUID'])[self.fields_to_aggregate].sum()
        self.df[sum_col] = self.df.drop_duplicates(
            ['PatientID',
             'StudyInstanceUID'])[self.fields_to_aggregate].sum(axis=1)
        sum_sten = self.df.drop_duplicates(['PatientID', 'StudyInstanceUID'])
        sum_sten = sum_sten[["PatientID", "StudyInstanceUID", sum_col]]
        self.df = self.df.drop(sum_col, axis=1)
        df2 = self.df.merge(
            sum_sten, left_on=['PatientID', 'StudyInstanceUID'],
            right_on=['PatientID', 'StudyInstanceUID'],
            how='right')[['rowid'] + [sum_col]]
        records = df2.to_records()
        return records

    def __call__(self):
        self.get_df_from_query()
       # data_types = ["int8"] * len(self.self.fields_to_aggregate)
        # add_columns(self.sql_config, self.self.fields_to_aggregate, data_types)
        records = self.threshold_segments()
        self.update_colum_wrap(records, self.fields_to_aggregate)
        # add_columns(self.sql_config, ["antalsignifikantestenoser_pred"],
        #            ["int8"])
        records = self.compute_nr_stenosis(
                                           "antalsignifikantestenoser_pred")
        self.update_colum_wrap(records, ["antalsignifikantestenoser_pred"])
        plotStenoserTrueVsPred(self.sql_config, ["antalsignifikantestenoser"],
                     ["antalsignifikantestenoser_pred"], self.output_dir)


if __name__ == '__main__':
    sql_config = {'database':
                  'mydb',
                  'username':
                  "sauroman",
                  'password':
                  "123qweasd",
                  'host':
                  "localhost",
                  'port':
                  "5432",
                  'table_name':
                  "classification_config_angio_SEP_May20_18-20-38_sauromanPC_dicom",
                  'query':
                  "SELECT * FROM ?table_name WHERE (\"sten_procent_2_midt_rca_transformed_confidences\" IS NOT NULL) AND (\"sten_procent_3_prox_rca_transformed_confidences\" IS NOT NULL);"
#                   """select distinct on (\"PatientID\", \"StudyInstanceUID\") \"classification_config_angio_SEP_May20_18-20-38_sauromanPC_dicom\".*
# from \"classification_config_angio_SEP_May20_18-20-38_sauromanPC_dicom\"
# order by "PatientID\", \"StudyInstanceUID\";"""
                  }
    fields_to_count = [
        "sten_procent_3_prox_rca_transformed_confidences_aggregated",
        "sten_procent_2_midt_rca_transformed_confidences_aggregated"]
    output_dir = "plot_scatter_test"
    mkFolder(output_dir)
    count = CountSignificantStenoses(sql_config, fields_to_count, output_dir)
    count()
