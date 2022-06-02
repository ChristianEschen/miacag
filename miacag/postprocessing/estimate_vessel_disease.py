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

# DEPENDENCY: I need a model that can detect if the system is right/left / co-dominant
@dataclass
class EstimateVesselDisease(Aggregator):
    sql_config: dict
    fields_to_aggregate: list
    output_dir: str

    def __post_init__(self):
        self.thresholded_cols_list = [
            i[:i.find('_confidences')] + "_thres"
            for i in self.fields_to_aggregate]

    #1 gebet: LAD (incl. sidegrene); Ikke dominant CX (incl. sidegrene); Dominant CX, kun distalt for midt; Dominant CX – kun OM; eller Dominant RCA (incl. sidegrene)
    def lad_plus_branches(self, threshold_segments):
        vessel_list = ['sten_proc_6_prox_lad_transformed_thres',
                       'sten_proc_7_midt_lad_transformed_thres',
                       'sten_proc_8_dist_lad_transformed_thres',
                       'sten_9_d1_transformed_thres',
                       'sten_10_d2_transformed_thres']
        canlist(set.intersection(set(list1), set(list2)))

        if threshold_segments in vessel_list:
            return 1
        else:    # DEPENDENCY: I need a model that can detect if the system is right/left / co-dominant



    def compute_nr_vessel_disease(self):
        nr_vessels = self.compute_RCA_disease() + \
                     self.compute_LCX_disease() + \
                     self.compute_LAD_disease()

        dict{0: "Atheroma"}

    def __call__(self):
        self.get_df_from_query()
        data_types = ["int8"] * len(self.thresholded_cols_list)
        add_columns(self.sql_config, self.thresholded_cols_list, data_types)
        # TODO: calculate all possible configurations:
        # TODO: determine if 1 vessel
        # TODO: determine if 2 vessel
        # TODO: determine if 3 vessel
        # TODO: update database
        # TODO: plot confusion_matrix and maybe cohens kappa

if __name__ == '__main__':
    # DEPENDENCY: I need a model that can detect if the system is right/left / co-dominant
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
               #   "SELECT * FROM ?table_name WHERE (\"sten_procent_2_midt_rca_transformed_confidences\" IS NOT NULL) AND (\"sten_procent_3_prox_rca_transformed_confidences\" IS NOT NULL);"
                  """select distinct on (\"PatientID\", \"StudyInstanceUID\") \"classification_config_angio_SEP_May20_18-20-38_sauromanPC_dicom\".*
from \"classification_config_angio_SEP_May20_18-20-38_sauromanPC_dicom\"
order by "PatientID\", \"StudyInstanceUID\";"""
                  }
    fields_to_count = [
        "sten_procent_3_prox_rca_transformed_confidences_aggregated",
        "sten_procent_2_midt_rca_transformed_confidences_aggregated"]
    output_dir = "plot_scatter_test"
    mkFolder(output_dir)

    d = {
        'labels': [1, 0, 1, 1, 0, 0, 1, 0],
        'confidences': [
            "{0:0;1:0.2}",
            "{0:0;1:0.3}",
            "{0:0;1:0.8}",
            "{0:0;1:0.2}",
            "{0:0;1:0.1}",
            "{0:0;1:0.1}",
            "{0:0;1:0.9}",
            "{0:0;1:0.10}"]
        }
    d = {
        'sten_proc_1_prox_rca_transformed_thres': [0, 0, 1, 0],
        'sten_proc_2_midt_rca_transformed_thres': [0, 1, 0, 0],
        'sten_proc_3_dist_rca_transformed_thres': [0, 1, 0, 0],
        'sten_proc_4_pda_transformed_thres': [0, 0, 1, 0],
        'sten_proc_5_lm_transformed_thres': [0, 0, 1, 0],
        'sten_proc_6_prox_lad_transformed_thres': [0, 0, 0, 0],
        'sten_proc_7_midt_lad_transformed_thres': [0, 1, 0, 0],
        'sten_proc_8_dist_lad_transformed_thres': [0, 0, 0, 0],
        'sten_proc_9_d1_transformed_thres': [0, 0, 0, 0],
        'sten_proc_10_d2_transformed_thres': [0, 0, 0, 0],
        'sten_proc_11_prox_lcx_transformed_thres': [0, 0, 0, 0],
        'sten_proc_12_om1_transformed_thres': [0, 0, 0, 0],
        'sten_proc_13_midt_lcx_transformed_thres': [0, 0, 0, 0],
        'sten_proc_14_om2_transformed_thres': [0, 0, 0, 0],
        'sten_proc_15_dist_lcx_transformed_thres': [0, 0, 0, 0],
        'sten_proc_16_pla_rca_transformed_thres': [0, 0, 0, 0]
        'dominance'
          }
    df = pd.DataFrame(data=d)
    count = CountSignificantStenoses(sql_config, fields_to_count, output_dir)
    count()
