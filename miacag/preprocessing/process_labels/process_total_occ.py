import dataclasses
import numpy as np
import pandas as pd

# process total occlusions
# this processing module has problems since it ignores collaterals
class ProcessLabelsOCC():
    def __init__(self, df, labels_names, config):
        self.df = df
        self.config = config
        self.labels_names = labels_names

    def processor(self, df, field_to_change, top_field):
        if field_to_change in df:
            if top_field in df:
                df[field_to_change] = \
                    np.where((
                            df[top_field] == 1),
                            1, df[field_to_change])
        return df


    def get_subset_of_fields(self):
        self.df_co_dom = self.df[
            self.df['dominans'] == "Balanceret (PDA fra RCA/PLA fra LCX)"]
        self.df_r_dom = self.df[
            self.df['dominans'] == "Højre dominans (PDA+PLA fra RCA)"]
        # if not listet. assume it is r-dominant
        self.df_r_dom_nan = self.df[self.df['dominans'].isna()]
        self.df_r_dom = pd.concat((self.df_r_dom, self.df_r_dom_nan))
        self.df_l_dom = self.df[
            self.df['dominans'] == "Venstre dominans (PDA+PLA fra LCX)"]

    def process_fields(self):
        self.get_subset_of_fields()
        self.df_r_dom = self.process_r_dom(self.df_r_dom)
        self.df_l_dom = self.process_l_dom(self.df_l_dom)
        self.df_co_dom = self.process_co_dom(self.df_co_dom)
        self.df = pd.concat(
            (self.df_co_dom, self.df_r_dom, self.df_l_dom)
            )

    def __call__(self):
        self.process_fields()
        return self.df

    def process_r_dom(self, df):
        # rca, prox-> 1.0
        df = self.processor(
            df,
            'sten_proc_2_midt_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_1_prox_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        # rca, midt-> 1.0
        df = self.processor(
            df,
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_2_midt_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_2_midt_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_2_midt_rca_transformed')

        # rca, dist-> 1.0
        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_3_dist_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_3_dist_rca_transformed')

        # lca
        # lca lm->1.0
        df = self.processor(
            df,
            'sten_proc_6_prox_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_9_d1_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_11_prox_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_12_om1_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_5_lm_transformed')
        # lca lad prox->1.0
        df = self.processor(
            df,
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_9_d1_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_6_prox_lad_transformed')

        # lca lad midt->1.0
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_7_midt_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_7_midt_lad_transformed')

        # lca lcx prox->1.0
        df = self.processor(
            df,
            'sten_proc_12_om1_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')

        # lca lcx midt->1.0
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_13_midt_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_13_midt_lcx_transformed')
        return df

    def process_l_dom(self, df):
       # rca, prox-> 1.0
        df = self.processor(
            df,
            'sten_proc_2_midt_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_1_prox_rca_transformed')


        # rca, midt-> 1.0
        df = self.processor(
            df,
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_2_midt_rca_transformed')


        # lca
        # lca lm->1.0
        df = self.processor(
            df,
            'sten_proc_6_prox_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_9_d1_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_11_prox_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_12_om1_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_5_lm_transformed')

        # lca lad prox->1.0
        df = self.processor(
            df,
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_9_d1_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_6_prox_lad_transformed')

        # lca lad midt->1.0
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_7_midt_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_7_midt_lad_transformed')

        # lca lcx prox->1.0
        df = self.processor(
            df,
            'sten_proc_12_om1_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_11_prox_lcx_transformed')
        # lca lcx midt->1.0
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_13_midt_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_13_midt_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_13_midt_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_13_midt_lcx_transformed')
        
        # lca lcx dist->1.0
        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_15_dist_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_15_dist_lcx_transformed')
        return df

    def process_co_dom(self, df):
        # rca
        df = self.processor(
            df,
            'sten_proc_2_midt_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_1_prox_rca_transformed')


        # rca, midt-> 1.0
        df = self.processor(
            df,
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_2_midt_rca_transformed')

        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_2_midt_rca_transformed')


        # rca, dist-> 1.0
        df = self.processor(
            df,
            'sten_proc_4_pda_transformed',
            'sten_proc_3_dist_rca_transformed')


        # lca
                # lca
        # lca lm->1.0
        df = self.processor(
            df,
            'sten_proc_6_prox_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_9_d1_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_11_prox_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_12_om1_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_5_lm_transformed')
        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_5_lm_transformed')

        # lca lad prox->1.0
        df = self.processor(
            df,
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_9_d1_transformed',
            'sten_proc_6_prox_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_6_prox_lad_transformed')

        # lca lad midt->1.0
        df = self.processor(
            df,
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_7_midt_lad_transformed')
        df = self.processor(
            df,
            'sten_proc_10_d2_transformed',
            'sten_proc_7_midt_lad_transformed')

        # lca lcx prox->1.0
        df = self.processor(
            df,
            'sten_proc_12_om1_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_11_prox_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')

        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_11_prox_lcx_transformed')
        # lca lcx midt->1.0
        df = self.processor(
            df,
            'sten_proc_14_om2_transformed',
            'sten_proc_13_midt_lcx_transformed')
        df = self.processor(
            df,
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_13_midt_lcx_transformed')

        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_13_midt_lcx_transformed')

        # lca lcx dist->1.0

        df = self.processor(
            df,
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_15_dist_lcx_transformed')
        return df

