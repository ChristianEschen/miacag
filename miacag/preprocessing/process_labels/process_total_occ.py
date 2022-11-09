import dataclasses
import numpy as np

# process total occlusions
# this processing module has problems since it ignores collaterals
class ProcessLabelsOCC():
    def __init__(self, df, labels_names, config):
        self.df = df
        self.config = config
        self.labels_names = labels_names

    def processor(self, field_to_change, top_field):
        if field_to_change in self.df:
            if top_field in self.df:
                self.df[field_to_change] = \
                    np.where((
                            self.df[top_field] == 1),
                            1, self.df[field_to_change])


    def process_fields(self):
        if self.config['dominans'] == 'r_dom':
            self.process_r_dom()
        elif self.config['dominans'] == 'l_dom':
            self.process_l_dom()
        elif self.config['dominans'] == 'co_dom':
            self.process_co_dom()
        else:
            raise ValueError(
                'dominanse type is not implemented' +
                self.config['dominans'])

    def __call__(self):
        self.process_fields()
        return self.df

    def process_r_dom(self):
        # rca, prox-> 1.0
        self.processor(
            'sten_proc_2_midt_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        self.processor(
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_1_prox_rca_transformed')

        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        # rca, midt-> 1.0
        self.processor(
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_2_midt_rca_transformed')

        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_2_midt_rca_transformed')

        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_2_midt_rca_transformed')

        # rca, dist-> 1.0
        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_3_dist_rca_transformed')

        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_3_dist_rca_transformed')

        # lca
        # lca lm->1.0
        self.processor(
            'sten_proc_6_prox_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_9_d1_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_11_prox_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_12_om1_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_5_lm_transformed')
        # lca lad prox->1.0
        self.processor(
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_9_d1_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_6_prox_lad_transformed')

        # lca lad midt->1.0
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_7_midt_lad_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_7_midt_lad_transformed')

        # lca lcx prox->1.0
        self.processor(
            'sten_proc_12_om1_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')

        # lca lcx midt->1.0
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_13_midt_lcx_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_13_midt_lcx_transformed')
        return

    def process_l_dom(self):
       # rca, prox-> 1.0
        self.processor(
            'sten_proc_2_midt_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        self.processor(
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_1_prox_rca_transformed')


        # rca, midt-> 1.0
        self.processor(
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_2_midt_rca_transformed')


        # lca
        # lca lm->1.0
        self.processor(
            'sten_proc_6_prox_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_9_d1_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_11_prox_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_12_om1_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_5_lm_transformed')

        # lca lad prox->1.0
        self.processor(
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_9_d1_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_6_prox_lad_transformed')

        # lca lad midt->1.0
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_7_midt_lad_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_7_midt_lad_transformed')

        # lca lcx prox->1.0
        self.processor(
            'sten_proc_12_om1_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_11_prox_lcx_transformed')
        # lca lcx midt->1.0
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_13_midt_lcx_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_13_midt_lcx_transformed')
        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_13_midt_lcx_transformed')
        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_13_midt_lcx_transformed')
        
        # lca lcx dist->1.0
        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_15_dist_lcx_transformed')
        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_15_dist_lcx_transformed')
        return

    def process_co_dom(self):
        # rca
        self.processor(
            'sten_proc_2_midt_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        self.processor(
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_1_prox_rca_transformed')

        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_1_prox_rca_transformed')


        # rca, midt-> 1.0
        self.processor(
            'sten_proc_3_dist_rca_transformed',
            'sten_proc_2_midt_rca_transformed')

        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_2_midt_rca_transformed')


        # rca, dist-> 1.0
        self.processor(
            'sten_proc_4_pda_transformed',
            'sten_proc_3_dist_rca_transformed')


        # lca
                # lca
        # lca lm->1.0
        self.processor(
            'sten_proc_6_prox_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_9_d1_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_11_prox_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_12_om1_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_14_om2_transformed'
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_5_lm_transformed')
        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_5_lm_transformed')

        # lca lad prox->1.0
        self.processor(
            'sten_proc_7_midt_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_9_d1_transformed',
            'sten_proc_6_prox_lad_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_6_prox_lad_transformed')

        # lca lad midt->1.0
        self.processor(
            'sten_proc_8_dist_lad_transformed',
            'sten_proc_7_midt_lad_transformed')
        self.processor(
            'sten_proc_10_d2_transformed',
            'sten_proc_7_midt_lad_transformed')

        # lca lcx prox->1.0
        self.processor(
            'sten_proc_12_om1_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_13_midt_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_11_prox_lcx_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_11_prox_lcx_transformed')

        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_11_prox_lcx_transformed')
        # lca lcx midt->1.0
        self.processor(
            'sten_proc_14_om2_transformed',
            'sten_proc_13_midt_lcx_transformed')
        self.processor(
            'sten_proc_15_dist_lcx_transformed',
            'sten_proc_13_midt_lcx_transformed')

        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_13_midt_lcx_transformed')

        # lca lcx dist->1.0

        self.processor(
            'sten_proc_16_pla_rca_transformed',
            'sten_proc_15_dist_lcx_transformed')
        return

