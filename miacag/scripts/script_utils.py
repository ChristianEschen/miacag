from typing import Any
import copy
from miacag.model_utils.get_loss_func import find_matches_two_lists
import re
import os
from miacag.utils.script_utils import mkFolder
class ConfigManipulator():
    
    def __init__(self, config, task_index):
        self.config = copy.deepcopy(config)
       
        self.task_index = task_index

    def get_correct_index(self, index):
        self.indexes = [i for i, e in enumerate(self.config['task_indicator']) if e == self.task_index]
        
    
    def seperate_lca_and_rca(self):
        self.config_copy_lca  = copy.deepcopy(self.config)
        self.config_copy_rca  = copy.deepcopy(self.config)
        identify_cor_artery_rca = ['_1_prox',  '_2_mi' , '_3_dist', '_4_pda_transformed', '_16_pla_rca']
        identify_cor_artery_lca = [ '_4_pda_lca', '_5_lm', '_6_prox', '_7_mi', '_8_dist',
                            '_9_d1', '_10_d2', '_11_prox', '_12_om', '_13_midt', '_14_om', '_15_dist', '_16_pla_lca']


        idx_lca = find_matches_two_lists(self.config['labels_names'], identify_cor_artery_lca)

        idx_rca = find_matches_two_lists(self.config['labels_names'], identify_cor_artery_rca)
        self.config_copy_rca["labels_names"] = [self.config_copy_rca["labels_names"][i] for i in idx_rca]
        self.config_copy_lca["labels_names"] = [self.config_copy_lca["labels_names"][i] for i in idx_lca]
        self.config_copy_lca["artery_type"] = "lca"
        self.config_copy_rca["artery_type"] = "rca"
        
        # losses
        self.config_copy_rca["loss"]["name"] = [self.config_copy_rca["loss"]["name"][i] for i in idx_rca]
        self.config_copy_lca["loss"]["name"] = [self.config_copy_lca["loss"]["name"][i] for i in idx_lca]
        
        # eval metrics
        self.config_copy_rca["eval_metric_train"]["name"] = [self.config_copy_rca["eval_metric_train"]["name"][i] for i in idx_rca]
        self.config_copy_lca["eval_metric_train"]["name"] = [self.config_copy_lca["eval_metric_train"]["name"][i] for i in idx_lca]
        
        self.config_copy_rca["eval_metric_val"]["name"] = [self.config_copy_rca["eval_metric_val"]["name"][i] for i in idx_rca]
        self.config_copy_lca["eval_metric_val"]["name"] = [self.config_copy_lca["eval_metric_val"]["name"][i] for i in idx_lca]

        self.config_copy_rca["model"]["num_classes"] = [self.config_copy_rca["model"]["num_classes"][i] for i in idx_rca]
        self.config_copy_lca["model"]["num_classes"] = [self.config_copy_lca["model"]["num_classes"][i] for i in idx_lca]

        
        self.config_copy_rca["output_directory"] = os.path.join(self.config_copy_rca["output_directory"], "rca")
        self.config_copy_lca["output_directory"] = os.path.join(self.config_copy_lca["output_directory"], "lca")
        # add path for rca and lca outputs
        # make output directories for rca and lca if not exist
        mkFolder(self.config_copy_rca["output_directory"])
        mkFolder(self.config_copy_lca["output_directory"])
        return [self.config_copy_lca, self.config_copy_rca]
    
    def change_query_lca_rca_train_test(self, config, idx):
        # this is train
        if config["artery_type"] == "rca": # rca
            config["query"] = config["query_rca"]
        elif config["artery_type"] == "lca": # lca
            config["query"] =config["query_lca"] 
        # test
        if config["artery_type"] == "rca": # rca
            config["query_test"] = config["query_rca_test"]
        elif config["artery_type"] == "lca": # lca
            config["query_test"] =config["query_lca_test"]
        return config
    
        
    def replace_substring(self, test_str, s1, s2):
        # Escaping parentheses in the strings for regex
        s1_escaped = re.escape(s1)
        s2_escaped = re.escape(s2)
        
        # Replacing all occurrences of substring s1 with s2
        test_str = re.sub(s1_escaped, s2, test_str)  # Use s2 as it is, assuming you don't need to escape it
        return test_str


    
    def manipulate_lca_rca(self, artey_type):
        # if artery_type = lca
        # elif artery_type = rca
        return None
        
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        #self.config_list = []
        if self.config['labels_names'][0].startswith('sten_') or self.config['labels_names'][0].startswith('ffr_') or self.config['labels_names'][0].startswith('timi_'):

            self.config_list = self.seperate_lca_and_rca()
            for i in range(len(self.config_list)):
                self.config_list[i] = self.change_query_lca_rca_train_test(self.config_list[i], i)
               # self.config_list[i] = self.change_query_lca_rca_plotting(self.config_list[i])
             #   self.config_list[i] = self.change_query_plotting(self.config_list[i])
        else:
            self.config_list = []
            self.config_list.append(copy.deepcopy(self.config))
            #self.config_list.append(self.change_query_plotting(config_copy))
        return self.config_list
    