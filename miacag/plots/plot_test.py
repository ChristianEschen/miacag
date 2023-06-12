from miacag.plots.plotter import make_plots, make_plots_compare_max_mean
if __name__ == '__main__':
    path_rca_bce = "/home/alatar/miacag/output/outputs_stenosis_identi/classification_config_angio_SEP_Jan21_15-32-06/plots/train/probas_trues.csv"
    path_lca_bce = "/home/alatar/miacag/output/outputs_stenosis_identi/classification_config_angio_SEP_Jan21_15-37-00/plots/train/probas_trues.csv"
    path_lca = "/home/alatar/miacag/output/outputs_stenosis_reg/classification_config_angio_SEP_Jan21_15-19-24/plots/train/probas_trues.csv"
    path_rca = "/home/alatar/miacag/output/outputs_stenosis_reg/classification_config_angio_SEP_Jan21_15-38-35/plots/train/probas_trues.csv"
    output_path = "/home/alatar/miacag/output/outputs_stenosis_identi/classification_config_angio_SEP_Jan21_15-32-06/plots/comb"
    output_path_comb = "/home/alatar/miacag/output/outputs_stenosis_identi/classification_config_angio_SEP_Jan21_15-32-06/plots/comb_mean_max"
    make_plots(path_rca_bce, path_lca_bce, path_rca, path_lca, output_path)
    
    make_plots_compare_max_mean(path_rca_bce, path_lca_bce, path_rca, path_lca,
                                path_rca_bce, path_lca_bce, path_rca, path_lca,
                                output_path_comb)
    