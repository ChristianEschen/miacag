import os

def checkExpExists(exp, output_folder):
    exp_exists = False
    runned_exps = os.listdir(output_folder)
    exp_basename = os.path.basename(exp)
    for runned_exp in runned_exps:
        if exp_basename in runned_exp:
            test_plot = os.path.join(
                output_folder,
                runned_exp,
                "plots/test/roc.pdf")
            file_exists = os.path.exists(test_plot)
            if file_exists:
                exp_exists = True
    return exp_exists


if __name__ == '__main__':
    exp_folder = "/home/sauroman/angiography_data/runs/test"
    output_folder = "classification_config_angio_2"
    exp_name = "DILLERRERE"
    checkExpExists(exp_folder, exp_name)
