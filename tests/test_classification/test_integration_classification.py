import os
import socket
from trainer import main
import argparse
import sys
import shutil
try:
    # python 3.4+ should use builtin unittest.mock not mock package
    from unittest.mock import patch
except ImportError:
    from mock import patch
# def test_pipeline():
#     os.chdir(r"tests/test_classification")
#     os.system("snakemake --cores 2")
#     return None


def test_classification_angio_3d():
    logfile = 'logfile.txt'
    testargs = [
        sys.argv[0],
        "--ValdataRoot", "data/angio/minc_file_path/",
        "--ValdataCSV", "data/angio/val.csv",
        '--logfile', logfile,
        '--TraindataRoot', "data/angio/minc_file_path/",
        '--TraindataCSV', "data/angio/train.csv",
        '--config',
        'tests/test_classification/configs/config_3d_angio.yaml',
        '--cpu', 'True',
        '--use_DDP', 'False']

    with patch.object(sys, 'argv', testargs):
        main()
    assert os.path.isfile(logfile)
    tensorboard_dir = open(logfile, "r").read()
    assert os.path.isdir(tensorboard_dir)
    os.remove(logfile)
    shutil.rmtree(tensorboard_dir)
    return None
