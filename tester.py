import torch
from dataloader.get_dataloader import get_dataloader_test
from configs.options import TestOptions
from metrics.metrics_utils import init_metrics, increment_metrics, \
     normalize_metrics
from model_utils.get_loss_func import get_loss_func
from model_utils.get_test_pipeline import TestPipeline
from configs.config import load_config
from trainer import get_device
from models.BuildModel import ModelBuilder
import os


def read_log_file(config):
    path = open(config['logfile'], "r").read()
    config = load_config(os.path.join(path, 'config.yaml'))
    config['model']['pretrain_model'] = path
    return config


def main():
    config = vars(TestOptions().parse())
   # config = load_config(config)
    config = read_log_file(config)
    config['loaders']['mode'] = 'testing'
    # torch.distributed.init_process_group(
    #     backend="nccl" if config["cpu"] == "False" else "Gloo")
    device = get_device(config)

    BuildModel = ModelBuilder(config)
    model = BuildModel()
    model.to(device)

    # Get data loader
    test_loader = get_dataloader_test(config)

    # Get loss func
    criterion = get_loss_func(config)

    running_metric_test, config['eval_metric_val']['name'] = \
        init_metrics(config['eval_metric_val']['name'], config)
    running_loss_test, _ = init_metrics(config['loss']['name'],
                                        config,
                                        mode='loss')
    pipeline = TestPipeline()
    pipeline.get_test_pipeline(model, criterion, config, test_loader,
                               device, init_metrics, increment_metrics,
                               normalize_metrics,
                               running_metric_test, running_loss_test)

if __name__ == '__main__':
    main()
