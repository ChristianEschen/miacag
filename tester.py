import torch
from dataloader.get_dataloader import get_dataloader_test
from configs.options import BaseOptions
from metrics.metrics_utils import init_metrics, increment_metrics, \
     normalize_metrics
from models.image2scalar_utils.utils_3D.test_utils import build_model
from model_utils.get_loss_func import get_loss_func
from model_utils.get_test_pipeline import TestPipeline
from configs.config import load_config


def main():
    config = BaseOptions().parse()
    config = load_config(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config, device)

    # Get data loader
    test_loader = get_dataloader_test(config)

    # Get loss func
    criterion = get_loss_func(config)

    running_metric_test, config['eval_metric']['name'] = \
        init_metrics(config['eval_metric']['name'],
                     config['model']['classes'])
    running_loss_test, _ = init_metrics(config['loss']['name'],
                                        config['model']['classes'],
                                        mode='loss')
    pipeline = TestPipeline()
    pipeline.get_test_pipeline(model, criterion, config, test_loader,
                               device, init_metrics, increment_metrics,
                               normalize_metrics,
                               running_metric_test, running_loss_test)


if __name__ == '__main__':
    main()
