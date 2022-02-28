import torch
from mia.dataloader.get_dataloader import get_dataloader_test
from mia.configs.options import TestOptions
from mia.metrics.metrics_utils import init_metrics, normalize_metrics
from mia.model_utils.get_loss_func import get_loss_func
from mia.model_utils.get_test_pipeline import TestPipeline
from mia.configs.config import load_config
from mia.trainer import get_device
from mia.models.BuildModel import ModelBuilder
import os
from mia.model_utils.train_utils import set_random_seeds


def read_log_file(config_input):
    config = load_config(
        os.path.join(config_input['output_directory'], 'config.yaml'))
    # config = load_config(config_train_out['config'])
    # config.update(config_input)
    config['model']['pretrain_model'] = config['output_directory']
    return config


def convert_string_to_tuple(field):
    res = []
    temp = []
    for token in field.split(", "):
        num = int(token.replace("(", "").replace(")", ""))
        temp.append(num)
        if ")" in token:
            res.append(tuple(temp))
            temp = []
    return res[0]

def test(config):
    config['loaders']['mode'] = 'testing'
    if config['loaders']['val_method']['saliency'] == 'False':
        config['loaders']['val_method']["samples"] = 2
    set_random_seeds(random_seed=config['manual_seed'])
    
    device = get_device(config)

    if config["cpu"] == "False":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    BuildModel = ModelBuilder(config, device)
    #if torch.distributed.get_rank() == 0:
    model = BuildModel()
    if config['use_DDP'] == 'True':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if config["cpu"] == "False" else None)
    # Get data loader
    test_loader = get_dataloader_test(config)

    # Get loss func
    criterion = get_loss_func(config)
    running_loss_test = init_metrics(config['loss']['name'])
    running_metric_test = init_metrics(
                config['eval_metric_val']['name'])


    # running_metric_test, config['eval_metric_val']['name'] = \
    #     init_metrics(config['eval_metric_val']['name'], config)
    # running_loss_test, _ = init_metrics(config['loss']['name'],
    #                                     config,
    #                                     mode='loss')
    pipeline = TestPipeline()
    pipeline.get_test_pipeline(model, criterion, config, test_loader,
                               device, init_metrics,
                               normalize_metrics,
                               running_metric_test, running_loss_test)


if __name__ == '__main__':
    config = vars(TestOptions().parse())
    config = read_log_file(config)
    if config['use_DDP'] == 'True':
        torch.distributed.init_process_group(
            backend="nccl" if config["cpu"] == "False" else "Gloo",
            init_method="env://")
    test(config)
