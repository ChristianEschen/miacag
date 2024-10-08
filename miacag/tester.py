import torch
from miacag.dataloader.get_dataloader import get_dataloader_test
from miacag.configs.options import TestOptions
from miacag.metrics.metrics_utils import init_metrics, normalize_metrics
from miacag.model_utils.get_loss_func import get_loss_func
from miacag.model_utils.get_test_pipeline import TestPipeline
from miacag.configs.config import load_config
from miacag.trainer import get_device
from miacag.models.BuildModel import ModelBuilder
import os
from miacag.model_utils.train_utils import set_random_seeds
from miacag.models.modules import get_loss_names_groups


def read_log_file(config_input):
    config = load_config(
        os.path.join(config_input['output_directory'], 'config.yaml'))
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
    # if config['loaders']['val_method']['saliency'] == 'False':
    config['loaders']['val_method']["samples"] = 1
    config['loaders']['batchSize'] = 1
    if config["task_type"] == "mil_classification":
        config['loaders']['val_method']["samples"] = 1

    set_random_seeds(random_seed=config['manual_seed'])

    device = get_device(config)

    if config["cpu"] == "False":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    config['loss']['groups_names'], config['loss']['groups_counts'], \
        config['loss']['group_idx'], config['groups_weights'] \
        = get_loss_names_groups(config)
        
    # Get data loader
    test_loader, config = get_dataloader_test(config)

    BuildModel = ModelBuilder(config, device)
    model = BuildModel()
    if config['use_DDP'] == 'True':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if config["cpu"] == "False" else None)

    # Get loss func
    criterion = get_loss_func(config)
    config['loss']['name'] = config['loss']['name'] + ['total']
    running_loss_test = init_metrics(config['loss']['name'],
                                     config,
                                     device,
                                     ptype='loss')
    running_metric_test = init_metrics(
                config['eval_metric_val']['name'],
                config,
                device)

    pipeline = TestPipeline()
    import time
    start = time.time()
    pipeline.get_test_pipeline(model, criterion, config, test_loader,
                               device, init_metrics,
                               normalize_metrics,
                               running_metric_test, running_loss_test)
    print('time elapsed:', time.time() - start)


if __name__ == '__main__':
    config = vars(TestOptions().parse())
    config = read_log_file(config)
    if config['use_DDP'] == 'True':
        torch.distributed.init_process_group(
            backend="nccl" if config["cpu"] == "False" else "Gloo",
            init_method="env://")
    test(config)
