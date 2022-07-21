import torch
from miacag.dataloader.get_dataloader import get_dataloader_test
from miacag.configs.options import TestOptions
from miacag.metrics.metrics_utils import init_metrics, normalize_metrics
from miacag.model_utils.get_loss_func import get_loss_func
from miacag.model_utils.predict_utils import Predictor
from miacag.configs.config import load_config
from miacag.trainer import get_device
from miacag.models.BuildModel import ModelBuilder
import os
from miacag.model_utils.train_utils import set_random_seeds
from miacag.tester import read_log_file, convert_string_to_tuple
import shutil


def maybe_remove_persistence_data(config):
    cachDir = os.path.join(
        config['model']['pretrain_model'],
        'persistent_cache')
    if torch.distributed.get_rank() == 0:
        if os.path.exists(cachDir):
            shutil.rmtree(cachDir)


def pred(config, model_path):
    config['loaders']['mode'] = 'prediction'

    if config['loaders']['val_method']['saliency'] == 'False':
        config['loaders']['val_method']["samples"] = 10
    if config['task_type'] == 'mil_classification':
        config['loaders']['batchSize'] = 1
        config['loaders']['val_method']["samples"] = 10
    else:
        config['loaders']['val_method']["samples"] = 1

    if config['loaders']['val_method']['saliency'] == 'True':
        config['loaders']['batchSize'] = 1
        config['loaders']['val_method']["samples"] = 1
    set_random_seeds(random_seed=config['manual_seed'])

    device = get_device(config)

    if config["cpu"] == "False":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    config['model']['pretrain_model'] = model_path
    maybe_remove_persistence_data(config)

    BuildModel = ModelBuilder(config, device)
    model = BuildModel()
    if config['use_DDP'] == 'True':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if config["cpu"] == "False" else None)
    # Get data loader
    test_loader = get_dataloader_test(config)

    # Get loss func
    criterion = get_loss_func(config)

    pipeline = Predictor(model, criterion, config, device, test_loader)
    pipeline.get_predictor_pipeline()


if __name__ == '__main__':
    config = vars(TestOptions().parse())
    config = read_log_file(config)
    if config['use_DDP'] == 'True':
        torch.distributed.init_process_group(
            backend="nccl" if config["cpu"] == "False" else "Gloo",
            init_method="env://")
    pred(config)
