from dataloader.get_dataloader import get_dataloader_train
import torch
from models.BuildModel import ModelBuilder
from configs.config import load_config
from torch.utils.tensorboard import SummaryWriter
from model_utils.get_optimizer import get_optimizer
from model_utils.get_loss_func import get_loss_func
from configs.options import TrainOptions
import os
from model_utils.train_utils import set_random_seeds, train_one_epoch
from metrics.metrics_utils import init_metrics, flatten, \
    unroll_list_in_dict
from model_utils.eval_utils import val_one_epoch


def get_device(config):
    if config["cpu"] == "False":
        if config['loaders']['mode'] == 'training':
            device = "cuda:{}".format(config['local_rank'])
        else:
            device = "cuda:0"
        device = "cpu"
    device = torch.device(device)
    return device


def main():
    config = TrainOptions().parse()
    config = load_config(config)
    writer = SummaryWriter(comment="_" + config['tensorboard_comment'])

    set_random_seeds(random_seed=config['manual_seed'])
    torch.distributed.init_process_group(
        backend="nccl" if config["cpu"] == "False" else "Gloo")
    device = get_device(config)

    BuildModel = ModelBuilder(config)
    model = BuildModel()
    model.to(device)

    if config['cpu'] == "False":
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[config['local_rank']],
            output_device=config['local_rank'],
            find_unused_parameters=True)

    # Get data loaders
    train_loader, val_loader = get_dataloader_train(config)

    # Get loss func
    criterion = get_loss_func(config)
    # Get optimizer
    optimizer, lr_scheduler = get_optimizer(config,
                                            model,
                                            len(train_loader))

    # Use AMP for speedup:
    scaler = torch.cuda.amp.GradScaler() \
        if config['loaders']['use_amp'] else None
#  ---- Start training loop ----#
    for epoch in range(0, config['trainer']['epochs']):
        print('epoch nr', epoch)
        running_loss_train, _ = init_metrics(config['loss']['name'],
                                             config,
                                             mode='loss')
        running_metric_train, config['eval_metric_train']['name'] = \
            init_metrics(config['eval_metric_train']['name'],
                         config)
        running_loss_val, _ = init_metrics(config['loss']['name'],
                                           config,
                                           mode='loss')
        running_metric_val, config['eval_metric_val']['name'] = \
            init_metrics(config['eval_metric_val']['name'],
                         config)

        #  validation one epoch (but not necessarily each)
        if epoch % config['trainer']['validate_frequency'] == 0:
            metric_dict_val = val_one_epoch(model, criterion, config,
                                            val_loader, device,
                                            running_metric_val,
                                            running_loss_val, writer, epoch)

        # train one epoch
        train_one_epoch(model, criterion,
                        train_loader, device, epoch,
                        optimizer, lr_scheduler,
                        running_metric_train, running_loss_train,
                        writer, config, scaler)

    config = unroll_list_in_dict(flatten(config))
    metric_dict_val = {str(key)+'/val': val
                       for key, val in metric_dict_val.items()}

    writer.add_hparams(config, metric_dict=metric_dict_val)
    writer.flush()
    writer.close()
    model_file_path = os.path.join(writer.log_dir, 'model.pt')
    torch.save(model.module.state_dict(), model_file_path)

    if config["task_type"] in ["representation_learning"]:
        model_encoder_file_path = os.path.join(
            writer.log_dir, 'model_encoder.pt')
        torch.save(model.module.encoder.state_dict(),
                   model_encoder_file_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
