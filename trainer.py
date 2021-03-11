from dataloader.get_dataloader import get_dataloader_train
import torch
from model_utils.get_model import get_model
from configs.config import load_config
from torch.utils.tensorboard import SummaryWriter
from model_utils.get_optimizer import get_optimizer
from model_utils.get_loss_func import get_loss_func
from configs.options import BaseOptions
import os
from model_utils.train_utils import set_random_seeds, train_one_epoch
from metrics.metrics_utils import init_metrics, flatten, \
    unroll_list_in_dict
from model_utils.eval_utils import val_one_epoch


def main():
    config = BaseOptions().parse()
    config = load_config(config)
    writer = SummaryWriter(comment="_" + config['tensorboard_comment'])

    model_file_path = os.path.join(writer.log_dir, 'model.pt')

    set_random_seeds(random_seed=config['manual_seed'])
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:{}".format(config['local_rank']))

    model = get_model(config)
    model.to(device)

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
                                             config['model']['classes'],
                                             mode='loss')
        running_metric_train, config['eval_metric']['name'] = \
            init_metrics(config['eval_metric']['name'],
                         config['model']['classes'])
        running_loss_val, _ = init_metrics(config['loss']['name'],
                                           config['model']['classes'],
                                           mode='loss')
        running_metric_val, config['eval_metric']['name'] = \
            init_metrics(config['eval_metric']['name'],
                         config['model']['classes'])

        #  validation one epoch (but not necessarily each)
        if epoch % config['trainer']['validate_frequency'] == 0:
            metric_dict_val = val_one_epoch(model, criterion, config,
                                            val_loader, device,
                                            running_metric_val,
                                            running_loss_val, writer, epoch)

        # train one epoch
        train_one_epoch(model, criterion,
                        train_loader, device, epoch,
                        optimizer, lr_scheduler, config['eval_metric']['name'],
                        running_metric_train, running_loss_train,
                        writer, config, scaler)

    config = unroll_list_in_dict(flatten(config))
    metric_dict_val = {str(key)+'/val': val
                       for key, val in metric_dict_val.items()}

    writer.add_hparams(config, metric_dict=metric_dict_val)
    writer.flush()
    writer.close()
    torch.save(model.state_dict(),
               model_file_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
