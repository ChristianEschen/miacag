from dataloader.get_dataloader import get_dataloader_train
import torch
from models.BuildModel import ModelBuilder
from configs.config import load_config, maybe_create_tensorboard_logdir
from torch.utils.tensorboard import SummaryWriter
from model_utils.get_optimizer import get_optimizer
from model_utils.get_loss_func import get_loss_func
from configs.options import TrainOptions
from model_utils.train_utils import set_random_seeds, train_one_epoch, early_stopping, \
    get_device, saver, save_model
from metrics.metrics_utils import init_metrics
from model_utils.eval_utils import val_one_epoch
import time
import torch.distributed as dist
from monai.utils import set_determinism
import os


def main():
    config = vars(TrainOptions().parse())
    config = load_config(config['config'], config)
    config['loaders']['mode'] = 'training'
    config = maybe_create_tensorboard_logdir(config)

    set_random_seeds(random_seed=config['manual_seed'])
    set_determinism(seed=config['manual_seed'])
    if config['use_DDP'] == 'True':
        torch.distributed.init_process_group(
            backend="nccl" if config["cpu"] == "False" else "Gloo",
            init_method="env://"
            )

    device = get_device(config)

    if torch.distributed.get_rank() == 0:
        writer = SummaryWriter(config['output_directory'])
    else:
        writer = False
    if config["cpu"] == "False":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    BuildModel = ModelBuilder(config, device)
    model = BuildModel()
    
    

    # Get data loaders
    train_loader, val_loader, train_ds, _ = get_dataloader_train(config)

    # Get loss func
    criterion = get_loss_func(config)
    # Get optimizer
    optimizer, lr_scheduler = get_optimizer(config,
                                            model,
                                            len(train_loader))

    # Use AMP for speedup:
    scaler = torch.cuda.amp.GradScaler() \
        if config['loaders']['use_amp'] else None
    best_val_loss, best_val_epoch = None, None
    early_stop = False
    if config['cache_num'] != 'None':
        train_ds.start()

    starter = time.time()

    # running_metric_train = init_metrics(
    #     config['eval_metric_train']['name'])
    # running_metric_val = init_metrics(
    #     config['eval_metric_val']['name'])
    running_loss_train = init_metrics(config['loss']['name'])
    running_metric_train = init_metrics(
            config['eval_metric_train']['name'])
    running_loss_val = init_metrics(config['loss']['name'])
    running_metric_val = init_metrics(
                config['eval_metric_val']['name'])

    #  ---- Start training loop ----#
    for epoch in range(0, config['trainer']['epochs']):
        print('epoch nr', epoch)
         # train one epoch
        start = time.time()
        
        train_one_epoch(model, criterion,
                        train_loader, device, epoch,
                        optimizer, lr_scheduler,
                        running_metric_train, running_loss_train,
                        writer, config, scaler)
                        
        #  validation one epoch (but not necessarily each)
        if config['cache_num'] != 'None':
            train_ds.update_cache()

        if epoch % config['trainer']['validate_frequency'] == 0:
            metric_dict_val = val_one_epoch(model, criterion, config,
                                            val_loader, device,
                                            running_metric_val,
                                            running_loss_val, writer, epoch)
            # early stopping
            early_stop, best_val_loss, best_val_epoch = early_stopping(
                best_val_loss, best_val_epoch,
                metric_dict_val['CE'],
                epoch, config['trainer']['max_stagnation'])
            config['best_val_epoch'] = best_val_epoch
            # save model
            if best_val_epoch == epoch:
                if torch.distributed.get_rank() == 0:
                    save_model(model, writer, config)
            if early_stop is True:
                break

    if config['cache_num'] != 'None':
        train_ds.shutdown()

    if early_stop is False:
        if torch.distributed.get_rank() == 0:
            save_model(model, writer, config)
    if torch.distributed.get_rank() == 0:
        saver(metric_dict_val, writer, config)
    print('Finished Training')
    print('training loop (s)', time.time()-starter)
    dist.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
