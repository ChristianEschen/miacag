from miacag.dataloader.get_dataloader import get_dataloader_train
import torch
from miacag.models.BuildModel import ModelBuilder
from miacag.configs.config import load_config, maybe_create_tensorboard_logdir
from torch.utils.tensorboard import SummaryWriter
from miacag.model_utils.get_optimizer import get_optimizer
from miacag.model_utils.get_loss_func import get_loss_func
from miacag.configs.options import TrainOptions
from miacag.model_utils.train_utils import set_random_seeds, \
    train_one_epoch, early_stopping, \
    get_device, saver, save_model
from miacag.metrics.metrics_utils import init_metrics
from miacag.model_utils.eval_utils import val_one_epoch
import time
import torch.distributed as dist
from monai.utils import set_determinism
from miacag.models.modules import get_loss_names_groups
import gc   

def should_validate(last_validation_time, val_frequency=30):
    current_time = time.time()
    elapsed_since_last_validation = current_time - last_validation_time
    if elapsed_since_last_validation >= val_frequency:  # 5 hours
        return 1
    else:
        return 0


def train(config):
    print('training')
    config['loaders']['mode'] = 'training'
    set_random_seeds(random_seed=config['manual_seed'])
    set_determinism(seed=config['manual_seed'])

    device = get_device(config)

    if torch.distributed.get_rank() == 0:
        writer = SummaryWriter(config['output_directory'])
    else:
        writer = False
    if config["cpu"] == "False":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    # Get metrics
    config['loss']['groups_names'], config['loss']['groups_counts'], \
        config['loss']['group_idx'], config['groups_weights'] \
        = get_loss_names_groups(config)
    running_loss_train = init_metrics(config['loss']['name'], config, device,
                                      ptype='loss')
    running_metric_train = init_metrics(
            config['eval_metric_train']['name'], config, device)
    running_loss_val = init_metrics(config['loss']['name'], config, device,
                                    ptype='loss')
    running_metric_val = init_metrics(
                config['eval_metric_val']['name'], config, device)

    # Get model
    BuildModel = ModelBuilder(config, device)
    model = BuildModel()

    # Get data loaders
    train_loader, val_loader, train_ds, _ = get_dataloader_train(config)

    # Get loss func
    criterion_train = get_loss_func(config, train=True)
    criterion_val = get_loss_func(config, train=False)
    # Get optimizer
    optimizer, lr_scheduler = get_optimizer(config,
                                            model,
                                            len(train_loader))

    # Use AMP for speedup:
    scaler = torch.cuda.amp.GradScaler() \
        if config['loaders']['use_amp'] else None
    best_val_loss, best_val_epoch = None, None
    early_stop = False
    if config['cache_num'] not in ['standard', 'None']:
        train_ds.start()

    starter = time.time()
    last_validation_time = starter  # Last validation time

    #  ---- Start training loop ----#
    iter_minibatch = 0
    
    for epoch in range(0, config['trainer']['epochs']):
        # print only every 100 epochs
        #if epoch % 100 == 0:
        print('epoch nr', epoch)

        # Check elapsed time since last validation
        if dist.get_rank() == 0:
            validate_now = should_validate(last_validation_time,
                                           config['trainer']['validate_frequency'])
        else:
            validate_now = 0

        # Broadcast decision to all nodes
        validate_now = torch.tensor(validate_now).to(device)

        dist.broadcast(validate_now, src=0)
        validate_now = validate_now.item()  # Convert back to Python bool

        # validate every 10 epochs if epoch % 10 == 0:
        if epoch == 0 or validate_now:
            metric_dict_val = val_one_epoch(model, criterion_val, config,
                                            val_loader, device,
                                            running_metric_val,
                                            running_loss_val, writer, epoch)
            # early stopping
            early_stop, best_val_loss, best_val_epoch = early_stopping(
                best_val_loss, best_val_epoch,
                metric_dict_val['total'],
                epoch, config['trainer']['max_stagnation'])
            config['best_val_epoch'] = best_val_epoch
            # save model
            if best_val_epoch == epoch:
                if torch.distributed.get_rank() == 0:
                    save_model(model, writer, config)
            last_validation_time = time.time()  # Reset last validation time
            if early_stop is True:
                break
        # train one epoch
        iter_minibatch = train_one_epoch(model, criterion_train,
                        train_loader, device, epoch,
                        optimizer, lr_scheduler,
                        running_metric_train, running_loss_train,
                        writer, config, scaler, iter_minibatch)

        if config['cache_num'] not in  ['standard', 'None']:
            train_ds.update_cache()
           # train_ds.set_data()

    if config['cache_num'] not in ['standard', 'None']:
        train_ds.shutdown()
    print('Finished Training')
    if torch.distributed.get_rank() == 0:
        saver(metric_dict_val, writer, config)

    print('training loop (s)', time.time()-starter)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    


if __name__ == '__main__':
    config = vars(TrainOptions().parse())
    config = load_config(config['config'], config)
    config = maybe_create_tensorboard_logdir(config)
    torch.distributed.init_process_group(
            backend="nccl" if config["cpu"] == "False" else "Gloo",
            init_method="env://"
            )
    train(config)
